"""Versioned monotone calibration policy for Phase 5.4.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.ml.backtesting import (
    BacktestError,
    BacktestFold,
    assert_calibration_ranking_preserved,
    build_fold_partitions,
)
from src.ml.contract import TARGET_COLUMN
from src.ml.enriched import PHASE54_CUTOFF_TIMESTAMP


CALIBRATION_POLICY_VERSION = "monotone_temporal_v1.0.0"
MIN_ISOTONIC_LEVELS = 2
MAX_ISOTONIC_LARGEST_TIE_FRACTION = 0.95


@dataclass(frozen=True)
class PolicyFoldPartitions:
    """Four disjoint temporal partitions for fit, calibration, threshold and test."""

    fold: BacktestFold
    train: pd.DataFrame
    calibration_fit: pd.DataFrame
    threshold_selection: pd.DataFrame
    evaluation: pd.DataFrame
    calibration_split_at: pd.Timestamp


@dataclass(frozen=True)
class CalibrationPolicyResult:
    """Selected estimator plus serializable diagnostics and status."""

    estimator: Any
    selected_method: str
    calibration_status: str
    diagnostics: dict[str, Any]


def build_policy_fold_partitions(
    dataset: pd.DataFrame,
    fold: BacktestFold,
) -> PolicyFoldPartitions:
    """Split the historical calibration window into fit and threshold months."""

    base = build_fold_partitions(dataset, fold)
    split_at = fold.calibration_start + pd.DateOffset(months=1)
    if not fold.calibration_start < split_at < fold.evaluation_start:
        raise BacktestError("Fenêtre de calibration impossible à scinder.")

    purchase = pd.to_datetime(dataset["purchase_timestamp"], errors="coerce")
    observed = pd.to_datetime(dataset["label_observed_at"], errors="coerce")
    calibration_fit = dataset.loc[
        purchase.ge(fold.calibration_start)
        & purchase.lt(split_at)
        & observed.lt(split_at)
    ].copy()
    threshold_selection = dataset.loc[
        purchase.ge(split_at)
        & purchase.lt(fold.evaluation_start)
        & observed.lt(fold.evaluation_start)
    ].copy()
    partitions = {
        "train": base.train,
        "calibration_fit": calibration_fit.reset_index(drop=True),
        "threshold_selection": threshold_selection.reset_index(drop=True),
        "evaluation": base.evaluation,
    }
    for name, frame in partitions.items():
        if frame.empty or set(frame[TARGET_COLUMN].astype(int).unique()) != {0, 1}:
            raise BacktestError(f"Partition 5.4.2 {name} insuffisante pour {fold.name}.")
        if frame["order_id"].duplicated().any():
            raise BacktestError(f"Doublon dans la partition 5.4.2 {name}.")
    identifiers = {
        name: set(frame["order_id"].astype(str))
        for name, frame in partitions.items()
    }
    names = tuple(identifiers)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if identifiers[left] & identifiers[right]:
                raise BacktestError(f"Chevauchement 5.4.2 entre {left} et {right}.")
    if not partitions["train"]["label_observed_at"].lt(
        fold.training_as_of
    ).all():
        raise BacktestError("Label train immature dans le protocole 5.4.2.")
    if not partitions["calibration_fit"]["label_observed_at"].lt(split_at).all():
        raise BacktestError("Label calibration immature dans le protocole 5.4.2.")
    if not partitions["threshold_selection"]["label_observed_at"].lt(
        fold.evaluation_start
    ).all():
        raise BacktestError("Label de seuil immature dans le protocole 5.4.2.")
    if purchase.ge(PHASE54_CUTOFF_TIMESTAMP).any():
        raise BacktestError("Le protocole 5.4.2 contient le holdout final.")

    return PolicyFoldPartitions(
        fold=fold,
        train=partitions["train"],
        calibration_fit=partitions["calibration_fit"],
        threshold_selection=partitions["threshold_selection"],
        evaluation=partitions["evaluation"],
        calibration_split_at=split_at,
    )


def ranking_diagnostics(
    raw_probability: np.ndarray,
    calibrated_probability: np.ndarray,
) -> dict[str, Any]:
    """Describe monotonicity and ties without using evaluation labels."""

    raw = np.asarray(raw_probability, dtype=np.float64)
    calibrated = np.asarray(calibrated_probability, dtype=np.float64)
    if raw.ndim != 1 or raw.size == 0 or calibrated.shape != raw.shape:
        raise BacktestError("Scores incompatibles pour le diagnostic 5.4.2.")
    if not np.isfinite(raw).all() or not np.isfinite(calibrated).all():
        raise BacktestError("Score non fini dans le diagnostic 5.4.2.")

    order = np.argsort(raw, kind="stable")
    sorted_raw = raw[order]
    sorted_calibrated = calibrated[order]
    distinct = np.diff(sorted_raw) > 1e-12
    reversed_boundaries = int(
        np.sum(distinct & (np.diff(sorted_calibrated) < -1e-12))
    )
    values, counts = np.unique(calibrated, return_counts=True)
    unique_levels = int(len(values))
    largest_tie_fraction = float(counts.max() / len(calibrated))
    if np.unique(raw).size > 1 and unique_levels > 1:
        spearman = float(spearmanr(raw, calibrated).statistic)
        kendall = float(kendalltau(raw, calibrated).statistic)
    else:
        spearman = None
        kendall = None
    return {
        "rows": int(len(raw)),
        "rank_preserved": reversed_boundaries == 0,
        "reversed_boundaries": reversed_boundaries,
        "spearman": spearman,
        "kendall": kendall,
        "raw_unique_scores": int(np.unique(raw).size),
        "calibrated_unique_scores": unique_levels,
        "calibrated_tied_rows": int(len(calibrated) - unique_levels),
        "largest_calibrated_tie_fraction": largest_tie_fraction,
    }


def fit_monotone_calibration_policy(
    base_pipeline: Any,
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_calibration_fit: pd.DataFrame,
    y_calibration_fit: pd.Series,
) -> CalibrationPolicyResult:
    """Apply the frozen sigmoid/isotonic/raw hierarchy before evaluation."""

    base_pipeline.fit(x_train, y_train)
    if not np.array_equal(np.asarray(base_pipeline.classes_), np.array([0, 1])):
        raise BacktestError("La classe positive de retard doit être encodée 1.")
    raw_probability = base_pipeline.predict_proba(x_calibration_fit)[:, 1]
    attempts: dict[str, dict[str, Any]] = {}
    estimators: dict[str, Any] = {"raw": base_pipeline}

    for method in ("sigmoid", "isotonic"):
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(base_pipeline),
            method=method,
            ensemble=False,
        )
        calibrated.fit(x_calibration_fit, y_calibration_fit)
        calibrated_probability = calibrated.predict_proba(x_calibration_fit)[:, 1]
        diagnostic = ranking_diagnostics(raw_probability, calibrated_probability)
        diagnostic["method"] = method
        diagnostic["admissible"] = bool(diagnostic["rank_preserved"])
        if method == "sigmoid":
            calibrator = calibrated.calibrated_classifiers_[0].calibrators[0]
            diagnostic["coefficient_a"] = float(calibrator.a_)
            diagnostic["intercept_b"] = float(calibrator.b_)
            diagnostic["increasing"] = bool(calibrator.a_ < 0)
            diagnostic["admissible"] = bool(
                diagnostic["admissible"] and diagnostic["increasing"]
            )
        else:
            diagnostic["minimum_unique_levels"] = MIN_ISOTONIC_LEVELS
            diagnostic["maximum_largest_tie_fraction"] = (
                MAX_ISOTONIC_LARGEST_TIE_FRACTION
            )
            diagnostic["admissible"] = bool(
                diagnostic["admissible"]
                and diagnostic["calibrated_unique_scores"] >= MIN_ISOTONIC_LEVELS
                and diagnostic["largest_calibrated_tie_fraction"]
                <= MAX_ISOTONIC_LARGEST_TIE_FRACTION
            )
        if diagnostic["rank_preserved"]:
            assert_calibration_ranking_preserved(
                raw_probability,
                calibrated_probability,
                method=method,
            )
        attempts[method] = diagnostic
        estimators[method] = calibrated

    if attempts["sigmoid"]["admissible"]:
        selected_method = "sigmoid"
        status = "calibrated_sigmoid"
    elif attempts["isotonic"]["admissible"]:
        selected_method = "isotonic"
        status = "calibrated_isotonic_fallback"
    else:
        selected_method = "raw"
        status = "uncalibrated_raw_fallback"

    diagnostics = {
        "policy_version": CALIBRATION_POLICY_VERSION,
        "selection_order": ["sigmoid", "isotonic", "raw"],
        "selected_method": selected_method,
        "calibration_status": status,
        "positive_class": 1,
        "selection_uses_evaluation_labels": False,
        "attempts": attempts,
    }
    return CalibrationPolicyResult(
        estimator=estimators[selected_method],
        selected_method=selected_method,
        calibration_status=status,
        diagnostics=diagnostics,
    )


def policy_fold_summary(partitions: PolicyFoldPartitions) -> dict[str, Any]:
    """Return auditable dates and class counts for all four partitions."""

    def describe(frame: pd.DataFrame) -> dict[str, Any]:
        delayed = int(frame[TARGET_COLUMN].sum())
        return {
            "rows": int(len(frame)),
            "delayed": delayed,
            "on_time": int(len(frame) - delayed),
            "purchase_min": pd.Timestamp(frame["purchase_timestamp"].min()).isoformat(),
            "purchase_max": pd.Timestamp(frame["purchase_timestamp"].max()).isoformat(),
            "label_observed_max": pd.Timestamp(
                frame["label_observed_at"].max()
            ).isoformat(),
        }

    return {
        "name": partitions.fold.name,
        "training_as_of_exclusive": partitions.fold.training_as_of.isoformat(),
        "calibration_fit_start_inclusive": (
            partitions.fold.calibration_start.isoformat()
        ),
        "calibration_fit_label_as_of_exclusive": (
            partitions.calibration_split_at.isoformat()
        ),
        "threshold_start_inclusive": partitions.calibration_split_at.isoformat(),
        "threshold_label_as_of_exclusive": (
            partitions.fold.evaluation_start.isoformat()
        ),
        "evaluation_start_inclusive": partitions.fold.evaluation_start.isoformat(),
        "evaluation_end_exclusive": partitions.fold.evaluation_end.isoformat(),
        "train": describe(partitions.train),
        "calibration_fit": describe(partitions.calibration_fit),
        "threshold_selection": describe(partitions.threshold_selection),
        "evaluation": describe(partitions.evaluation),
    }
