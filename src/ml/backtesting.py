"""Leakage-safe Phase 5.4 data loading, temporal folds and calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.ml.contract import STRICT_FEATURE_COLUMNS, TARGET_COLUMN
from src.ml.contract_v2 import (
    ENRICHED_BOOLEAN_FEATURES,
    ENRICHED_CATEGORICAL_FEATURES,
    ENRICHED_FEATURE_COLUMNS,
    ENRICHED_NUMERIC_FEATURES,
)
from src.ml.enriched import PHASE54_CUTOFF_TIMESTAMP, validate_enriched_dataset
from src.ml.evaluation import best_f1_threshold, evaluate_probabilities
from src.ml.training import (
    build_model_pipeline,
    build_typed_model_pipeline,
    feature_matrix,
    model_definitions,
    target_vector,
)
from src.ml.validation import validate_strict_dataset


FORBIDDEN_PHASE54_FILENAMES = frozenset({"strict_v1_test.parquet"})


class BacktestError(RuntimeError):
    """Raised when a Phase 5.4 temporal safety invariant is violated."""


@dataclass(frozen=True)
class BacktestFold:
    """One expanding-origin fold with a separate two-month calibrator window."""

    name: str
    training_as_of: pd.Timestamp
    calibration_start: pd.Timestamp
    calibration_as_of: pd.Timestamp
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp


@dataclass(frozen=True)
class FoldPartitions:
    """Disjoint train, calibration and retrospective evaluation partitions."""

    fold: BacktestFold
    train: pd.DataFrame
    calibration: pd.DataFrame
    evaluation: pd.DataFrame


def assert_phase54_safe_path(path: Path) -> None:
    """Reject final-test paths before any filesystem read occurs."""

    if path.name.lower() in FORBIDDEN_PHASE54_FILENAMES or "test" in path.name.lower():
        raise BacktestError(f"Accès interdit au test final : {path.name}")


def load_phase54_strict_dataset(input_dir: Path) -> pd.DataFrame:
    """Load only the descriptive train and Jan-Apr validation Parquets."""

    paths = (
        input_dir / "strict_v1_train.parquet",
        input_dir / "strict_v1_validation.parquet",
    )
    frames: list[pd.DataFrame] = []
    for path in paths:
        assert_phase54_safe_path(path)
        if not path.is_file():
            raise BacktestError(f"Partition sûre absente : {path}")
        frame = pd.read_parquet(path)
        validate_strict_dataset(frame)
        frames.append(frame)
    dataset = pd.concat(frames, ignore_index=True)
    if dataset["order_id"].duplicated().any():
        raise BacktestError("Chevauchement entre les partitions Phase 5.4 sûres.")
    if pd.to_datetime(dataset["purchase_timestamp"]).ge(
        PHASE54_CUTOFF_TIMESTAMP
    ).any():
        raise BacktestError("Une ligne du test final a été chargée.")
    return dataset.sort_values(
        ["purchase_timestamp", "order_id"]
    ).reset_index(drop=True)


def monthly_backtest_folds() -> tuple[BacktestFold, ...]:
    """Return the four reproducible Jan-Apr 2018 pre-test folds."""

    folds: list[BacktestFold] = []
    for evaluation_start in pd.date_range(
        "2018-01-01", "2018-04-01", freq="MS"
    ):
        evaluation_start = pd.Timestamp(evaluation_start)
        calibration_start = evaluation_start - pd.DateOffset(months=2)
        folds.append(
            BacktestFold(
                name=f"evaluation_{evaluation_start:%Y_%m}",
                training_as_of=calibration_start,
                calibration_start=calibration_start,
                calibration_as_of=evaluation_start,
                evaluation_start=evaluation_start,
                evaluation_end=evaluation_start + pd.DateOffset(months=1),
            )
        )
    return tuple(folds)


def build_fold_partitions(
    dataset: pd.DataFrame,
    fold: BacktestFold,
) -> FoldPartitions:
    """Apply purchase windows and label-maturity rules to a safe dataset."""

    purchase = pd.to_datetime(dataset["purchase_timestamp"], errors="coerce")
    observed = pd.to_datetime(dataset["label_observed_at"], errors="coerce")
    if purchase.isna().any() or observed.isna().any():
        raise BacktestError("Timestamp invalide dans le backtest.")
    if purchase.ge(PHASE54_CUTOFF_TIMESTAMP).any():
        raise BacktestError("Le backtest contient le test final.")
    if not (
        fold.training_as_of
        == fold.calibration_start
        < fold.calibration_as_of
        == fold.evaluation_start
        < fold.evaluation_end
        <= PHASE54_CUTOFF_TIMESTAMP
    ):
        raise BacktestError("Dates de fold incohérentes ou postérieures au cutoff.")

    train_mask = purchase.lt(fold.training_as_of) & observed.lt(
        fold.training_as_of
    )
    calibration_mask = (
        purchase.ge(fold.calibration_start)
        & purchase.lt(fold.evaluation_start)
        & observed.lt(fold.calibration_as_of)
    )
    evaluation_mask = purchase.ge(fold.evaluation_start) & purchase.lt(
        fold.evaluation_end
    )
    partitions = {
        "train": dataset.loc[train_mask].copy(),
        "calibration": dataset.loc[calibration_mask].copy(),
        "evaluation": dataset.loc[evaluation_mask].copy(),
    }
    for name, frame in partitions.items():
        if frame.empty:
            raise BacktestError(f"Partition {name} vide pour {fold.name}.")
        if set(frame[TARGET_COLUMN].astype(int).unique()) != {0, 1}:
            raise BacktestError(f"Partition {name} sans les deux classes.")

    identifiers = {
        name: set(frame["order_id"].astype(str))
        for name, frame in partitions.items()
    }
    if (
        identifiers["train"] & identifiers["calibration"]
        or identifiers["train"] & identifiers["evaluation"]
        or identifiers["calibration"] & identifiers["evaluation"]
    ):
        raise BacktestError("Chevauchement d'order_id dans le fold.")
    if not partitions["train"]["label_observed_at"].lt(
        fold.training_as_of
    ).all():
        raise BacktestError("Label train immature.")
    if not partitions["calibration"]["label_observed_at"].lt(
        fold.evaluation_start
    ).all():
        raise BacktestError("Label calibration immature.")

    return FoldPartitions(
        fold=fold,
        train=partitions["train"].reset_index(drop=True),
        calibration=partitions["calibration"].reset_index(drop=True),
        evaluation=partitions["evaluation"].reset_index(drop=True),
    )


def feature_columns_for_contract(contract: str) -> tuple[str, ...]:
    if contract == "strict_v1":
        return STRICT_FEATURE_COLUMNS
    if contract == "enriched_v2":
        return ENRICHED_FEATURE_COLUMNS
    raise BacktestError(f"Contrat inconnu : {contract}")


def build_phase54_pipeline(contract: str, model_name: str) -> Any:
    """Reuse Phase 5.3 model definitions with the requested contract."""

    definitions = model_definitions()
    if model_name not in definitions:
        raise BacktestError(f"Modèle inconnu : {model_name}")
    definition = definitions[model_name]
    if contract == "strict_v1":
        return build_model_pipeline(definition)
    if contract == "enriched_v2":
        return build_typed_model_pipeline(
            definition,
            numeric_features=ENRICHED_NUMERIC_FEATURES,
            categorical_features=ENRICHED_CATEGORICAL_FEATURES,
            boolean_features=ENRICHED_BOOLEAN_FEATURES,
        )
    raise BacktestError(f"Contrat inconnu : {contract}")


def contract_matrix(frame: pd.DataFrame, contract: str) -> pd.DataFrame:
    """Return only model features after validating the selected contract."""

    if contract == "strict_v1":
        return feature_matrix(frame)
    if contract == "enriched_v2":
        validate_enriched_dataset(frame)
        return frame.loc[:, list(ENRICHED_FEATURE_COLUMNS)].copy()
    raise BacktestError(f"Contrat inconnu : {contract}")


def fit_calibration_variants(
    base_pipeline: Any,
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_calibration: pd.DataFrame,
    y_calibration: pd.Series,
) -> dict[str, Any]:
    """Fit base model on train and calibrators only on the later window."""

    base_pipeline.fit(x_train, y_train)
    if not np.array_equal(np.asarray(base_pipeline.classes_), np.array([0, 1])):
        raise BacktestError("La classe positive de retard doit être encodée 1.")
    raw_calibration_probability = base_pipeline.predict_proba(x_calibration)[:, 1]
    variants: dict[str, Any] = {"raw": base_pipeline}
    for method in ("sigmoid", "isotonic"):
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(base_pipeline),
            method=method,
            ensemble=False,
        )
        calibrated.fit(x_calibration, y_calibration)
        assert_calibration_ranking_preserved(
            raw_calibration_probability,
            calibrated.predict_proba(x_calibration)[:, 1],
            method=method,
        )
        variants[method] = calibrated
    return variants


def assert_calibration_ranking_preserved(
    raw_probability: np.ndarray,
    calibrated_probability: np.ndarray,
    *,
    method: str,
    tolerance: float = 1e-12,
) -> None:
    """Reject a calibrator that reverses the ordering of distinct raw scores."""

    raw = np.asarray(raw_probability, dtype=np.float64)
    calibrated = np.asarray(calibrated_probability, dtype=np.float64)
    if raw.ndim != 1 or calibrated.shape != raw.shape or raw.size == 0:
        raise BacktestError("Scores incompatibles pour l'audit de classement.")
    if not np.isfinite(raw).all() or not np.isfinite(calibrated).all():
        raise BacktestError("Score non fini dans l'audit de classement.")

    order = np.argsort(raw, kind="stable")
    sorted_raw = raw[order]
    sorted_calibrated = calibrated[order]
    distinct_raw = np.diff(sorted_raw) > tolerance
    reversed_boundaries = distinct_raw & (
        np.diff(sorted_calibrated) < -tolerance
    )
    if reversed_boundaries.any():
        raise BacktestError(
            f"La calibration {method} inverse le classement des scores "
            f"à {int(reversed_boundaries.sum())} frontière(s)."
        )


def evaluate_calibration_variants(
    variants: dict[str, Any],
    *,
    x_calibration: pd.DataFrame,
    y_calibration: pd.Series,
    x_evaluation: pd.DataFrame,
    y_evaluation: pd.Series,
    historical_threshold: float,
) -> dict[str, Any]:
    """Choose thresholds on calibration and apply them unchanged to evaluation."""

    results: dict[str, Any] = {}
    for method, estimator in variants.items():
        calibration_probability = estimator.predict_proba(x_calibration)[:, 1]
        evaluation_probability = estimator.predict_proba(x_evaluation)[:, 1]
        if not np.isfinite(evaluation_probability).all():
            raise BacktestError("Probabilité de backtest non finie.")
        candidate_threshold, calibration_at_candidate = best_f1_threshold(
            y_calibration, calibration_probability
        )
        results[method] = {
            "candidate_threshold": float(candidate_threshold),
            "threshold_selected_before_evaluation": True,
            "calibration_at_candidate_threshold": calibration_at_candidate,
            "evaluation": {
                "threshold_0_5": evaluate_probabilities(
                    y_evaluation, evaluation_probability, threshold=0.5
                ),
                "historical_v1_threshold": evaluate_probabilities(
                    y_evaluation,
                    evaluation_probability,
                    threshold=historical_threshold,
                ),
                "calibration_selected_threshold": evaluate_probabilities(
                    y_evaluation,
                    evaluation_probability,
                    threshold=candidate_threshold,
                ),
            },
            "evaluation_probabilities": evaluation_probability,
        }
    return results


def fold_summary(partitions: FoldPartitions) -> dict[str, Any]:
    """Return exact dates, maturity cutoffs and class distributions."""

    def describe(frame: pd.DataFrame) -> dict[str, Any]:
        positive = int(frame[TARGET_COLUMN].sum())
        return {
            "rows": int(len(frame)),
            "delayed": positive,
            "on_time": int(len(frame) - positive),
            "delay_rate": float(positive / len(frame)),
            "purchase_min": pd.Timestamp(frame["purchase_timestamp"].min()).isoformat(),
            "purchase_max": pd.Timestamp(frame["purchase_timestamp"].max()).isoformat(),
            "label_observed_max": pd.Timestamp(
                frame["label_observed_at"].max()
            ).isoformat(),
        }

    fold = partitions.fold
    return {
        "name": fold.name,
        "training_as_of_exclusive": fold.training_as_of.isoformat(),
        "calibration_purchase_start_inclusive": fold.calibration_start.isoformat(),
        "calibration_label_as_of_exclusive": fold.calibration_as_of.isoformat(),
        "evaluation_start_inclusive": fold.evaluation_start.isoformat(),
        "evaluation_end_exclusive": fold.evaluation_end.isoformat(),
        "train": describe(partitions.train),
        "calibration": describe(partitions.calibration),
        "evaluation": describe(partitions.evaluation),
    }
