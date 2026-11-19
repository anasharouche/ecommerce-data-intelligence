"""Preregistered historical calibration and deterministic top-k policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from src.ml.backtesting import (
    BacktestError,
    BacktestFold,
    assert_calibration_ranking_preserved,
)
from src.ml.calibration_policy import (
    MAX_ISOTONIC_LARGEST_TIE_FRACTION,
    MIN_ISOTONIC_LEVELS,
    ranking_diagnostics,
)
from src.ml.contract import TARGET_COLUMN
from src.ml.enriched import PHASE54_CUTOFF_TIMESTAMP


CAMPAIGN_VERSION = "historical_calibration_alert_policy_v1.0.0"
HISTORICAL_CALIBRATION_MONTHS = 3
MIN_CALIBRATION_POSITIVES = 30
TOPK_CAPACITIES = (0.05, 0.10, 0.20)


@dataclass(frozen=True)
class HistoricalFoldPartitions:
    """Leakage-safe train, expanded calibration and retrospective evaluation."""

    fold: BacktestFold
    training_as_of: pd.Timestamp
    train: pd.DataFrame
    calibration: pd.DataFrame
    evaluation: pd.DataFrame


@dataclass(frozen=True)
class CalibrationStrategies:
    """Fitted raw and admissible calibration strategies plus diagnostics."""

    estimators: dict[str, Any]
    diagnostics: dict[str, dict[str, Any]]


def build_expanded_historical_partitions(
    dataset: pd.DataFrame,
    fold: BacktestFold,
) -> HistoricalFoldPartitions:
    """Build the preregistered three-month historical calibration window."""

    training_as_of = fold.evaluation_start - pd.DateOffset(
        months=HISTORICAL_CALIBRATION_MONTHS
    )
    purchase = pd.to_datetime(dataset["purchase_timestamp"], errors="coerce")
    observed = pd.to_datetime(dataset["label_observed_at"], errors="coerce")
    if purchase.isna().any() or observed.isna().any():
        raise BacktestError("Timestamp invalide dans la campagne 5.4.3.")
    if purchase.ge(PHASE54_CUTOFF_TIMESTAMP).any():
        raise BacktestError("La campagne 5.4.3 contient le holdout final.")
    if not (
        training_as_of
        < fold.evaluation_start
        < fold.evaluation_end
        <= PHASE54_CUTOFF_TIMESTAMP
    ):
        raise BacktestError("Dates de fold 5.4.3 incohérentes.")

    partitions = {
        "train": dataset.loc[
            purchase.lt(training_as_of) & observed.lt(training_as_of)
        ].copy(),
        "calibration": dataset.loc[
            purchase.ge(training_as_of)
            & purchase.lt(fold.evaluation_start)
            & observed.lt(fold.evaluation_start)
        ].copy(),
        "evaluation": dataset.loc[
            purchase.ge(fold.evaluation_start)
            & purchase.lt(fold.evaluation_end)
        ].copy(),
    }
    for name, frame in partitions.items():
        if frame.empty or set(frame[TARGET_COLUMN].astype(int).unique()) != {0, 1}:
            raise BacktestError(f"Partition historique {name} insuffisante.")
        if frame["order_id"].duplicated().any():
            raise BacktestError(f"Doublon dans la partition historique {name}.")
    identifiers = {
        name: set(frame["order_id"].astype(str))
        for name, frame in partitions.items()
    }
    if (
        identifiers["train"] & identifiers["calibration"]
        or identifiers["train"] & identifiers["evaluation"]
        or identifiers["calibration"] & identifiers["evaluation"]
    ):
        raise BacktestError("Chevauchement des partitions historiques 5.4.3.")
    if not partitions["train"]["label_observed_at"].lt(training_as_of).all():
        raise BacktestError("Label train immature dans la campagne 5.4.3.")
    if not partitions["calibration"]["label_observed_at"].lt(
        fold.evaluation_start
    ).all():
        raise BacktestError("Label calibration immature dans la campagne 5.4.3.")

    return HistoricalFoldPartitions(
        fold=fold,
        training_as_of=training_as_of,
        train=partitions["train"].reset_index(drop=True),
        calibration=partitions["calibration"].reset_index(drop=True),
        evaluation=partitions["evaluation"].reset_index(drop=True),
    )


def historical_partition_summary(
    partitions: HistoricalFoldPartitions,
) -> dict[str, Any]:
    """Return feasibility evidence before any model is fitted."""

    def describe(frame: pd.DataFrame) -> dict[str, Any]:
        delayed = int(frame[TARGET_COLUMN].sum())
        return {
            "rows": int(len(frame)),
            "delayed": delayed,
            "on_time": int(len(frame) - delayed),
            "delay_rate": float(delayed / len(frame)),
            "purchase_min": pd.Timestamp(frame["purchase_timestamp"].min()).isoformat(),
            "purchase_max": pd.Timestamp(frame["purchase_timestamp"].max()).isoformat(),
            "label_observed_max": pd.Timestamp(
                frame["label_observed_at"].max()
            ).isoformat(),
        }

    calibration = describe(partitions.calibration)
    return {
        "fold": partitions.fold.name,
        "evaluation_origin": partitions.fold.evaluation_start.isoformat(),
        "training_as_of_exclusive": partitions.training_as_of.isoformat(),
        "calibration_start_inclusive": partitions.training_as_of.isoformat(),
        "calibration_label_as_of_exclusive": (
            partitions.fold.evaluation_start.isoformat()
        ),
        "evaluation_end_exclusive": partitions.fold.evaluation_end.isoformat(),
        "train": describe(partitions.train),
        "calibration": calibration,
        "evaluation": describe(partitions.evaluation),
        "calibration_minimum_positive_rows": MIN_CALIBRATION_POSITIVES,
        "calibration_feasible": calibration["delayed"]
        >= MIN_CALIBRATION_POSITIVES,
        "partitions_disjoint": True,
        "labels_mature_before_use": True,
    }


def fit_expanded_calibration_strategies(
    base_pipeline: Any,
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_calibration: pd.DataFrame,
    y_calibration: pd.Series,
) -> CalibrationStrategies:
    """Fit A0/A1/A2 once using only the expanded historical calibration set."""

    base_pipeline.fit(x_train, y_train)
    if not np.array_equal(np.asarray(base_pipeline.classes_), np.array([0, 1])):
        raise BacktestError("La classe positive de retard doit être encodée 1.")
    positive_rows = int(np.asarray(y_calibration, dtype=np.int8).sum())
    raw_probability = base_pipeline.predict_proba(x_calibration)[:, 1]
    estimators: dict[str, Any] = {"uncalibrated_raw": base_pipeline}
    diagnostics: dict[str, dict[str, Any]] = {
        "uncalibrated_raw": {
            "strategy": "uncalibrated_raw",
            "status": "uncalibrated_raw",
            "admissible": True,
            "calibration_rows": int(len(y_calibration)),
            "calibration_positive_rows": positive_rows,
            "probabilities_are_calibrated": False,
        }
    }
    enough_support = positive_rows >= MIN_CALIBRATION_POSITIVES

    for method, strategy in (
        ("sigmoid", "sigmoid_expanded"),
        ("isotonic", "isotonic_expanded"),
    ):
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(base_pipeline), method=method, ensemble=False
        )
        calibrated.fit(x_calibration, y_calibration)
        probability = calibrated.predict_proba(x_calibration)[:, 1]
        diagnostic = ranking_diagnostics(raw_probability, probability)
        diagnostic.update(
            {
                "strategy": strategy,
                "calibration_rows": int(len(y_calibration)),
                "calibration_positive_rows": positive_rows,
                "minimum_positive_rows": MIN_CALIBRATION_POSITIVES,
                "sufficient_support": enough_support,
                "probability_min": float(probability.min()),
                "probability_max": float(probability.max()),
                "finite_probabilities": bool(np.isfinite(probability).all()),
                "probabilities_in_0_1": bool(
                    ((probability >= 0) & (probability <= 1)).all()
                ),
            }
        )
        admissible = bool(
            enough_support
            and diagnostic["rank_preserved"]
            and diagnostic["finite_probabilities"]
            and diagnostic["probabilities_in_0_1"]
        )
        if method == "sigmoid":
            calibrator = calibrated.calibrated_classifiers_[0].calibrators[0]
            diagnostic["coefficient_a"] = float(calibrator.a_)
            diagnostic["intercept_b"] = float(calibrator.b_)
            diagnostic["increasing"] = bool(calibrator.a_ < 0)
            admissible = bool(admissible and diagnostic["increasing"])
        else:
            diagnostic["minimum_unique_levels"] = MIN_ISOTONIC_LEVELS
            diagnostic["maximum_largest_tie_fraction"] = (
                MAX_ISOTONIC_LARGEST_TIE_FRACTION
            )
            admissible = bool(
                admissible
                and diagnostic["calibrated_unique_scores"] >= MIN_ISOTONIC_LEVELS
                and diagnostic["largest_calibrated_tie_fraction"]
                <= MAX_ISOTONIC_LARGEST_TIE_FRACTION
            )
        diagnostic["admissible"] = admissible
        diagnostic["status"] = "admissible" if admissible else "rejected"
        if diagnostic["rank_preserved"]:
            assert_calibration_ranking_preserved(
                raw_probability, probability, method=strategy
            )
        diagnostics[strategy] = diagnostic
        estimators[strategy] = calibrated

    return CalibrationStrategies(estimators=estimators, diagnostics=diagnostics)


def deterministic_topk_metrics(
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
    order_ids: pd.Series | np.ndarray,
    *,
    capacity: float,
) -> dict[str, Any]:
    """Evaluate an exact-capacity retrospective batch policy with stable ties."""

    y = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(scores, dtype=np.float64)
    identifiers = np.asarray(order_ids, dtype=str)
    if y.ndim != 1 or probability.shape != y.shape or identifiers.shape != y.shape:
        raise ValueError("Dimensions incompatibles pour top-k.")
    if len(y) == 0 or len(np.unique(identifiers)) != len(identifiers):
        raise ValueError("Le lot top-k doit être non vide et unique par commande.")
    if not 0 < capacity <= 1:
        raise ValueError("La capacité top-k doit être dans ]0, 1].")
    if not np.isfinite(probability).all():
        raise ValueError("Score top-k non fini.")

    alert_count = min(len(y), max(1, int(np.ceil(len(y) * capacity))))
    order = np.lexsort((identifiers, -probability))
    selected = order[:alert_count]
    alerted = np.zeros(len(y), dtype=bool)
    alerted[selected] = True
    detected = int(y[alerted].sum())
    total_positive = int(y.sum())
    false_alerts = int(alert_count - detected)
    missed = int(total_positive - detected)
    precision = float(detected / alert_count)
    recall = float(detected / total_positive) if total_positive else 0.0
    prevalence = float(total_positive / len(y))
    lift = float(precision / prevalence) if prevalence else 0.0
    boundary_score = float(probability[order[alert_count - 1]])
    tied_at_boundary = probability == boundary_score
    return {
        "capacity": float(capacity),
        "rounding": "ceil",
        "rows": int(len(y)),
        "alert_count": alert_count,
        "alert_rate": float(alert_count / len(y)),
        "precision_at_k": precision,
        "recall_at_k": recall,
        "lift_at_k": lift,
        "delayed_rows": total_positive,
        "detected_delays": detected,
        "false_alerts": false_alerts,
        "missed_delays": missed,
        "boundary_score": boundary_score,
        "rows_tied_at_boundary": int(tied_at_boundary.sum()),
        "selected_rows_tied_at_boundary": int(
            np.sum(tied_at_boundary[selected])
        ),
        "tie_breaker": "order_id_ascending",
        "budget_expanded_for_ties": False,
    }
