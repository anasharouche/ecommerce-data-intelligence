"""Validation-only metrics, thresholds, robustness, and interpretation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def evaluate_probabilities(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate binary probabilities with Average Precision as PR-AUC."""

    y = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.ndim != 1 or len(probability) != len(y):
        raise ValueError("Dimensions de probabilités incompatibles.")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Les probabilités doivent être finies entre 0 et 1.")
    if not 0 <= threshold <= 1:
        raise ValueError("Le seuil doit être compris entre 0 et 1.")

    predictions = probability >= threshold
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    has_both_classes = len(np.unique(y)) == 2
    return {
        "rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "positive_rate": float(y.mean()),
        "prediction_positive_rate": float(predictions.mean()),
        "alert_count": int(predictions.sum()),
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": (
            float(roc_auc_score(y, probability)) if has_both_classes else None
        ),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "expected_calibration_error": expected_calibration_error(
            y, probability, bins=10
        ),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }


def expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Return equal-frequency ECE weighted by the number of observations."""

    calibration = calibration_bins(y_true, probabilities, bins=bins)
    total = sum(item["rows"] for item in calibration)
    if total == 0:
        raise ValueError("ECE impossible sur un ensemble vide.")
    return float(
        sum(
            item["rows"]
            * abs(item["mean_probability"] - item["observed_rate"])
            for item in calibration
        )
        / total
    )


def best_f1_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    """Choose the validation-only threshold maximizing F1."""

    y = np.asarray(y_true, dtype=np.int8)
    probability = np.asarray(probabilities, dtype=np.float64)
    precision, recall, thresholds = precision_recall_curve(y, probability)
    if thresholds.size == 0:
        threshold = 0.5
    else:
        denominator = precision[:-1] + recall[:-1]
        f1_values = np.divide(
            2 * precision[:-1] * recall[:-1],
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        threshold = float(thresholds[int(np.argmax(f1_values))])
    return threshold, evaluate_probabilities(y, probability, threshold=threshold)


def threshold_diagnostics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    candidate_threshold: float,
) -> list[dict[str, Any]]:
    """Evaluate interpretable validation thresholds plus the candidate."""

    thresholds = sorted(
        set([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
            + [round(float(candidate_threshold), 10)])
    )
    return [
        evaluate_probabilities(y_true, probabilities, threshold=threshold)
        for threshold in thresholds
    ]


def calibration_bins(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> list[dict[str, Any]]:
    """Return quantile calibration bins with support counts."""

    frame = pd.DataFrame(
        {
            "target": np.asarray(y_true, dtype=np.int8),
            "probability": np.asarray(probabilities, dtype=np.float64),
        }
    )
    unique = int(frame["probability"].nunique())
    if unique <= 1:
        return [
            {
                "bin": 0,
                "rows": int(len(frame)),
                "mean_probability": float(frame["probability"].mean()),
                "observed_rate": float(frame["target"].mean()),
            }
        ]
    quantiles = min(bins, unique)
    frame["bin"] = pd.qcut(
        frame["probability"], q=quantiles, labels=False, duplicates="drop"
    )
    grouped = frame.groupby("bin", observed=True)
    return [
        {
            "bin": int(bin_id),
            "rows": int(len(group)),
            "mean_probability": float(group["probability"].mean()),
            "observed_rate": float(group["target"].mean()),
        }
        for bin_id, group in grouped
    ]


def subgroup_metrics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    group_values: pd.Series,
    threshold: float,
    minimum_rows: int,
) -> list[dict[str, Any]]:
    """Evaluate sufficiently large validation subgroups."""

    working = pd.DataFrame(
        {
            "group": group_values.astype("string").to_numpy(),
            "target": frame["is_delayed"].to_numpy(),
            "probability": np.asarray(probabilities, dtype=np.float64),
        },
        index=frame.index,
    )
    results: list[dict[str, Any]] = []
    for group_name, group in working.groupby("group", dropna=False):
        if len(group) < minimum_rows:
            continue
        metrics = evaluate_probabilities(
            group["target"], group["probability"], threshold=threshold
        )
        metrics["group"] = str(group_name)
        results.append(metrics)
    return sorted(results, key=lambda item: item["group"])


def transformed_feature_importance(
    pipeline: Pipeline,
) -> list[dict[str, Any]] | None:
    """Return coefficients or impurity importances when exposed."""

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    if hasattr(classifier, "coef_"):
        values = np.asarray(classifier.coef_)[0]
        records = [
            {
                "feature": str(name),
                "signed_importance": float(value),
                "absolute_importance": float(abs(value)),
            }
            for name, value in zip(names, values, strict=True)
        ]
        return sorted(
            records, key=lambda item: item["absolute_importance"], reverse=True
        )
    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_)
        records = [
            {"feature": str(name), "importance": float(value)}
            for name, value in zip(names, values, strict=True)
        ]
        return sorted(records, key=lambda item: item["importance"], reverse=True)
    return None
