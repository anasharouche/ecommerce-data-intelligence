"""One-shot Phase 5.5 evaluation of the frozen pre-holdout candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from config.paths import BASE_DIR
from src.ml.backtesting import (
    BacktestError,
    contract_matrix,
    load_phase54_strict_dataset,
)
from src.ml.contract import (
    FEATURE_SCHEMA_VERSION,
    STRICT_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.ml.drift import drift_report
from src.ml.evaluation import expected_calibration_error
from src.ml.historical_calibration import deterministic_topk_metrics
from src.ml.run_phase543 import validate_artifact_manifest
from src.ml.training import BOOLEAN_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.ml.validation import feature_missing_rates, validate_strict_dataset


PHASE = "5.5"
PROTOCOL_VERSION = "final_holdout_v1.0.0"
EXPECTED_CONFIG_SHA256 = "3c20ab70f9b434f6698f41a1ba2bbc0491f7f420aa4fad5b59b0c84c90eb8c39"
CANDIDATE_VERSION = "strict_v1_logistic_top10_v1"
CANDIDATE_DIR = (
    Path(BASE_DIR)
    / "models"
    / "delivery_delay"
    / "phase_5_4_3"
    / "historical_calibration_alert_policy_v1"
)
DEFAULT_INPUT_DIR = Path(BASE_DIR) / "data" / "processed" / "ml"
DEFAULT_OUTPUT_DIR = (
    Path(BASE_DIR) / "models" / "delivery_delay" / "phase_5_5" / "final_holdout_v1"
)
HOLDOUT_FILENAME = "strict_v1_test.parquet"
HOLDOUT_START = pd.Timestamp("2018-05-01")
HOLDOUT_END = pd.Timestamp("2018-09-01")
TOP10_CAPACITY = 0.10
BOOTSTRAP_REPLICATIONS = 1000
BOOTSTRAP_RANDOM_STATE = 42


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _atomic_bytes(payload: bytes, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_json(payload: Any, destination: Path) -> None:
    _atomic_bytes(_json_bytes(payload), destination)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _score_distribution(probability: np.ndarray) -> dict[str, float]:
    values = np.asarray(probability, dtype=np.float64)
    quantiles = np.quantile(values, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
        **{
            f"q{int(level * 100):02d}": float(value)
            for level, value in zip(
                (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99),
                quantiles,
                strict=True,
            )
        },
    }


def safe_probability_metrics(
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
) -> dict[str, Any]:
    """Compute diagnostics while marking ranking metrics undefined for one class."""

    y = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or scores.shape != y.shape or len(y) == 0:
        raise ValueError("Dimensions finales incompatibles.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Score final invalide.")
    both_classes = np.unique(y).size == 2
    return {
        "rows": int(len(y)),
        "delayed_rows": int(y.sum()),
        "prevalence": float(y.mean()),
        "average_precision": (
            float(average_precision_score(y, scores)) if both_classes else None
        ),
        "roc_auc": float(roc_auc_score(y, scores)) if both_classes else None,
        "brier_score": float(brier_score_loss(y, scores)),
        "log_loss": float(log_loss(y, scores, labels=[0, 1])),
        "expected_calibration_error": expected_calibration_error(y, scores, bins=10),
        "score_distribution": _score_distribution(scores),
        "ranking_metrics_defined": both_classes,
        "scores_are_calibrated": False,
    }


def deterministic_topk_mask(
    scores: np.ndarray,
    order_ids: pd.Series | np.ndarray,
    *,
    capacity: float = TOP10_CAPACITY,
) -> np.ndarray:
    """Return the frozen exact-capacity mask used for monthly batch alerts."""

    probability = np.asarray(scores, dtype=np.float64)
    identifiers = np.asarray(order_ids, dtype=str)
    if probability.ndim != 1 or identifiers.shape != probability.shape:
        raise ValueError("Dimensions top-k incompatibles.")
    if len(probability) == 0 or len(np.unique(identifiers)) != len(identifiers):
        raise ValueError("Lot top-k vide ou non unique.")
    if not 0 < capacity <= 1 or not np.isfinite(probability).all():
        raise ValueError("Capacité ou score top-k invalide.")
    k = min(len(probability), max(1, int(np.ceil(len(probability) * capacity))))
    order = np.lexsort((identifiers, -probability))
    mask = np.zeros(len(probability), dtype=bool)
    mask[order[:k]] = True
    return mask


def score_frozen_candidate(estimator: Any, features: pd.DataFrame) -> np.ndarray:
    """Score without invoking fit, fit_transform or any state-changing method."""

    before = joblib.hash(estimator)
    probability = np.asarray(estimator.predict_proba(features)[:, 1], dtype=np.float64)
    after = joblib.hash(estimator)
    if before != after:
        raise BacktestError("Le scoring a modifié l'état du candidat figé.")
    return probability


def _synthetic_contract_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "purchase_hour": [9, 18],
            "purchase_day_of_week": [0, 5],
            "purchase_month": [1, 4],
            "purchase_quarter": [1, 2],
            "purchase_is_weekend": [False, True],
            "customer_state": ["SP", "RJ"],
            "customer_latitude": [-23.55, np.nan],
            "customer_longitude": [-46.63, np.nan],
            "customer_has_gps": [True, False],
        }
    ).loc[:, list(STRICT_FEATURE_COLUMNS)]


def validate_strict_dtypes(dataset: pd.DataFrame) -> dict[str, str]:
    """Validate the persisted strict-v1 dtypes without coercing holdout values."""

    expected = {
        "order_id": "string",
        "purchase_timestamp": "datetime64[us]",
        "label_observed_at": "datetime64[us]",
        "purchase_hour": "int8",
        "purchase_day_of_week": "int8",
        "purchase_month": "int8",
        "purchase_quarter": "int8",
        "purchase_is_weekend": "bool",
        "customer_state": "string",
        "customer_latitude": "Float64",
        "customer_longitude": "Float64",
        "customer_has_gps": "bool",
        TARGET_COLUMN: "int8",
    }
    actual = {column: str(dtype) for column, dtype in dataset.dtypes.items()}
    mismatches = {
        column: {"expected": dtype, "actual": actual.get(column)}
        for column, dtype in expected.items()
        if actual.get(column) != dtype
    }
    if mismatches:
        raise BacktestError(f"Types strict_v1 inattendus : {mismatches}")
    return actual


def run_preflight(candidate_dir: Path = CANDIDATE_DIR) -> dict[str, Any]:
    """Validate every frozen input without reading the final holdout."""

    manifest_path = candidate_dir / "artifact_manifest.json"
    metadata_path = candidate_dir / "candidate_metadata.json"
    pipeline_path = candidate_dir / "candidate_pipeline.joblib"
    config_path = candidate_dir / "experiment_config.json"
    for path in (manifest_path, metadata_path, pipeline_path, config_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    validate_artifact_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config_sha256 = _sha256(config_path)
    if config_sha256 != EXPECTED_CONFIG_SHA256:
        raise BacktestError("Hash de configuration Phase 5.4.3 inattendu.")
    pipeline_sha256 = _sha256(pipeline_path)
    if pipeline_sha256 != manifest["artifacts"]["candidate_pipeline"]["sha256"]:
        raise BacktestError("Hash du candidat différent du manifeste.")

    first = joblib.load(pipeline_path)
    second = joblib.load(pipeline_path)
    if not isinstance(first, Pipeline) or list(first.named_steps) != [
        "preprocessor",
        "classifier",
    ]:
        raise BacktestError("Le candidat n'est pas le Pipeline V1 attendu.")
    classifier = first.named_steps["classifier"]
    checks = {
        "candidate_exists": True,
        "manifest_valid": True,
        "configuration_hash_valid": config_sha256 == EXPECTED_CONFIG_SHA256,
        "pipeline_hash_valid": True,
        "candidate_version_valid": metadata["candidate_version"] == CANDIDATE_VERSION,
        "contract_valid": metadata["contract"] == FEATURE_SCHEMA_VERSION,
        "feature_order_valid": list(first.feature_names_in_)
        == list(STRICT_FEATURE_COLUMNS),
        "positive_class_valid": TARGET_COLUMN == "is_delayed",
        "classes_valid": np.array_equal(first.classes_, np.array([0, 1])),
        "random_state_valid": classifier.random_state == 42,
        "class_weight_valid": classifier.class_weight == "balanced",
        "model_valid": type(classifier).__name__ == "LogisticRegression",
        "no_calibration_valid": metadata["calibration_strategy"]
        == "uncalibrated_raw"
        and not metadata["scores_are_calibrated"],
        "top10_capacity_valid": metadata["topk_policy"]["capacity"] == 0.10,
        "top10_rounding_valid": metadata["topk_policy"]["rounding"] == "ceil",
        "tie_breaker_valid": metadata["topk_policy"]["tie_breaker"]
        == "order_id_ascending",
        "single_order_not_supported": not metadata["topk_policy"][
            "single_order_decision_supported"
        ],
    }
    synthetic = _synthetic_contract_rows()
    first_probability = score_frozen_candidate(first, synthetic)
    second_probability = score_frozen_candidate(second, synthetic)
    checks["reload_reproducible"] = bool(
        np.array_equal(first_probability, second_probability)
    )
    checks["missing_values_supported_by_frozen_pipeline"] = bool(
        np.isfinite(first_probability).all()
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise BacktestError("Préflight échoué : " + ", ".join(failed))
    return {
        "phase": PHASE,
        "candidate_version": CANDIDATE_VERSION,
        "candidate_pipeline_sha256": pipeline_sha256,
        "experiment_config_sha256": config_sha256,
        "feature_contract": FEATURE_SCHEMA_VERSION,
        "feature_order": list(STRICT_FEATURE_COLUMNS),
        "classes": first.classes_.tolist(),
        "checks": checks,
        "all_checks_passed": True,
        "holdout_opened": False,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_evaluation_protocol(preflight: dict[str, Any]) -> dict[str, Any]:
    """Build the immutable protocol before any final-test read."""

    return {
        "phase": PHASE,
        "protocol_version": PROTOCOL_VERSION,
        "candidate": {
            "version": CANDIDATE_VERSION,
            "pipeline_sha256": preflight["candidate_pipeline_sha256"],
            "contract": FEATURE_SCHEMA_VERSION,
            "features": list(STRICT_FEATURE_COLUMNS),
            "model": "LogisticRegression",
            "class_weight": "balanced",
            "random_state": 42,
            "calibration": "none",
            "score_status": "uncalibrated_raw",
        },
        "holdout": {
            "source": f"data/processed/ml/{HOLDOUT_FILENAME}",
            "read_count": 1,
            "purchase_start_inclusive": HOLDOUT_START.isoformat(),
            "purchase_end_exclusive": HOLDOUT_END.isoformat(),
            "expected_rows_diagnostic_only": 25352,
        },
        "positive_class": {"column": TARGET_COLUMN, "value": 1},
        "metrics": [
            "average_precision",
            "roc_auc",
            "brier_score",
            "log_loss",
            "expected_calibration_error",
            "prevalence",
            "score_distribution",
            "precision_at_10_percent",
            "recall_at_10_percent",
            "lift_at_10_percent",
        ],
        "top10_policy": {
            "grouping": "purchase_calendar_month",
            "capacity": TOP10_CAPACITY,
            "rounding": "ceil(month_rows * 0.10)",
            "minimum_for_nonempty_month": 1,
            "ranking": "score_descending_then_order_id_ascending",
            "ties": "do_not_expand_capacity",
            "operational_scope": "retrospective_monthly_batch_simulation",
        },
        "missing_values": {
            "validation": "strict_v1 dataset invariants",
            "numeric": "frozen training median imputer",
            "categorical": "frozen most-frequent imputer and unknown-category ignore",
            "boolean": "frozen boolean conversion and most-frequent imputer",
            "no_holdout_fit_or_fit_transform": True,
        },
        "monthly_aggregation": {
            "months": ["2018-05", "2018-06", "2018-07", "2018-08"],
            "global_top10": "concatenate alerts selected independently in each month",
            "monthly_macro_mean_reported_separately": True,
            "single_class": "AP and ROC-AUC reported as null; no invented value",
            "zero_positive_top10": (
                "precision, recall and lift reported as 0; alert capacity unchanged"
            ),
        },
        "drift": {
            "reference": "frozen candidate training population before 2018-02-01",
            "methods": [
                "score_psi",
                "feature_psi",
                "customer_state_psi",
                "gps_coverage",
                "monthly_prevalence",
                "score_distribution",
            ],
            "holdout_never_used_as_reference": True,
        },
        "uncertainty": {
            "method": "calendar-month block bootstrap with replacement",
            "replications": BOOTSTRAP_REPLICATIONS,
            "random_state": BOOTSTRAP_RANDOM_STATE,
            "confidence_level": 0.95,
            "metrics": [
                "average_precision",
                "roc_auc",
                "brier_score",
                "top10_precision",
                "top10_recall",
                "top10_lift",
            ],
        },
        "stop_rule": {
            "one_final_evaluation_only": True,
            "no_retraining": True,
            "no_recalibration": True,
            "no_threshold_optimization": True,
            "no_feature_change": True,
            "no_second_holdout_campaign": True,
        },
        "production_ready": False,
        "created_before_holdout_open": True,
    }


def prepare_phase55(
    *, candidate_dir: Path = CANDIDATE_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> dict[str, Any]:
    """Run preflight and freeze the protocol without reading the holdout."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Le dossier Phase 5.5 n'est pas vide.")
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = run_preflight(candidate_dir)
    protocol = build_evaluation_protocol(preflight)
    protocol_bytes = _json_bytes(protocol)
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    _atomic_bytes(protocol_bytes, output_dir / "evaluation_protocol.json")
    _atomic_bytes(
        (protocol_sha256 + "  evaluation_protocol.json\n").encode("utf-8"),
        output_dir / "evaluation_protocol.sha256",
    )
    preflight["evaluation_protocol_sha256"] = protocol_sha256
    _atomic_json(preflight, output_dir / "preflight_report.json")
    return preflight


def verify_frozen_protocol(output_dir: Path) -> str:
    """Verify protocol immutability before and after final evaluation."""

    protocol_path = output_dir / "evaluation_protocol.json"
    hash_path = output_dir / "evaluation_protocol.sha256"
    expected = hash_path.read_text(encoding="utf-8").split()[0]
    actual = _sha256(protocol_path)
    if actual != expected:
        raise BacktestError("Le protocole Phase 5.5 a été modifié.")
    return actual


def _monthly_top10(
    frame: pd.DataFrame,
    probability: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    months = pd.to_datetime(frame["purchase_timestamp"]).dt.to_period("M").astype(str)
    alerts = np.zeros(len(frame), dtype=bool)
    results: dict[str, Any] = {}
    for month in ("2018-05", "2018-06", "2018-07", "2018-08"):
        positions = np.flatnonzero(months.to_numpy() == month)
        if len(positions) == 0:
            results[month] = {"rows": 0, "defined": False}
            continue
        month_frame = frame.iloc[positions]
        month_probability = probability[positions]
        month_mask = deterministic_topk_mask(
            month_probability, month_frame["order_id"], capacity=TOP10_CAPACITY
        )
        alerts[positions[month_mask]] = True
        result = deterministic_topk_metrics(
            month_frame[TARGET_COLUMN],
            month_probability,
            month_frame["order_id"],
            capacity=TOP10_CAPACITY,
        )
        result["defined"] = True
        results[month] = result
    return results, alerts


def _aggregate_monthly_top10(
    y: np.ndarray,
    alerts: np.ndarray,
    monthly: dict[str, Any],
) -> dict[str, Any]:
    total_positive = int(y.sum())
    alert_count = int(alerts.sum())
    detected = int(y[alerts].sum())
    false_alerts = alert_count - detected
    missed = total_positive - detected
    precision = float(detected / alert_count) if alert_count else 0.0
    recall = float(detected / total_positive) if total_positive else 0.0
    prevalence = float(y.mean())
    defined = [item for item in monthly.values() if item.get("defined")]
    return {
        "monthly_batch_totals": {
            "rows": int(len(y)),
            "alert_count": alert_count,
            "alert_rate": float(alert_count / len(y)),
            "precision_at_10_percent": precision,
            "recall_at_10_percent": recall,
            "lift_at_10_percent": float(precision / prevalence) if prevalence else 0.0,
            "delayed_rows": total_positive,
            "detected_delays": detected,
            "false_alerts": false_alerts,
            "missed_delays": missed,
        },
        "macro_monthly_mean": {
            "precision_at_10_percent": float(
                np.mean([item["precision_at_k"] for item in defined])
            ),
            "recall_at_10_percent": float(
                np.mean([item["recall_at_k"] for item in defined])
            ),
            "lift_at_10_percent": float(
                np.mean([item["lift_at_k"] for item in defined])
            ),
            "alert_rate": float(np.mean([item["alert_rate"] for item in defined])),
        },
    }


def _block_bootstrap(
    frame: pd.DataFrame,
    probability: np.ndarray,
    monthly_top10: dict[str, Any],
) -> dict[str, Any]:
    months = pd.to_datetime(frame["purchase_timestamp"]).dt.to_period("M").astype(str)
    month_names = np.array(["2018-05", "2018-06", "2018-07", "2018-08"])
    blocks: dict[str, dict[str, Any]] = {}
    for month in month_names:
        positions = np.flatnonzero(months.to_numpy() == month)
        item = monthly_top10[month]
        blocks[month] = {
            "y": frame.iloc[positions][TARGET_COLUMN].to_numpy(dtype=np.int8),
            "probability": probability[positions],
            "alert_count": item["alert_count"],
            "detected": item["detected_delays"],
        }
    rng = np.random.default_rng(BOOTSTRAP_RANDOM_STATE)
    samples: dict[str, list[float]] = {
        name: []
        for name in (
            "average_precision",
            "roc_auc",
            "brier_score",
            "top10_precision",
            "top10_recall",
            "top10_lift",
        )
    }
    for _ in range(BOOTSTRAP_REPLICATIONS):
        selected = rng.choice(month_names, size=len(month_names), replace=True)
        y = np.concatenate([blocks[month]["y"] for month in selected])
        scores = np.concatenate([blocks[month]["probability"] for month in selected])
        if np.unique(y).size == 2:
            samples["average_precision"].append(float(average_precision_score(y, scores)))
            samples["roc_auc"].append(float(roc_auc_score(y, scores)))
        samples["brier_score"].append(float(brier_score_loss(y, scores)))
        alerts = int(sum(blocks[month]["alert_count"] for month in selected))
        detected = int(sum(blocks[month]["detected"] for month in selected))
        positives = int(y.sum())
        precision = detected / alerts
        recall = detected / positives if positives else 0.0
        prevalence = float(y.mean())
        samples["top10_precision"].append(float(precision))
        samples["top10_recall"].append(float(recall))
        samples["top10_lift"].append(
            float(precision / prevalence) if prevalence else 0.0
        )

    intervals: dict[str, Any] = {}
    for name, values in samples.items():
        array = np.asarray(values, dtype=np.float64)
        intervals[name] = (
            {
                "lower_2_5_percent": float(np.quantile(array, 0.025)),
                "median": float(np.quantile(array, 0.50)),
                "upper_97_5_percent": float(np.quantile(array, 0.975)),
                "successful_replications": int(len(array)),
            }
            if len(array)
            else {
                "lower_2_5_percent": None,
                "median": None,
                "upper_97_5_percent": None,
                "successful_replications": 0,
            }
        )
    return {
        "method": "calendar_month_block_bootstrap_with_replacement",
        "replications": BOOTSTRAP_REPLICATIONS,
        "random_state": BOOTSTRAP_RANDOM_STATE,
        "confidence_level": 0.95,
        "block_count": 4,
        "intervals": intervals,
        "limitations": (
            "Only four calendar-month blocks; intervals are descriptive and coarse."
        ),
    }


def evaluate_holdout_once(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    candidate_dir: Path = CANDIDATE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Open and evaluate the final holdout exactly once under the frozen protocol."""

    protocol_sha256 = verify_frozen_protocol(output_dir)
    preflight = json.loads((output_dir / "preflight_report.json").read_text(encoding="utf-8"))
    if not preflight.get("all_checks_passed"):
        raise BacktestError("Préflight Phase 5.5 non validé.")
    opened_path = output_dir / "evaluation_opened.json"
    if opened_path.exists():
        raise BacktestError("Le holdout Phase 5.5 a déjà été ouvert.")
    forbidden_existing = (
        "holdout_summary.json",
        "global_metrics.json",
        "monthly_metrics.json",
        "top10_results.json",
        "drift_analysis.json",
        "comparison_with_development.json",
        "final_decision.json",
        "artifact_manifest.json",
    )
    if any((output_dir / name).exists() for name in forbidden_existing):
        raise BacktestError("Des résultats Phase 5.5 existent déjà.")
    validate_artifact_manifest(candidate_dir / "artifact_manifest.json")
    pipeline_path = candidate_dir / "candidate_pipeline.joblib"
    if _sha256(pipeline_path) != preflight["candidate_pipeline_sha256"]:
        raise BacktestError("Le candidat a changé après le préflight.")

    # Development data is loaded before the irreversible holdout-open marker.
    development = load_phase54_strict_dataset(input_dir)
    development_ids = set(development["order_id"].astype(str))
    candidate_train = development.loc[
        pd.to_datetime(development["purchase_timestamp"]).lt("2018-02-01")
        & pd.to_datetime(development["label_observed_at"]).lt("2018-02-01")
    ].copy()
    if len(candidate_train) != 47794 or int(candidate_train[TARGET_COLUMN].sum()) != 2387:
        raise BacktestError("Référence historique du candidat incohérente.")

    _atomic_json(
        {
            "phase": PHASE,
            "protocol_sha256": protocol_sha256,
            "opened_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": HOLDOUT_FILENAME,
            "read_count_authorized": 1,
            "irreversible_stop_marker": True,
        },
        opened_path,
    )

    # This is the single authorized final-test read.
    holdout_path = input_dir / HOLDOUT_FILENAME
    holdout = pd.read_parquet(holdout_path)
    validate_strict_dataset(holdout)
    feature_dtypes = validate_strict_dtypes(holdout)
    purchase = pd.to_datetime(holdout["purchase_timestamp"], errors="coerce")
    if purchase.isna().any() or purchase.lt(HOLDOUT_START).any() or purchase.ge(HOLDOUT_END).any():
        raise BacktestError("Période réelle du holdout incohérente.")
    if set(holdout["order_id"].astype(str)) & development_ids:
        raise BacktestError("Chevauchement entre développement et holdout.")
    if list(holdout.columns) != [
        "order_id",
        "purchase_timestamp",
        "label_observed_at",
        *STRICT_FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]:
        raise BacktestError("Ordre des colonnes holdout inattendu.")

    candidate = joblib.load(pipeline_path)
    x_holdout = contract_matrix(holdout, "strict_v1")
    y_holdout = holdout[TARGET_COLUMN].astype("int8").to_numpy()
    probability = score_frozen_candidate(candidate, x_holdout)
    global_metrics = safe_probability_metrics(y_holdout, probability)

    monthly_top10, alert_mask = _monthly_top10(holdout, probability)
    top10_aggregate = _aggregate_monthly_top10(
        y_holdout, alert_mask, monthly_top10
    )
    top10_results = {
        "policy": {
            "capacity": TOP10_CAPACITY,
            "rounding": "ceil",
            "tie_breaker": "order_id_ascending",
            "scope": "retrospective_monthly_batch",
        },
        "monthly": monthly_top10,
        **top10_aggregate,
    }

    month_values = purchase.dt.to_period("M").astype(str)
    monthly_metrics: dict[str, Any] = {}
    for month in ("2018-05", "2018-06", "2018-07", "2018-08"):
        positions = np.flatnonzero(month_values.to_numpy() == month)
        if len(positions) == 0:
            monthly_metrics[month] = {"rows": 0, "defined": False}
            continue
        metrics = safe_probability_metrics(y_holdout[positions], probability[positions])
        metrics["defined"] = True
        metrics["top10"] = monthly_top10[month]
        monthly_metrics[month] = metrics

    x_reference = contract_matrix(candidate_train, "strict_v1")
    reference_probability = score_frozen_candidate(candidate, x_reference)
    drift = drift_report(
        x_reference,
        x_holdout,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        boolean_features=BOOLEAN_FEATURES,
        reference_scores=reference_probability,
        current_scores=probability,
    )
    drift.update(
        {
            "reference": {
                "definition": "frozen candidate train before 2018-02-01",
                "rows": int(len(candidate_train)),
                "delayed": int(candidate_train[TARGET_COLUMN].sum()),
                "holdout_used_as_reference": False,
            },
            "customer_gps_coverage_reference": float(
                candidate_train["customer_has_gps"].mean()
            ),
            "customer_gps_coverage_holdout": float(
                holdout["customer_has_gps"].mean()
            ),
            "customer_state_reference_distribution": {
                str(key): float(value)
                for key, value in candidate_train["customer_state"]
                .value_counts(normalize=True)
                .sort_index()
                .items()
            },
            "customer_state_holdout_distribution": {
                str(key): float(value)
                for key, value in holdout["customer_state"]
                .value_counts(normalize=True)
                .sort_index()
                .items()
            },
            "monthly_prevalence": {
                month: (
                    monthly_metrics[month]["prevalence"]
                    if monthly_metrics[month].get("defined")
                    else None
                )
                for month in monthly_metrics
            },
            "monthly_score_distribution": {
                month: (
                    monthly_metrics[month]["score_distribution"]
                    if monthly_metrics[month].get("defined")
                    else None
                )
                for month in monthly_metrics
            },
        }
    )

    uncertainty = _block_bootstrap(holdout, probability, monthly_top10)
    development_reference = {
        "aggregation": "macro_mean_across_january_to_april_folds",
        "average_precision": 0.167930,
        "roc_auc": 0.628907,
        "precision_at_10_percent": 0.1865,
        "recall_at_10_percent": 0.2068,
        "lift_at_10_percent": 2.067,
    }
    final_top10 = top10_aggregate["monthly_batch_totals"]
    comparison = {
        "aggregation_warning": (
            "Development values are macro fold means; holdout global ranking metrics "
            "are computed on concatenated rows and top10 on concatenated monthly alerts."
        ),
        "development_reference": development_reference,
        "holdout": {
            "average_precision_global": global_metrics["average_precision"],
            "roc_auc_global": global_metrics["roc_auc"],
            "precision_at_10_percent_monthly_batches": final_top10[
                "precision_at_10_percent"
            ],
            "recall_at_10_percent_monthly_batches": final_top10[
                "recall_at_10_percent"
            ],
            "lift_at_10_percent_monthly_batches": final_top10[
                "lift_at_10_percent"
            ],
            "monthly_macro_top10": top10_aggregate["macro_monthly_mean"],
        },
        "absolute_differences_descriptive_only": {
            "average_precision": (
                float(
                    global_metrics["average_precision"]
                    - development_reference["average_precision"]
                )
                if global_metrics["average_precision"] is not None
                else None
            ),
            "roc_auc": (
                float(global_metrics["roc_auc"] - development_reference["roc_auc"])
                if global_metrics["roc_auc"] is not None
                else None
            ),
            "top10_precision": float(
                final_top10["precision_at_10_percent"]
                - development_reference["precision_at_10_percent"]
            ),
            "top10_recall": float(
                final_top10["recall_at_10_percent"]
                - development_reference["recall_at_10_percent"]
            ),
            "top10_lift": float(
                final_top10["lift_at_10_percent"]
                - development_reference["lift_at_10_percent"]
            ),
        },
        "uncertainty": uncertainty,
    }

    holdout_summary = {
        "rows": int(len(holdout)),
        "expected_rows_diagnostic_only": 25352,
        "row_count_matches_expectation": len(holdout) == 25352,
        "delayed_rows": int(y_holdout.sum()),
        "on_time_rows": int(len(holdout) - y_holdout.sum()),
        "prevalence": float(y_holdout.mean()),
        "purchase_min": purchase.min().isoformat(),
        "purchase_max": purchase.max().isoformat(),
        "label_observed_max": pd.to_datetime(holdout["label_observed_at"]).max().isoformat(),
        "unique_order_ids": bool(not holdout["order_id"].duplicated().any()),
        "development_overlap": 0,
        "feature_missing_rates": feature_missing_rates(holdout),
        "persisted_dtypes": feature_dtypes,
        "strict_dtypes_valid": True,
        "classes": sorted(holdout[TARGET_COLUMN].astype(int).unique().tolist()),
        "source_read_count": 1,
    }
    final_decision = {
        "phase": PHASE,
        "evaluation_experimentally_valid": True,
        "technical_evaluation": {
            "decision": "GO",
            "assessment": "valid_frozen_one_shot_evaluation",
        },
        "generalization": {
            "decision": "NO-GO",
            "assessment": "weak_and_temporally_unstable_on_final_holdout",
            "post_holdout_interpretation_not_a_selection_rule": True,
            "average_precision": global_metrics["average_precision"],
            "roc_auc": global_metrics["roc_auc"],
            "top10_lift": final_top10["lift_at_10_percent"],
            "worst_month_top10_lift": min(
                item["lift_at_k"]
                for item in monthly_top10.values()
                if item.get("defined")
            ),
        },
        "production": {
            "decision": "NO-GO",
            "assessment": "not_production_ready",
        },
        "production_ready": False,
        "candidate_changed": False,
        "retraining_performed": False,
        "recalibration_performed": False,
        "threshold_optimization_performed": False,
        "top10_capacity_changed": False,
        "protocol_sha256": protocol_sha256,
        "holdout_opened_once": True,
        "second_evaluation_authorized": False,
        "limitations": [
            "uncalibrated scores",
            "monthly batch top-k is not single-order real-time inference",
            "no validated business cost or operational alert budget",
            "population conditional on ultimately delivered orders",
            "not production ready",
        ],
    }
    dependency_versions = {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
    }

    _atomic_json(holdout_summary, output_dir / "holdout_summary.json")
    _atomic_json(global_metrics, output_dir / "global_metrics.json")
    _atomic_json(monthly_metrics, output_dir / "monthly_metrics.json")
    _atomic_json(top10_results, output_dir / "top10_results.json")
    _atomic_json(drift, output_dir / "drift_analysis.json")
    _atomic_json(comparison, output_dir / "comparison_with_development.json")
    _atomic_json(final_decision, output_dir / "final_decision.json")
    _atomic_json(dependency_versions, output_dir / "dependency_versions.json")
    if verify_frozen_protocol(output_dir) != protocol_sha256:
        raise BacktestError("Le protocole a changé pendant l'évaluation.")

    artifact_paths = [
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    manifest = {
        "phase": PHASE,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256,
        "candidate_pipeline_sha256": preflight["candidate_pipeline_sha256"],
        "holdout_opened_once": True,
        "holdout_source_read_count": 1,
        "candidate_saved_or_modified": False,
        "artifacts": {
            path.name: {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(artifact_paths)
        },
    }
    _atomic_json(manifest, output_dir / "artifact_manifest.json")
    validate_artifact_manifest(output_dir / "artifact_manifest.json")
    return {
        "holdout_summary": holdout_summary,
        "global_metrics": global_metrics,
        "monthly_metrics": monthly_metrics,
        "top10_results": top10_results,
        "drift_analysis": drift,
        "comparison_with_development": comparison,
        "final_decision": final_decision,
        "artifact_manifest": manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5.5 final holdout evaluation.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=CANDIDATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare:
        report = prepare_phase55(
            candidate_dir=args.candidate_dir.resolve(),
            output_dir=args.output_dir.resolve(),
        )
        print(
            "Préflight Phase 5.5 réussi | protocole="
            + report["evaluation_protocol_sha256"]
            + " | holdout fermé"
        )
    else:
        result = evaluate_holdout_once(
            input_dir=args.input_dir.resolve(),
            candidate_dir=args.candidate_dir.resolve(),
            output_dir=args.output_dir.resolve(),
        )
        average_precision = result["global_metrics"]["average_precision"]
        average_precision_display = (
            f"{average_precision:.6f}" if average_precision is not None else "undefined"
        )
        print(
            f"Évaluation finale terminée | lignes={result['holdout_summary']['rows']} "
            f"| AP={average_precision_display} "
            "| seconde évaluation interdite"
        )


if __name__ == "__main__":
    main()
