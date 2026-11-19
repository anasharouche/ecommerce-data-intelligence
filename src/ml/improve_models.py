"""Phase 5.4 conditional features, calibration and temporal backtesting CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
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

from config.paths import BASE_DIR
from src.ml.backtesting import (
    build_fold_partitions,
    build_phase54_pipeline,
    contract_matrix,
    evaluate_calibration_variants,
    fit_calibration_variants,
    fold_summary,
    load_phase54_strict_dataset,
    monthly_backtest_folds,
)
from src.ml.contract import FEATURE_SCHEMA_VERSION, STRICT_FEATURE_COLUMNS, TARGET_COLUMN
from src.ml.contract_v2 import (
    ENRICHED_BOOLEAN_FEATURES,
    ENRICHED_CATEGORICAL_FEATURES,
    ENRICHED_FEATURE_COLUMNS,
    ENRICHED_FEATURE_SCHEMA_VERSION,
    ENRICHED_NUMERIC_FEATURES,
    enriched_feature_contract_as_dict,
)
from src.ml.drift import drift_report
from src.ml.enriched import build_enriched_dataset, extract_enriched_sources
from src.ml.evaluation import best_f1_threshold, calibration_bins, subgroup_metrics
from src.ml.training import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    target_vector,
)


logger = logging.getLogger(__name__)
RANDOM_STATE = 42
HISTORICAL_V1_THRESHOLD = 0.45956620240681
DEFAULT_INPUT_DIR = Path(BASE_DIR) / "data" / "processed" / "ml"
DEFAULT_OUTPUT_DIR = (
    Path(BASE_DIR)
    / "models"
    / "delivery_delay"
    / "phase_5_4"
    / "conditional_v2_backtest_v1"
)

EXPERIMENTS = (
    ("strict_v1_logistic_regression", "strict_v1", "logistic_regression"),
    ("enriched_v2_logistic_regression", "enriched_v2", "logistic_regression"),
    ("enriched_v2_random_forest", "enriched_v2", "random_forest"),
    ("enriched_v2_hist_gradient_boosting", "enriched_v2", "hist_gradient_boosting"),
)


def _feature_availability_audit() -> dict[str, Any]:
    """Return the evidence-based operational availability decisions."""

    conditional = [
        {
            "domain": "delivery promise",
            "features": ["promised_delivery_days"],
            "source": [
                "dw.fact_orders.purchase_timestamp",
                "dw.fact_orders.estimated_delivery_timestamp",
            ],
            "source_grain": "order",
            "transformation": "calendar-day difference",
            "business_assumption": "promise fixed before scoring",
            "historical_evidence": "column present but no availability timestamp",
            "leakage_risk": "high if promise is generated after scoring",
            "decision": "conditional",
        },
        {
            "domain": "basket values",
            "features": [
                "item_count",
                "total_item_value",
                "total_freight_value",
                "freight_ratio",
            ],
            "source": ["dw.fact_order_items.order_item_id", "price", "freight_value"],
            "source_grain": "order item, aggregated before join",
            "transformation": "counts, sums and freight / price",
            "business_assumption": "final basket and freight known at creation",
            "historical_evidence": "warehouse values have no event availability timestamp",
            "leakage_risk": "availability leakage if finalized after creation",
            "decision": "conditional",
        },
        {
            "domain": "seller topology",
            "features": ["seller_count", "seller_state_count", "is_multi_seller"],
            "source": ["dw.fact_order_items.seller_key", "dw.dim_seller.seller_state"],
            "source_grain": "order item and seller, aggregated per order",
            "transformation": "distinct counts",
            "business_assumption": "seller allocation known at creation",
            "historical_evidence": "allocation timestamp unavailable",
            "leakage_risk": "availability leakage and non-historized dimension",
            "decision": "conditional",
        },
        {
            "domain": "seller-customer distance",
            "features": [
                "distance_km_mean",
                "distance_km_max",
                "distance_km_weighted_by_item_value",
                "distance_coverage_ratio",
                "distance_incomplete",
            ],
            "source": ["dw.dim_customer coordinates", "dw.dim_seller coordinates"],
            "source_grain": "item route, aggregated per order",
            "transformation": "Haversine in km, mean/max/price-weighted mean",
            "business_assumption": "seller allocation and GPS lookup known at creation",
            "historical_evidence": "GPS dimensions are static, not historized",
            "leakage_risk": "snapshot drift rather than direct target leakage",
            "decision": "conditional",
        },
        {
            "domain": "product category and physical attributes",
            "features": [
                "category_count",
                "dominant_category",
                "total_product_weight_g",
                "total_product_volume_cm3",
                "product_physical_coverage_ratio",
                "product_physical_incomplete",
            ],
            "source": ["dw.fact_order_items.product_key", "dw.dim_product"],
            "source_grain": "order item and product, aggregated per order",
            "transformation": "distinct/dominant category, item sums in g and cm3",
            "business_assumption": "versioned product catalog known at creation",
            "historical_evidence": "dimension is not historized",
            "leakage_risk": "future catalog state and checkout availability",
            "decision": "conditional",
        },
    ]
    return {
        "strict_v1": {
            "decision": "authorized_under_existing_project_assumptions",
            "features": list(STRICT_FEATURE_COLUMNS),
        },
        "conditional_v2": conditional,
        "excluded": [
            {
                "features": [
                    "delivered_timestamp",
                    "carrier timestamps",
                    "order_status",
                    "reviews",
                    "approval timestamps",
                    "full-history seller/category target aggregates",
                    "raw identifiers",
                ],
                "decision": "excluded",
                "reason": "post-outcome, post-creation, target leakage or memorization",
            }
        ],
        "production_validation_required": True,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataframe_sha256(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _atomic_json(payload: Any, destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
            newline="\n",
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_joblib(payload: Any, destination: Path, *, overwrite: bool) -> None:
    if destination.exists() and not overwrite:
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
        joblib.dump(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _prepare_output(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(path.is_file() for path in output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Artefacts Phase 5.4 déjà présents : {output_dir}. Utiliser --overwrite."
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def _group_types(contract: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if contract == "strict_v1":
        return NUMERIC_FEATURES, CATEGORICAL_FEATURES, BOOLEAN_FEATURES
    return (
        ENRICHED_NUMERIC_FEATURES,
        ENRICHED_CATEGORICAL_FEATURES,
        ENRICHED_BOOLEAN_FEATURES,
    )


def _aggregate_comparison(backtests: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for experiment_name, experiment in backtests.items():
        for method in ("raw", "sigmoid", "isotonic"):
            probability_metrics = [
                fold["methods"][method]["evaluation"]["threshold_0_5"]
                for fold in experiment["folds"].values()
            ]
            threshold_metrics = [
                fold["methods"][method]["evaluation"][
                    "calibration_selected_threshold"
                ]
                for fold in experiment["folds"].values()
            ]
            mean = lambda rows, key: float(np.mean([row[key] for row in rows]))
            records.append(
                {
                    "experiment": experiment_name,
                    "contract": experiment["contract"],
                    "model": experiment["model"],
                    "calibration_method": method,
                    "fold_count": len(probability_metrics),
                    "mean_average_precision": mean(
                        probability_metrics, "average_precision"
                    ),
                    "mean_roc_auc": mean(probability_metrics, "roc_auc"),
                    "mean_brier_score": mean(probability_metrics, "brier_score"),
                    "mean_log_loss": mean(probability_metrics, "log_loss"),
                    "mean_ece": mean(
                        probability_metrics, "expected_calibration_error"
                    ),
                    "mean_precision_at_prior_threshold": mean(
                        threshold_metrics, "precision"
                    ),
                    "mean_recall_at_prior_threshold": mean(
                        threshold_metrics, "recall"
                    ),
                    "mean_f1_at_prior_threshold": mean(threshold_metrics, "f1"),
                    "mean_false_positive_rate": mean(
                        threshold_metrics, "false_positive_rate"
                    ),
                    "mean_alert_count": mean(threshold_metrics, "alert_count"),
                    "mean_alert_rate": mean(
                        threshold_metrics, "prediction_positive_rate"
                    ),
                    "minimum_fold_average_precision": float(
                        min(row["average_precision"] for row in probability_metrics)
                    ),
                    "maximum_fold_average_precision": float(
                        max(row["average_precision"] for row in probability_metrics)
                    ),
                }
            )
    return sorted(
        records,
        key=lambda row: (
            -row["mean_average_precision"],
            row["mean_brier_score"],
        ),
    )


def _select_candidate(comparison: list[dict[str, Any]]) -> dict[str, Any]:
    calibrated_candidates = [
        row for row in comparison if row["calibration_method"] != "raw"
    ]
    best_ap = max(
        row["mean_average_precision"] for row in calibrated_candidates
    )
    near_best = [
        row
        for row in calibrated_candidates
        if row["mean_average_precision"] >= best_ap - 0.005
    ]
    best_brier = min(row["mean_brier_score"] for row in near_best)
    stable_sigmoid = [
        row
        for row in near_best
        if row["calibration_method"] == "sigmoid"
        and row["mean_brier_score"] <= best_brier + 0.005
    ]
    if stable_sigmoid:
        selected = min(
            stable_sigmoid,
            key=lambda row: (
                -row["mean_average_precision"],
                row["mean_log_loss"],
                row["mean_brier_score"],
            ),
        )
    else:
        selected = min(
            near_best,
            key=lambda row: (
                row["mean_brier_score"],
                row["mean_ece"],
                row["mean_log_loss"],
            ),
        )
    return {
        **selected,
        "selection_policy": (
            "Among sigmoid/isotonic variants, highest mean temporal-fold "
            "Average Precision; within 0.005, prefer sigmoid when its mean "
            "Brier score is within 0.005 of the best calibrated value, then "
            "use log loss. This avoids isotonic 0/1 extremes under drift. "
            "Raw scores remain ranking diagnostics only."
        ),
        "best_mean_average_precision": float(best_ap),
        "near_best_variants": [
            {
                "experiment": row["experiment"],
                "calibration_method": row["calibration_method"],
            }
            for row in near_best
        ],
        "production_ready": False,
        "test_accessed": False,
        "test_evaluated": False,
    }


def _prepare_final_candidate(
    dataset: pd.DataFrame,
    *,
    contract: str,
    model_name: str,
    calibration_method: str,
) -> tuple[Any, dict[str, Any]]:
    """Fit the frozen pre-test candidate as of 2018-05-01 without test access."""

    training_as_of = pd.Timestamp("2018-03-01")
    candidate_as_of = pd.Timestamp("2018-05-01")
    purchase = pd.to_datetime(dataset["purchase_timestamp"])
    observed = pd.to_datetime(dataset["label_observed_at"])
    train = dataset[
        purchase.lt(training_as_of) & observed.lt(training_as_of)
    ].copy()
    calibration = dataset[
        purchase.ge(training_as_of)
        & purchase.lt(candidate_as_of)
        & observed.lt(candidate_as_of)
    ].copy()
    if set(train["order_id"]) & set(calibration["order_id"]):
        raise RuntimeError("Chevauchement du candidat final.")
    x_train = contract_matrix(train, contract)
    x_calibration = contract_matrix(calibration, contract)
    y_train = target_vector(train)
    y_calibration = target_vector(calibration)
    variants = fit_calibration_variants(
        build_phase54_pipeline(contract, model_name),
        x_train=x_train,
        y_train=y_train,
        x_calibration=x_calibration,
        y_calibration=y_calibration,
    )
    estimator = variants[calibration_method]
    calibration_probability = estimator.predict_proba(x_calibration)[:, 1]
    threshold, threshold_metrics = best_f1_threshold(
        y_calibration, calibration_probability
    )
    return estimator, {
        "training_as_of_exclusive": training_as_of.isoformat(),
        "candidate_as_of_exclusive": candidate_as_of.isoformat(),
        "train_rows": int(len(train)),
        "train_delayed": int(y_train.sum()),
        "calibration_rows": int(len(calibration)),
        "calibration_delayed": int(y_calibration.sum()),
        "candidate_threshold": float(threshold),
        "calibration_metrics_at_candidate_threshold": threshold_metrics,
        "calibration_bins": calibration_bins(y_calibration, calibration_probability),
        "test_accessed": False,
        "test_evaluated": False,
    }


def _geographic_robustness(
    dataset: pd.DataFrame,
    *,
    contract: str,
    model_name: str,
    calibration_method: str,
) -> dict[str, Any]:
    """Reproduce the selected variant and evaluate sufficiently large states."""

    results: dict[str, Any] = {}
    for fold in monthly_backtest_folds():
        partitions = build_fold_partitions(dataset, fold)
        x_train = contract_matrix(partitions.train, contract)
        x_calibration = contract_matrix(partitions.calibration, contract)
        x_evaluation = contract_matrix(partitions.evaluation, contract)
        y_train = target_vector(partitions.train)
        y_calibration = target_vector(partitions.calibration)
        variants = fit_calibration_variants(
            build_phase54_pipeline(contract, model_name),
            x_train=x_train,
            y_train=y_train,
            x_calibration=x_calibration,
            y_calibration=y_calibration,
        )
        estimator = variants[calibration_method]
        calibration_probability = estimator.predict_proba(x_calibration)[:, 1]
        threshold, _ = best_f1_threshold(y_calibration, calibration_probability)
        evaluation_probability = estimator.predict_proba(x_evaluation)[:, 1]
        results[fold.name] = {
            "threshold_selected_on_prior_calibration": float(threshold),
            "minimum_state_rows": 200,
            "states": subgroup_metrics(
                partitions.evaluation,
                evaluation_probability,
                group_values=partitions.evaluation["customer_state"],
                threshold=threshold,
                minimum_rows=200,
            ),
        }
    return results


def run_phase54(
    *,
    input_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """Execute all safe Phase 5.4 experiments and persist separate artifacts."""

    _prepare_output(output_dir, overwrite=overwrite)
    strict_dataset = load_phase54_strict_dataset(input_dir)
    order_source, item_source = extract_enriched_sources()
    enriched_build = build_enriched_dataset(order_source, item_source)
    enriched_dataset = enriched_build.dataset

    strict_index = strict_dataset.set_index("order_id")
    enriched_index = enriched_dataset.set_index("order_id")
    if set(strict_index.index) != set(enriched_index.index):
        raise RuntimeError("V1 et V2 ne couvrent pas les mêmes commandes sûres.")
    aligned = enriched_index.loc[strict_index.index]
    if not strict_index[TARGET_COLUMN].equals(aligned[TARGET_COLUMN]):
        raise RuntimeError("Les cibles V1 et V2 divergent.")

    datasets = {"strict_v1": strict_dataset, "enriched_v2": enriched_dataset}
    fold_definitions = monthly_backtest_folds()
    fold_summaries: dict[str, Any] = {}
    backtests: dict[str, Any] = {}
    drift_results: dict[str, Any] = {}

    for experiment_name, contract, model_name in EXPERIMENTS:
        logger.info("Expérience : %s", experiment_name)
        dataset = datasets[contract]
        experiment_folds: dict[str, Any] = {}
        experiment_drift: dict[str, Any] = {}
        numeric, categorical, boolean = _group_types(contract)
        for fold in fold_definitions:
            partitions = build_fold_partitions(dataset, fold)
            if fold.name not in fold_summaries:
                fold_summaries[fold.name] = fold_summary(partitions)
            x_train = contract_matrix(partitions.train, contract)
            x_calibration = contract_matrix(partitions.calibration, contract)
            x_evaluation = contract_matrix(partitions.evaluation, contract)
            y_train = target_vector(partitions.train)
            y_calibration = target_vector(partitions.calibration)
            y_evaluation = target_vector(partitions.evaluation)
            variants = fit_calibration_variants(
                build_phase54_pipeline(contract, model_name),
                x_train=x_train,
                y_train=y_train,
                x_calibration=x_calibration,
                y_calibration=y_calibration,
            )
            evaluated = evaluate_calibration_variants(
                variants,
                x_calibration=x_calibration,
                y_calibration=y_calibration,
                x_evaluation=x_evaluation,
                y_evaluation=y_evaluation,
                historical_threshold=HISTORICAL_V1_THRESHOLD,
            )
            raw_evaluation_scores = evaluated["raw"].pop(
                "evaluation_probabilities"
            )
            for method in ("sigmoid", "isotonic"):
                evaluated[method].pop("evaluation_probabilities")
            raw_train_scores = variants["raw"].predict_proba(x_train)[:, 1]
            experiment_drift[fold.name] = drift_report(
                x_train,
                x_evaluation,
                numeric_features=numeric,
                categorical_features=categorical,
                boolean_features=boolean,
                reference_scores=raw_train_scores,
                current_scores=raw_evaluation_scores,
            )
            experiment_folds[fold.name] = {"methods": evaluated}
        backtests[experiment_name] = {
            "contract": contract,
            "model": model_name,
            "folds": experiment_folds,
        }
        drift_results[experiment_name] = experiment_drift

    comparison = _aggregate_comparison(backtests)
    selection = _select_candidate(comparison)
    selected_dataset = datasets[selection["contract"]]
    geographic_robustness = _geographic_robustness(
        selected_dataset,
        contract=selection["contract"],
        model_name=selection["model"],
        calibration_method=selection["calibration_method"],
    )
    candidate_pipeline, candidate_preparation = _prepare_final_candidate(
        selected_dataset,
        contract=selection["contract"],
        model_name=selection["model"],
        calibration_method=selection["calibration_method"],
    )
    selection["candidate_preparation"] = candidate_preparation
    selection["artifact"] = "candidate_pipeline.joblib"

    dataset_audit = {
        "strict_v1_rows": int(len(strict_dataset)),
        "enriched_v2_rows": int(len(enriched_dataset)),
        "same_order_ids": True,
        "same_targets": True,
        "on_time_rows": int((strict_dataset[TARGET_COLUMN] == 0).sum()),
        "delayed_rows": int((strict_dataset[TARGET_COLUMN] == 1).sum()),
        "maximum_purchase_timestamp": pd.Timestamp(
            strict_dataset["purchase_timestamp"].max()
        ).isoformat(),
        "source_order_rows_before_cutoff": enriched_build.source_order_rows,
        "source_item_rows_before_cutoff": enriched_build.source_item_rows,
        "conditional_feature_count": len(ENRICHED_FEATURE_COLUMNS) - len(
            STRICT_FEATURE_COLUMNS
        ),
        "total_v2_feature_count": len(ENRICHED_FEATURE_COLUMNS),
        "missing_rates_v2": {
            column: float(enriched_dataset[column].isna().mean())
            for column in ENRICHED_FEATURE_COLUMNS
        },
        "strict_dataframe_sha256": _dataframe_sha256(strict_dataset),
        "enriched_dataframe_sha256": _dataframe_sha256(enriched_dataset),
        "test_file_read": False,
    }
    experiment_config = {
        "phase": "5.4",
        "random_state": RANDOM_STATE,
        "strict_contract": FEATURE_SCHEMA_VERSION,
        "conditional_contract": ENRICHED_FEATURE_SCHEMA_VERSION,
        "safe_cutoff_exclusive": "2018-05-01T00:00:00",
        "calibration_window_months": 2,
        "historical_v1_threshold": HISTORICAL_V1_THRESHOLD,
        "experiments": [
            {"name": name, "contract": contract, "model": model}
            for name, contract, model in EXPERIMENTS
        ],
        "calibration_methods": ["raw", "sigmoid", "isotonic"],
        "versions": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_accessed": False,
        "test_evaluated": False,
    }

    reports: dict[str, Any] = {
        "feature_contract_v2": enriched_feature_contract_as_dict(),
        "feature_availability_audit": _feature_availability_audit(),
        "experiment_config": experiment_config,
        "dataset_audit": dataset_audit,
        "fold_definitions": fold_summaries,
        "backtest_results": backtests,
        "model_calibration_comparison": comparison,
        "drift_analysis": drift_results,
        "geographic_robustness": geographic_robustness,
        "selected_candidate": selection,
    }
    json_paths: dict[str, Path] = {}
    for name, payload in reports.items():
        path = output_dir / f"{name}.json"
        _atomic_json(payload, path, overwrite=overwrite)
        json_paths[name] = path
    candidate_path = output_dir / "candidate_pipeline.joblib"
    _atomic_joblib(candidate_pipeline, candidate_path, overwrite=overwrite)

    artifact_paths = {**json_paths, "candidate_pipeline": candidate_path}
    manifest = {
        "phase": "5.4",
        "selected_experiment": selection["experiment"],
        "selected_calibration": selection["calibration_method"],
        "test_accessed": False,
        "test_evaluated": False,
        "artifacts": {
            name: {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in sorted(artifact_paths.items())
        },
    }
    _atomic_json(
        manifest, output_dir / "artifact_manifest.json", overwrite=overwrite
    )
    return {**reports, "artifact_manifest": manifest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exécuter la Phase 5.4 avant mai 2018 sans lire ni évaluer le test final."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s | %(message)s",
    )
    result = run_phase54(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        overwrite=args.overwrite,
    )
    candidate = result["selected_candidate"]
    logger.info(
        "Candidat Phase 5.4 : %s / %s | calibration=%s | test intact",
        candidate["contract"],
        candidate["model"],
        candidate["calibration_method"],
    )


if __name__ == "__main__":
    main()
