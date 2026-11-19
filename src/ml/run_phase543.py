"""Single preregistered Phase 5.4.3 development campaign."""

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
    BacktestError,
    build_phase54_pipeline,
    contract_matrix,
    load_phase54_strict_dataset,
    monthly_backtest_folds,
)
from src.ml.contract import FEATURE_SCHEMA_VERSION, TARGET_COLUMN
from src.ml.contract_v2 import ENRICHED_FEATURE_SCHEMA_VERSION
from src.ml.drift import drift_report
from src.ml.enriched import build_enriched_dataset, extract_enriched_sources
from src.ml.evaluation import evaluate_probabilities
from src.ml.historical_calibration import (
    CAMPAIGN_VERSION,
    HISTORICAL_CALIBRATION_MONTHS,
    MIN_CALIBRATION_POSITIVES,
    TOPK_CAPACITIES,
    build_expanded_historical_partitions,
    deterministic_topk_metrics,
    fit_expanded_calibration_strategies,
    historical_partition_summary,
)
from src.ml.training import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    target_vector,
)
from src.ml.contract_v2 import (
    ENRICHED_BOOLEAN_FEATURES,
    ENRICHED_CATEGORICAL_FEATURES,
    ENRICHED_NUMERIC_FEATURES,
)


logger = logging.getLogger(__name__)
PHASE = "5.4.3"
DEFAULT_INPUT_DIR = Path(BASE_DIR) / "data" / "processed" / "ml"
DEFAULT_OUTPUT_DIR = (
    Path(BASE_DIR)
    / "models"
    / "delivery_delay"
    / "phase_5_4_3"
    / "historical_calibration_alert_policy_v1"
)
EXPERIMENTS = (
    ("strict_v1_logistic_regression", "strict_v1"),
    ("enriched_v2_logistic_regression", "enriched_v2"),
)
PROBABILITY_METRICS = (
    "rows",
    "positive_rows",
    "positive_rate",
    "average_precision",
    "roc_auc",
    "brier_score",
    "log_loss",
    "expected_calibration_error",
)


def _atomic_json(payload: Any, destination: Path) -> None:
    if destination.exists():
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


def _atomic_joblib(payload: Any, destination: Path) -> None:
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
        joblib.dump(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_artifact_manifest(manifest_path: Path) -> None:
    """Reject a missing, resized or hash-mismatched campaign artifact."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, record in manifest["artifacts"].items():
        artifact = manifest_path.parent / record["filename"]
        if not artifact.is_file():
            raise BacktestError(f"Artefact absent du manifeste : {name}.")
        if artifact.stat().st_size != record["bytes"]:
            raise BacktestError(f"Taille divergente pour l'artefact : {name}.")
        if _sha256(artifact) != record["sha256"]:
            raise BacktestError(f"SHA-256 divergent pour l'artefact : {name}.")


def _dataframe_sha256(frame: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _feature_groups(
    contract: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if contract == "strict_v1":
        return NUMERIC_FEATURES, CATEGORICAL_FEATURES, BOOLEAN_FEATURES
    return (
        ENRICHED_NUMERIC_FEATURES,
        ENRICHED_CATEGORICAL_FEATURES,
        ENRICHED_BOOLEAN_FEATURES,
    )


def _metric_projection(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics[key] for key in PROBABILITY_METRICS}


def _aggregate_probability_metrics(
    folds: list[dict[str, Any]],
    y_concat: np.ndarray,
    probability_concat: np.ndarray,
) -> dict[str, Any]:
    rows = np.asarray([item["rows"] for item in folds], dtype=np.float64)
    total = float(rows.sum())
    keys = PROBABILITY_METRICS[2:]
    concatenated = _metric_projection(
        evaluate_probabilities(y_concat, probability_concat, threshold=0.5)
    )
    return {
        "macro_fold_mean": {
            key: float(np.mean([float(item[key]) for item in folds])) for key in keys
        },
        "row_weighted_fold_mean": {
            key: float(
                sum(float(item[key]) * float(item["rows"]) for item in folds)
                / total
            )
            for key in keys
        },
        "concatenated_out_of_sample": concatenated,
    }


def _aggregate_topk(
    fold_rows: list[dict[str, Any]],
    y_concat: np.ndarray,
    probability_concat: np.ndarray,
    identifiers_concat: np.ndarray,
    capacity: float,
) -> dict[str, Any]:
    metric_keys = ("precision_at_k", "recall_at_k", "lift_at_k", "alert_rate")
    total_rows = float(sum(item["rows"] for item in fold_rows))
    return {
        "macro_fold_mean": {
            key: float(np.mean([float(item[key]) for item in fold_rows]))
            for key in metric_keys
        },
        "row_weighted_fold_mean": {
            key: float(
                sum(float(item[key]) * float(item["rows"]) for item in fold_rows)
                / total_rows
            )
            for key in metric_keys
        },
        "totals_across_monthly_batches": {
            "rows": int(sum(item["rows"] for item in fold_rows)),
            "alert_count": int(sum(item["alert_count"] for item in fold_rows)),
            "detected_delays": int(
                sum(item["detected_delays"] for item in fold_rows)
            ),
            "false_alerts": int(sum(item["false_alerts"] for item in fold_rows)),
            "missed_delays": int(sum(item["missed_delays"] for item in fold_rows)),
        },
        "concatenated_out_of_sample_single_batch": deterministic_topk_metrics(
            y_concat,
            probability_concat,
            identifiers_concat,
            capacity=capacity,
        ),
    }


def _candidate_partitions(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_as_of = pd.Timestamp("2018-02-01")
    candidate_as_of = pd.Timestamp("2018-05-01")
    purchase = pd.to_datetime(dataset["purchase_timestamp"], errors="coerce")
    observed = pd.to_datetime(dataset["label_observed_at"], errors="coerce")
    train = dataset.loc[
        purchase.lt(training_as_of) & observed.lt(training_as_of)
    ].copy()
    calibration = dataset.loc[
        purchase.ge(training_as_of)
        & purchase.lt(candidate_as_of)
        & observed.lt(candidate_as_of)
    ].copy()
    if set(train["order_id"].astype(str)) & set(calibration["order_id"].astype(str)):
        raise BacktestError("Chevauchement des partitions candidat 5.4.3.")
    for name, frame in (("train", train), ("calibration", calibration)):
        if frame.empty or frame["order_id"].duplicated().any():
            raise BacktestError(f"Partition candidat invalide : {name}.")
        if set(frame[TARGET_COLUMN].astype(int).unique()) != {0, 1}:
            raise BacktestError(f"Classes candidat insuffisantes : {name}.")
    if not train["label_observed_at"].lt(training_as_of).all():
        raise BacktestError("Label train candidat immature.")
    if not calibration["label_observed_at"].lt(candidate_as_of).all():
        raise BacktestError("Label calibration candidat immature.")
    return train.reset_index(drop=True), calibration.reset_index(drop=True)


def _describe(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "delayed": int(frame[TARGET_COLUMN].sum()),
        "purchase_min": pd.Timestamp(frame["purchase_timestamp"].min()).isoformat(),
        "purchase_max": pd.Timestamp(frame["purchase_timestamp"].max()).isoformat(),
        "label_observed_max": pd.Timestamp(
            frame["label_observed_at"].max()
        ).isoformat(),
    }


def run_phase543(*, input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Execute the one allowed preregistered Phase 5.4.3 campaign."""

    config_path = output_dir / "experiment_config.json"
    if not config_path.is_file():
        raise FileNotFoundError("Configuration préenregistrée absente.")
    extra_existing = [path for path in output_dir.iterdir() if path != config_path]
    if extra_existing:
        raise FileExistsError(
            "La campagne 5.4.3 a déjà produit des artefacts; réexécution interdite."
        )
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    if config.get("campaign_version") != CAMPAIGN_VERSION:
        raise ValueError("Version de campagne préenregistrée inattendue.")
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()

    strict_dataset = load_phase54_strict_dataset(input_dir)
    order_source, item_source = extract_enriched_sources()
    enriched_build = build_enriched_dataset(order_source, item_source)
    enriched_dataset = enriched_build.dataset
    strict_index = strict_dataset.set_index("order_id")
    enriched_index = enriched_dataset.set_index("order_id")
    if set(strict_index.index) != set(enriched_index.index):
        raise BacktestError("V1 et V2 ne couvrent pas les mêmes commandes.")
    if not strict_index[TARGET_COLUMN].equals(
        enriched_index.loc[strict_index.index, TARGET_COLUMN]
    ):
        raise BacktestError("Les cibles V1 et V2 divergent.")
    datasets = {"strict_v1": strict_dataset, "enriched_v2": enriched_dataset}

    # Feasibility is persisted before any model fit.
    feasibility_folds: dict[str, Any] = {}
    partitions_by_contract: dict[str, dict[str, Any]] = {}
    for contract, dataset in datasets.items():
        partitions_by_contract[contract] = {}
        for fold in monthly_backtest_folds():
            partitions = build_expanded_historical_partitions(dataset, fold)
            partitions_by_contract[contract][fold.name] = partitions
            if contract == "strict_v1":
                feasibility_folds[fold.name] = historical_partition_summary(partitions)
            else:
                strict_partitions = partitions_by_contract["strict_v1"][fold.name]
                for name in ("train", "calibration", "evaluation"):
                    if set(getattr(strict_partitions, name)["order_id"]) != set(
                        getattr(partitions, name)["order_id"]
                    ):
                        raise BacktestError("Population V1/V2 non appariée en 5.4.3.")
    feasibility = {
        "campaign_version": CAMPAIGN_VERSION,
        "computed_before_model_fit": True,
        "historical_calibration_months": HISTORICAL_CALIBRATION_MONTHS,
        "minimum_calibration_positive_rows": MIN_CALIBRATION_POSITIVES,
        "same_v1_v2_orders_and_targets": True,
        "all_folds_feasible": all(
            row["calibration_feasible"] for row in feasibility_folds.values()
        ),
        "folds": feasibility_folds,
        "holdout_accessed": False,
    }
    _atomic_json(feasibility, output_dir / "feasibility_report.json")
    _atomic_json(feasibility_folds, output_dir / "temporal_folds.json")

    ranking_metrics: dict[str, Any] = {}
    calibration_comparison: dict[str, Any] = {}
    topk_results: dict[str, Any] = {}
    drift_analysis: dict[str, Any] = {}
    strategy_admissibility: dict[str, dict[str, bool]] = {}

    for experiment_name, contract in EXPERIMENTS:
        logger.info("Campagne 5.4.3 : %s", experiment_name)
        fold_metrics: dict[str, Any] = {}
        fold_calibration: dict[str, Any] = {}
        fold_topk: dict[str, Any] = {}
        fold_drift: dict[str, Any] = {}
        concatenated: dict[str, dict[str, list[np.ndarray]]] = {}
        admissibility: dict[str, list[bool]] = {
            "uncalibrated_raw": [],
            "sigmoid_expanded": [],
            "isotonic_expanded": [],
        }
        numeric, categorical, boolean = _feature_groups(contract)
        for fold in monthly_backtest_folds():
            partitions = partitions_by_contract[contract][fold.name]
            x_train = contract_matrix(partitions.train, contract)
            x_calibration = contract_matrix(partitions.calibration, contract)
            x_evaluation = contract_matrix(partitions.evaluation, contract)
            y_train = target_vector(partitions.train)
            y_calibration = target_vector(partitions.calibration)
            y_evaluation = target_vector(partitions.evaluation)
            strategies = fit_expanded_calibration_strategies(
                build_phase54_pipeline(contract, "logistic_regression"),
                x_train=x_train,
                y_train=y_train,
                x_calibration=x_calibration,
                y_calibration=y_calibration,
            )
            fold_calibration[fold.name] = strategies.diagnostics
            fold_metrics[fold.name] = {}
            fold_topk[fold.name] = {}
            for strategy_name, estimator in strategies.estimators.items():
                diagnostic = strategies.diagnostics[strategy_name]
                admissibility[strategy_name].append(bool(diagnostic["admissible"]))
                if not diagnostic["admissible"]:
                    fold_metrics[fold.name][strategy_name] = {
                        "admissible": False,
                        "reason": diagnostic["status"],
                    }
                    continue
                probability = estimator.predict_proba(x_evaluation)[:, 1]
                metrics = _metric_projection(
                    evaluate_probabilities(y_evaluation, probability, threshold=0.5)
                )
                metrics["admissible"] = True
                fold_metrics[fold.name][strategy_name] = metrics
                fold_topk[fold.name][strategy_name] = {
                    f"top_{int(capacity * 100)}_percent": deterministic_topk_metrics(
                        y_evaluation,
                        probability,
                        partitions.evaluation["order_id"],
                        capacity=capacity,
                    )
                    for capacity in TOPK_CAPACITIES
                }
                bucket = concatenated.setdefault(
                    strategy_name, {"y": [], "probability": [], "order_id": []}
                )
                bucket["y"].append(y_evaluation.to_numpy(dtype=np.int8))
                bucket["probability"].append(probability)
                bucket["order_id"].append(
                    partitions.evaluation["order_id"].astype(str).to_numpy()
                )
            raw_estimator = strategies.estimators["uncalibrated_raw"]
            raw_train_probability = raw_estimator.predict_proba(x_train)[:, 1]
            raw_evaluation_probability = raw_estimator.predict_proba(x_evaluation)[:, 1]
            drift = drift_report(
                x_train,
                x_evaluation,
                numeric_features=numeric,
                categorical_features=categorical,
                boolean_features=boolean,
                reference_scores=raw_train_probability,
                current_scores=raw_evaluation_probability,
            )
            drift["geographic_diagnostics"] = {
                "customer_state_psi": drift["feature_psi"]["customer_state"],
                "customer_gps_coverage_train": float(
                    partitions.train["customer_has_gps"].mean()
                ),
                "customer_gps_coverage_evaluation": float(
                    partitions.evaluation["customer_has_gps"].mean()
                ),
            }
            if contract == "enriched_v2":
                drift["geographic_diagnostics"].update(
                    {
                        "distance_coverage_mean_train": float(
                            partitions.train["distance_coverage_ratio"].mean()
                        ),
                        "distance_coverage_mean_evaluation": float(
                            partitions.evaluation["distance_coverage_ratio"].mean()
                        ),
                    }
                )
            fold_drift[fold.name] = drift

        aggregates: dict[str, Any] = {}
        topk_aggregates: dict[str, Any] = {}
        for strategy_name, arrays in concatenated.items():
            y_concat = np.concatenate(arrays["y"])
            probability_concat = np.concatenate(arrays["probability"])
            identifiers_concat = np.concatenate(arrays["order_id"])
            probability_rows = [
                fold_metrics[fold.name][strategy_name]
                for fold in monthly_backtest_folds()
                if fold_metrics[fold.name][strategy_name].get("admissible")
            ]
            aggregates[strategy_name] = _aggregate_probability_metrics(
                probability_rows, y_concat, probability_concat
            )
            topk_aggregates[strategy_name] = {}
            for capacity in TOPK_CAPACITIES:
                key = f"top_{int(capacity * 100)}_percent"
                topk_rows = [
                    fold_topk[fold.name][strategy_name][key]
                    for fold in monthly_backtest_folds()
                    if strategy_name in fold_topk[fold.name]
                ]
                topk_aggregates[strategy_name][key] = _aggregate_topk(
                    topk_rows,
                    y_concat,
                    probability_concat,
                    identifiers_concat,
                    capacity,
                )
        ranking_metrics[experiment_name] = {
            "contract": contract,
            "model": "logistic_regression",
            "folds": fold_metrics,
            "aggregates": aggregates,
        }
        calibration_comparison[experiment_name] = fold_calibration
        topk_results[experiment_name] = {
            "folds": fold_topk,
            "aggregates": topk_aggregates,
            "monthly_batch_simulation_only": True,
        }
        drift_analysis[experiment_name] = fold_drift
        strategy_admissibility[experiment_name] = {
            name: bool(all(values)) for name, values in admissibility.items()
        }

    _atomic_json(calibration_comparison, output_dir / "calibration_comparison.json")
    _atomic_json(ranking_metrics, output_dir / "ranking_metrics.json")
    _atomic_json(topk_results, output_dir / "topk_policy_results.json")
    _atomic_json(drift_analysis, output_dir / "drift_analysis.json")

    strict_name = "strict_v1_logistic_regression"
    strict_raw_folds = ranking_metrics[strict_name]["folds"]
    strict_topk_folds = topk_results[strict_name]["folds"]
    raw_roc_ok = all(
        strict_raw_folds[fold.name]["uncalibrated_raw"]["roc_auc"] > 0.5
        for fold in monthly_backtest_folds()
    )
    raw_ap_ok = all(
        strict_raw_folds[fold.name]["uncalibrated_raw"]["average_precision"]
        > strict_raw_folds[fold.name]["uncalibrated_raw"]["positive_rate"]
        for fold in monthly_backtest_folds()
    )
    raw_top10_ok = all(
        strict_topk_folds[fold.name]["uncalibrated_raw"]["top_10_percent"][
            "lift_at_k"
        ]
        > 1.0
        for fold in monthly_backtest_folds()
    )
    blockers: list[str] = []
    if not feasibility["all_folds_feasible"]:
        blockers.append("expanded_calibration_infeasible")
    if not raw_roc_ok:
        blockers.append("strict_v1_raw_roc_not_above_0_5_on_every_fold")
    if not raw_ap_ok:
        blockers.append("strict_v1_raw_ap_not_above_prevalence_on_every_fold")
    if not raw_top10_ok:
        blockers.append("strict_v1_raw_top10_lift_not_above_1_on_every_fold")

    decision = "GO" if not blockers else "NO_GO"
    candidate: Any | None = None
    candidate_report: dict[str, Any] = {"prepared": False}
    if decision == "GO":
        train, calibration = _candidate_partitions(strict_dataset)
        candidate_strategies = fit_expanded_calibration_strategies(
            build_phase54_pipeline("strict_v1", "logistic_regression"),
            x_train=contract_matrix(train, "strict_v1"),
            y_train=target_vector(train),
            x_calibration=contract_matrix(calibration, "strict_v1"),
            y_calibration=target_vector(calibration),
        )
        sigmoid_all_folds = strategy_admissibility[strict_name]["sigmoid_expanded"]
        sigmoid_final = candidate_strategies.diagnostics["sigmoid_expanded"][
            "admissible"
        ]
        selected_strategy = (
            "sigmoid_expanded"
            if sigmoid_all_folds and sigmoid_final
            else "uncalibrated_raw"
        )
        candidate = candidate_strategies.estimators[selected_strategy]
        candidate_report = {
            "prepared": True,
            "candidate_version": "strict_v1_logistic_top10_v1",
            "contract": FEATURE_SCHEMA_VERSION,
            "model": "logistic_regression",
            "calibration_strategy": selected_strategy,
            "calibration_status": (
                "calibrated_sigmoid"
                if selected_strategy == "sigmoid_expanded"
                else "uncalibrated_raw"
            ),
            "scores_are_calibrated": selected_strategy == "sigmoid_expanded",
            "topk_policy": {
                "capacity": 0.10,
                "scope": "monthly_batch_simulation",
                "rounding": "ceil",
                "tie_breaker": "order_id_ascending",
                "requires_batch": True,
                "single_order_decision_supported": False,
            },
            "train": _describe(train),
            "calibration": _describe(calibration),
            "calibration_diagnostics": candidate_strategies.diagnostics,
            "random_state": 42,
            "reference_as_of_exclusive": "2018-05-01T00:00:00",
            "holdout_accessed": False,
            "holdout_evaluated": False,
            "production_ready": False,
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
    decision_report = {
        "phase": PHASE,
        "campaign_version": CAMPAIGN_VERSION,
        "decision": decision,
        "decision_scope": "single_experimental_final_holdout_evaluation",
        "production_ready": False,
        "blocking_reasons": blockers,
        "criteria": {
            "all_expanded_calibration_folds_feasible": feasibility[
                "all_folds_feasible"
            ],
            "strict_v1_raw_roc_above_0_5_every_fold": raw_roc_ok,
            "strict_v1_raw_ap_above_prevalence_every_fold": raw_ap_ok,
            "strict_v1_raw_top10_lift_above_1_every_fold": raw_top10_ok,
        },
        "strategy_admissibility": strategy_admissibility,
        "candidate": candidate_report,
        "stop_rule": {
            "development_campaigns_completed": 1,
            "additional_phase_5_4_campaign_authorized": False,
            "hyperparameter_tuning_after_results": False,
            "topk_capacities_changed_after_results": False,
        },
        "experiment_config_sha256": config_sha256,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_accessed": False,
        "holdout_evaluated": False,
        "phase_5_5_started": False,
    }
    _atomic_json(decision_report, output_dir / "decision_report.json")
    _atomic_json(dependency_versions, output_dir / "dependency_versions.json")
    if candidate is not None:
        candidate_path = output_dir / "candidate_pipeline.joblib"
        _atomic_joblib(candidate, candidate_path)
        loaded = joblib.load(candidate_path)
        _, calibration = _candidate_partitions(strict_dataset)
        x_check = contract_matrix(calibration, "strict_v1")
        if not np.allclose(
            candidate.predict_proba(x_check), loaded.predict_proba(x_check), atol=0, rtol=0
        ):
            raise BacktestError("Le candidat rechargé ne reproduit pas ses prédictions.")
        _atomic_json(candidate_report, output_dir / "candidate_metadata.json")

    artifact_paths = [
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    manifest = {
        "phase": PHASE,
        "campaign_version": CAMPAIGN_VERSION,
        "decision": decision,
        "candidate_prepared": candidate is not None,
        "experiment_config_sha256": config_sha256,
        "holdout_accessed": False,
        "holdout_evaluated": False,
        "artifacts": {
            path.stem: {
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
        "feasibility_report": feasibility,
        "ranking_metrics": ranking_metrics,
        "topk_policy_results": topk_results,
        "decision_report": decision_report,
        "artifact_manifest": manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exécuter l'unique campagne préenregistrée Phase 5.4.3."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(levelname)s | %(message)s"
    )
    result = run_phase543(
        input_dir=args.input_dir.resolve(), output_dir=args.output_dir.resolve()
    )
    logger.info(
        "Phase 5.4.3 terminée : décision=%s candidat=%s holdout intact",
        result["decision_report"]["decision"],
        result["decision_report"]["candidate"]["prepared"],
    )


if __name__ == "__main__":
    main()
