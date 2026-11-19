"""Phase 5.4.2 monotone calibration and versioned temporal backtesting."""

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
from src.ml.calibration_policy import (
    CALIBRATION_POLICY_VERSION,
    MAX_ISOTONIC_LARGEST_TIE_FRACTION,
    MIN_ISOTONIC_LEVELS,
    build_policy_fold_partitions,
    fit_monotone_calibration_policy,
    policy_fold_summary,
    ranking_diagnostics,
)
from src.ml.contract import FEATURE_SCHEMA_VERSION, TARGET_COLUMN
from src.ml.contract_v2 import (
    ENRICHED_FEATURE_COLUMNS,
    ENRICHED_FEATURE_SCHEMA_VERSION,
    enriched_feature_contract_as_dict,
)
from src.ml.drift import drift_report
from src.ml.enriched import build_enriched_dataset, extract_enriched_sources
from src.ml.evaluation import (
    best_f1_threshold,
    calibration_bins,
    evaluate_probabilities,
)
from src.ml.improve_models import EXPERIMENTS, HISTORICAL_V1_THRESHOLD
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
PHASE_VERSION = "5.4.2"
CANDIDATE_VERSION = "experimental_pre_holdout_v1"
MIN_CALIBRATION_POSITIVES = 30
MIN_THRESHOLD_POSITIVES = 30
DEFAULT_INPUT_DIR = Path(BASE_DIR) / "data" / "processed" / "ml"
DEFAULT_OUTPUT_DIR = (
    Path(BASE_DIR)
    / "models"
    / "delivery_delay"
    / "phase_5_4_2"
    / "monotone_temporal_v1"
)


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _aggregate_comparison(backtests: dict[str, Any]) -> list[dict[str, Any]]:
    comparison: list[dict[str, Any]] = []
    for experiment_name, experiment in backtests.items():
        folds = list(experiment["folds"].values())
        raw = [fold["raw_evaluation"] for fold in folds]
        selected = [fold["policy_evaluation"]["threshold_0_5"] for fold in folds]
        threshold = [
            fold["policy_evaluation"]["prior_selected_threshold"] for fold in folds
        ]
        alert_rates = [row["prediction_positive_rate"] for row in threshold]
        comparison.append(
            {
                "experiment": experiment_name,
                "contract": experiment["contract"],
                "model": experiment["model"],
                "fold_count": len(folds),
                "selected_methods": [fold["selected_method"] for fold in folds],
                "calibration_statuses": [
                    fold["calibration_status"] for fold in folds
                ],
                "mean_raw_average_precision": _mean(raw, "average_precision"),
                "mean_raw_roc_auc": _mean(raw, "roc_auc"),
                "minimum_raw_roc_auc": float(min(row["roc_auc"] for row in raw)),
                "mean_policy_average_precision": _mean(
                    selected, "average_precision"
                ),
                "mean_policy_roc_auc": _mean(selected, "roc_auc"),
                "mean_policy_brier_score": _mean(selected, "brier_score"),
                "mean_policy_log_loss": _mean(selected, "log_loss"),
                "mean_policy_ece": _mean(
                    selected, "expected_calibration_error"
                ),
                "mean_precision_at_prior_threshold": _mean(
                    threshold, "precision"
                ),
                "mean_recall_at_prior_threshold": _mean(threshold, "recall"),
                "mean_f1_at_prior_threshold": _mean(threshold, "f1"),
                "mean_balanced_accuracy_at_prior_threshold": _mean(
                    threshold, "balanced_accuracy"
                ),
                "mean_alert_count": _mean(threshold, "alert_count"),
                "mean_alert_rate": float(np.mean(alert_rates)),
                "minimum_alert_rate": float(min(alert_rates)),
                "maximum_alert_rate": float(max(alert_rates)),
                "alert_rate_standard_deviation": float(np.std(alert_rates)),
                "minimum_calibration_positive_rows": int(
                    min(fold["calibration_fit_positive_rows"] for fold in folds)
                ),
                "minimum_threshold_positive_rows": int(
                    min(fold["threshold_selection_positive_rows"] for fold in folds)
                ),
                "minimum_required_calibration_positive_rows": (
                    MIN_CALIBRATION_POSITIVES
                ),
                "minimum_required_threshold_positive_rows": MIN_THRESHOLD_POSITIVES,
                "partition_support_sufficient": all(
                    fold["calibration_fit_positive_rows"]
                    >= MIN_CALIBRATION_POSITIVES
                    and fold["threshold_selection_positive_rows"]
                    >= MIN_THRESHOLD_POSITIVES
                    for fold in folds
                ),
                "all_evaluation_rankings_preserved": all(
                    fold["evaluation_rank_diagnostics"]["rank_preserved"]
                    for fold in folds
                ),
            }
        )
    return sorted(
        comparison,
        key=lambda row: (
            -row["mean_raw_average_precision"],
            -row["mean_raw_roc_auc"],
        ),
    )


def _select_experiment(comparison: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in comparison
        if row["all_evaluation_rankings_preserved"]
        and row["minimum_raw_roc_auc"] > 0.5
        and row["partition_support_sufficient"]
    ]
    if not eligible:
        diagnostic_leader = comparison[0]
        return {
            "admissible": False,
            "reason": (
                "Les sous-fenêtres historiques distinctes de calibration et de "
                "seuil contiennent moins de 30 retards sur au moins un fold; "
                "les volumes d'alertes sont trop instables pour figer un candidat."
            ),
            "diagnostic_leader": diagnostic_leader["experiment"],
            "diagnostic_leader_contract": diagnostic_leader["contract"],
            "diagnostic_leader_model": diagnostic_leader["model"],
            "diagnostic_leader_mean_raw_average_precision": diagnostic_leader[
                "mean_raw_average_precision"
            ],
            "diagnostic_leader_mean_raw_roc_auc": diagnostic_leader[
                "mean_raw_roc_auc"
            ],
            "diagnostic_leader_alert_rate_range": [
                diagnostic_leader["minimum_alert_rate"],
                diagnostic_leader["maximum_alert_rate"],
            ],
            "production_ready": False,
            "test_accessed": False,
            "test_evaluated": False,
        }
    best_ap = max(row["mean_raw_average_precision"] for row in eligible)
    near_best = [
        row for row in eligible if row["mean_raw_average_precision"] >= best_ap - 0.005
    ]
    complexity = {
        "logistic_regression": 1,
        "hist_gradient_boosting": 2,
        "random_forest": 3,
    }
    selected = min(
        near_best,
        key=lambda row: (
            complexity[row["model"]],
            0 if row["contract"] == "strict_v1" else 1,
            row["mean_policy_brier_score"],
        ),
    )
    return {
        **selected,
        "admissible": True,
        "selection_policy": (
            "Highest mean raw temporal Average Precision; within 0.005, prefer "
            "the lower-complexity model, then strict_v1, then lower policy Brier."
        ),
        "near_best_experiments": [row["experiment"] for row in near_best],
        "production_ready": False,
        "test_accessed": False,
        "test_evaluated": False,
    }


def _candidate_partitions(dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
    purchase = pd.to_datetime(dataset["purchase_timestamp"], errors="coerce")
    observed = pd.to_datetime(dataset["label_observed_at"], errors="coerce")
    boundaries = {
        "train_as_of": pd.Timestamp("2018-03-01"),
        "calibration_as_of": pd.Timestamp("2018-04-01"),
        "candidate_as_of": pd.Timestamp("2018-05-01"),
    }
    result = {
        "train": dataset.loc[
            purchase.lt(boundaries["train_as_of"])
            & observed.lt(boundaries["train_as_of"])
        ].copy(),
        "calibration_fit": dataset.loc[
            purchase.ge(boundaries["train_as_of"])
            & purchase.lt(boundaries["calibration_as_of"])
            & observed.lt(boundaries["calibration_as_of"])
        ].copy(),
        "threshold_selection": dataset.loc[
            purchase.ge(boundaries["calibration_as_of"])
            & purchase.lt(boundaries["candidate_as_of"])
            & observed.lt(boundaries["candidate_as_of"])
        ].copy(),
    }
    identifiers: dict[str, set[str]] = {}
    for name, frame in result.items():
        if frame.empty or set(frame[TARGET_COLUMN].astype(int).unique()) != {0, 1}:
            raise BacktestError(f"Partition candidat 5.4.2 insuffisante : {name}.")
        if frame["order_id"].duplicated().any():
            raise BacktestError(f"Doublon candidat 5.4.2 : {name}.")
        identifiers[name] = set(frame["order_id"].astype(str))
    if (
        identifiers["train"] & identifiers["calibration_fit"]
        or identifiers["train"] & identifiers["threshold_selection"]
        or identifiers["calibration_fit"] & identifiers["threshold_selection"]
    ):
        raise BacktestError("Chevauchement des partitions candidat 5.4.2.")
    return {name: frame.reset_index(drop=True) for name, frame in result.items()}


def _describe_partition(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "delayed": int(frame[TARGET_COLUMN].sum()),
        "purchase_min": pd.Timestamp(frame["purchase_timestamp"].min()).isoformat(),
        "purchase_max": pd.Timestamp(frame["purchase_timestamp"].max()).isoformat(),
        "label_observed_max": pd.Timestamp(
            frame["label_observed_at"].max()
        ).isoformat(),
    }


def _prepare_candidate(
    dataset: pd.DataFrame,
    *,
    contract: str,
    model_name: str,
) -> tuple[Any, dict[str, Any]]:
    partitions = _candidate_partitions(dataset)
    train = partitions["train"]
    calibration_fit = partitions["calibration_fit"]
    threshold_selection = partitions["threshold_selection"]
    policy = fit_monotone_calibration_policy(
        build_phase54_pipeline(contract, model_name),
        x_train=contract_matrix(train, contract),
        y_train=target_vector(train),
        x_calibration_fit=contract_matrix(calibration_fit, contract),
        y_calibration_fit=target_vector(calibration_fit),
    )
    threshold_probability = policy.estimator.predict_proba(
        contract_matrix(threshold_selection, contract)
    )[:, 1]
    threshold, threshold_metrics = best_f1_threshold(
        target_vector(threshold_selection), threshold_probability
    )
    return policy.estimator, {
        "candidate_version": CANDIDATE_VERSION,
        "contract": contract,
        "model": model_name,
        "calibration_policy_version": CALIBRATION_POLICY_VERSION,
        "selected_method": policy.selected_method,
        "calibration_status": policy.calibration_status,
        "policy_diagnostics": policy.diagnostics,
        "train": _describe_partition(train),
        "calibration_fit": _describe_partition(calibration_fit),
        "threshold_selection": _describe_partition(threshold_selection),
        "candidate_threshold": float(threshold),
        "threshold_selection_metrics": threshold_metrics,
        "threshold_selection_calibration_bins": calibration_bins(
            target_vector(threshold_selection), threshold_probability
        ),
        "independent_candidate_evaluation_performed": False,
        "production_ready": False,
        "test_accessed": False,
        "test_evaluated": False,
    }


def run_phase542(
    *,
    input_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    """Run corrected pre-holdout backtests and persist a new artifact version."""

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    backtests: dict[str, Any] = {}
    fold_definitions: dict[str, Any] = {}
    calibration_diagnostics: dict[str, Any] = {}
    threshold_analysis: dict[str, Any] = {}
    drift_analysis: dict[str, Any] = {}

    for experiment_name, contract, model_name in EXPERIMENTS:
        logger.info("Backtest corrigé : %s", experiment_name)
        dataset = datasets[contract]
        experiment_folds: dict[str, Any] = {}
        experiment_calibration: dict[str, Any] = {}
        experiment_thresholds: dict[str, Any] = {}
        experiment_drift: dict[str, Any] = {}
        numeric, categorical, boolean = _feature_groups(contract)
        for fold in monthly_backtest_folds():
            partitions = build_policy_fold_partitions(dataset, fold)
            fold_definitions.setdefault(fold.name, policy_fold_summary(partitions))
            x_train = contract_matrix(partitions.train, contract)
            x_calibration = contract_matrix(partitions.calibration_fit, contract)
            x_threshold = contract_matrix(partitions.threshold_selection, contract)
            x_evaluation = contract_matrix(partitions.evaluation, contract)
            y_train = target_vector(partitions.train)
            y_calibration = target_vector(partitions.calibration_fit)
            y_threshold = target_vector(partitions.threshold_selection)
            y_evaluation = target_vector(partitions.evaluation)
            policy = fit_monotone_calibration_policy(
                build_phase54_pipeline(contract, model_name),
                x_train=x_train,
                y_train=y_train,
                x_calibration_fit=x_calibration,
                y_calibration_fit=y_calibration,
            )
            raw_estimator = policy.estimator
            if policy.selected_method != "raw":
                raw_estimator = policy.estimator.calibrated_classifiers_[0].estimator.estimator
            raw_train_probability = raw_estimator.predict_proba(x_train)[:, 1]
            raw_evaluation_probability = raw_estimator.predict_proba(x_evaluation)[:, 1]
            threshold_probability = policy.estimator.predict_proba(x_threshold)[:, 1]
            evaluation_probability = policy.estimator.predict_proba(x_evaluation)[:, 1]
            threshold, threshold_metrics = best_f1_threshold(
                y_threshold, threshold_probability
            )
            rank_evaluation = ranking_diagnostics(
                raw_evaluation_probability, evaluation_probability
            )
            if not rank_evaluation["rank_preserved"]:
                raise BacktestError(
                    f"Classement inversé après politique sur {experiment_name}/{fold.name}."
                )
            experiment_folds[fold.name] = {
                "selected_method": policy.selected_method,
                "calibration_status": policy.calibration_status,
                "threshold_selected_before_evaluation": True,
                "candidate_threshold": float(threshold),
                "calibration_fit_positive_rows": int(y_calibration.sum()),
                "threshold_selection_positive_rows": int(y_threshold.sum()),
                "raw_evaluation": evaluate_probabilities(
                    y_evaluation, raw_evaluation_probability, threshold=0.5
                ),
                "policy_evaluation": {
                    "threshold_0_5": evaluate_probabilities(
                        y_evaluation, evaluation_probability, threshold=0.5
                    ),
                    "historical_v1_threshold": evaluate_probabilities(
                        y_evaluation,
                        evaluation_probability,
                        threshold=HISTORICAL_V1_THRESHOLD,
                    ),
                    "prior_selected_threshold": evaluate_probabilities(
                        y_evaluation, evaluation_probability, threshold=threshold
                    ),
                },
                "evaluation_rank_diagnostics": rank_evaluation,
            }
            experiment_calibration[fold.name] = policy.diagnostics
            experiment_thresholds[fold.name] = {
                "threshold": float(threshold),
                "selection_partition_metrics": threshold_metrics,
                "selection_partition_calibration_bins": calibration_bins(
                    y_threshold, threshold_probability
                ),
                "evaluation_alert_count": experiment_folds[fold.name][
                    "policy_evaluation"
                ]["prior_selected_threshold"]["alert_count"],
                "evaluation_alert_rate": experiment_folds[fold.name][
                    "policy_evaluation"
                ]["prior_selected_threshold"]["prediction_positive_rate"],
            }
            experiment_drift[fold.name] = drift_report(
                x_train,
                x_evaluation,
                numeric_features=numeric,
                categorical_features=categorical,
                boolean_features=boolean,
                reference_scores=raw_train_probability,
                current_scores=raw_evaluation_probability,
            )
        backtests[experiment_name] = {
            "contract": contract,
            "model": model_name,
            "folds": experiment_folds,
        }
        calibration_diagnostics[experiment_name] = experiment_calibration
        threshold_analysis[experiment_name] = experiment_thresholds
        drift_analysis[experiment_name] = experiment_drift

    comparison = _aggregate_comparison(backtests)
    selection = _select_experiment(comparison)
    candidate: Any | None = None
    candidate_report: dict[str, Any] = {
        **selection,
        "candidate_prepared": False,
    }
    if selection["admissible"]:
        candidate, preparation = _prepare_candidate(
            datasets[selection["contract"]],
            contract=selection["contract"],
            model_name=selection["model"],
        )
        candidate_report.update(
            {
                "candidate_prepared": True,
                "candidate_artifact": "candidate_pipeline.joblib",
                "preparation": preparation,
            }
        )

    policy_config = {
        "phase": PHASE_VERSION,
        "version": CALIBRATION_POLICY_VERSION,
        "positive_class": 1,
        "calibration_fit_window_months": 1,
        "threshold_selection_window_months": 1,
        "selection_order": ["sigmoid_increasing", "isotonic_monotone", "raw"],
        "sigmoid_rule": "accept only coefficient_a < 0 and no rank inversion",
        "isotonic_rule": {
            "rank_preserved": True,
            "minimum_unique_levels": MIN_ISOTONIC_LEVELS,
            "maximum_largest_tie_fraction": MAX_ISOTONIC_LARGEST_TIE_FRACTION,
        },
        "raw_fallback_status": "uncalibrated_raw_fallback",
        "minimum_calibration_positive_rows_for_candidate": (
            MIN_CALIBRATION_POSITIVES
        ),
        "minimum_threshold_positive_rows_for_candidate": MIN_THRESHOLD_POSITIVES,
        "minimum_support_rule": (
            "Conservative statistical reliability guard, not a business cost rule."
        ),
        "evaluation_labels_used_for_policy_selection": False,
        "test_accessed": False,
        "test_evaluated": False,
    }
    experiment_config = {
        "phase": PHASE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "safe_cutoff_exclusive": "2018-05-01T00:00:00",
        "random_state": 42,
        "strict_contract": FEATURE_SCHEMA_VERSION,
        "conditional_contract": ENRICHED_FEATURE_SCHEMA_VERSION,
        "historical_v1_threshold": HISTORICAL_V1_THRESHOLD,
        "experiments": [
            {"name": name, "contract": contract, "model": model}
            for name, contract, model in EXPERIMENTS
        ],
        "versions": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
        },
        "test_accessed": False,
        "test_evaluated": False,
    }
    dataset_audit = {
        "strict_rows": int(len(strict_dataset)),
        "enriched_rows": int(len(enriched_dataset)),
        "same_order_ids": True,
        "same_targets": True,
        "maximum_purchase_timestamp": pd.Timestamp(
            strict_dataset["purchase_timestamp"].max()
        ).isoformat(),
        "strict_dataframe_sha256": _dataframe_sha256(strict_dataset),
        "enriched_dataframe_sha256": _dataframe_sha256(enriched_dataset),
        "test_file_read": False,
    }
    candidate_contract = {
        "candidate_version": CANDIDATE_VERSION,
        "candidate_prepared": candidate is not None,
        "feature_contract": None,
        "calibration_policy_version": CALIBRATION_POLICY_VERSION,
        "production_ready": False,
    }
    if candidate is not None:
        candidate_contract["feature_contract"] = (
            enriched_feature_contract_as_dict()
            if selection.get("contract") == "enriched_v2"
            else {"feature_schema_version": FEATURE_SCHEMA_VERSION}
        )
    reports = {
        "calibration_policy": policy_config,
        "experiment_config": experiment_config,
        "dataset_audit": dataset_audit,
        "fold_definitions": fold_definitions,
        "backtest_results": backtests,
        "model_comparison": comparison,
        "calibration_diagnostics": calibration_diagnostics,
        "threshold_analysis": threshold_analysis,
        "drift_analysis": drift_analysis,
        "selected_candidate": candidate_report,
        "candidate_contract": candidate_contract,
    }
    paths: dict[str, Path] = {}
    for name, payload in reports.items():
        path = output_dir / f"{name}.json"
        _atomic_json(payload, path, overwrite=overwrite)
        paths[name] = path
    if candidate is not None:
        candidate_path = output_dir / "candidate_pipeline.joblib"
        _atomic_joblib(candidate, candidate_path, overwrite=overwrite)
        paths["candidate_pipeline"] = candidate_path
    manifest = {
        "phase": PHASE_VERSION,
        "policy_version": CALIBRATION_POLICY_VERSION,
        "candidate_prepared": candidate is not None,
        "test_accessed": False,
        "test_evaluated": False,
        "artifacts": {
            name: {
                "filename": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in sorted(paths.items())
        },
    }
    _atomic_json(manifest, output_dir / "artifact_manifest.json", overwrite=overwrite)
    return {**reports, "artifact_manifest": manifest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exécuter la Phase 5.4.2 sans ouvrir le holdout final."
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
    result = run_phase542(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        overwrite=args.overwrite,
    )
    selected = result["selected_candidate"]
    logger.info(
        "Phase 5.4.2 : admissible=%s candidat=%s test intact",
        selected.get("admissible"),
        selected.get("candidate_prepared"),
    )


if __name__ == "__main__":
    main()
