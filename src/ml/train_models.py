"""CLI for temporal training and validation-only model comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from config.paths import BASE_DIR
from src.ml.contract import FEATURE_SCHEMA_VERSION, STRICT_FEATURE_COLUMNS
from src.ml.evaluation import (
    best_f1_threshold,
    calibration_bins,
    evaluate_probabilities,
    subgroup_metrics,
    threshold_diagnostics,
    transformed_feature_importance,
)
from src.ml.training import (
    GPS_FEATURES,
    TrainingScenarioConfig,
    build_model_pipeline,
    feature_matrix,
    load_training_scenario,
    model_definitions,
    target_vector,
)


logger = logging.getLogger(__name__)
DEFAULT_INPUT_DIR = Path(BASE_DIR) / "data" / "processed" / "ml"
DEFAULT_OUTPUT_DIR = (
    Path(BASE_DIR)
    / "models"
    / "delivery_delay"
    / "strict_v1"
    / "historical_as_of_20180101"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _json_safe_params(estimator: Any) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key, value in estimator.get_params(deep=False).items():
        if value is None or isinstance(value, (str, int, float, bool)):
            params[key] = value
        else:
            params[key] = repr(value)
    return params


def _input_hashes(input_dir: Path, config: TrainingScenarioConfig) -> dict[str, str]:
    paths = {
        "train_as_of": input_dir
        / (
            "strict_v1_labels_available_as_of_"
            f"{config.as_of_date.strftime('%Y%m%dT%H%M%S')}.parquet"
        ),
        "validation": input_dir / "strict_v1_validation.parquet",
        "test_metadata_only": input_dir / "strict_v1_test.parquet",
        "feature_contract": input_dir / "strict_v1_feature_contract.json",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _select_candidate(
    metrics: dict[str, dict[str, Any]],
    definitions: dict[str, Any],
    tolerance: float,
) -> tuple[str, dict[str, Any]]:
    """Choose the simplest non-dummy model within an AP tolerance."""

    candidates = [name for name in metrics if name != "dummy_classifier"]
    best_ap = max(metrics[name]["threshold_0_5"]["average_precision"] for name in candidates)
    near_best = [
        name
        for name in candidates
        if metrics[name]["threshold_0_5"]["average_precision"]
        >= best_ap - tolerance
    ]
    selected = min(near_best, key=lambda name: definitions[name].complexity_rank)
    return selected, {
        "policy": (
            "Highest validation Average Precision, then simplest model within "
            f"an absolute tolerance of {tolerance:.3f}."
        ),
        "best_average_precision": float(best_ap),
        "near_best_models": near_best,
        "selected_model": selected,
        "test_evaluated": False,
    }


def _prepare_output(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        existing = [path for path in output_dir.iterdir() if path.is_file()]
        if existing and not overwrite:
            raise FileExistsError(
                f"Le répertoire contient déjà {len(existing)} artefact(s) : "
                f"{output_dir}. Utiliser --overwrite."
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def run_training(
    *,
    input_dir: Path,
    output_dir: Path,
    overwrite: bool,
    run_gps_ablation: bool,
) -> dict[str, Any]:
    """Train, validate, select a candidate, and persist trusted artifacts."""

    config = TrainingScenarioConfig()
    _prepare_output(output_dir, overwrite=overwrite)
    scenario = load_training_scenario(input_dir, config)
    definitions = model_definitions(config.random_state)

    x_train = feature_matrix(scenario.train)
    y_train = target_vector(scenario.train)
    x_validation = feature_matrix(scenario.validation)
    y_validation = target_vector(scenario.validation)

    pipelines: dict[str, Any] = {}
    probabilities: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, Any]] = {}
    importances: dict[str, Any] = {}
    threshold_reports: dict[str, Any] = {}

    for name, definition in definitions.items():
        logger.info("Entraînement : %s", name)
        pipeline = build_model_pipeline(definition)
        started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started

        validation_probability = pipeline.predict_proba(x_validation)[:, 1]
        train_probability = pipeline.predict_proba(x_train)[:, 1]
        threshold, threshold_metrics = best_f1_threshold(
            y_validation, validation_probability
        )
        metrics[name] = {
            "fit_seconds": float(fit_seconds),
            "threshold_0_5": evaluate_probabilities(
                y_validation, validation_probability, threshold=0.5
            ),
            "train_threshold_0_5": evaluate_probabilities(
                y_train, train_probability, threshold=0.5
            ),
            "best_f1_threshold": float(threshold),
            "validation_at_best_f1": threshold_metrics,
        }
        threshold_reports[name] = threshold_diagnostics(
            y_validation, validation_probability, threshold
        )
        importances[name] = transformed_feature_importance(pipeline)
        pipelines[name] = pipeline
        probabilities[name] = validation_probability

    selected_name, selection = _select_candidate(
        metrics, definitions, config.selection_ap_tolerance
    )
    selected_pipeline = pipelines[selected_name]
    selected_probability = probabilities[selected_name]
    selected_threshold = float(metrics[selected_name]["best_f1_threshold"])

    validation_month = scenario.validation["purchase_timestamp"].dt.to_period("M").astype(str)
    monthly = subgroup_metrics(
        scenario.validation,
        selected_probability,
        group_values=validation_month,
        threshold=selected_threshold,
        minimum_rows=1,
    )
    by_state = subgroup_metrics(
        scenario.validation,
        selected_probability,
        group_values=scenario.validation["customer_state"],
        threshold=selected_threshold,
        minimum_rows=config.minimum_state_rows,
    )
    calibration = calibration_bins(y_validation, selected_probability, bins=10)

    gps_ablation: dict[str, Any] | None = None
    if run_gps_ablation:
        ablation_features = tuple(
            name for name in STRICT_FEATURE_COLUMNS if name not in GPS_FEATURES
        )
        definition = definitions["logistic_regression"]
        pipeline = build_model_pipeline(
            definition, feature_columns=ablation_features
        )
        pipeline.fit(
            feature_matrix(scenario.train, ablation_features), y_train
        )
        probability = pipeline.predict_proba(
            feature_matrix(scenario.validation, ablation_features)
        )[:, 1]
        threshold, candidate_metrics = best_f1_threshold(y_validation, probability)
        gps_ablation = {
            "model": "logistic_regression",
            "features": list(ablation_features),
            "threshold_0_5": evaluate_probabilities(
                y_validation, probability, threshold=0.5
            ),
            "best_f1_threshold": float(threshold),
            "validation_at_best_f1": candidate_metrics,
        }

    model_files: dict[str, Path] = {}
    for name, pipeline in pipelines.items():
        destination = output_dir / f"{name}.joblib"
        _atomic_joblib(pipeline, destination, overwrite=overwrite)
        model_files[name] = destination

    training_config = {
        "scenario": config.name,
        "as_of_date": config.as_of_date.isoformat(),
        "validation_end": config.validation_end.isoformat(),
        "test_end": config.test_end.isoformat(),
        "random_state": config.random_state,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": list(STRICT_FEATURE_COLUMNS),
        "target": "is_delayed",
        "train_rows": int(len(scenario.train)),
        "validation_rows": int(len(scenario.validation)),
        "test_rows_metadata_validation_only": int(len(scenario.test)),
        "test_evaluated": False,
        "class_distribution": {
            "train_delayed": int(y_train.sum()),
            "train_on_time": int(len(y_train) - y_train.sum()),
            "validation_delayed": int(y_validation.sum()),
            "validation_on_time": int(len(y_validation) - y_validation.sum()),
        },
        "models": {
            name: _json_safe_params(definition.estimator)
            for name, definition in definitions.items()
        },
        "input_sha256": _input_hashes(input_dir, config),
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
    }
    robustness = {
        "selected_model": selected_name,
        "selected_threshold": selected_threshold,
        "monthly_validation": monthly,
        "state_validation_minimum_rows": config.minimum_state_rows,
        "state_validation": by_state,
        "calibration_bins": calibration,
        "gps_ablation": gps_ablation,
    }
    candidate = {
        **selection,
        "candidate_threshold": selected_threshold,
        "threshold_policy": (
            "Maximum F1 on chronological validation only; reference threshold "
            "0.5 remains reported. No business cost was assumed."
        ),
        "validation_metrics_at_candidate_threshold": metrics[selected_name][
            "validation_at_best_f1"
        ],
        "artifact": model_files[selected_name].name,
        "production_ready": False,
    }

    json_files = {
        "training_config": output_dir / "training_config.json",
        "validation_metrics": output_dir / "validation_metrics.json",
        "threshold_analysis": output_dir / "threshold_analysis.json",
        "robustness_analysis": output_dir / "robustness_analysis.json",
        "feature_importance": output_dir / "feature_importance.json",
        "selected_candidate": output_dir / "selected_candidate.json",
    }
    payloads = {
        "training_config": training_config,
        "validation_metrics": metrics,
        "threshold_analysis": threshold_reports,
        "robustness_analysis": robustness,
        "feature_importance": importances,
        "selected_candidate": candidate,
    }
    for name, destination in json_files.items():
        _atomic_json(payloads[name], destination, overwrite=overwrite)

    artifacts = {**model_files, **json_files}
    manifest = {
        "scenario": config.name,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "selected_model": selected_name,
        "test_evaluated": False,
        "artifacts": {
            name: {
                "filename": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in sorted(artifacts.items())
        },
    }
    manifest_path = output_dir / "artifact_manifest.json"
    _atomic_json(manifest, manifest_path, overwrite=overwrite)

    return {
        "training_config": training_config,
        "metrics": metrics,
        "selection": candidate,
        "robustness": robustness,
        "manifest": manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entraîner et comparer les modèles strict_v1 sur validation "
            "chronologique, sans évaluer le test final."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-gps-ablation", action="store_true")
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
    result = run_training(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        overwrite=args.overwrite,
        run_gps_ablation=args.run_gps_ablation,
    )
    logger.info(
        "Candidat : %s | seuil validation : %.6f | test évalué : non",
        result["selection"]["selected_model"],
        result["selection"]["candidate_threshold"],
    )


if __name__ == "__main__":
    main()
