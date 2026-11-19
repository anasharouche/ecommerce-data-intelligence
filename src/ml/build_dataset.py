"""CLI for the strict delivery-delay ML dataset pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config.paths import BASE_DIR
from src.ml.contract import (
    FEATURE_SCHEMA_VERSION,
    STRICT_FEATURE_COLUMNS,
    TARGET_COLUMN,
    feature_contract_as_dict,
)
from src.ml.dataset import (
    PopulationResult,
    build_supervised_population,
    exclusion_summary,
    extract_order_source,
)
from src.ml.features import build_strict_features
from src.ml.splits import (
    TemporalSplitConfig,
    labels_available_as_of,
    split_temporally,
)
from src.ml.validation import (
    feature_missing_rates,
    validate_split_order,
    validate_strict_dataset,
)


logger = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path(BASE_DIR) / "data" / "processed" / "ml"


@dataclass(frozen=True)
class BuildArtifacts:
    """In-memory artifacts before export."""

    dataset: pd.DataFrame
    splits: dict[str, pd.DataFrame]
    available_as_of: pd.DataFrame | None
    population: PopulationResult
    summary: dict[str, Any]


def build_artifacts(
    source: pd.DataFrame,
    *,
    as_of_date: str | None = None,
    split_config: TemporalSplitConfig | None = None,
) -> BuildArtifacts:
    """Build and validate all in-memory strict-v1 artifacts."""

    population = build_supervised_population(source)
    dataset = build_strict_features(population.supervised_source)
    validate_strict_dataset(dataset)
    splits = split_temporally(dataset, split_config)
    validate_split_order(splits)

    available = None
    if as_of_date is not None:
        available = labels_available_as_of(dataset, as_of_date)
        validate_strict_dataset(available)

    delayed = int(dataset[TARGET_COLUMN].sum())
    on_time = int(len(dataset) - delayed)
    summary: dict[str, Any] = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "source_rows": int(population.source_row_count),
        "eligible_rows": int(len(dataset)),
        "excluded_rows": int(len(population.exclusions)),
        "delayed_rows": delayed,
        "on_time_rows": on_time,
        "duplicate_order_ids": int(dataset["order_id"].duplicated().sum()),
        "strict_feature_count": len(STRICT_FEATURE_COLUMNS),
        "strict_features": list(STRICT_FEATURE_COLUMNS),
        "feature_missing_rates": feature_missing_rates(dataset),
        "partitions": {
            name: {
                "rows": int(len(frame)),
                "delayed_rows": int(frame[TARGET_COLUMN].sum()),
                "on_time_rows": int(len(frame) - frame[TARGET_COLUMN].sum()),
                "purchase_min": frame["purchase_timestamp"].min().isoformat(),
                "purchase_max": frame["purchase_timestamp"].max().isoformat(),
            }
            for name, frame in splits.items()
        },
        "exclusions": exclusion_summary(population.exclusions),
        "as_of_date": as_of_date,
        "labels_available_as_of_rows": (
            int(len(available)) if available is not None else None
        ),
    }
    return BuildArtifacts(
        dataset=dataset,
        splits=splits,
        available_as_of=available,
        population=population,
        summary=summary,
    )


def _as_of_suffix(as_of_date: str) -> str:
    """Return a filesystem-safe representation of an as-of timestamp."""

    timestamp = pd.Timestamp(as_of_date)
    return timestamp.strftime("%Y%m%dT%H%M%S")


def _atomic_parquet_write(
    frame: pd.DataFrame,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    """Write Parquet atomically and refuse silent replacement."""

    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Le fichier existe déjà : {destination}. Utiliser --overwrite."
        )
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
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_json_write(
    payload: dict[str, Any],
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    """Write UTF-8 JSON atomically and refuse silent replacement."""

    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Le fichier existe déjà : {destination}. Utiliser --overwrite."
        )
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


def _sha256(path: Path) -> str:
    """Calculate the SHA-256 hash of an exported artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_display_path(path: Path) -> str:
    """Return a repository-relative path when possible."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(BASE_DIR))
    except ValueError:
        return str(resolved)


def export_artifacts(
    artifacts: BuildArtifacts,
    output_dir: Path,
    *,
    overwrite: bool,
) -> dict[str, Path]:
    """Export Parquet datasets, contract, exclusions, and build summary."""

    files: dict[str, Path] = {
        "dataset": output_dir / "strict_v1_supervised.parquet",
        "train": output_dir / "strict_v1_train.parquet",
        "validation": output_dir / "strict_v1_validation.parquet",
        "test": output_dir / "strict_v1_test.parquet",
        "contract": output_dir / "strict_v1_feature_contract.json",
        "exclusions": output_dir / "strict_v1_exclusions_report.json",
        "summary": output_dir / "strict_v1_build_summary.json",
    }
    if artifacts.available_as_of is not None:
        as_of_date = artifacts.summary["as_of_date"]
        files["available_as_of"] = output_dir / (
            "strict_v1_labels_available_as_of_"
            f"{_as_of_suffix(as_of_date)}.parquet"
        )

    existing = [path for path in files.values() if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Export refusé : les fichiers suivants existent déjà. "
            "Utiliser --overwrite :\n" + joined
        )

    _atomic_parquet_write(
        artifacts.dataset, files["dataset"], overwrite=overwrite
    )
    for name, frame in artifacts.splits.items():
        _atomic_parquet_write(frame, files[name], overwrite=overwrite)
    if artifacts.available_as_of is not None:
        _atomic_parquet_write(
            artifacts.available_as_of,
            files["available_as_of"],
            overwrite=overwrite,
        )

    _atomic_json_write(
        feature_contract_as_dict(), files["contract"], overwrite=overwrite
    )
    _atomic_json_write(
        exclusion_summary(artifacts.population.exclusions),
        files["exclusions"],
        overwrite=overwrite,
    )

    summary = dict(artifacts.summary)
    summary["artifacts"] = {
        name: {
            "path": _artifact_display_path(path),
            "sha256": _sha256(path),
        }
        for name, path in files.items()
        if name != "summary"
    }
    _atomic_json_write(summary, files["summary"], overwrite=overwrite)
    return files


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Construire le dataset ML strict V1 en lecture seule."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Répertoire des artefacts Parquet et JSON.",
    )
    parser.add_argument(
        "--as-of-date",
        help=(
            "Date de référence optionnelle. Exporte les labels observés "
            "strictement avant cette date."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Autoriser explicitement le remplacement des artefacts existants.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args()


def main() -> None:
    """Execute the read-only extraction, build, validation, and export."""

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s | %(message)s",
    )
    source = extract_order_source()
    artifacts = build_artifacts(source, as_of_date=args.as_of_date)
    files = export_artifacts(
        artifacts,
        args.output_dir.resolve(),
        overwrite=args.overwrite,
    )

    logger.info("Dataset éligible : %s", len(artifacts.dataset))
    logger.info(
        "Cible : %s retards, %s livraisons à temps",
        artifacts.summary["delayed_rows"],
        artifacts.summary["on_time_rows"],
    )
    for name, frame in artifacts.splits.items():
        logger.info("Partition %s : %s lignes", name, len(frame))
    if artifacts.available_as_of is not None:
        logger.info(
            "Labels disponibles au %s : %s lignes",
            args.as_of_date,
            len(artifacts.available_as_of),
        )
    logger.info("Artefacts exportés : %s", len(files))


if __name__ == "__main__":
    main()
