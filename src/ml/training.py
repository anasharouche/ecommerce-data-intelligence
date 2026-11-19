"""Preprocessing, model definitions, and temporal training scenario loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.ml.contract import (
    FEATURE_SCHEMA_VERSION,
    STRICT_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.ml.validation import validate_strict_dataset


RANDOM_STATE = 42

NUMERIC_FEATURES = (
    "purchase_hour",
    "purchase_day_of_week",
    "purchase_month",
    "purchase_quarter",
    "customer_latitude",
    "customer_longitude",
)
CATEGORICAL_FEATURES = ("customer_state",)
BOOLEAN_FEATURES = ("purchase_is_weekend", "customer_has_gps")
GPS_FEATURES = (
    "customer_latitude",
    "customer_longitude",
    "customer_has_gps",
)


class TrainingError(RuntimeError):
    """Raised when the training scenario or contract is invalid."""


@dataclass(frozen=True)
class TrainingScenarioConfig:
    """Reproducible temporal scenario for the first historical experiment."""

    name: str = "historical_as_of_20180101"
    as_of_date: pd.Timestamp = pd.Timestamp("2018-01-01")
    validation_end: pd.Timestamp = pd.Timestamp("2018-05-01")
    test_end: pd.Timestamp = pd.Timestamp("2018-09-01")
    random_state: int = RANDOM_STATE
    selection_ap_tolerance: float = 0.005
    minimum_state_rows: int = 500


@dataclass(frozen=True)
class ScenarioData:
    """Validated train, validation, and untouched test partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    contract: dict[str, Any]


@dataclass(frozen=True)
class ModelDefinition:
    """Estimator and preprocessing requirements for one candidate."""

    name: str
    estimator: Any
    dense_output: bool
    scale_numeric: bool
    complexity_rank: int


def _boolean_to_float(values: Any) -> np.ndarray:
    """Convert booleans and missing values to a numeric matrix."""

    frame = pd.DataFrame(values).replace({True: 1.0, False: 0.0})
    return frame.to_numpy(dtype=np.float64, na_value=np.nan)


def _selected_groups(
    feature_columns: tuple[str, ...],
) -> tuple[list[str], list[str], list[str]]:
    """Return feature groups in contract order and reject unknown columns."""

    selected = set(feature_columns)
    unknown = selected - set(STRICT_FEATURE_COLUMNS)
    if unknown:
        raise TrainingError(
            "Features hors contrat : " + ", ".join(sorted(unknown))
        )
    numeric = [name for name in NUMERIC_FEATURES if name in selected]
    categorical = [name for name in CATEGORICAL_FEATURES if name in selected]
    boolean = [name for name in BOOLEAN_FEATURES if name in selected]
    grouped = numeric + categorical + boolean
    if set(grouped) != selected:
        missing = sorted(selected - set(grouped))
        raise TrainingError(
            "Features sans groupe de prétraitement : " + ", ".join(missing)
        )
    return numeric, categorical, boolean


def build_preprocessor(
    *,
    feature_columns: tuple[str, ...] = STRICT_FEATURE_COLUMNS,
    dense_output: bool,
    scale_numeric: bool,
) -> ColumnTransformer:
    """Build a leakage-safe preprocessor fitted only by a sklearn Pipeline."""

    numeric, categorical, boolean = _selected_groups(feature_columns)
    return build_typed_preprocessor(
        numeric_features=tuple(numeric),
        categorical_features=tuple(categorical),
        boolean_features=tuple(boolean),
        dense_output=dense_output,
        scale_numeric=scale_numeric,
    )


def build_typed_preprocessor(
    *,
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
    boolean_features: tuple[str, ...],
    dense_output: bool,
    scale_numeric: bool,
) -> ColumnTransformer:
    """Build a preprocessor for an explicit versioned feature contract."""

    groups = numeric_features + categorical_features + boolean_features
    if len(groups) != len(set(groups)):
        raise TrainingError("Une feature appartient à plusieurs groupes de types.")
    if not groups:
        raise TrainingError("Le préprocesseur exige au moins une feature.")

    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        transformers.append(
            ("numeric", Pipeline(numeric_steps), list(numeric_features))
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=not dense_output,
                                dtype=np.float64,
                            ),
                        ),
                    ]
                ),
                list(categorical_features),
            )
        )
    if boolean_features:
        transformers.append(
            (
                "boolean",
                Pipeline(
                    [
                        (
                            "to_float",
                            FunctionTransformer(
                                _boolean_to_float,
                                feature_names_out="one-to-one",
                            ),
                        ),
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                    ]
                ),
                list(boolean_features),
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0 if dense_output else 0.3,
        verbose_feature_names_out=False,
    )


def model_definitions(random_state: int = RANDOM_STATE) -> dict[str, ModelDefinition]:
    """Return the four bounded first-comparison model definitions."""

    return {
        "dummy_classifier": ModelDefinition(
            name="dummy_classifier",
            estimator=DummyClassifier(strategy="prior"),
            dense_output=False,
            scale_numeric=False,
            complexity_rank=0,
        ),
        "logistic_regression": ModelDefinition(
            name="logistic_regression",
            estimator=LogisticRegression(
                class_weight="balanced",
                max_iter=1_000,
                solver="lbfgs",
                random_state=random_state,
            ),
            dense_output=False,
            scale_numeric=True,
            complexity_rank=1,
        ),
        "random_forest": ModelDefinition(
            name="random_forest",
            estimator=RandomForestClassifier(
                n_estimators=250,
                max_depth=12,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
            dense_output=False,
            scale_numeric=False,
            complexity_rank=3,
        ),
        "hist_gradient_boosting": ModelDefinition(
            name="hist_gradient_boosting",
            estimator=HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=150,
                max_leaf_nodes=31,
                min_samples_leaf=30,
                l2_regularization=1.0,
                class_weight="balanced",
                early_stopping=False,
                random_state=random_state,
            ),
            dense_output=True,
            scale_numeric=False,
            complexity_rank=2,
        ),
    }


def build_model_pipeline(
    definition: ModelDefinition,
    *,
    feature_columns: tuple[str, ...] = STRICT_FEATURE_COLUMNS,
) -> Pipeline:
    """Build a complete preprocessing and classifier pipeline."""

    return Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(
                    feature_columns=feature_columns,
                    dense_output=definition.dense_output,
                    scale_numeric=definition.scale_numeric,
                ),
            ),
            # A definition can be reused by an optional ablation. Clone the
            # estimator so fitting one pipeline cannot mutate another one.
            ("classifier", clone(definition.estimator)),
        ]
    )


def build_typed_model_pipeline(
    definition: ModelDefinition,
    *,
    numeric_features: tuple[str, ...],
    categorical_features: tuple[str, ...],
    boolean_features: tuple[str, ...],
) -> Pipeline:
    """Build an isolated pipeline for a non-V1 versioned feature contract."""

    return Pipeline(
        [
            (
                "preprocessor",
                build_typed_preprocessor(
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    boolean_features=boolean_features,
                    dense_output=definition.dense_output,
                    scale_numeric=definition.scale_numeric,
                ),
            ),
            ("classifier", clone(definition.estimator)),
        ]
    )


def _as_of_filename(as_of_date: pd.Timestamp) -> str:
    return (
        "strict_v1_labels_available_as_of_"
        f"{as_of_date.strftime('%Y%m%dT%H%M%S')}.parquet"
    )


def _load_contract(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "strict_v1_feature_contract.json"
    if not path.is_file():
        raise TrainingError(f"Contrat de features absent : {path}")
    with path.open(encoding="utf-8") as stream:
        contract = json.load(stream)

    if contract.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise TrainingError(
            "Version de contrat incompatible : "
            f"{contract.get('feature_schema_version')}"
        )
    names = tuple(item["name"] for item in contract.get("features", []))
    if names != STRICT_FEATURE_COLUMNS:
        raise TrainingError("La liste ou l'ordre des features diverge du contrat Python.")
    return contract


def load_training_scenario(
    input_dir: Path,
    config: TrainingScenarioConfig | None = None,
) -> ScenarioData:
    """Load and validate the mature train, validation, and untouched test."""

    config = config or TrainingScenarioConfig()
    input_dir = input_dir.resolve()
    contract = _load_contract(input_dir)
    paths = {
        "train": input_dir / _as_of_filename(config.as_of_date),
        "validation": input_dir / "strict_v1_validation.parquet",
        "test": input_dir / "strict_v1_test.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise TrainingError("Parquet absents : " + ", ".join(missing))

    frames = {name: pd.read_parquet(path) for name, path in paths.items()}
    validate_training_scenario_frames(frames, config)
    return ScenarioData(
        train=frames["train"],
        validation=frames["validation"],
        test=frames["test"],
        contract=contract,
    )


def validate_training_scenario_frames(
    frames: dict[str, pd.DataFrame],
    config: TrainingScenarioConfig | None = None,
) -> None:
    """Validate feature contract, dates, labels, and partition disjointness."""

    config = config or TrainingScenarioConfig()
    required = {"train", "validation", "test"}
    if set(frames) != required:
        raise TrainingError(f"Partitions attendues : {sorted(required)}")
    for frame in frames.values():
        validate_strict_dataset(frame)

    train = frames["train"]
    validation = frames["validation"]
    test = frames["test"]
    if not train["label_observed_at"].lt(config.as_of_date).all():
        raise TrainingError("Le train contient un label non observable à as_of_date.")
    if not train["purchase_timestamp"].lt(config.as_of_date).all():
        raise TrainingError("Le train contient une commande créée après as_of_date.")
    if not (
        validation["purchase_timestamp"].ge(config.as_of_date).all()
        and validation["purchase_timestamp"].lt(config.validation_end).all()
    ):
        raise TrainingError("La validation sort de la fenêtre configurée.")
    if not (
        test["purchase_timestamp"].ge(config.validation_end).all()
        and test["purchase_timestamp"].lt(config.test_end).all()
    ):
        raise TrainingError("Le test sort de la fenêtre configurée.")

    id_sets = {
        name: set(frame["order_id"].astype(str))
        for name, frame in frames.items()
    }
    if (
        id_sets["train"] & id_sets["validation"]
        or id_sets["train"] & id_sets["test"]
        or id_sets["validation"] & id_sets["test"]
    ):
        raise TrainingError("Chevauchement d'order_id entre partitions.")


def feature_matrix(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...] = STRICT_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Return the exact ordered contract feature matrix."""

    _selected_groups(feature_columns)
    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        raise TrainingError("Features absentes : " + ", ".join(missing))
    return frame.loc[:, list(feature_columns)].copy()


def target_vector(frame: pd.DataFrame) -> pd.Series:
    """Return the binary target as an integer series."""

    target = frame[TARGET_COLUMN]
    if target.isna().any() or not set(target.astype(int).unique()).issubset({0, 1}):
        raise TrainingError("Cible absente ou non binaire.")
    return target.astype("int8").copy()
