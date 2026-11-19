"""Quality and leakage validations for the strict ML dataset."""

from __future__ import annotations

import pandas as pd

from src.ml.contract import (
    METADATA_COLUMNS,
    STRICT_FEATURE_COLUMNS,
    TARGET_COLUMN,
)


class ValidationError(ValueError):
    """Raised when a strict dataset invariant is violated."""


FORBIDDEN_FEATURE_COLUMNS = frozenset(
    {
        "order_id",
        "purchase_timestamp",
        "label_observed_at",
        "order_status",
        "delivered_timestamp",
        "estimated_delivery_timestamp",
        "approved_date_key",
        "carrier_date_key",
        "delivered_date_key",
        "delivery_days",
        "delivery_delay_days",
        "warehouse_is_delayed",
        "review_score",
        "review_creation_date",
        TARGET_COLUMN,
    }
)


def validate_strict_dataset(dataset: pd.DataFrame) -> None:
    """Validate grain, schema, target, GPS consistency, and leakage rules."""

    expected = (
        list(METADATA_COLUMNS)
        + list(STRICT_FEATURE_COLUMNS)
        + [TARGET_COLUMN]
    )
    if list(dataset.columns) != expected:
        raise ValidationError(
            "Schéma strict inattendu. Attendu : " + ", ".join(expected)
        )
    if dataset.empty:
        raise ValidationError("Le dataset strict est vide.")
    if dataset["order_id"].isna().any():
        raise ValidationError("order_id contient une valeur nulle.")
    if dataset["order_id"].duplicated().any():
        raise ValidationError("order_id n'est pas unique.")
    if dataset["purchase_timestamp"].isna().any():
        raise ValidationError("purchase_timestamp contient une valeur nulle.")
    if dataset["label_observed_at"].isna().any():
        raise ValidationError("label_observed_at contient une valeur nulle.")
    if dataset[TARGET_COLUMN].isna().any():
        raise ValidationError("La cible contient une valeur inconnue.")
    classes = set(dataset[TARGET_COLUMN].astype(int).unique())
    if not classes.issubset({0, 1}):
        raise ValidationError(f"Classes de cible invalides : {classes}")
    if dataset["customer_state"].isna().any():
        raise ValidationError("customer_state contient une valeur nulle.")
    if not dataset["customer_state"].str.fullmatch(r"[A-Z]{2}").all():
        raise ValidationError("customer_state ne respecte pas le format attendu.")

    bounded_features = {
        "purchase_hour": (0, 23),
        "purchase_day_of_week": (0, 6),
        "purchase_month": (1, 12),
        "purchase_quarter": (1, 4),
    }
    for column, (minimum, maximum) in bounded_features.items():
        if not dataset[column].between(minimum, maximum).all():
            raise ValidationError(f"{column} sort des bornes attendues.")

    gps = dataset["customer_has_gps"].astype(bool)
    coordinates_present = dataset[
        ["customer_latitude", "customer_longitude"]
    ].notna().all(axis=1)
    if not gps.equals(coordinates_present):
        raise ValidationError("Le flag GPS et les coordonnées divergent.")

    leaked = FORBIDDEN_FEATURE_COLUMNS.intersection(STRICT_FEATURE_COLUMNS)
    if leaked:
        raise ValidationError(
            "Colonnes interdites dans les features : "
            + ", ".join(sorted(leaked))
        )


def validate_split_order(
    splits: dict[str, pd.DataFrame],
) -> None:
    """Validate chronological order between train, validation, and test."""

    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]
    if train.empty or validation.empty or test.empty:
        raise ValidationError("Une partition temporelle est vide.")
    if train["purchase_timestamp"].max() >= validation["purchase_timestamp"].min():
        raise ValidationError("Train et validation se chevauchent dans le temps.")
    if validation["purchase_timestamp"].max() >= test["purchase_timestamp"].min():
        raise ValidationError("Validation et test se chevauchent dans le temps.")


def feature_missing_rates(dataset: pd.DataFrame) -> dict[str, float]:
    """Return missing-value rates for model features only."""

    rates = dataset.loc[:, list(STRICT_FEATURE_COLUMNS)].isna().mean()
    return {name: round(float(value), 8) for name, value in rates.items()}
