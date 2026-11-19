"""Deterministic strict-v1 feature engineering."""

from __future__ import annotations

import pandas as pd

from src.ml.contract import (
    METADATA_COLUMNS,
    STRICT_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.ml.dataset import DatasetError


REQUIRED_FEATURE_SOURCE_COLUMNS = (
    "order_id",
    "purchase_timestamp",
    "delivered_timestamp",
    "customer_state",
    "customer_latitude",
    "customer_longitude",
    "customer_flag_has_gps",
    TARGET_COLUMN,
)


def build_strict_features(supervised_source: pd.DataFrame) -> pd.DataFrame:
    """Build metadata, strict model features, and target at order grain."""

    missing = sorted(
        set(REQUIRED_FEATURE_SOURCE_COLUMNS) - set(supervised_source.columns)
    )
    if missing:
        raise DatasetError(
            "Colonnes requises pour les features absentes : "
            + ", ".join(missing)
        )

    purchase = pd.to_datetime(
        supervised_source["purchase_timestamp"], errors="coerce"
    )
    label_observed = pd.to_datetime(
        supervised_source["delivered_timestamp"], errors="coerce"
    )
    if purchase.isna().any() or label_observed.isna().any():
        raise DatasetError(
            "Timestamp invalide dans la population supervisée."
        )

    latitude = pd.to_numeric(
        supervised_source["customer_latitude"], errors="coerce"
    ).astype("Float64")
    longitude = pd.to_numeric(
        supervised_source["customer_longitude"], errors="coerce"
    ).astype("Float64")
    source_gps_flag = (
        supervised_source["customer_flag_has_gps"]
        .fillna(False)
        .astype(bool)
    )
    valid_coordinates = (
        source_gps_flag
        & latitude.between(-90, 90, inclusive="both").fillna(False)
        & longitude.between(-180, 180, inclusive="both").fillna(False)
    )

    dataset = pd.DataFrame(index=supervised_source.index)
    dataset["order_id"] = supervised_source["order_id"].astype("string")
    dataset["purchase_timestamp"] = purchase
    dataset["label_observed_at"] = label_observed
    dataset["purchase_hour"] = purchase.dt.hour.astype("int8")
    dataset["purchase_day_of_week"] = purchase.dt.day_of_week.astype("int8")
    dataset["purchase_month"] = purchase.dt.month.astype("int8")
    dataset["purchase_quarter"] = purchase.dt.quarter.astype("int8")
    dataset["purchase_is_weekend"] = (
        purchase.dt.day_of_week.ge(5).astype(bool)
    )
    dataset["customer_state"] = (
        supervised_source["customer_state"]
        .astype("string")
        .str.strip()
        .str.upper()
    )
    dataset["customer_latitude"] = latitude.where(valid_coordinates)
    dataset["customer_longitude"] = longitude.where(valid_coordinates)
    dataset["customer_has_gps"] = valid_coordinates.astype(bool)
    dataset[TARGET_COLUMN] = supervised_source[TARGET_COLUMN].astype("int8")

    ordered_columns = (
        list(METADATA_COLUMNS)
        + list(STRICT_FEATURE_COLUMNS)
        + [TARGET_COLUMN]
    )
    return dataset.loc[:, ordered_columns].reset_index(drop=True)


def model_matrix(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return only the contract-approved model feature matrix."""

    return dataset.loc[:, list(STRICT_FEATURE_COLUMNS)].copy()
