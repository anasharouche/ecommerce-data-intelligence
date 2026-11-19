"""Conditional enriched feature construction with a hard pre-test cutoff."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import psycopg

from src.ml.contract import METADATA_COLUMNS, STRICT_FEATURE_COLUMNS, TARGET_COLUMN
from src.ml.contract_v2 import (
    ENRICHED_FEATURE_COLUMNS,
    ENRICHED_ONLY_FEATURE_COLUMNS,
    PHASE54_CUTOFF,
)
from src.ml.dataset import (
    DatasetError,
    PopulationResult,
    _connection_parameters,
    build_supervised_population,
)
from src.ml.features import build_strict_features
from src.ml.validation import FORBIDDEN_FEATURE_COLUMNS, validate_strict_dataset


logger = logging.getLogger(__name__)
PHASE54_CUTOFF_TIMESTAMP = pd.Timestamp(PHASE54_CUTOFF)


ORDER_SOURCE_QUERY = """
SELECT
    fo.order_id,
    fo.order_status,
    fo.purchase_timestamp,
    fo.delivered_timestamp,
    fo.estimated_delivery_timestamp,
    fo.is_delayed AS warehouse_is_delayed,
    dc.customer_state,
    dc.latitude AS customer_latitude,
    dc.longitude AS customer_longitude,
    dc.flag_has_gps AS customer_flag_has_gps,
    so.flag_delivery_before_purchase
FROM dw.fact_orders AS fo
JOIN dw.dim_customer AS dc
    ON dc.customer_key = fo.customer_key
JOIN staging.orders_clean AS so
    ON so.order_id = fo.order_id
WHERE fo.purchase_timestamp < %(cutoff)s
ORDER BY fo.purchase_timestamp, fo.order_id
"""


ITEM_SOURCE_QUERY = """
SELECT
    foi.order_id,
    foi.order_item_id,
    ds.seller_id,
    foi.price,
    foi.freight_value,
    ds.seller_state,
    ds.latitude AS seller_latitude,
    ds.longitude AS seller_longitude,
    ds.flag_has_gps AS seller_flag_has_gps,
    dc.latitude AS customer_latitude,
    dc.longitude AS customer_longitude,
    dc.flag_has_gps AS customer_flag_has_gps,
    dp.product_category_name_english AS product_category,
    dp.product_weight_g,
    dp.product_length_cm,
    dp.product_height_cm,
    dp.product_width_cm
FROM dw.fact_order_items AS foi
JOIN dw.fact_orders AS fo
    ON fo.order_key = foi.order_key
JOIN dw.dim_customer AS dc
    ON dc.customer_key = fo.customer_key
JOIN dw.dim_seller AS ds
    ON ds.seller_key = foi.seller_key
JOIN dw.dim_product AS dp
    ON dp.product_key = foi.product_key
WHERE fo.purchase_timestamp < %(cutoff)s
ORDER BY foi.order_id, foi.order_item_id
"""


@dataclass(frozen=True)
class EnrichedBuildResult:
    """Safe pre-test dataset and source audit counts."""

    dataset: pd.DataFrame
    population: PopulationResult
    item_aggregates: pd.DataFrame
    source_order_rows: int
    source_item_rows: int


def _fetch_frame(cursor: Any, query: str) -> pd.DataFrame:
    cursor.execute(query, {"cutoff": PHASE54_CUTOFF_TIMESTAMP.to_pydatetime()})
    rows = cursor.fetchall()
    columns = [column.name for column in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def extract_enriched_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read only pre-May-2018 orders and items in one read-only transaction."""

    logger.info(
        "Extraction enrichie READ ONLY avec cutoff exclusif %s",
        PHASE54_CUTOFF,
    )
    with psycopg.connect(**_connection_parameters()) as connection:
        with connection.transaction():
            connection.execute("SET TRANSACTION READ ONLY")
            connection.execute("SET LOCAL statement_timeout = '120s'")
            with connection.cursor() as cursor:
                orders = _fetch_frame(cursor, ORDER_SOURCE_QUERY)
                items = _fetch_frame(cursor, ITEM_SOURCE_QUERY)

    purchase = pd.to_datetime(orders["purchase_timestamp"], errors="coerce")
    if purchase.isna().any() or purchase.ge(PHASE54_CUTOFF_TIMESTAMP).any():
        raise DatasetError("La source Phase 5.4 a franchi la frontière du test final.")
    if orders["order_id"].duplicated().any():
        raise DatasetError("La source commande enrichie n'est pas unique.")
    if items.duplicated(["order_id", "order_item_id"]).any():
        raise DatasetError("La source article contient une clé composite dupliquée.")
    return orders, items


def haversine_km(
    latitude_one: pd.Series | np.ndarray,
    longitude_one: pd.Series | np.ndarray,
    latitude_two: pd.Series | np.ndarray,
    longitude_two: pd.Series | np.ndarray,
) -> np.ndarray:
    """Calculate vectorized great-circle distance in kilometers."""

    lat1 = np.radians(np.asarray(latitude_one, dtype=np.float64))
    lon1 = np.radians(np.asarray(longitude_one, dtype=np.float64))
    lat2 = np.radians(np.asarray(latitude_two, dtype=np.float64))
    lon2 = np.radians(np.asarray(longitude_two, dtype=np.float64))
    delta_latitude = lat2 - lat1
    delta_longitude = lon2 - lon1
    value = (
        np.sin(delta_latitude / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_longitude / 2.0) ** 2
    )
    return 2.0 * 6371.0088 * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))


def aggregate_order_items(items: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one item source independently to one row per order."""

    required = {
        "order_id",
        "order_item_id",
        "seller_id",
        "price",
        "freight_value",
        "seller_state",
        "seller_latitude",
        "seller_longitude",
        "seller_flag_has_gps",
        "customer_latitude",
        "customer_longitude",
        "customer_flag_has_gps",
        "product_category",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    }
    missing = sorted(required - set(items.columns))
    if missing:
        raise DatasetError("Colonnes article enrichies absentes : " + ", ".join(missing))
    if items.empty:
        raise DatasetError("La source article enrichie est vide.")
    if items[["order_id", "order_item_id"]].isna().any().any():
        raise DatasetError("Une clé article est absente.")
    if items.duplicated(["order_id", "order_item_id"]).any():
        raise DatasetError("La source article contient un produit cartésien.")

    frame = items.copy()
    numeric_columns = (
        "price",
        "freight_value",
        "seller_latitude",
        "seller_longitude",
        "customer_latitude",
        "customer_longitude",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["price", "freight_value"]].isna().any().any():
        raise DatasetError("Prix ou fret article absent.")
    if frame["price"].le(0).any() or frame["freight_value"].lt(0).any():
        raise DatasetError("Prix ou fret article hors bornes.")

    customer_valid = (
        frame["customer_flag_has_gps"].fillna(False).astype(bool)
        & frame["customer_latitude"].between(-90, 90).fillna(False)
        & frame["customer_longitude"].between(-180, 180).fillna(False)
    )
    seller_valid = (
        frame["seller_flag_has_gps"].fillna(False).astype(bool)
        & frame["seller_latitude"].between(-90, 90).fillna(False)
        & frame["seller_longitude"].between(-180, 180).fillna(False)
    )
    distance_valid = customer_valid & seller_valid
    frame["distance_km"] = np.nan
    frame.loc[distance_valid, "distance_km"] = haversine_km(
        frame.loc[distance_valid, "customer_latitude"],
        frame.loc[distance_valid, "customer_longitude"],
        frame.loc[distance_valid, "seller_latitude"],
        frame.loc[distance_valid, "seller_longitude"],
    )
    frame["distance_weighted_numerator"] = (
        frame["distance_km"] * frame["price"]
    )
    frame["distance_weight"] = frame["price"].where(distance_valid)

    positive_dimensions = (
        frame["product_length_cm"].gt(0)
        & frame["product_height_cm"].gt(0)
        & frame["product_width_cm"].gt(0)
    )
    positive_weight = frame["product_weight_g"].gt(0)
    frame["product_volume_cm3"] = (
        frame["product_length_cm"]
        * frame["product_height_cm"]
        * frame["product_width_cm"]
    ).where(positive_dimensions)
    frame["product_weight_g_valid"] = frame["product_weight_g"].where(
        positive_weight
    )
    frame["product_physical_valid"] = positive_dimensions & positive_weight
    frame["product_category"] = (
        frame["product_category"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace("", pd.NA)
        .fillna("unknown")
    )
    frame["seller_state"] = (
        frame["seller_state"].astype("string").str.strip().str.upper()
    )

    grouped = frame.groupby("order_id", sort=True, observed=True)
    aggregate = grouped.agg(
        item_count=("order_item_id", "size"),
        total_item_value=("price", "sum"),
        total_freight_value=("freight_value", "sum"),
        seller_count=("seller_id", "nunique"),
        seller_state_count=("seller_state", "nunique"),
        distance_km_mean=("distance_km", "mean"),
        distance_km_max=("distance_km", "max"),
        distance_covered_items=("distance_km", "count"),
        distance_weighted_numerator=("distance_weighted_numerator", "sum"),
        distance_weight=("distance_weight", "sum"),
        category_count=("product_category", "nunique"),
        total_product_weight_g=("product_weight_g_valid", "sum"),
        total_product_volume_cm3=("product_volume_cm3", "sum"),
        product_physical_covered_items=("product_physical_valid", "sum"),
    ).reset_index()
    category_value = (
        frame.groupby(["order_id", "product_category"], observed=True)
        .agg(category_value=("price", "sum"), category_items=("order_item_id", "size"))
        .reset_index()
        .sort_values(
            ["order_id", "category_value", "category_items", "product_category"],
            ascending=[True, False, False, True],
            kind="mergesort",
        )
        .drop_duplicates("order_id")
        .loc[:, ["order_id", "product_category"]]
        .rename(columns={"product_category": "dominant_category"})
    )
    aggregate = aggregate.merge(
        category_value, on="order_id", how="left", validate="one_to_one"
    )
    aggregate["freight_ratio"] = (
        aggregate["total_freight_value"] / aggregate["total_item_value"]
    )
    aggregate["is_multi_seller"] = aggregate["seller_count"].gt(1)
    aggregate["distance_coverage_ratio"] = (
        aggregate["distance_covered_items"] / aggregate["item_count"]
    )
    aggregate["distance_incomplete"] = aggregate[
        "distance_coverage_ratio"
    ].lt(1.0)
    aggregate["distance_km_weighted_by_item_value"] = (
        aggregate["distance_weighted_numerator"] / aggregate["distance_weight"]
    )
    aggregate["product_physical_coverage_ratio"] = (
        aggregate["product_physical_covered_items"] / aggregate["item_count"]
    )
    aggregate["product_physical_incomplete"] = aggregate[
        "product_physical_coverage_ratio"
    ].lt(1.0)

    output = aggregate.loc[:, ["order_id", *ENRICHED_ONLY_FEATURE_COLUMNS[1:]]]
    if output["order_id"].duplicated().any():
        raise DatasetError("L'agrégat article n'est pas unique par commande.")
    return output.reset_index(drop=True)


def build_enriched_dataset(
    order_source: pd.DataFrame,
    item_source: pd.DataFrame,
) -> EnrichedBuildResult:
    """Build the conditional V2 prototype on exactly the eligible safe orders."""

    purchase = pd.to_datetime(order_source["purchase_timestamp"], errors="coerce")
    if purchase.isna().any() or purchase.ge(PHASE54_CUTOFF_TIMESTAMP).any():
        raise DatasetError("Une commande Phase 5.4 appartient au test final.")
    population = build_supervised_population(order_source)
    strict = build_strict_features(population.supervised_source)
    validate_strict_dataset(strict)
    item_aggregates = aggregate_order_items(item_source)

    promise = population.supervised_source.loc[
        :, ["order_id", "purchase_timestamp", "estimated_delivery_timestamp"]
    ].copy()
    promise["promised_delivery_days"] = (
        pd.to_datetime(promise["estimated_delivery_timestamp"]).dt.normalize()
        - pd.to_datetime(promise["purchase_timestamp"]).dt.normalize()
    ).dt.days
    if promise["promised_delivery_days"].isna().any() or promise[
        "promised_delivery_days"
    ].le(0).any():
        raise DatasetError("Horizon promis manquant ou non positif.")

    dataset = strict.merge(
        promise.loc[:, ["order_id", "promised_delivery_days"]],
        on="order_id",
        how="left",
        validate="one_to_one",
    ).merge(
        item_aggregates,
        on="order_id",
        how="left",
        validate="one_to_one",
    )
    ordered = [
        *METADATA_COLUMNS,
        *ENRICHED_FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]
    dataset = dataset.loc[:, ordered]
    validate_enriched_dataset(dataset)
    return EnrichedBuildResult(
        dataset=dataset,
        population=population,
        item_aggregates=item_aggregates,
        source_order_rows=len(order_source),
        source_item_rows=len(item_source),
    )


def validate_enriched_dataset(dataset: pd.DataFrame) -> None:
    """Validate V2 grain, schema, cutoff, leakage and aggregate bounds."""

    expected = [
        *METADATA_COLUMNS,
        *ENRICHED_FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]
    if list(dataset.columns) != expected:
        raise DatasetError("Schéma enrichi inattendu.")
    if dataset.empty or dataset["order_id"].duplicated().any():
        raise DatasetError("Le grain enrichi n'est pas une commande unique.")
    if pd.to_datetime(dataset["purchase_timestamp"]).ge(
        PHASE54_CUTOFF_TIMESTAMP
    ).any():
        raise DatasetError("Le dataset enrichi contient le test final.")
    leaked = set(ENRICHED_FEATURE_COLUMNS) & set(FORBIDDEN_FEATURE_COLUMNS)
    if leaked:
        raise DatasetError("Fuite dans V2 : " + ", ".join(sorted(leaked)))
    required_positive = (
        "promised_delivery_days",
        "item_count",
        "total_item_value",
        "seller_count",
        "category_count",
    )
    if dataset.loc[:, list(required_positive)].isna().any().any():
        raise DatasetError("Un agrégat enrichi obligatoire est absent.")
    if (dataset.loc[:, list(required_positive)] <= 0).any().any():
        raise DatasetError("Un agrégat enrichi obligatoire n'est pas positif.")
    for column in (
        "distance_coverage_ratio",
        "product_physical_coverage_ratio",
    ):
        if not dataset[column].between(0, 1).all():
            raise DatasetError(f"{column} sort de [0, 1].")
    if not tuple(dataset.loc[:, list(STRICT_FEATURE_COLUMNS)].columns) == (
        STRICT_FEATURE_COLUMNS
    ):
        raise DatasetError("Le sous-contrat strict V1 a changé.")


def enriched_model_matrix(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return only the ordered conditional V2 model features."""

    validate_enriched_dataset(dataset)
    return dataset.loc[:, list(ENRICHED_FEATURE_COLUMNS)].copy()
