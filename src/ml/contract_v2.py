"""Conditional enriched-v2 feature contract for Phase 5.4 experiments."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.ml.contract import (
    METADATA_COLUMNS,
    STRICT_FEATURE_COLUMNS,
    STRICT_FEATURE_SPECS,
    TARGET_COLUMN,
    FeatureSpec,
)


ENRICHED_FEATURE_SCHEMA_VERSION = "enriched_v2.0.0"
ENRICHED_OPERATIONAL_STATUS = "conditional_not_production_validated"
PHASE54_CUTOFF = "2018-05-01T00:00:00"


ENRICHED_FEATURE_SPECS = (
    FeatureSpec(
        name="promised_delivery_days",
        dtype="int16",
        source=(
            "dw.fact_orders.estimated_delivery_timestamp and "
            "dw.fact_orders.purchase_timestamp"
        ),
        transformation="Calendar-day difference between promise and purchase.",
        temporal_availability=(
            "Conditional: Olist does not timestamp when the promise was fixed."
        ),
        missing_value_policy="Reject non-positive or missing horizons.",
    ),
    FeatureSpec(
        name="item_count",
        dtype="int16",
        source="dw.fact_order_items.order_item_id",
        transformation="Count item rows after order-level aggregation.",
        temporal_availability="Conditional: checkout payload availability unproven.",
        missing_value_policy="Reject orders without item aggregates.",
    ),
    FeatureSpec(
        name="total_item_value",
        dtype="float64",
        source="dw.fact_order_items.price",
        transformation="Sum item prices at order grain.",
        temporal_availability="Conditional: checkout payload availability unproven.",
        missing_value_policy="Reject missing aggregate; item prices are non-null.",
    ),
    FeatureSpec(
        name="total_freight_value",
        dtype="float64",
        source="dw.fact_order_items.freight_value",
        transformation="Sum freight values at order grain.",
        temporal_availability="Conditional: checkout payload availability unproven.",
        missing_value_policy="Reject missing aggregate; freight is non-null.",
    ),
    FeatureSpec(
        name="freight_ratio",
        dtype="float64",
        source="Derived from order-level item and freight totals.",
        transformation="total_freight_value / total_item_value.",
        temporal_availability="Conditional like both source totals.",
        missing_value_policy="Null only if total item value is not positive.",
    ),
    FeatureSpec(
        name="seller_count",
        dtype="int16",
        source="dw.fact_order_items.seller_key",
        transformation="Count distinct sellers per order.",
        temporal_availability="Conditional: seller assignment at creation unproven.",
        missing_value_policy="Reject orders without item aggregates.",
    ),
    FeatureSpec(
        name="seller_state_count",
        dtype="int16",
        source="dw.fact_order_items.seller_key -> dw.dim_seller.seller_state",
        transformation="Count distinct non-null seller states per order.",
        temporal_availability="Conditional: seller assignment at creation unproven.",
        missing_value_policy="Zero only when every seller state is missing.",
    ),
    FeatureSpec(
        name="is_multi_seller",
        dtype="bool",
        source="Derived from seller_count.",
        transformation="True when seller_count is greater than one.",
        temporal_availability="Conditional like seller_count.",
        missing_value_policy="Always populated after aggregation.",
    ),
    FeatureSpec(
        name="distance_km_mean",
        dtype="Float64",
        source="dw.dim_customer and dw.dim_seller latitude/longitude.",
        transformation="Mean item-level Haversine distance at order grain.",
        temporal_availability=(
            "Conditional: seller assignment and static GPS lookups are unproven."
        ),
        missing_value_policy="Keep null and expose distance coverage flags.",
    ),
    FeatureSpec(
        name="distance_km_max",
        dtype="Float64",
        source="dw.dim_customer and dw.dim_seller latitude/longitude.",
        transformation="Maximum item-level Haversine distance at order grain.",
        temporal_availability="Conditional like distance_km_mean.",
        missing_value_policy="Keep null and expose distance coverage flags.",
    ),
    FeatureSpec(
        name="distance_km_weighted_by_item_value",
        dtype="Float64",
        source="Coordinates plus dw.fact_order_items.price.",
        transformation="Price-weighted mean Haversine distance.",
        temporal_availability="Conditional like distance and item price.",
        missing_value_policy="Keep null when no covered positive-price item exists.",
    ),
    FeatureSpec(
        name="distance_coverage_ratio",
        dtype="float64",
        source="Derived from valid item-level customer/seller coordinates.",
        transformation="Items with valid distance divided by item_count.",
        temporal_availability="Conditional like the distance inputs.",
        missing_value_policy="Always populated between zero and one.",
    ),
    FeatureSpec(
        name="distance_incomplete",
        dtype="bool",
        source="Derived from distance_coverage_ratio.",
        transformation="True when distance coverage is below one.",
        temporal_availability="Conditional like the distance inputs.",
        missing_value_policy="Always populated.",
    ),
    FeatureSpec(
        name="category_count",
        dtype="int16",
        source="dw.fact_order_items.product_key -> dw.dim_product category.",
        transformation="Count distinct normalized categories per order.",
        temporal_availability="Conditional: catalog snapshot at creation unproven.",
        missing_value_policy="Missing categories map to stable 'unknown'.",
    ),
    FeatureSpec(
        name="dominant_category",
        dtype="string",
        source="dw.fact_order_items plus dw.dim_product category.",
        transformation=(
            "Highest summed item price, then item count, then lexical tie-break."
        ),
        temporal_availability="Conditional: catalog snapshot at creation unproven.",
        missing_value_policy="Stable 'unknown' category.",
    ),
    FeatureSpec(
        name="total_product_weight_g",
        dtype="Float64",
        source="dw.dim_product.product_weight_g joined at item grain.",
        transformation="Sum known item weights in grams.",
        temporal_availability="Conditional: versioned catalog not available.",
        missing_value_policy="Partial sum plus physical coverage indicators.",
    ),
    FeatureSpec(
        name="total_product_volume_cm3",
        dtype="Float64",
        source=(
            "dw.dim_product.product_length_cm, product_height_cm and "
            "product_width_cm"
        ),
        transformation="Sum known length × height × width in cubic centimeters.",
        temporal_availability="Conditional: versioned catalog not available.",
        missing_value_policy="Partial sum plus physical coverage indicators.",
    ),
    FeatureSpec(
        name="product_physical_coverage_ratio",
        dtype="float64",
        source="Derived from non-null product weight and dimensions.",
        transformation="Fully covered physical item rows divided by item_count.",
        temporal_availability="Conditional like catalog attributes.",
        missing_value_policy="Always populated between zero and one.",
    ),
    FeatureSpec(
        name="product_physical_incomplete",
        dtype="bool",
        source="Derived from product_physical_coverage_ratio.",
        transformation="True when physical coverage is below one.",
        temporal_availability="Conditional like catalog attributes.",
        missing_value_policy="Always populated.",
    ),
)

ENRICHED_ONLY_FEATURE_COLUMNS = tuple(
    spec.name for spec in ENRICHED_FEATURE_SPECS
)
ENRICHED_FEATURE_COLUMNS = STRICT_FEATURE_COLUMNS + ENRICHED_ONLY_FEATURE_COLUMNS

ENRICHED_NUMERIC_FEATURES = (
    "purchase_hour",
    "purchase_day_of_week",
    "purchase_month",
    "purchase_quarter",
    "customer_latitude",
    "customer_longitude",
    "promised_delivery_days",
    "item_count",
    "total_item_value",
    "total_freight_value",
    "freight_ratio",
    "seller_count",
    "seller_state_count",
    "distance_km_mean",
    "distance_km_max",
    "distance_km_weighted_by_item_value",
    "distance_coverage_ratio",
    "category_count",
    "total_product_weight_g",
    "total_product_volume_cm3",
    "product_physical_coverage_ratio",
)
ENRICHED_CATEGORICAL_FEATURES = ("customer_state", "dominant_category")
ENRICHED_BOOLEAN_FEATURES = (
    "purchase_is_weekend",
    "customer_has_gps",
    "is_multi_seller",
    "distance_incomplete",
    "product_physical_incomplete",
)


def enriched_feature_contract_as_dict() -> dict[str, Any]:
    """Return the conditional V2 contract without altering strict V1."""

    return {
        "feature_schema_version": ENRICHED_FEATURE_SCHEMA_VERSION,
        "operational_status": ENRICHED_OPERATIONAL_STATUS,
        "prototype_cutoff_exclusive": PHASE54_CUTOFF,
        "grain": "one row per order_id",
        "metadata_columns": list(METADATA_COLUMNS),
        "target": {
            "name": TARGET_COLUMN,
            "definition": (
                "1 when delivered_timestamp::date is after "
                "estimated_delivery_timestamp::date, otherwise 0"
            ),
        },
        "strict_v1_preserved": {
            "feature_count": len(STRICT_FEATURE_SPECS),
            "features": [asdict(spec) for spec in STRICT_FEATURE_SPECS],
        },
        "conditional_features": [
            {**asdict(spec), "decision": "conditional"}
            for spec in ENRICHED_FEATURE_SPECS
        ],
        "prohibited_feature_families": [
            "actual delivery or carrier timestamps",
            "final order status",
            "reviews and satisfaction",
            "post-purchase approval timestamps",
            "full-history seller or category target aggregates",
            "raw business identifiers",
        ],
    }
