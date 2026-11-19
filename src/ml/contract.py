"""Versioned feature contract for the strict delivery-delay dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


FEATURE_SCHEMA_VERSION = "strict_v1.0.0"
TARGET_COLUMN = "is_delayed"

METADATA_COLUMNS = (
    "order_id",
    "purchase_timestamp",
    "label_observed_at",
)


@dataclass(frozen=True)
class FeatureSpec:
    """Describe one deterministic model feature."""

    name: str
    dtype: str
    source: str
    transformation: str
    temporal_availability: str
    missing_value_policy: str


STRICT_FEATURE_SPECS = (
    FeatureSpec(
        name="purchase_hour",
        dtype="int8",
        source="dw.fact_orders.purchase_timestamp",
        transformation="Hour in local source time, from 0 to 23.",
        temporal_availability="Available at order creation.",
        missing_value_policy="Not nullable; reject an invalid purchase timestamp.",
    ),
    FeatureSpec(
        name="purchase_day_of_week",
        dtype="int8",
        source="dw.fact_orders.purchase_timestamp",
        transformation="Monday=0 through Sunday=6.",
        temporal_availability="Available at order creation.",
        missing_value_policy="Not nullable; reject an invalid purchase timestamp.",
    ),
    FeatureSpec(
        name="purchase_month",
        dtype="int8",
        source="dw.fact_orders.purchase_timestamp",
        transformation="Calendar month from 1 to 12.",
        temporal_availability="Available at order creation.",
        missing_value_policy="Not nullable; reject an invalid purchase timestamp.",
    ),
    FeatureSpec(
        name="purchase_quarter",
        dtype="int8",
        source="dw.fact_orders.purchase_timestamp",
        transformation="Calendar quarter from 1 to 4.",
        temporal_availability="Available at order creation.",
        missing_value_policy="Not nullable; reject an invalid purchase timestamp.",
    ),
    FeatureSpec(
        name="purchase_is_weekend",
        dtype="bool",
        source="dw.fact_orders.purchase_timestamp",
        transformation="True for Saturday or Sunday.",
        temporal_availability="Available at order creation.",
        missing_value_policy="Not nullable; reject an invalid purchase timestamp.",
    ),
    FeatureSpec(
        name="customer_state",
        dtype="string",
        source="dw.dim_customer.customer_state",
        transformation="Upper-case two-letter state code.",
        temporal_availability=(
            "Available from the customer shipping address at order creation."
        ),
        missing_value_policy="Not nullable; reject a missing state.",
    ),
    FeatureSpec(
        name="customer_latitude",
        dtype="Float64",
        source="dw.dim_customer.latitude",
        transformation=(
            "ZIP-prefix centroid retained only when both coordinates and the "
            "GPS quality flag are valid."
        ),
        temporal_availability=(
            "Deterministic lookup from the customer shipping ZIP prefix."
        ),
        missing_value_policy=(
            "Keep null and use customer_has_gps; no statistical imputation here."
        ),
    ),
    FeatureSpec(
        name="customer_longitude",
        dtype="Float64",
        source="dw.dim_customer.longitude",
        transformation=(
            "ZIP-prefix centroid retained only when both coordinates and the "
            "GPS quality flag are valid."
        ),
        temporal_availability=(
            "Deterministic lookup from the customer shipping ZIP prefix."
        ),
        missing_value_policy=(
            "Keep null and use customer_has_gps; no statistical imputation here."
        ),
    ),
    FeatureSpec(
        name="customer_has_gps",
        dtype="bool",
        source=(
            "dw.dim_customer.flag_has_gps, latitude and longitude"
        ),
        transformation="True only when the flag and both coordinates are valid.",
        temporal_availability=(
            "Deterministic lookup from the customer shipping ZIP prefix."
        ),
        missing_value_policy="Always populated.",
    ),
)

STRICT_FEATURE_COLUMNS = tuple(spec.name for spec in STRICT_FEATURE_SPECS)


def feature_contract_as_dict() -> dict[str, Any]:
    """Return a JSON-serializable feature contract."""

    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "grain": "one row per order_id",
        "metadata_columns": list(METADATA_COLUMNS),
        "target": {
            "name": TARGET_COLUMN,
            "dtype": "int8",
            "definition": (
                "1 when delivered_timestamp::date is after "
                "estimated_delivery_timestamp::date, otherwise 0"
            ),
        },
        "features": [asdict(spec) for spec in STRICT_FEATURE_SPECS],
    }
