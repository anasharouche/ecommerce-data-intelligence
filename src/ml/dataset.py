"""PostgreSQL extraction and supervised-population construction."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from dotenv import load_dotenv

from config.paths import BASE_DIR


logger = logging.getLogger(__name__)


SOURCE_QUERY = """
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
ORDER BY fo.purchase_timestamp, fo.order_id
"""


DATETIME_COLUMNS = (
    "purchase_timestamp",
    "delivered_timestamp",
    "estimated_delivery_timestamp",
)

REQUIRED_SOURCE_COLUMNS = (
    "order_id",
    "order_status",
    "purchase_timestamp",
    "delivered_timestamp",
    "estimated_delivery_timestamp",
    "warehouse_is_delayed",
    "customer_state",
    "customer_latitude",
    "customer_longitude",
    "customer_flag_has_gps",
    "flag_delivery_before_purchase",
)


class DatasetError(RuntimeError):
    """Raised when source data violate the dataset contract."""


@dataclass(frozen=True)
class PopulationResult:
    """Result of target construction and eligibility filtering."""

    supervised_source: pd.DataFrame
    exclusions: pd.DataFrame
    source_row_count: int


def _connection_parameters() -> dict[str, Any]:
    """Read local connection settings without logging their values."""

    load_dotenv(dotenv_path=Path(BASE_DIR) / ".env")
    required = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise DatasetError(
            "Variables PostgreSQL manquantes : " + ", ".join(missing)
        )

    return {
        "host": os.environ["DB_HOST"],
        "port": int(os.environ["DB_PORT"]),
        "dbname": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
        "connect_timeout": 10,
    }


def extract_order_source() -> pd.DataFrame:
    """Extract the strict source snapshot in a read-only transaction."""

    logger.info("Extraction PostgreSQL en lecture seule")
    with psycopg.connect(**_connection_parameters()) as connection:
        with connection.transaction():
            connection.execute("SET TRANSACTION READ ONLY")
            connection.execute("SET LOCAL statement_timeout = '120s'")
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM dw.fact_orders")
                expected_orders = int(cursor.fetchone()[0])
                cursor.execute(SOURCE_QUERY)
                rows = cursor.fetchall()
                columns = [column.name for column in cursor.description]

    frame = pd.DataFrame(rows, columns=columns)
    if len(frame) != expected_orders:
        raise DatasetError(
            "La jointure d'extraction a modifié le nombre de commandes : "
            f"attendu={expected_orders}, obtenu={len(frame)}."
        )
    if frame["order_id"].duplicated().any():
        raise DatasetError(
            "La jointure d'extraction a dupliqué un ou plusieurs order_id."
        )
    logger.info("Commandes extraites : %s", len(frame))
    return frame


def _assign_reason(
    reasons: pd.Series,
    condition: pd.Series,
    value: str,
) -> None:
    """Assign the first matching exclusion reason in a stable order."""

    mask = reasons.isna() & condition.fillna(False)
    reasons.loc[mask] = value


def build_supervised_population(source: pd.DataFrame) -> PopulationResult:
    """Calculate the calendar-day target and filter observable labels.

    Exclusions are mutually exclusive and assigned according to a documented
    priority. The warehouse target is used only as a consistency check.
    """

    missing_columns = sorted(set(REQUIRED_SOURCE_COLUMNS) - set(source.columns))
    if missing_columns:
        raise DatasetError(
            "Colonnes source absentes : " + ", ".join(missing_columns)
        )

    frame = source.loc[:, list(REQUIRED_SOURCE_COLUMNS)].copy()
    for column in DATETIME_COLUMNS:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")

    reasons = pd.Series(pd.NA, index=frame.index, dtype="string")
    _assign_reason(reasons, frame["order_id"].isna(), "missing_order_id")
    _assign_reason(
        reasons,
        frame["purchase_timestamp"].isna(),
        "missing_purchase_timestamp",
    )
    _assign_reason(
        reasons,
        frame["order_status"].ne("delivered"),
        "status_not_delivered",
    )
    _assign_reason(
        reasons,
        frame["delivered_timestamp"].isna(),
        "missing_delivered_timestamp",
    )
    _assign_reason(
        reasons,
        frame["estimated_delivery_timestamp"].isna(),
        "missing_estimated_delivery_timestamp",
    )
    _assign_reason(
        reasons,
        frame["flag_delivery_before_purchase"].fillna(False).astype(bool)
        | frame["delivered_timestamp"].lt(frame["purchase_timestamp"]),
        "delivery_before_purchase",
    )
    _assign_reason(
        reasons,
        frame["estimated_delivery_timestamp"]
        .dt.normalize()
        .le(frame["purchase_timestamp"].dt.normalize()),
        "invalid_estimated_delivery_date",
    )

    frame["exclusion_reason"] = reasons
    eligible = frame[frame["exclusion_reason"].isna()].copy()

    delivered_date = eligible["delivered_timestamp"].dt.normalize()
    estimated_date = eligible["estimated_delivery_timestamp"].dt.normalize()
    eligible["is_delayed"] = delivered_date.gt(estimated_date).astype("int8")

    warehouse_target = eligible["warehouse_is_delayed"]
    if warehouse_target.isna().any():
        raise DatasetError(
            "La cible du Data Warehouse est absente pour une commande éligible."
        )
    warehouse_target = warehouse_target.astype(bool).astype("int8")
    mismatches = eligible["is_delayed"].ne(warehouse_target)
    if mismatches.any():
        raise DatasetError(
            "La cible recalculée diverge du Data Warehouse pour "
            f"{int(mismatches.sum())} commande(s)."
        )

    exclusions = frame[frame["exclusion_reason"].notna()].copy()
    return PopulationResult(
        supervised_source=eligible.reset_index(drop=True),
        exclusions=exclusions.reset_index(drop=True),
        source_row_count=len(frame),
    )


def exclusion_summary(exclusions: pd.DataFrame) -> dict[str, Any]:
    """Summarize exclusions without exporting post-outcome row details."""

    reason_counts = (
        exclusions["exclusion_reason"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    status_counts = (
        exclusions["order_status"]
        .fillna("<missing>")
        .value_counts()
        .sort_index()
        .to_dict()
    )
    return {
        "excluded_rows": int(len(exclusions)),
        "by_reason": {str(key): int(value) for key, value in reason_counts.items()},
        "by_order_status": {
            str(key): int(value) for key, value in status_counts.items()
        },
    }
