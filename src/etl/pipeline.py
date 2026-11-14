
import logging

import pandas as pd

from config.paths import INTERIM_DATA_DIR
from src.etl.extract import extract_all
from src.etl.transform import (
    transform_orders,
    transform_order_items,
    transform_order_payments,
    aggregate_order_payments,
    transform_order_reviews,
    aggregate_order_reviews,
    transform_products,
    transform_geolocation,
    build_geolocation_zip,
    transform_customers,
    transform_sellers,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


def validate_table(
    name: str,
    df: pd.DataFrame,
    expected_rows: int | None = None
) -> None:

    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(
            f"{name}: {len(df)} lignes au lieu de "
            f"{expected_rows}"
        )

    if df.columns.duplicated().any():
        raise ValueError(
            f"{name}: noms de colonnes dupliqués"
        )

    logger.info(
        "Validation %s : %s lignes, %s colonnes",
        name,
        len(df),
        len(df.columns)
    )


def run_pipeline() -> None:

    logger.info("=== DÉBUT DU PIPELINE ETL ===")

    # 1. Extraction
    raw = extract_all()

    # 2. Transformations
    geo_clean = transform_geolocation(
        raw["geolocation"]
    )

    geo_zip = build_geolocation_zip(geo_clean)

    tables = {
        "orders_clean": transform_orders(
            raw["orders"]
        ),

        "order_items_clean": transform_order_items(
            raw["order_items"]
        ),

        "products_clean": transform_products(
            raw["products"],
            raw["category_translation"]
        ),

        "geolocation_zip": geo_zip,

        "customers_clean": transform_customers(
            raw["customers"],
            geo_zip
        ),

        "sellers_clean": transform_sellers(
            raw["sellers"],
            geo_zip
        ),
    }

    payments_clean = transform_order_payments(
        raw["order_payments"]
    )

    reviews_clean = transform_order_reviews(
        raw["order_reviews"]
    )

    tables["order_payments_clean"] = payments_clean

    tables["order_payments_aggregated"] = (
        aggregate_order_payments(payments_clean)
    )

    tables["order_reviews_clean"] = reviews_clean

    tables["order_reviews_aggregated"] = (
        aggregate_order_reviews(reviews_clean)
    )

    # 3. Contrôles de conservation des lignes
    expected_rows = {
        "orders_clean": len(raw["orders"]),
        "order_items_clean": len(raw["order_items"]),
        "order_payments_clean": len(raw["order_payments"]),
        "order_reviews_clean": len(raw["order_reviews"]),
        "products_clean": len(raw["products"]),
        "customers_clean": len(raw["customers"]),
        "sellers_clean": len(raw["sellers"]),
    }

    for name, df in tables.items():
        validate_table(
            name,
            df,
            expected_rows.get(name)
        )

    # 4. Export Parquet
    INTERIM_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for name, df in tables.items():

        output_path = (
            INTERIM_DATA_DIR / f"{name}.parquet"
        )

        df.to_parquet(
            output_path,
            index=False,
            engine="pyarrow"
        )

        logger.info(
            "Export : %s",
            output_path
        )

    logger.info(
        "=== PIPELINE ETL TERMINÉ : %s TABLES ===",
        len(tables)
    )


if __name__ == "__main__":
    run_pipeline()
