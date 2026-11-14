
import pandas as pd
import logging

from config.paths import RAW_DATA_DIR
from src.etl.extract import extract_csv
from src.etl.transform import transform_orders


def main():


    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s"
    )
    # Extraction
    orders = extract_csv(
        RAW_DATA_DIR / "olist_orders_dataset.csv"
    )

    # Transformation
    clean_orders = transform_orders(orders)

    print("\n=== TEST TRANSFORMATION ORDERS ===")

    print("Lignes originales :", len(orders))
    print("Lignes transformées :", len(clean_orders))

    print("\nTypes des colonnes :")
    print(clean_orders.dtypes)

    print("\nAnomalies détectées :")
    print(
        clean_orders[
            [
                "flag_delivery_before_purchase",
                "flag_delivered_without_date",
                "flag_canceled_with_delivery",
            ]
        ].sum()
    )

    print("\nValeurs manquantes après conversion :")
    print(
        clean_orders[
            [
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ]
        ].isna().sum()
    )


if __name__ == "__main__":
    main()
