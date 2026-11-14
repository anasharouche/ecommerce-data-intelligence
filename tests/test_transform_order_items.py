
import logging

import pandas as pd

from config.paths import RAW_DATA_DIR
from src.etl.extract import extract_csv
from src.etl.transform import transform_order_items


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)


def main():

    original = extract_csv(
        RAW_DATA_DIR / "olist_order_items_dataset.csv"
    )

    clean = transform_order_items(original)

    print("\n=== TEST TRANSFORMATION ORDER ITEMS ===")

    print("Lignes originales :", len(original))
    print("Lignes transformées :", len(clean))

    print("\nTypes des colonnes :")
    print(clean.dtypes)

    print("\nContrôles financiers :")
    print("Prix négatifs :", (clean["price"] < 0).sum())
    print(
        "Frais de livraison négatifs :",
        (clean["freight_value"] < 0).sum()
    )
    print(
        "Livraisons gratuites :",
        (clean["freight_value"] == 0).sum()
    )

    print("\nClés composites dupliquées :")
    print(
        clean.duplicated(
            subset=["order_id", "order_item_id"]
        ).sum()
    )

    print("\nDates invalides après conversion :")
    print(clean["shipping_limit_date"].isna().sum())

    print("\nMontants agrégés :")
    print(
        clean[
            ["price", "freight_value", "item_total_value"]
        ].sum()
    )

    # Assertions
    assert len(original) == len(clean)

    assert clean[
        "shipping_limit_date"
    ].notna().all()

    assert not clean.duplicated(
        subset=["order_id", "order_item_id"]
    ).any()

    assert (
        clean["item_total_value"]
        .round(2)
        == (
            clean["price"]
            + clean["freight_value"]
        ).round(2)
    ).all()

    # Le DataFrame original doit rester inchangé
    assert "item_total_value" not in original.columns

    print("\nTous les tests sont réussis.")


if __name__ == "__main__":
    main()
