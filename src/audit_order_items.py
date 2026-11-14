
import pandas as pd

from config.paths import RAW_DATA_DIR
from src.etl.extract import extract_csv


def main():

    items = extract_csv(
        RAW_DATA_DIR / "olist_order_items_dataset.csv"
    )

    print("\n=== AUDIT ORDER ITEMS ===")

    print("\n1. Dimensions :")
    print(items.shape)

    print("\n2. Valeurs manquantes :")
    print(items.isna().sum())

    print("\n3. Statistiques financières :")
    print(
        items[["price", "freight_value"]]
        .describe()
    )

    print("\n4. Valeurs négatives ou nulles :")

    for column in ["price", "freight_value"]:
        print(
            f"{column} négatif :",
            (items[column] < 0).sum()
        )
        print(
            f"{column} égal à zéro :",
            (items[column] == 0).sum()
        )

    print("\n5. Clé composite dupliquée :")
    print(
        items.duplicated(
            subset=["order_id", "order_item_id"]
        ).sum()
    )

    print("\n6. Nombre d'articles par commande :")

    items_per_order = (
        items.groupby("order_id")
        .size()
    )

    print(items_per_order.describe())

    print("\n7. Commandes avec plusieurs vendeurs :")

    sellers_per_order = (
        items.groupby("order_id")["seller_id"]
        .nunique()
    )

    print(
        "Commandes multi-vendeurs :",
        (sellers_per_order > 1).sum()
    )

    print("\n8. Dates d'expédition invalides :")

    converted_dates = pd.to_datetime(
        items["shipping_limit_date"],
        errors="coerce"
    )

    print(
        "Dates invalides :",
        (
            converted_dates.isna().sum()
            - items["shipping_limit_date"].isna().sum()
        )
    )


if __name__ == "__main__":
    main()
