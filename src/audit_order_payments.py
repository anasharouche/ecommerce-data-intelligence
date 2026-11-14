
import pandas as pd

from config.paths import RAW_DATA_DIR
from src.etl.extract import extract_csv


def main():

    payments = extract_csv(
        RAW_DATA_DIR / "olist_order_payments_dataset.csv"
    )

    print("\n=== AUDIT ORDER PAYMENTS ===")

    print("\n1. Dimensions :")
    print(payments.shape)

    print("\n2. Valeurs manquantes :")
    print(payments.isna().sum())

    print("\n3. Types de paiement :")
    print(payments["payment_type"].value_counts())

    print("\n4. Statistiques financières :")
    print(
        payments[
            ["payment_value", "payment_installments"]
        ].describe()
    )

    print("\n5. Paiements négatifs ou nuls :")
    print(
        "Négatifs :",
        (payments["payment_value"] < 0).sum()
    )
    print(
        "Nuls :",
        (payments["payment_value"] == 0).sum()
    )

    print("\n6. Nombre de paiements par commande :")

    payments_per_order = (
        payments.groupby("order_id")
        .size()
    )

    print(payments_per_order.describe())

    print(
        "Commandes avec plusieurs paiements :",
        (payments_per_order > 1).sum()
    )

    print("\n7. Clé composite dupliquée :")
    print(
        payments.duplicated(
            subset=["order_id", "payment_sequential"]
        ).sum()
    )

    print("\n8. Nombre de versements :")
    print(
        payments["payment_installments"]
        .value_counts()
        .sort_index()
    )

    print("\n9. Montant total des paiements :")
    print(payments["payment_value"].sum())

    print("\n10. Comparaison paiements / articles :")

    items = extract_csv(
        RAW_DATA_DIR / "olist_order_items_dataset.csv"
    )

    item_totals = (
        items.assign(
            item_total=items["price"] + items["freight_value"]
        )
        .groupby("order_id")["item_total"]
        .sum()
    )

    payment_totals = (
        payments.groupby("order_id")["payment_value"]
        .sum()
    )

    comparison = pd.concat(
        [
            item_totals.rename("items_total"),
            payment_totals.rename("payments_total")
        ],
        axis=1
    )

    # Arrondir à deux décimales pour comparer
    # les montants financiers.
    comparison["difference"] = (
        comparison["payments_total"]
        - comparison["items_total"]
    ).round(2)

    comparable = comparison.dropna(
        subset=["items_total", "payments_total"]
    )

    print(
        "Commandes comparables :",
        len(comparable)
    )

    print(
        "Commandes avec écart non nul :",
        (comparable["difference"] != 0).sum()
    )

    print(
        "Écart absolu maximum :",
        comparable["difference"].abs().max()
    )

    print("\nExemples des écarts les plus importants :")
    print(
        comparable.reindex(
            comparable["difference"].abs()
            .sort_values(ascending=False)
            .head(10)
            .index
        ).to_string()
    )


if __name__ == "__main__":
    main()
