
import logging

from config.paths import RAW_DATA_DIR
from src.etl.extract import extract_csv
from src.etl.transform import (
    transform_order_payments,
    aggregate_order_payments
)


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)


def main():

    original = extract_csv(
        RAW_DATA_DIR / "olist_order_payments_dataset.csv"
    )

    clean = transform_order_payments(original)

    aggregated = aggregate_order_payments(clean)

    print("\n=== TEST ORDER PAYMENTS ===")

    print("Paiements originaux :", len(original))
    print("Paiements nettoyés :", len(clean))
    print("Commandes agrégées :", len(aggregated))

    print("\nIndicateurs de qualité :")
    print("Paiements nuls :", clean["flag_zero_payment"].sum())
    print(
        "Versements nuls :",
        clean["flag_zero_installments"].sum()
    )
    print(
        "Types non définis :",
        clean["flag_undefined_payment_type"].sum()
    )

    print("\nMontants financiers :")

    original_total = round(
        original["payment_value"].sum(), 2
    )

    aggregated_total = round(
        aggregated["total_payment_value"].sum(), 2
    )

    print("Montant original :", original_total)
    print("Montant agrégé :", aggregated_total)

    print("\nCommandes multi-paiements :")
    print(
        (aggregated["payment_count"] > 1).sum()
    )

    print("\nClés dupliquées après agrégation :")
    print(aggregated["order_id"].duplicated().sum())

    assert len(original) == len(clean)
    assert len(aggregated) == 99440
    assert original_total == aggregated_total
    assert aggregated["order_id"].is_unique

    print("\nTous les tests sont réussis.")


if __name__ == "__main__":
    main()
