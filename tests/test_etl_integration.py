
import pandas as pd

from config.paths import INTERIM_DATA_DIR


EXPECTED_ROWS = {
    "orders_clean": 99441,
    "order_items_clean": 112650,
    "products_clean": 32951,
    "geolocation_zip": 19015,
    "customers_clean": 99441,
    "sellers_clean": 3095,
    "order_payments_clean": 103886,
    "order_payments_aggregated": 99440,
    "order_reviews_clean": 99224,
    "order_reviews_aggregated": 98673,
}


def main():

    print("\n=== TEST INTÉGRATION ETL ===")

    tables = {}

    # 1. Vérifier les fichiers et les dimensions
    for name, expected_rows in EXPECTED_ROWS.items():

        path = INTERIM_DATA_DIR / f"{name}.parquet"

        assert path.exists(), f"Fichier absent : {path}"

        df = pd.read_parquet(path)

        assert len(df) == expected_rows, (
            f"{name}: {len(df)} lignes "
            f"au lieu de {expected_rows}"
        )

        assert df.columns.is_unique, (
            f"{name}: colonnes dupliquées"
        )

        tables[name] = df

        print(
            f"{name}: {len(df)} lignes, "
            f"{len(df.columns)} colonnes - OK"
        )

    # 2. Vérifier les clés uniques
    print("\n=== CONTRÔLES DES CLÉS ===")

    unique_keys = {
        "orders_clean": ["order_id"],
        "order_items_clean": ["order_id", "order_item_id"],
        "products_clean": ["product_id"],
        "geolocation_zip": ["geolocation_zip_code_prefix"],
        "customers_clean": ["customer_id"],
        "sellers_clean": ["seller_id"],
        "order_payments_clean": [
            "order_id", "payment_sequential"
        ],
        "order_payments_aggregated": ["order_id"],
        "order_reviews_clean": ["review_id", "order_id"],
        "order_reviews_aggregated": ["order_id"],
    }

    for name, columns in unique_keys.items():

        df = tables[name]

        assert df[columns].notna().all().all(), (
            f"{name}: clé manquante"
        )

        assert not df.duplicated(
            subset=columns
        ).any(), f"{name}: clé dupliquée"

        print(f"{name}: clé valide")

    # 3. Vérifier les relations entre tables
    print("\n=== INTÉGRITÉ RÉFÉRENTIELLE ===")

    relationships = [
        (
            "orders_clean",
            "customer_id",
            "customers_clean",
            "customer_id"
        ),
        (
            "order_items_clean",
            "order_id",
            "orders_clean",
            "order_id"
        ),
        (
            "order_items_clean",
            "product_id",
            "products_clean",
            "product_id"
        ),
        (
            "order_items_clean",
            "seller_id",
            "sellers_clean",
            "seller_id"
        ),
        (
            "order_payments_clean",
            "order_id",
            "orders_clean",
            "order_id"
        ),
        (
            "order_reviews_clean",
            "order_id",
            "orders_clean",
            "order_id"
        ),
    ]

    for child, child_key, parent, parent_key in relationships:

        orphan_count = (
            ~tables[child][child_key].isin(
                tables[parent][parent_key]
            )
        ).sum()

        assert orphan_count == 0, (
            f"{child}: {orphan_count} références orphelines"
        )

        print(f"{child} -> {parent}: OK")

    # 4. Vérifier les agrégations financières
    print("\n=== CONTRÔLES FINANCIERS ===")

    payments_detail = tables["order_payments_clean"]
    payments_agg = tables["order_payments_aggregated"]

    total_detail = round(
        payments_detail["payment_value"].sum(), 2
    )

    total_agg = round(
        payments_agg["total_payment_value"].sum(), 2
    )

    assert total_detail == total_agg

    print("Paiements détaillés :", total_detail)
    print("Paiements agrégés :", total_agg)

    # 5. Vérifier les agrégations des avis
    print("\n=== CONTRÔLES DES AVIS ===")

    reviews_detail = tables["order_reviews_clean"]
    reviews_agg = tables["order_reviews_aggregated"]

    assert (
        reviews_agg["review_count"].sum()
        == len(reviews_detail)
    )

    print(
        "Avis détaillés :",
        len(reviews_detail)
    )

    print(
        "Avis après agrégation :",
        reviews_agg["review_count"].sum()
    )

    # 6. Vérifier les coordonnées géographiques
    print("\n=== CONTRÔLES GÉOGRAPHIQUES ===")

    customers = tables["customers_clean"]
    sellers = tables["sellers_clean"]

    assert customers["flag_has_gps"].sum() == 99162
    assert sellers["flag_has_gps"].sum() == 3088

    print(
        "Clients avec GPS :",
        customers["flag_has_gps"].sum()
    )

    print(
        "Vendeurs avec GPS :",
        sellers["flag_has_gps"].sum()
    )

    print("\nTOUS LES TESTS D'INTÉGRATION SONT RÉUSSIS.")


if __name__ == "__main__":
    main()
