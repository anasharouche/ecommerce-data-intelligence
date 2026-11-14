
import logging

from config.paths import RAW_DATA_DIR
from src.etl.extract import extract_csv
from src.etl.transform import transform_products


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)


def main():

    products = extract_csv(
        RAW_DATA_DIR / "olist_products_dataset.csv"
    )

    translations = extract_csv(
        RAW_DATA_DIR / "product_category_name_translation.csv"
    )

    clean = transform_products(
        products,
        translations
    )

    print("\n=== TEST TRANSFORMATION PRODUCTS ===")

    print("Produits originaux :", len(products))
    print("Produits transformés :", len(clean))

    print("\nIndicateurs de qualité :")

    print(
        "Catégories manquantes :",
        clean["flag_missing_category"].sum()
    )

    print(
        "Poids nuls :",
        clean["flag_zero_weight"].sum()
    )

    print(
        "Dimensions manquantes :",
        clean["flag_missing_dimensions"].sum()
    )

    print(
        "Produits sans traduction :",
        clean["flag_missing_translation"].sum()
    )

    print("\nExemples de catégories :")
    print(
        clean[
            [
                "product_category_name",
                "product_category_name_english"
            ]
        ].drop_duplicates().head(15).to_string(index=False)
    )

    print("\nClés dupliquées :")
    print(clean["product_id"].duplicated().sum())

    assert len(clean) == len(products)
    assert clean["product_id"].is_unique
    assert clean["flag_missing_category"].sum() == 610
    assert clean["flag_zero_weight"].sum() == 4
    assert clean["flag_missing_dimensions"].sum() == 2

    print("\nTous les tests sont réussis.")


if __name__ == "__main__":
    main()
