
import pandas as pd

from config.paths import RAW_DATA_DIR
from src.etl.extract import extract_csv


def main():

    products = extract_csv(
        RAW_DATA_DIR / "olist_products_dataset.csv"
    )

    translations = extract_csv(
        RAW_DATA_DIR / "product_category_name_translation.csv"
    )

    print("\n=== AUDIT PRODUCTS ===")

    print("\n1. Dimensions :")
    print(products.shape)

    print("\n2. Valeurs manquantes :")
    print(products.isna().sum())

    print("\n3. Clés product_id dupliquées :")
    print(products["product_id"].duplicated().sum())

    print("\n4. Catégories distinctes :")
    print(
        products["product_category_name"].nunique()
    )

    print("\n5. Catégories les plus fréquentes :")
    print(
        products["product_category_name"]
        .value_counts()
        .head(15)
    )

    print("\n6. Valeurs numériques :")

    numeric_columns = [
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm"
    ]

    print(
        products[numeric_columns].describe()
    )

    print("\n7. Valeurs négatives :")

    for column in numeric_columns:
        print(
            column,
            ":",
            (products[column] < 0).sum()
        )

    print("\n8. Dimensions et poids nuls :")

    dimension_columns = [
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm"
    ]

    for column in dimension_columns:
        print(
            column,
            ":",
            (products[column] == 0).sum()
        )

    print("\n9. Couverture des traductions :")

    categories = set(
        products["product_category_name"].dropna()
    )

    translated_categories = set(
        translations["product_category_name"].dropna()
    )

    missing_translations = (
        categories - translated_categories
    )

    print(
        "Catégories sans traduction :",
        len(missing_translations)
    )

    print(
        "Liste :",
        sorted(missing_translations)
    )

    print("\n10. Clés dupliquées dans les traductions :")
    print(
        translations[
            "product_category_name"
        ].duplicated().sum()
    )


if __name__ == "__main__":
    main()
