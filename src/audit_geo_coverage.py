
import pandas as pd

from config.paths import RAW_DATA_DIR
from src.etl.extract import extract_csv
from src.etl.transform import (
    transform_geolocation,
    build_geolocation_zip
)


def audit_coverage(
    dataframe: pd.DataFrame,
    zip_column: str,
    geo_zip: pd.DataFrame,
    label: str
):
    """
    Mesure la couverture géographique d'une table.
    """

    merged = dataframe.merge(
        geo_zip,
        how="left",
        left_on=zip_column,
        right_on="geolocation_zip_code_prefix",
        validate="many_to_one",
        indicator=True
    )

    total = len(merged)

    matched = (
        merged["_merge"] == "both"
    ).sum()

    usable = (
        merged["latitude"].notna()
        & merged["longitude"].notna()
    ).sum()

    missing_prefixes = (
        merged.loc[
            merged["_merge"] == "left_only",
            zip_column
        ]
        .dropna()
        .nunique()
    )

    print(f"\n=== COUVERTURE {label.upper()} ===")

    print("Lignes totales :", total)
    print("Lignes avec préfixe trouvé :", matched)
    print("Lignes avec coordonnées utilisables :", usable)
    print("Préfixes distincts absents :", missing_prefixes)

    print(
        "Taux de couverture GPS :",
        f"{usable / total * 100:.2f}%"
    )

    assert len(merged) == len(dataframe)


def main():

    geo = extract_csv(
        RAW_DATA_DIR / "olist_geolocation_dataset.csv"
    )

    geo_clean = transform_geolocation(geo)

    geo_zip = build_geolocation_zip(geo_clean)

    customers = extract_csv(
        RAW_DATA_DIR / "olist_customers_dataset.csv"
    )

    sellers = extract_csv(
        RAW_DATA_DIR / "olist_sellers_dataset.csv"
    )

    audit_coverage(
        customers,
        "customer_zip_code_prefix",
        geo_zip,
        "customers"
    )

    audit_coverage(
        sellers,
        "seller_zip_code_prefix",
        geo_zip,
        "sellers"
    )


if __name__ == "__main__":
    main()
