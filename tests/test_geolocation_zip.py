
import logging

from config.paths import RAW_DATA_DIR
from src.etl.extract import extract_csv
from src.etl.transform import (
    transform_geolocation,
    build_geolocation_zip
)


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)


def main():

    original = extract_csv(
        RAW_DATA_DIR / "olist_geolocation_dataset.csv"
    )

    clean = transform_geolocation(original)

    geo_zip = build_geolocation_zip(clean)

    print("\n=== TEST GEOLOCATION ZIP ===")

    print("Observations originales :", len(original))
    print("Observations nettoyées :", len(clean))

    print(
        "Coordonnées hors zone :",
        (~clean["flag_brazil_bbox"]).sum()
    )

    print("Préfixes agrégés :", len(geo_zip))

    print(
        "Préfixes dupliqués :",
        geo_zip[
            "geolocation_zip_code_prefix"
        ].duplicated().sum()
    )

    print(
        "Préfixes sans coordonnées admissibles :",
        geo_zip["latitude"].isna().sum()
    )

    print("\nStatistiques de qualité GPS :")
    print(geo_zip["gps_quality_ratio"].describe())

    print("\nExemples :")
    print(geo_zip.head(10).to_string(index=False))

    assert len(original) == len(clean)
    assert len(geo_zip) == 19015

    assert (
        geo_zip[
            "geolocation_zip_code_prefix"
        ].is_unique
    )

    assert (
        geo_zip["observation_count"].sum()
        == len(original)
    )


if __name__ == "__main__":
    main()
