
import pandas as pd

from config.paths import RAW_DATA_DIR
from src.etl.extract import extract_csv
from src.etl.transform import transform_locations


def main():

    geo = extract_csv(
        RAW_DATA_DIR / "olist_geolocation_dataset.csv"
    )

    geo = transform_locations(
        geo,
        "geolocation_city",
        "geolocation_state"
    )

    print("\n=== AUDIT GEOLOCATION ===")

    print("\n1. Nombre de villes distinctes :")
    print(geo["geolocation_city"].nunique())

    print("\n2. Variantes contenant 'paulo' :")

    variants = (
        geo.loc[
            geo["geolocation_city"].str.contains(
                "paulo",
                case=False,
                na=False
            ),
            "geolocation_city"
        ]
        .value_counts()
    )

    print(variants.head(30))

    print("\n3. Valeurs suspectes :")

    suspicious = geo[
        geo["geolocation_city"].str.contains(
            r"[^a-z\s\-']",
            regex=True,
            na=False
        )
    ]

    print("Lignes suspectes :", len(suspicious))

    print(
        suspicious["geolocation_city"]
        .value_counts()
        .head(30)
    )

    print("\n4. Doublons exacts :")
    print(geo.duplicated().sum())

    print("\n5. Préfixes postaux distincts :")
    print(
        geo["geolocation_zip_code_prefix"].nunique()
    )

    print("\n6. Préfixes avec plusieurs coordonnées :")

    coordinate_counts = (
        geo.groupby("geolocation_zip_code_prefix")
        .apply(
            lambda group: group[
                ["geolocation_lat", "geolocation_lng"]
            ].drop_duplicates().shape[0],
            include_groups=False
        )
    )

    print(
        "Préfixes avec plusieurs coordonnées :",
        (coordinate_counts > 1).sum()
    )

    print(
        "Maximum de coordonnées distinctes par préfixe :",
        coordinate_counts.max()
    )


if __name__ == "__main__":
    main()
