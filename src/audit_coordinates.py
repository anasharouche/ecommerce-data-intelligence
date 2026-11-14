
import pandas as pd

from config.paths import RAW_DATA_DIR
from src.etl.extract import extract_csv


def main():

    geo = extract_csv(
        RAW_DATA_DIR / "olist_geolocation_dataset.csv"
    )

    lat = pd.to_numeric(
        geo["geolocation_lat"],
        errors="coerce"
    )

    lng = pd.to_numeric(
        geo["geolocation_lng"],
        errors="coerce"
    )

    print("\n=== AUDIT DES COORDONNÉES GPS ===")

    print("\n1. Statistiques descriptives :")
    print(
        pd.DataFrame({
            "latitude": lat,
            "longitude": lng
        }).describe()
    )

    print("\n2. Coordonnées manquantes ou invalides :")
    print("Latitudes manquantes :", lat.isna().sum())
    print("Longitudes manquantes :", lng.isna().sum())

    print("\n3. Coordonnées hors limites GPS :")

    invalid_gps = (
        lat.notna()
        & lng.notna()
        & (
            ~lat.between(-90, 90)
            | ~lng.between(-180, 180)
        )
    )

    print("Lignes hors limites GPS :", invalid_gps.sum())

    print("\n4. Coordonnées hors zone géographique approximative du Brésil :")

    # Rectangle approximatif, volontairement large.
    # Ce filtre n'est pas une frontière officielle.
    outside_brazil_bbox = (
        lat.notna()
        & lng.notna()
        & (
            ~lat.between(-35, 6)
            | ~lng.between(-75, -30)
        )
    )

    print(
        "Lignes hors zone approximative :",
        outside_brazil_bbox.sum()
    )

    print("\n5. Exemples de coordonnées suspectes :")

    print(
        geo.loc[
            invalid_gps | outside_brazil_bbox,
            [
                "geolocation_zip_code_prefix",
                "geolocation_city",
                "geolocation_state",
                "geolocation_lat",
                "geolocation_lng"
            ]
        ].head(15).to_string(index=False)
    )

    print("\n6. Préfixes postaux avec coordonnées hors zone :")

    print(
        geo.loc[
            invalid_gps | outside_brazil_bbox,
            "geolocation_zip_code_prefix"
        ].nunique()
    )


if __name__ == "__main__":
    main()
