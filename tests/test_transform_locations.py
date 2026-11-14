
import logging

from config.paths import RAW_DATA_DIR
from src.etl.extract import extract_csv
from src.etl.transform import transform_locations


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)


TABLES = {
    "customers": (
        "olist_customers_dataset.csv",
        "customer_city",
        "customer_state"
    ),
    "sellers": (
        "olist_sellers_dataset.csv",
        "seller_city",
        "seller_state"
    ),
    "geolocation": (
        "olist_geolocation_dataset.csv",
        "geolocation_city",
        "geolocation_state"
    )
}


def main():

    for table_name, (filename, city_col, state_col) in TABLES.items():

        print(f"\n=== {table_name.upper()} ===")

        original = extract_csv(
            RAW_DATA_DIR / filename
        )

        transformed = transform_locations(
            original,
            city_col,
            state_col
        )

        print("Lignes avant :", len(original))
        print("Lignes après :", len(transformed))

        print("\nExemples de villes normalisées :")
        print(
            transformed[city_col]
            .drop_duplicates()
            .head(10)
            .to_list()
        )

        print("\nÉtats présents :")
        print(
            sorted(
                transformed[state_col]
                .dropna()
                .unique()
                .tolist()
            )
        )

        assert len(original) == len(transformed), (
            f"Nombre de lignes modifié dans {table_name}"
        )


if __name__ == "__main__":
    main()
