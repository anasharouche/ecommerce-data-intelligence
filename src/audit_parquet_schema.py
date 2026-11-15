
import pandas as pd

from config.paths import INTERIM_DATA_DIR


TABLES = [
    "orders_clean",
    "order_items_clean",
    "products_clean",
    "customers_clean",
    "sellers_clean",
    "geolocation_zip",
    "order_payments_clean",
    "order_payments_aggregated",
    "order_reviews_clean",
    "order_reviews_aggregated",
]


def main():

    for table in TABLES:

        path = INTERIM_DATA_DIR / f"{table}.parquet"

        df = pd.read_parquet(path)

        print(f"\n{'=' * 65}")
        print(f"TABLE : {table}")
        print(f"LIGNES : {len(df)}")
        print(f"{'=' * 65}")

        for column, dtype in df.dtypes.items():

            null_count = df[column].isna().sum()

            print(
                f"{column:<40} "
                f"{str(dtype):<20} "
                f"NULL: {null_count}"
            )


if __name__ == "__main__":
    main()
