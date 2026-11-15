
import logging
import os
from decimal import Decimal
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

from config.paths import INTERIM_DATA_DIR


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 5000

# Respect des dépendances entre tables.
TABLES = [
    "customers_clean",
    "sellers_clean",
    "products_clean",
    "geolocation_zip",
    "orders_clean",
    "order_items_clean",
    "order_payments_clean",
    "order_payments_aggregated",
    "order_reviews_clean",
    "order_reviews_aggregated",
]

# Colonnes financières stockées en NUMERIC(15,2).
MONEY_COLUMNS = {
    "order_items_clean": [
        "price",
        "freight_value",
        "item_total_value",
    ],
    "order_payments_clean": [
        "payment_value",
    ],
    "order_payments_aggregated": [
        "total_payment_value",
    ],
}

# Colonnes entières dont le Parquet utilise float64
# en raison de valeurs manquantes.
INTEGER_COLUMNS = {
    "products_clean": [
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
    ],
}


def get_connection():
    load_dotenv()

    required = [
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
    ]

    missing = [key for key in required if not os.getenv(key)]

    if missing:
        raise ValueError(
            f"Variables d'environnement manquantes : {missing}"
        )

    return psycopg.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ["DB_PORT"]),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )


def prepare_dataframe(table_name: str) -> pd.DataFrame:
    path = Path(INTERIM_DATA_DIR) / f"{table_name}.parquet"

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)

    for column in MONEY_COLUMNS.get(table_name, []):
        df[column] = df[column].map(
            lambda value: (
                None
                if pd.isna(value)
                else Decimal(str(round(float(value), 2)))
            )
        )

    if table_name == "order_items_clean":
        # Recalcul après conversion monétaire pour garantir
        # la cohérence avec la contrainte CHECK SQL.
        df["item_total_value"] = (
            df["price"] + df["freight_value"]
        )


    for column in INTEGER_COLUMNS.get(table_name, []):
        # Conversion des float64 en entiers nullable
        df[column] = df[column].astype("Int64")

    # Conversion finale vers des objets Python
    # et remplacement des valeurs manquantes par None
    df = df.astype(object)
    df = df.where(pd.notna(df), None)

    return df



def load_table(conn, table_name: str) -> int:
    df = prepare_dataframe(table_name)

    columns = list(df.columns)
    column_sql = ", ".join(f'"{column}"' for column in columns)

    copy_query = (
        f"COPY staging.{table_name} ({column_sql}) FROM STDIN"
    )

    logger.info(
        "Chargement de %s : %s lignes",
        table_name,
        len(df),
    )

    with conn.cursor() as cursor:
        with cursor.copy(copy_query) as copy:
            for start in range(0, len(df), BATCH_SIZE):
                batch = df.iloc[start:start + BATCH_SIZE]

                for row in batch.itertuples(
                    index=False,
                    name=None,
                ):
                    copy.write_row(row)

        cursor.execute(
            f"SELECT COUNT(*) FROM staging.{table_name}"
        )
        actual_count = cursor.fetchone()[0]

    if actual_count != len(df):
        raise RuntimeError(
            f"{table_name}: attendu={len(df)}, "
            f"obtenu={actual_count}"
        )

    logger.info(
        "Validation OK : %s (%s lignes)",
        table_name,
        actual_count,
    )

    return actual_count


def main():
    expected_total = 0

    logger.info("Début du chargement PostgreSQL")

    with get_connection() as conn:
        with conn.transaction():
            with conn.cursor() as cursor:
                # Sécurité : on exige des tables vides.
                for table_name in TABLES:
                    cursor.execute(
                        f"SELECT COUNT(*) FROM staging.{table_name}"
                    )
                    if cursor.fetchone()[0] != 0:
                        raise RuntimeError(
                            f"Table non vide : staging.{table_name}"
                        )

            for table_name in TABLES:
                expected_total += load_table(conn, table_name)

    logger.info(
        "Chargement terminé : %s lignes au total",
        expected_total,
    )


if __name__ == "__main__":
    main()
