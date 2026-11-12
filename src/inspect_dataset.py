
from pathlib import Path
import pandas as pd

# Chemin vers les données originales
DATA_DIR = Path("data/raw/olist")

# Fichiers attendus
EXPECTED_FILES = [
    "olist_orders_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv",
]

def inspect_dataset():
    print("=" * 70)
    print("OLIST DATASET - INITIAL INSPECTION")
    print("=" * 70)

    for filename in EXPECTED_FILES:
        file_path = DATA_DIR / filename

        print(f"\nFile: {filename}")

        if not file_path.exists():
            print("STATUS: MISSING")
            continue

        df = pd.read_csv(file_path)

        print(f"Rows: {df.shape[0]:,}")
        print(f"Columns: {df.shape[1]}")

        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nData types:")
        print(df.dtypes.to_string())

        print("\nMissing values:")
        print(df.isnull().sum().to_string())

        print(f"\nDuplicate rows: {df.duplicated().sum():,}")

        print("-" * 70)


if __name__ == "__main__":
    inspect_dataset()
