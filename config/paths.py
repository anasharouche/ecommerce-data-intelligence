
from pathlib import Path

# Racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent

# Répertoires de données
RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "olist"
INTERIM_DATA_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Correspondance entre les noms logiques et les fichiers CSV
OLIST_FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}
