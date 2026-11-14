
import logging

from config.paths import RAW_DATA_DIR
from src.etl.extract import extract_csv
from src.etl.transform import (
    transform_geolocation,
    build_geolocation_zip,
    transform_customers,
    transform_sellers
)


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)


def main():

    customers = extract_csv(
        RAW_DATA_DIR / "olist_customers_dataset.csv"
    )

    sellers = extract_csv(
        RAW_DATA_DIR / "olist_sellers_dataset.csv"
    )

    geolocation = extract_csv(
        RAW_DATA_DIR / "olist_geolocation_dataset.csv"
    )

    geo_clean = transform_geolocation(geolocation)

    geo_zip = build_geolocation_zip(geo_clean)

    print("\nColonnes géographiques disponibles :")
    print(geo_zip.columns.tolist())

    customers_clean = transform_customers(
        customers,
        geo_zip
    )

    sellers_clean = transform_sellers(
        sellers,
        geo_zip
    )

    print("\n=== TEST CUSTOMERS & SELLERS ===")

    print("\nClients :")
    print("Lignes originales :", len(customers))
    print("Lignes transformées :", len(customers_clean))
    print(
        "Clients uniques :",
        customers_clean["customer_unique_id"].nunique()
    )
    print(
        "Clients avec GPS :",
        customers_clean["flag_has_gps"].sum()
    )
    print(
        "Clients sans GPS :",
        (~customers_clean["flag_has_gps"]).sum()
    )

    print("\nVendeurs :")
    print("Lignes originales :", len(sellers))
    print("Lignes transformées :", len(sellers_clean))
    print(
        "Vendeurs avec GPS :",
        sellers_clean["flag_has_gps"].sum()
    )
    print(
        "Vendeurs sans GPS :",
        (~sellers_clean["flag_has_gps"]).sum()
    )

    # Tests
    assert len(customers_clean) == len(customers)
    assert len(sellers_clean) == len(sellers)

    assert customers_clean["customer_id"].is_unique
    assert sellers_clean["seller_id"].is_unique

    assert (
        customers_clean["customer_unique_id"].nunique()
        == 96096
    )

    assert customers_clean["flag_has_gps"].sum() == 99162
    assert sellers_clean["flag_has_gps"].sum() == 3088

    print("\nTous les tests sont réussis.")


if __name__ == "__main__":
    main()
