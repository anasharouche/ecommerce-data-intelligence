
from config.paths import RAW_DATA_DIR
from src.etl.extract import extract_csv


def main():

    customers = extract_csv(
        RAW_DATA_DIR / "olist_customers_dataset.csv"
    )

    sellers = extract_csv(
        RAW_DATA_DIR / "olist_sellers_dataset.csv"
    )

    orders = extract_csv(
        RAW_DATA_DIR / "olist_orders_dataset.csv"
    )

    print("\n=== AUDIT CUSTOMERS ===")

    print("\n1. Dimensions :")
    print(customers.shape)

    print("\n2. Valeurs manquantes :")
    print(customers.isna().sum())

    print("\n3. customer_id dupliqués :")
    print(customers["customer_id"].duplicated().sum())

    print("\n4. Clients uniques :")
    print(customers["customer_unique_id"].nunique())

    print("\n5. Nombre de customer_id par client unique :")

    ids_per_customer = (
        customers.groupby("customer_unique_id")
        ["customer_id"]
        .nunique()
    )

    print(ids_per_customer.describe())

    print(
        "Clients avec plusieurs customer_id :",
        (ids_per_customer > 1).sum()
    )

    print(
        "Maximum de customer_id par client :",
        ids_per_customer.max()
    )

    print("\n6. Commandes par client unique :")

    customer_orders = orders.merge(
        customers[
            ["customer_id", "customer_unique_id"]
        ],
        on="customer_id",
        how="left",
        validate="many_to_one"
    )

    orders_per_customer = (
        customer_orders.groupby("customer_unique_id")
        ["order_id"]
        .nunique()
    )

    print(orders_per_customer.describe())

    print(
        "Clients avec plusieurs commandes :",
        (orders_per_customer > 1).sum()
    )

    print("\n=== AUDIT SELLERS ===")

    print("\n7. Dimensions :")
    print(sellers.shape)

    print("\n8. Valeurs manquantes :")
    print(sellers.isna().sum())

    print("\n9. seller_id dupliqués :")
    print(sellers["seller_id"].duplicated().sum())

    print("\n10. Vendeurs par État :")
    print(
        sellers["seller_state"]
        .value_counts()
        .head(15)
    )


if __name__ == "__main__":
    main()
