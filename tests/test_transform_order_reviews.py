
import logging

from config.paths import RAW_DATA_DIR
from src.etl.extract import extract_csv
from src.etl.transform import (
    transform_order_reviews,
    aggregate_order_reviews
)


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)


def main():

    original = extract_csv(
        RAW_DATA_DIR / "olist_order_reviews_dataset.csv"
    )

    clean = transform_order_reviews(original)
    aggregated = aggregate_order_reviews(clean)

    print("\n=== TEST ORDER REVIEWS ===")

    print("Avis originaux :", len(original))
    print("Avis nettoyés :", len(clean))
    print("Commandes agrégées :", len(aggregated))

    print("\nIndicateurs de qualité :")
    print(
        "Titres présents :",
        clean["flag_has_title"].sum()
    )
    print(
        "Messages présents :",
        clean["flag_has_message"].sum()
    )
    print(
        "Délais négatifs :",
        clean["flag_negative_response_time"].sum()
    )
    print(
        "Délais supérieurs à 30 jours :",
        clean["flag_long_response_time"].sum()
    )

    print("\nStatistiques de satisfaction :")
    print(
        "Note moyenne :",
        clean["review_score"].mean()
    )
    print(
        "Commandes multi-avis :",
        (aggregated["review_count"] > 1).sum()
    )

    print("\nClés dupliquées après agrégation :")
    print(aggregated["order_id"].duplicated().sum())

    assert len(original) == len(clean)
    assert clean["review_score"].between(1, 5).all()
    assert aggregated["order_id"].is_unique
    assert (
        aggregated["review_count"].sum()
        == len(clean)
    )

    assert (
        clean["flag_negative_response_time"].sum()
        == 0
    )

    assert (
        (aggregated["review_count"] > 1).sum()
        == 547
    )

    print("\nTous les tests sont réussis.")


if __name__ == "__main__":
    main()
