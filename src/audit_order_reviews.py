
import pandas as pd

from config.paths import RAW_DATA_DIR
from src.etl.extract import extract_csv


def main():

    reviews = extract_csv(
        RAW_DATA_DIR / "olist_order_reviews_dataset.csv"
    )

    print("\n=== AUDIT ORDER REVIEWS ===")

    print("\n1. Dimensions :")
    print(reviews.shape)

    print("\n2. Valeurs manquantes :")
    print(reviews.isna().sum())

    print("\n3. Distribution des notes :")
    print(
        reviews["review_score"]
        .value_counts()
        .sort_index()
    )

    print("\n4. Statistiques des notes :")
    print(reviews["review_score"].describe())

    print("\n5. Notes invalides :")
    invalid_scores = (
        reviews["review_score"].isna()
        | ~reviews["review_score"].between(1, 5)
    )
    print(invalid_scores.sum())

    print("\n6. Identifiants review_id dupliqués :")
    print(reviews["review_id"].duplicated().sum())

    print("\n7. Clé composite dupliquée :")
    print(
        reviews.duplicated(
            subset=["review_id", "order_id"]
        ).sum()
    )

    print("\n8. Commandes avec plusieurs avis :")
    reviews_per_order = (
        reviews.groupby("order_id").size()
    )
    print(
        "Commandes multi-avis :",
        (reviews_per_order > 1).sum()
    )
    print(
        "Maximum d'avis par commande :",
        reviews_per_order.max()
    )

    print("\n9. Couverture des commentaires :")
    print(
        "Titres présents :",
        reviews["review_comment_title"].notna().sum()
    )
    print(
        "Messages présents :",
        reviews["review_comment_message"].notna().sum()
    )

    print("\n10. Validité des dates :")
    date_columns = [
        "review_creation_date",
        "review_answer_timestamp"
    ]

    for column in date_columns:
        converted = pd.to_datetime(
            reviews[column],
            errors="coerce"
        )

        invalid = (
            converted.isna().sum()
            - reviews[column].isna().sum()
        )

        print(
            column,
            "- dates invalides :",
            invalid
        )

    print("\n11. Réponses avant création de l'avis :")

    creation = pd.to_datetime(
        reviews["review_creation_date"],
        errors="coerce"
    )

    answer = pd.to_datetime(
        reviews["review_answer_timestamp"],
        errors="coerce"
    )

    print(
        (answer < creation).sum()
    )

    print("\n12. Temps de réponse aux avis (jours) :")

    response_days = (
        answer - creation
    ).dt.total_seconds() / 86400

    print(response_days.describe())

    print("\n13. Doublons exacts :")
    print(reviews.duplicated().sum())


if __name__ == "__main__":
    main()
