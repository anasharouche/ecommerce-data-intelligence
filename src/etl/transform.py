
import logging

import pandas as pd
import unicodedata


logger = logging.getLogger(__name__)


ORDER_DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


def transform_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie la table orders sans modifier le DataFrame original.

    Transformations :
    - Conversion des dates en datetime.
    - Identification des anomalies chronologiques.
    - Préservation de toutes les commandes.
    """

    df = orders.copy()

    initial_rows = len(df)


    # 1. Conversion et validation des colonnes de dates
    for column in ORDER_DATE_COLUMNS:

        missing_before = df[column].isna().sum()

        df[column] = pd.to_datetime(
            df[column],
            errors="coerce"
        )

        missing_after = df[column].isna().sum()

        invalid_dates = missing_after - missing_before

        if invalid_dates > 0:
            logger.warning(
                "%s : %s dates invalides détectées",
                column,
                invalid_dates
            )

        else:
            logger.info(
                "%s : conversion réussie, aucune date invalide",
                column
            )

    # 2. Contrôle de cohérence : achat après livraison
    df["flag_delivery_before_purchase"] = (
        df["order_delivered_customer_date"].notna()
        & (
            df["order_delivered_customer_date"]
            < df["order_purchase_timestamp"]
        )
    )

    # 3. Contrôle métier : commande livrée sans date de livraison
    df["flag_delivered_without_date"] = (
        df["order_status"].eq("delivered")
        & df["order_delivered_customer_date"].isna()
    )

    # 4. Contrôle métier : commande annulée avec date de livraison
    df["flag_canceled_with_delivery"] = (
        df["order_status"].eq("canceled")
        & df["order_delivered_customer_date"].notna()
    )

    # 5. Vérification du nombre de lignes
    assert len(df) == initial_rows, (
        "Le nombre de commandes a changé pendant la transformation."
    )

    logger.info(
        "Transformation orders terminée : %s lignes",
        len(df)
    )

    logger.info(
        "Livraisons avant achat : %s",
        df["flag_delivery_before_purchase"].sum()
    )

    logger.info(
        "Commandes livrées sans date : %s",
        df["flag_delivered_without_date"].sum()
    )

    logger.info(
        "Commandes annulées avec livraison : %s",
        df["flag_canceled_with_delivery"].sum()
    )

    return df


def normalize_city(value):
    """
    Normalise un nom de ville :
    - supprime les espaces superflus ;
    - convertit en minuscules ;
    - supprime les accents ;
    - conserve les valeurs manquantes.
    """
    if pd.isna(value):
        return value

    value = str(value).strip().lower()
    value = " ".join(value.split())

    value = unicodedata.normalize("NFKD", value)

    value = "".join(
        char for char in value
        if not unicodedata.combining(char)
    )

    return value


def transform_locations(
    df: pd.DataFrame,
    city_column: str,
    state_column: str
) -> pd.DataFrame:
    """
    Normalise les colonnes géographiques
    sans modifier le DataFrame original.
    """

    result = df.copy()

    for column in [city_column, state_column]:
        if column not in result.columns:
            raise ValueError(
                f"Colonne absente : {column}"
            )

    result[city_column] = result[city_column].apply(
        normalize_city
    )

    result[state_column] = (
        result[state_column]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    logger.info(
        "Normalisation géographique terminée : %s lignes",
        len(result)
    )

    return result


def transform_geolocation(
    geolocation: pd.DataFrame
) -> pd.DataFrame:
    """
    Normalise la géolocalisation et ajoute des
    indicateurs de qualité sans supprimer de lignes.
    """

    df = transform_locations(
        geolocation,
        "geolocation_city",
        "geolocation_state"
    )

    lat = pd.to_numeric(
        df["geolocation_lat"],
        errors="coerce"
    )

    lng = pd.to_numeric(
        df["geolocation_lng"],
        errors="coerce"
    )

    df["geolocation_lat"] = lat
    df["geolocation_lng"] = lng

    df["flag_valid_gps"] = (
        lat.between(-90, 90)
        & lng.between(-180, 180)
    )

    # Rectangle géographique approximatif du Brésil
    df["flag_brazil_bbox"] = (
        df["flag_valid_gps"]
        & lat.between(-35, 6)
        & lng.between(-75, -30)
    )

    logger.info(
        "Géolocalisation : %s coordonnées hors zone",
        (~df["flag_brazil_bbox"]).sum()
    )

    return df


def build_geolocation_zip(
    geolocation_clean: pd.DataFrame
) -> pd.DataFrame:
    """
    Produit une table agrégée avec une ligne
    par préfixe postal.

    Les coordonnées représentatives sont les
    médianes des coordonnées admissibles.
    """

    zip_col = "geolocation_zip_code_prefix"

    # Nombre total d'observations par préfixe
    total_counts = (
        geolocation_clean
        .groupby(zip_col)
        .size()
        .rename("observation_count")
    )

    # Coordonnées admissibles pour l'agrégation
    valid = geolocation_clean.loc[
        geolocation_clean["flag_brazil_bbox"]
    ]

    aggregated = (
        valid.groupby(zip_col)
        .agg(
            latitude=("geolocation_lat", "median"),
            longitude=("geolocation_lng", "median"),
            valid_observation_count=(
                "geolocation_lat",
                "size"
            )
        )
    )

    # Conserver tous les préfixes, même sans GPS valide
    result = (
        total_counts
        .to_frame()
        .join(aggregated, how="left")
        .reset_index()
    )

    result["valid_observation_count"] = (
        result["valid_observation_count"]
        .fillna(0)
        .astype("int64")
    )

    result["invalid_observation_count"] = (
        result["observation_count"]
        - result["valid_observation_count"]
    )

    result["gps_quality_ratio"] = (
        result["valid_observation_count"]
        / result["observation_count"]
    )

    logger.info(
        "Table géographique agrégée : %s préfixes",
        len(result)
    )

    return result


def transform_order_items(
    order_items: pd.DataFrame
) -> pd.DataFrame:
    """
    Nettoie la table order_items sans modifier
    les données originales.

    Transformations :
    - Conversion de shipping_limit_date.
    - Validation des valeurs financières.
    - Validation de la clé composite.
    - Ajout d'un montant total par article.
    """

    df = order_items.copy()

    initial_rows = len(df)

    # 1. Conversion de la date limite d'expédition
    missing_before = df["shipping_limit_date"].isna().sum()

    df["shipping_limit_date"] = pd.to_datetime(
        df["shipping_limit_date"],
        errors="coerce"
    )

    invalid_dates = (
        df["shipping_limit_date"].isna().sum()
        - missing_before
    )

    if invalid_dates > 0:
        logger.warning(
            "%s dates d'expédition invalides",
            invalid_dates
        )

    # 2. Validation des montants financiers
    for column in ["price", "freight_value"]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        invalid_amounts = (
            df[column].isna()
            | (df[column] < 0)
        )

        if column == "price":
            invalid_amounts |= df[column].eq(0)

        if invalid_amounts.any():
            raise ValueError(
                f"{invalid_amounts.sum()} montants "
                f"invalides dans {column}"
            )

    # 3. Validation de la clé composite
    if df.duplicated(
        subset=["order_id", "order_item_id"]
    ).any():
        raise ValueError(
            "Clé composite dupliquée dans order_items"
        )

    # 4. Montant total par article
    df["item_total_value"] = (
        df["price"] + df["freight_value"]
    ).round(2)

    # 5. Contrôle du nombre de lignes
    assert len(df) == initial_rows

    logger.info(
        "Transformation order_items terminée : %s lignes",
        len(df)
    )

    return df


def transform_order_payments(
    payments: pd.DataFrame
) -> pd.DataFrame:
    """
    Nettoie les paiements sans supprimer de lignes.

    - Validation de la clé composite.
    - Conversion des valeurs numériques.
    - Identification des paiements nuls.
    - Identification des versements nuls.
    """

    df = payments.copy()

    initial_rows = len(df)

    # 1. Validation de la clé composite
    if df.duplicated(
        subset=["order_id", "payment_sequential"]
    ).any():
        raise ValueError(
            "Clé composite dupliquée dans order_payments"
        )

    # 2. Conversion des colonnes numériques
    numeric_columns = [
        "payment_sequential",
        "payment_installments",
        "payment_value"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # 3. Vérification des valeurs invalides
    if df[numeric_columns].isna().any().any():
        raise ValueError(
            "Valeurs numériques invalides dans order_payments"
        )

    if (df["payment_value"] < 0).any():
        raise ValueError(
            "Montants de paiement négatifs détectés"
        )

    # 4. Indicateurs de qualité
    df["flag_zero_payment"] = (
        df["payment_value"] == 0
    )

    df["flag_zero_installments"] = (
        df["payment_installments"] == 0
    )

    df["flag_undefined_payment_type"] = (
        df["payment_type"] == "not_defined"
    )

    assert len(df) == initial_rows

    logger.info(
        "Transformation order_payments : %s lignes",
        len(df)
    )

    logger.info(
        "Paiements nuls : %s",
        df["flag_zero_payment"].sum()
    )

    logger.info(
        "Versements nuls : %s",
        df["flag_zero_installments"].sum()
    )

    return df


def aggregate_order_payments(
    payments_clean: pd.DataFrame
) -> pd.DataFrame:
    """
    Agrège les paiements au niveau de la commande.
    """

    aggregated = (
        payments_clean.groupby("order_id")
        .agg(
            total_payment_value=(
                "payment_value",
                "sum"
            ),
            payment_count=(
                "payment_sequential",
                "size"
            ),
            payment_method_count=(
                "payment_type",
                "nunique"
            ),
            max_installments=(
                "payment_installments",
                "max"
            ),
            has_zero_payment=(
                "flag_zero_payment",
                "max"
            ),
            has_undefined_payment_type=(
                "flag_undefined_payment_type",
                "max"
            )
        )
        .reset_index()
    )

    aggregated["total_payment_value"] = (
        aggregated["total_payment_value"].round(2)
    )

    logger.info(
        "Paiements agrégés : %s commandes",
        len(aggregated)
    )

    return aggregated


def transform_products(
    products: pd.DataFrame,
    translations: pd.DataFrame
) -> pd.DataFrame:
    """
    Nettoie les produits et ajoute les catégories traduites.

    - Conserve tous les produits.
    - Signale les catégories manquantes.
    - Préserve les poids et dimensions manquants.
    - Signale les poids nuls.
    - Ajoute les traductions disponibles.
    """

    df = products.copy()
    initial_rows = len(df)

    # 1. Validation de la clé primaire
    if df["product_id"].isna().any():
        raise ValueError("product_id contient des valeurs manquantes")

    if df["product_id"].duplicated().any():
        raise ValueError("product_id contient des doublons")

    # 2. Indicateur de catégorie manquante
    df["flag_missing_category"] = (
        df["product_category_name"].isna()
    )

    # 3. Catégorie technique pour les valeurs manquantes
    df["product_category_name"] = (
        df["product_category_name"]
        .fillna("unknown")
    )

    # 4. Validation des valeurs numériques
    numeric_columns = [
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        if (df[column] < 0).any():
            raise ValueError(
                f"Valeurs négatives détectées dans {column}"
            )

    # 5. Indicateurs de qualité
    df["flag_zero_weight"] = (
        df["product_weight_g"] == 0
    )

    df["flag_missing_dimensions"] = (
        df[
            [
                "product_weight_g",
                "product_length_cm",
                "product_height_cm",
                "product_width_cm"
            ]
        ].isna().any(axis=1)
    )

    # 6. Validation de la table de traduction
    if translations["product_category_name"].duplicated().any():
        raise ValueError(
            "Catégories dupliquées dans la table de traduction"
        )

    # 7. Jointure avec les traductions
    df = df.merge(
        translations,
        on="product_category_name",
        how="left",
        validate="many_to_one"
    )

    # 8. Indicateur de traduction manquante
    df["flag_missing_translation"] = (
        df["product_category_name_english"].isna()
        & ~df["flag_missing_category"]
    )

    # 9. Valeur de repli pour la catégorie anglaise
    df["product_category_name_english"] = (
        df["product_category_name_english"]
        .fillna(df["product_category_name"])
    )

    # 10. Validation du nombre de lignes
    assert len(df) == initial_rows

    logger.info(
        "Transformation products terminée : %s lignes",
        len(df)
    )

    logger.info(
        "Catégories manquantes : %s",
        df["flag_missing_category"].sum()
    )

    logger.info(
        "Poids nuls : %s",
        df["flag_zero_weight"].sum()
    )

    logger.info(
        "Dimensions manquantes : %s",
        df["flag_missing_dimensions"].sum()
    )

    logger.info(
        "Produits sans traduction : %s",
        df["flag_missing_translation"].sum()
    )

    return df


def transform_order_reviews(
    reviews: pd.DataFrame
) -> pd.DataFrame:
    """
    Nettoie les avis clients sans supprimer de lignes.
    """

    df = reviews.copy()
    initial_rows = len(df)

    # 1. Validation de la clé composite
    if df[["review_id", "order_id"]].isna().any().any():
        raise ValueError("Clé d'avis manquante")

    if df.duplicated(
        subset=["review_id", "order_id"]
    ).any():
        raise ValueError(
            "Clé composite dupliquée dans order_reviews"
        )

    # 2. Validation des notes
    df["review_score"] = pd.to_numeric(
        df["review_score"],
        errors="coerce"
    )

    if (
        df["review_score"].isna().any()
        or not df["review_score"].between(1, 5).all()
    ):
        raise ValueError("Notes d'avis invalides")

    # 3. Conversion des dates
    date_columns = [
        "review_creation_date",
        "review_answer_timestamp"
    ]

    for column in date_columns:
        missing_before = df[column].isna().sum()

        df[column] = pd.to_datetime(
            df[column],
            errors="coerce"
        )

        invalid_dates = (
            df[column].isna().sum()
            - missing_before
        )

        if invalid_dates > 0:
            logger.warning(
                "%s dates invalides dans %s",
                invalid_dates,
                column
            )

    # 4. Indicateurs de commentaires
    df["flag_has_title"] = (
        df["review_comment_title"].notna()
        & df["review_comment_title"].str.strip().ne("")
    )

    df["flag_has_message"] = (
        df["review_comment_message"].notna()
        & df["review_comment_message"].str.strip().ne("")
    )

    # 5. Délai entre création et réponse
    df["review_response_days"] = (
        (
            df["review_answer_timestamp"]
            - df["review_creation_date"]
        ).dt.total_seconds() / 86400
    )

    # 6. Indicateurs temporels
    df["flag_negative_response_time"] = (
        df["review_response_days"] < 0
    )

    df["flag_long_response_time"] = (
        df["review_response_days"] > 30
    )

    assert len(df) == initial_rows

    logger.info(
        "Transformation order_reviews : %s lignes",
        len(df)
    )

    return df


def aggregate_order_reviews(
    reviews_clean: pd.DataFrame
) -> pd.DataFrame:
    """
    Agrège les avis au niveau de la commande.
    """

    aggregated = (
        reviews_clean.groupby("order_id")
        .agg(
            review_count=("review_id", "size"),
            avg_review_score=("review_score", "mean"),
            min_review_score=("review_score", "min"),
            max_review_score=("review_score", "max"),
            has_review_message=("flag_has_message", "max"),
            has_review_title=("flag_has_title", "max"),
            avg_response_days=(
                "review_response_days",
                "mean"
            )
        )
        .reset_index()
    )

    aggregated["avg_review_score"] = (
        aggregated["avg_review_score"].round(2)
    )

    logger.info(
        "Avis agrégés : %s commandes",
        len(aggregated)
    )

    return aggregated


def transform_customers(
    customers: pd.DataFrame,
    geolocation_zip: pd.DataFrame
) -> pd.DataFrame:
    """
    Nettoie les clients et enrichit leurs données
    avec les coordonnées géographiques.
    """

    df = customers.copy()
    initial_rows = len(df)

    # 1. Validation des identifiants
    if df["customer_id"].isna().any():
        raise ValueError("customer_id manquant")

    if df["customer_id"].duplicated().any():
        raise ValueError("customer_id dupliqué")

    if df["customer_unique_id"].isna().any():
        raise ValueError("customer_unique_id manquant")

    # 2. Normalisation des localisations
    df = transform_locations(
        df,
        city_column="customer_city",
        state_column="customer_state"
    )

    # 3. Enrichissement géographique
    geo_columns = [
        "geolocation_zip_code_prefix",
        "latitude",
        "longitude",
        "gps_quality_ratio"
    ]

    df = df.merge(
        geolocation_zip[geo_columns],
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
        validate="many_to_one"
    )

    # 4. Indicateur de couverture GPS
    df["flag_has_gps"] = (
        df["latitude"].notna()
        & df["longitude"].notna()
    )

    # 5. Suppression de la clé technique de jointure
    df = df.drop(
        columns=["geolocation_zip_code_prefix"]
    )

    assert len(df) == initial_rows

    logger.info(
        "Transformation customers : %s lignes",
        len(df)
    )

    logger.info(
        "Clients avec GPS : %s",
        df["flag_has_gps"].sum()
    )

    return df


def transform_sellers(
    sellers: pd.DataFrame,
    geolocation_zip: pd.DataFrame
) -> pd.DataFrame:
    """
    Nettoie les vendeurs et enrichit leurs données
    avec les coordonnées géographiques.
    """

    df = sellers.copy()
    initial_rows = len(df)

    # 1. Validation des identifiants
    if df["seller_id"].isna().any():
        raise ValueError("seller_id manquant")

    if df["seller_id"].duplicated().any():
        raise ValueError("seller_id dupliqué")

    # 2. Normalisation des localisations
    df = transform_locations(
        df,
        city_column="seller_city",
        state_column="seller_state"
    )

    # 3. Enrichissement géographique
    geo_columns = [
        "geolocation_zip_code_prefix",
        "latitude",
        "longitude",
        "gps_quality_ratio"
    ]

    df = df.merge(
        geolocation_zip[geo_columns],
        left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
        validate="many_to_one"
    )

    # 4. Indicateur de couverture GPS
    df["flag_has_gps"] = (
        df["latitude"].notna()
        & df["longitude"].notna()
    )

    # 5. Suppression de la clé technique de jointure
    df = df.drop(
        columns=["geolocation_zip_code_prefix"]
    )

    assert len(df) == initial_rows

    logger.info(
        "Transformation sellers : %s lignes",
        len(df)
    )

    logger.info(
        "Vendeurs avec GPS : %s",
        df["flag_has_gps"].sum()
    )

    return df
