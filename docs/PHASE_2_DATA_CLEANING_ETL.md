
# Phase 2 — Data Cleaning & ETL

## E-Commerce Data Intelligence Platform

### 1. Objectif

Cette phase vise à construire un pipeline ETL reproductible
à partir du dataset public Olist Brazilian E-Commerce.

Le pipeline extrait neuf fichiers CSV, applique des règles
de transformation et de validation, puis génère dix tables
au format Parquet.

### 2. Architecture

Sources CSV (data/raw/olist/)
    |
    v
Extraction (src/etl/extract.py)
    |
    v
Transformation (src/etl/transform.py)
    |
    v
Validation (src/etl/pipeline.py + tests/)
    |
    v
Tables Parquet (data/interim/)

Les fichiers CSV originaux ne sont pas modifiés.

### 3. Technologies

- Python 3.12
- Pandas
- NumPy
- PyArrow
- Parquet
- Logging
- Git / GitHub

### 4. Transformations réalisées

#### 4.1 Orders

- Conversion des cinq colonnes temporelles.
- Conservation des 99 441 commandes.
- Identification de 8 commandes livrées sans date de livraison.
- Identification de 6 commandes annulées avec date de livraison.
- Conservation des anomalies à l'aide d'indicateurs.

#### 4.2 Order Items

- Conservation de 112 650 articles.
- Validation de la clé (order_id, order_item_id).
- Validation des prix et des frais de livraison.
- Conservation de 383 articles avec livraison gratuite.
- Création de item_total_value = price + freight_value.
- Identification de 1 278 commandes multi-vendeurs lors de l'audit.

#### 4.3 Order Payments

- Conservation de 103 886 paiements.
- Validation de la clé (order_id, payment_sequential).
- Identification de 9 paiements nuls.
- Identification de 2 valeurs payment_installments égales à zéro.
- Identification de 3 types de paiement non définis.
- Agrégation au niveau de 99 440 commandes.
- Conservation du montant total de 16 008 872,12 BRL.

L'audit financier a identifié 576 commandes présentant
un écart entre les paiements et la somme des articles
et des frais de livraison, parmi 98 665 commandes comparables.

Ces écarts sont conservés pour analyse ultérieure.

#### 4.4 Order Reviews

- Conservation de 99 224 avis.
- Validation de la clé (review_id, order_id).
- Conversion des dates.
- Identification des commentaires non vides.
- Calcul du délai entre création et réponse.
- Identification de 682 délais supérieurs à 30 jours.
- Agrégation au niveau de 98 673 commandes.
- Conservation des 547 commandes multi-avis.

Les avis ne seront pas utilisés comme variables prédictives
pour une prédiction de retard effectuée au moment de l'achat,
afin d'éviter les fuites de données.

#### 4.5 Products

- Conservation de 32 951 produits.
- Identification de 610 catégories manquantes.
- Utilisation de la catégorie technique "unknown".
- Identification de 4 produits de poids nul.
- Identification de 2 produits avec données physiques manquantes.
- Jointure avec la table des traductions.
- Identification de 13 produits appartenant à des catégories
  sans traduction disponible.

Les valeurs manquantes et les valeurs nulles sont distinguées.

#### 4.6 Geolocation

- Normalisation des villes et des États.
- Identification de 29 coordonnées hors d'une zone
  géographique approximative couvrant le Brésil.
- Agrégation par préfixe postal.
- Calcul des coordonnées médianes à partir des observations
  retenues par le filtre géographique.
- Production de 19 015 préfixes postaux uniques.
- Calcul d'un indicateur de qualité GPS.

La zone géographique utilisée est un filtre approximatif,
et non une validation officielle des frontières du Brésil.

#### 4.7 Customers

- Conservation de 99 441 enregistrements.
- Identification de 96 096 clients uniques.
- Normalisation des villes et des États.
- Enrichissement avec les coordonnées géographiques.
- 99 162 clients avec coordonnées exploitables.
- 279 clients sans coordonnées exploitables.

#### 4.8 Sellers

- Conservation de 3 095 vendeurs.
- Normalisation des villes et des États.
- Enrichissement géographique.
- 3 088 vendeurs avec coordonnées exploitables.
- 7 vendeurs sans coordonnées exploitables.

### 5. Tables produites

| Table | Lignes |
|---|---:|
| orders_clean | 99 441 |
| order_items_clean | 112 650 |
| products_clean | 32 951 |
| geolocation_zip | 19 015 |
| customers_clean | 99 441 |
| sellers_clean | 3 095 |
| order_payments_clean | 103 886 |
| order_payments_aggregated | 99 440 |
| order_reviews_clean | 99 224 |
| order_reviews_aggregated | 98 673 |

Toutes les tables sont enregistrées dans data/interim/.

### 6. Tests et contrôles

Les tests couvrent :

- Conservation des nombres de lignes.
- Validation des clés uniques et composites.
- Absence de références orphelines sur six relations.
- Conservation des montants financiers après agrégation.
- Conservation du nombre total d'avis.
- Vérification de la couverture GPS.
- Lecture des dix fichiers Parquet générés.

Le test d'intégration a été exécuté avec succès.

Commande :

python -m tests.test_etl_integration

### 7. Exécution du pipeline

Activer l'environnement virtuel, installer les dépendances
du projet, puis exécuter :

python -m src.etl.pipeline

Le pipeline lit les CSV du dossier data/raw/olist/
et écrit les tables Parquet dans data/interim/.

### 8. Limites et améliorations futures

- Les contrôles sont validés sur le dataset Olist actuel.
- Certaines anomalies sont signalées sans correction arbitraire.
- Les écarts financiers nécessitent une analyse métier.
- La normalisation des villes ne corrige pas toutes les variantes.
- Les coordonnées GPS restent des estimations par préfixe postal.
- Le pipeline réécrit les fichiers Parquet existants.
- L'écriture atomique et la gestion des exécutions partielles
  restent à améliorer.
- Les tests de reproductibilité et les contrôles génériques
  sur de nouveaux jeux de données restent à développer.

### 9. Résultat

La Phase 2 fournit un pipeline ETL fonctionnel et testé,
avec dix tables Parquet prêtes à servir de sources à la
modélisation du Data Warehouse PostgreSQL.

### 10. Prochaine phase

Phase 3 — Data Warehouse PostgreSQL :

- Définition des granularités.
- Conception du modèle dimensionnel.
- Création des dimensions et tables de faits.
- Chargement des données.
- Contrôles SQL de qualité et de cohérence.
