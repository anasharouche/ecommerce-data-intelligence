# Phase 2 — ETL & Data Processing

> **Statut : Phase 2 terminée et testée.**
> Le pipeline extrait les neuf CSV Olist, applique des transformations traçables et produit dix tables Parquet destinées au chargement PostgreSQL.

## Table des matières

1. [Présentation et objectifs](#section-1)
2. [Architecture ETL](#section-2)
3. [Entrées, sorties et technologies](#section-3)
4. [Extraction et orchestration](#section-4)
5. [Transformations détaillées](#section-5)
6. [Résultats et tables produites](#section-6)
7. [Tests et validations](#section-7)
8. [Anomalies et décisions techniques](#section-8)
9. [Exécution et reproductibilité](#section-9)
10. [Limites](#section-10)
11. [Livrables](#section-11)
12. [Conclusion et transition](#section-12)

<a id="section-1"></a>

## 1. Présentation et objectifs

La Phase 2 transforme les constats de la [Phase 1](PHASE_1_DATA_EXPLORATION.md) en règles de traitement reproductibles. Elle conserve les données utiles, ajoute des indicateurs de qualité, normalise les types et prépare des tables intermédiaires adaptées au chargement dans PostgreSQL.

Ses objectifs sont :

- centraliser l'extraction des neuf fichiers CSV ;
- convertir les types, notamment les dates et montants ;
- valider les clés et relations essentielles ;
- signaler les anomalies sans correction arbitraire ;
- agréger paiements, avis et géolocalisation aux grains nécessaires ;
- produire dix fichiers Parquet stables et lisibles ;
- fournir des tests unitaires ciblés et un test d'intégration.

La phase ne charge pas encore le Data Warehouse et ne met pas en place d'orchestrateur externe. Le pipeline est un module Python exécutable dans un ordre déterministe.

<a id="section-2"></a>

## 2. Architecture ETL

```mermaid
flowchart LR
    A["CSV bruts · data/raw/olist"] --> B["Extract · extract.py"]
    B --> C["Transform · transform.py"]
    C --> D["Validation · pipeline.py et tests"]
    D --> E["10 tables Parquet · data/interim"]
    E --> F["Chargement PostgreSQL · Phase 3"]
```

| Couche | Responsabilité | Fichier principal |
|---|---|---|
| Configuration | Résoudre les chemins et associer les noms logiques aux CSV | [`config/paths.py`](../config/paths.py) |
| Extraction | Vérifier l'existence, lire les CSV et détecter les fichiers vides | [`src/etl/extract.py`](../src/etl/extract.py) |
| Transformation | Normaliser, enrichir, agréger et ajouter les indicateurs de qualité | [`src/etl/transform.py`](../src/etl/transform.py) |
| Pipeline | Enchaîner les traitements, contrôler les volumes et écrire les Parquet | [`src/etl/pipeline.py`](../src/etl/pipeline.py) |
| Validation | Contrôler les domaines et la cohérence globale des sorties | [`tests/`](../tests/) |

<a id="section-3"></a>

## 3. Entrées, sorties et technologies

### 3.1 Sources

Les neuf fichiers décrits en Phase 1 sont lus depuis `data/raw/olist/`. `extract_csv()` impose un fichier existant, utilise l'encodage UTF-8, désactive l'inférence par blocs avec `low_memory=False` et convertit un CSV vide en erreur explicite.

### 3.2 Sorties

Les tables sont écrites dans `data/interim/` au format **Apache Parquet** avec PyArrow, sans index Pandas. Parquet préserve mieux les types qu'un nouvel export CSV, réduit les coûts de lecture et fournit une interface stable entre l'ETL et le chargement PostgreSQL.

### 3.3 Technologies

| Usage | Technologie |
|---|---|
| Langage | Python 3.12 |
| Traitement tabulaire | Pandas, NumPy |
| Stockage intermédiaire | Parquet, PyArrow |
| Journalisation | Module standard `logging` |
| Validation | Assertions et scripts Python exécutables |
| Gestion des dépendances | [`requirements.txt`](../requirements.txt) |

<a id="section-4"></a>

## 4. Extraction et orchestration

`extract_all()` parcourt la correspondance `OLIST_FILES` et retourne un dictionnaire de neuf DataFrames. `run_pipeline()` applique ensuite les transformations dans l'ordre requis : la géolocalisation agrégée est construite avant l'enrichissement des clients et vendeurs ; les paiements et avis détaillés sont transformés avant leur agrégation.

Avant l'écriture, `validate_table()` contrôle :

- le nombre de lignes attendu pour les tables détaillées ;
- l'absence de noms de colonnes dupliqués ;
- la disponibilité de toutes les tables produites.

Le dossier `data/interim/` est créé s'il n'existe pas, puis les fichiers Parquet sont écrits ou remplacés. Le pipeline ne fournit pas encore de reprise sur erreur, d'écriture atomique ou de planification externe.

<a id="section-5"></a>

## 5. Transformations détaillées

### 5.1 Commandes

`transform_orders()` conserve les **99 441 commandes** et convertit cinq colonnes temporelles avec `errors="coerce"`. Les conversions n'ajoutent aucune date invalide. Trois indicateurs sont créés :

- `flag_delivery_before_purchase` : 0 cas ;
- `flag_delivered_without_date` : 8 cas ;
- `flag_canceled_with_delivery` : 6 cas.

Les valeurs manquantes légitimes restent nulles : 160 approbations, 1 783 remises au transporteur et 2 965 livraisons client non renseignées.

### 5.2 Articles de commande

`transform_order_items()` :

- conserve **112 650 articles** ;
- convertit `shipping_limit_date` sans date invalide ;
- exige des prix strictement positifs et des frais de livraison positifs ou nuls ;
- valide la clé (`order_id`, `order_item_id`) ;
- calcule `item_total_value = price + freight_value`.

Les montants obtenus sont :

| Mesure | Montant (BRL) |
|---|---:|
| Prix des articles | 13 591 643,70 |
| Frais de livraison | 2 251 909,54 |
| Total articles + livraison | 15 843 553,24 |

L'audit recense **383 articles avec livraison gratuite** et **1 278 commandes multivendeurs**. Ces cas sont valides et conservés.

### 5.3 Paiements

`transform_order_payments()` valide la clé (`order_id`, `payment_sequential`), convertit les champs numériques et refuse les montants négatifs. Il conserve **103 886 paiements** et ajoute trois indicateurs :

| Indicateur | Résultat |
|---|---:|
| Paiements nuls | 9 |
| Nombre de versements égal à zéro | 2 |
| Type `not_defined` | 3 |

`aggregate_order_payments()` produit **99 440 commandes** avec total payé, nombre de transactions, nombre de méthodes, maximum de versements et indicateurs agrégés. Les **2 961 commandes multi-paiements** sont correctement regroupées. Le total de **16 008 872,12 BRL** est identique avant et après agrégation.

L'audit compare paiements et articles sur 98 665 commandes communes : **576** présentent un écart non nul après arrondi à deux décimales. Ces écarts ne sont pas corrigés arbitrairement, car paiement et valeur des articles représentent des mesures distinctes.

### 5.4 Avis clients

`transform_order_reviews()` retient la clé composite (`review_id`, `order_id`), valide les notes de 1 à 5 et convertit les deux dates sans erreur. Il crée des indicateurs de titre, message, réponse négative et réponse longue.

| Résultat | Valeur |
|---|---:|
| Avis détaillés conservés | 99 224 |
| Titres non vides | 11 566 |
| Messages non vides | 40 950 |
| Délais de réponse négatifs | 0 |
| Délais supérieurs à 30 jours | 682 |
| Note moyenne | 4,0864 |
| Commandes avec plusieurs avis | 547 |

Les nombres de titres et messages utilisent une condition plus stricte que `notna()` : les chaînes vides après suppression des espaces ne sont pas considérées comme du contenu.

`aggregate_order_reviews()` produit **98 673 commandes** et conserve, pour chacune, le nombre d'avis, les notes moyenne/minimale/maximale et les indicateurs de commentaire.

### 5.5 Produits et catégories

`transform_products()` conserve **32 951 produits**, valide l'unicité de `product_id` et distingue valeurs manquantes et valeurs nulles métier.

- 610 catégories manquantes reçoivent la valeur technique `unknown`, avec un indicateur dédié.
- 4 produits possèdent un poids égal à zéro.
- 2 produits ont au moins une dimension physique manquante.
- 2 catégories sources ne possèdent pas de traduction, soit 13 produits ; leur nom portugais sert de repli.
- Aucune valeur numérique négative n'est acceptée.

La jointure de traduction est validée `many_to_one` afin d'empêcher une multiplication des produits.

### 5.6 Géolocalisation

Les villes sont normalisées par suppression des espaces superflus, passage en minuscules et retrait des accents. Les codes d'État sont nettoyés et mis en majuscules.

`transform_geolocation()` valide les limites GPS et applique un rectangle volontairement approximatif couvrant le Brésil. Sur **1 000 163 observations**, 29 coordonnées se trouvent hors de cette zone ; le filtre ne prétend pas représenter une frontière officielle.

`build_geolocation_zip()` produit **19 015 préfixes postaux uniques**. Les coordonnées représentatives sont les médianes des observations admissibles. Tous les préfixes sont conservés, y compris 4 préfixes sans coordonnées admissibles. `gps_quality_ratio` mesure la part d'observations utilisables.

### 5.7 Clients et vendeurs

Les transformations valident les identifiants, normalisent villes et États, puis enrichissent les enregistrements avec la table géographique par préfixe postal.

| Population | Lignes | Entités uniques | Coordonnées disponibles | Sans coordonnées |
|---|---:|---:|---:|---:|
| Clients | 99 441 | 96 096 `customer_unique_id` | 99 162 | 279 |
| Vendeurs | 3 095 | 3 095 `seller_id` | 3 088 | 7 |

Les jointures sont validées `many_to_one` et ne changent pas les nombres de lignes.

<a id="section-6"></a>

## 6. Résultats et tables produites

| Table Parquet | Lignes | Grain |
|---|---:|---|
| `orders_clean` | 99 441 | Une commande |
| `order_items_clean` | 112 650 | Un article de commande |
| `products_clean` | 32 951 | Un produit |
| `geolocation_zip` | 19 015 | Un préfixe postal |
| `customers_clean` | 99 441 | Un `customer_id` |
| `sellers_clean` | 3 095 | Un vendeur |
| `order_payments_clean` | 103 886 | Une transaction de paiement |
| `order_payments_aggregated` | 99 440 | Une commande payée |
| `order_reviews_clean` | 99 224 | Un couple avis–commande |
| `order_reviews_aggregated` | 98 673 | Une commande évaluée |
| **Total** | **767 816** |  |

Les tables détaillées conservent les lignes sources. Les tables agrégées ajoutent des grains adaptés au chargement de commandes et aux analyses ultérieures.

<a id="section-7"></a>

## 7. Tests et validations

### 7.1 Couverture

| Domaine | Script |
|---|---|
| Commandes | [`test_transform_orders.py`](../tests/test_transform_orders.py) |
| Articles | [`test_transform_order_items.py`](../tests/test_transform_order_items.py) |
| Paiements | [`test_transform_order_payments.py`](../tests/test_transform_order_payments.py) |
| Avis | [`test_transform_order_reviews.py`](../tests/test_transform_order_reviews.py) |
| Produits | [`test_transform_products.py`](../tests/test_transform_products.py) |
| Localisations | [`test_transform_locations.py`](../tests/test_transform_locations.py) |
| Clients et vendeurs | [`test_transform_customers_sellers.py`](../tests/test_transform_customers_sellers.py) |
| Géolocalisation agrégée | [`test_geolocation_zip.py`](../tests/test_geolocation_zip.py) |
| Intégration globale | [`test_etl_integration.py`](../tests/test_etl_integration.py) |

Ces fichiers sont des modules exécutables avec une fonction `main()` et des assertions ; ils ne sont pas organisés comme une suite de fonctions `pytest`.

### 7.2 Test d'intégration

Le test d'intégration vérifie :

- la présence et les volumes des dix Parquet ;
- l'unicité et la non-nullité des clés ;
- six relations sans références orphelines ;
- la conservation du total des paiements ;
- la conservation des 99 224 avis après agrégation ;
- la couverture GPS de 99 162 clients et 3 088 vendeurs.

Il a été exécuté avec succès sur les sorties présentes. Les tests de transformation ciblés confirment également les montants, anomalies et volumes détaillés dans ce rapport.

<a id="section-8"></a>

## 8. Anomalies et décisions techniques

| Constat | Décision implémentée |
|---|---|
| Dates ou statuts de livraison incohérents | Conserver la ligne et ajouter un indicateur de qualité |
| Commentaires d'avis facultatifs | Créer des indicateurs de contenu non vide, sans invalider l'avis |
| `review_id` non unique | Valider la clé composite (`review_id`, `order_id`) |
| Paiements et avis multiples | Conserver le détail et produire une table agrégée au grain commande |
| Catégorie produit manquante | Utiliser `unknown` et conserver `flag_missing_category` |
| Traduction absente | Reprendre le nom portugais et ajouter un indicateur |
| Coordonnées atypiques | Exclure du calcul médian, conserver leur compte dans les indicateurs |
| Écarts paiement–articles | Conserver les deux mesures et documenter l'écart |
| Avis pour le futur ML | Ne pas les utiliser si la prédiction intervient au moment de l'achat |

<a id="section-9"></a>

## 9. Exécution et reproductibilité

Après activation de l'environnement et installation des dépendances :

```powershell
python -m src.etl.pipeline
```

Le test d'intégration se lance avec :

```powershell
python -m tests.test_etl_integration
```

Le pipeline lit les sources brutes et réécrit les dix fichiers Parquet correspondants. Il journalise les étapes, mais ne dispose pas encore d'un orchestrateur, d'un catalogue de runs ni d'une stratégie de reprise partielle.

<a id="section-10"></a>

## 10. Limites

- Les règles et volumes sont validés sur le dataset Olist actuel.
- Certaines anomalies sont signalées plutôt que corrigées faute de règle métier démontrée.
- Le rectangle géographique est un filtre de qualité approximatif.
- La normalisation des villes ne résout pas toutes les variantes sémantiques.
- Les coordonnées par préfixe postal sont des estimations médianes.
- Les fichiers Parquet sont réécrits, sans écriture atomique.
- Les scripts de test reposent sur des données locales exclues de Git.
- L'orchestration, les chargements incrémentaux et la gestion formelle des exécutions restent à développer.

<a id="section-11"></a>

## 11. Livrables

| Livrable | Emplacement |
|---|---|
| Configuration des données | [`config/paths.py`](../config/paths.py) |
| Extraction | [`src/etl/extract.py`](../src/etl/extract.py) |
| Transformations | [`src/etl/transform.py`](../src/etl/transform.py) |
| Pipeline | [`src/etl/pipeline.py`](../src/etl/pipeline.py) |
| Audits spécialisés | [`src/`](../src/) |
| Tests | [`tests/`](../tests/) |
| Dépendances | [`requirements.txt`](../requirements.txt) |
| Tables intermédiaires | `data/interim/*.parquet` — artefacts locaux non versionnés |

<a id="section-12"></a>

## 12. Conclusion et transition

La Phase 2 fournit un pipeline ETL reproductible qui transforme neuf CSV en dix tables Parquet contrôlées, sans effacer les anomalies utiles. Les clés, volumes, montants, avis et enrichissements géographiques sont validés avant toute insertion en base.

Ces sorties constituent l'entrée de la **Phase 3 — PostgreSQL Data Warehouse**, qui matérialise les contraintes relationnelles, les dimensions, les faits et les contrôles SQL.

**Statut final : Phase 2 terminée et testée.**
