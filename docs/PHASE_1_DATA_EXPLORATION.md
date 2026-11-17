# Phase 1 — Data Exploration & Data Quality

> **Statut : Phase 1 terminée.**
> Cette phase explore les données brutes Olist et établit les règles de qualité nécessaires aux traitements ultérieurs. Elle identifie les anomalies sans modifier les fichiers sources.

## Table des matières

1. [Présentation et objectifs](#section-1)
2. [Contexte technique et source des données](#section-2)
3. [Démarche d'exploration](#section-3)
4. [Inventaire du dataset](#section-4)
5. [Structure, clés et relations](#section-5)
6. [Valeurs manquantes](#section-6)
7. [Doublons et multiplicité des avis](#section-7)
8. [Commandes et cohérence temporelle](#section-8)
9. [Synthèse des anomalies et décisions](#section-9)
10. [Validations et limites](#section-10)
11. [Livrables](#section-11)
12. [Conclusion et transition](#section-12)

<a id="section-1"></a>

## 1. Présentation et objectifs

L'**E-Commerce Data Intelligence Platform** s'appuie sur le dataset public **Brazilian E-Commerce Public Dataset by Olist**. L'objectif global est de construire une plateforme de Data Engineering et d'analyse capable de suivre les performances e-commerce, puis de préparer un cas d'usage de Machine Learning consacré à la prédiction des retards de livraison.

La Phase 1 précède toute transformation. Elle vise à :

- comprendre la structure des neuf fichiers CSV ;
- mesurer leurs dimensions et examiner leurs types ;
- identifier les clés candidates et les relations ;
- analyser les valeurs manquantes et les doublons ;
- repérer les anomalies métier et temporelles ;
- définir les précautions nécessaires au futur ETL et au modèle prédictif.

La distinction est essentielle : **explorer n'est pas nettoyer**. Les constats de cette phase décrivent les données brutes ; les corrections, enrichissements et indicateurs de qualité sont implémentés à partir de la [Phase 2](PHASE_2_DATA_CLEANING_ETL.md).

<a id="section-2"></a>

## 2. Contexte technique et source des données

| Élément | Référence |
|---|---|
| Dataset | [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) |
| Environnement documenté | Windows 11, Python 3.12.10, environnement virtuel `venv` |
| Analyse tabulaire | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Support d'exploration | Jupyter Notebook |
| Données sources | CSV dans `data/raw/olist/` |

Le notebook principal est [`01_olist_data_exploration.ipynb`](../notebooks/01_olist_data_exploration.ipynb). Le script [`inspect_dataset.py`](../src/inspect_dataset.py) fournit en complément une inspection reproductible des fichiers attendus, de leurs colonnes, types, valeurs manquantes et doublons complets.

Les CSV originaux sont considérés comme immuables. Ils sont exclus du versionnement Git et ne sont pas réécrits pendant l'exploration.

<a id="section-3"></a>

## 3. Démarche d'exploration

```mermaid
flowchart LR
    A["9 fichiers CSV Olist"] --> B["Chargement Pandas"]
    B --> C["Dimensions et types"]
    C --> D["Valeurs manquantes"]
    D --> E["Clés et doublons"]
    E --> F["Intégrité référentielle"]
    F --> G["Anomalies métier et temporelles"]
    G --> H["Règles pour l'ETL et le ML"]
```

Le notebook charge explicitement les neuf fichiers, puis exécute des contrôles ciblés. Les résultats enregistrés dans ses sorties constituent la trace principale de cette phase. Les audits Python de la Phase 2 approfondissent ensuite certains domaines, sans changer les constats bruts présentés ici.

<a id="section-4"></a>

## 4. Inventaire du dataset

| Jeu de données | Fichier source | Lignes | Colonnes | Rôle |
|---|---|---:|---:|---|
| Commandes | `olist_orders_dataset.csv` | 99 441 | 8 | Statut et cycle de vie des commandes |
| Clients | `olist_customers_dataset.csv` | 99 441 | 5 | Identifiants et localisation des clients |
| Articles de commande | `olist_order_items_dataset.csv` | 112 650 | 7 | Produits, vendeurs, prix et transport |
| Produits | `olist_products_dataset.csv` | 32 951 | 9 | Caractéristiques et catégories produit |
| Vendeurs | `olist_sellers_dataset.csv` | 3 095 | 4 | Identifiants et localisation des vendeurs |
| Paiements | `olist_order_payments_dataset.csv` | 103 886 | 5 | Transactions et modes de paiement |
| Avis | `olist_order_reviews_dataset.csv` | 99 224 | 7 | Notes, commentaires et dates de réponse |
| Géolocalisation | `olist_geolocation_dataset.csv` | 1 000 163 | 5 | Coordonnées par préfixe postal |
| Traduction des catégories | `product_category_name_translation.csv` | 71 | 2 | Traductions portugais–anglais |
| **Total** |  | **1 560 862** |  |  |

Les volumes ne décrivent pas tous la même granularité. Il existe davantage de lignes d'articles et de paiements que de commandes, ce qui annonce des relations un-à-plusieurs qu'il faudra préserver lors des agrégations.

<a id="section-5"></a>

## 5. Structure, clés et relations

### 5.1 Clés candidates

| Table | Clé contrôlée | Valeurs nulles | Doublons de la clé |
|---|---|---:|---:|
| `orders` | `order_id` | 0 | 0 |
| `customers` | `customer_id` | 0 | 0 |
| `order_items` | (`order_id`, `order_item_id`) | 0 | 0 |
| `products` | `product_id` | 0 | 0 |
| `sellers` | `seller_id` | 0 | 0 |
| `payments` | (`order_id`, `payment_sequential`) | 0 | 0 |
| `reviews` | `review_id` seul | 0 | Non unique |
| `reviews` | (`review_id`, `order_id`) | 0 | 0 |

Le premier contrôle du notebook teste `review_id` seul et obtient **814 occurrences supplémentaires après la première occurrence**. L'analyse approfondie établit ensuite que la combinaison (`review_id`, `order_id`) est unique. Cette clé composite devient donc la clé métier retenue pour les avis dans les phases suivantes.

### 5.2 Intégrité référentielle

| Relation vérifiée | Valeurs enfant sans parent |
|---|---:|
| `orders.customer_id` → `customers.customer_id` | 0 |
| `order_items.order_id` → `orders.order_id` | 0 |
| `order_items.product_id` → `products.product_id` | 0 |
| `order_items.seller_id` → `sellers.seller_id` | 0 |
| `payments.order_id` → `orders.order_id` | 0 |
| `reviews.order_id` → `orders.order_id` | 0 |

Ces contrôles comparent les ensembles d'identifiants. Ils confirment l'absence de références orphelines sur les six relations testées, mais ne remplacent pas les contraintes SQL du futur Data Warehouse.

<a id="section-6"></a>

## 6. Valeurs manquantes

| Table | Colonne | Valeurs manquantes | Proportion |
|---|---|---:|---:|
| `reviews` | `review_comment_title` | 87 656 | 88,34 % |
| `reviews` | `review_comment_message` | 58 247 | 58,70 % |
| `orders` | `order_delivered_customer_date` | 2 965 | 2,98 % |
| `orders` | `order_delivered_carrier_date` | 1 783 | 1,79 % |
| `orders` | `order_approved_at` | 160 | 0,16 % |
| `products` | Catégorie, longueur du nom, description et photos | 610 par colonne | 1,85 % |
| `products` | Poids, longueur, hauteur et largeur | 2 par colonne | 0,006 % |

Les commentaires et titres d'avis sont facultatifs : leur absence n'invalide pas un avis. Les dates de livraison manquantes, en revanche, affectent la possibilité de calculer un retard et doivent être interprétées avec le statut de commande. Les valeurs produit manquantes nécessitent des indicateurs de qualité plutôt qu'une imputation arbitraire.

<a id="section-7"></a>

## 7. Doublons et multiplicité des avis

### 7.1 Quatre notions à ne pas confondre

| Notion | Résultat | Interprétation |
|---|---:|---|
| Doublons complets dans `geolocation` | 261 831 | Lignes entièrement identiques parmi 1 000 163 observations |
| Occurrences supplémentaires de `review_id` | 814 | Résultat de `duplicated(review_id).sum()` ; chaque occurrence après la première est comptée |
| Lignes appartenant à un `review_id` répété | 1 603 | Toutes les lignes concernées, obtenues avec `keep=False` |
| Valeurs distinctes de `review_id` répétées | 789 | Nombre d'identifiants non uniques |
| Doublons de (`review_id`, `order_id`) | 0 | La clé composite reste unique |
| Commandes possédant plusieurs avis | 547 | Groupes `order_id` comportant plus d'une ligne |
| Occurrences supplémentaires de `order_id` | 551 | Quatre commandes ont trois avis, les autres groupes multiples en ont deux |

Un identifiant `review_id` répété n'implique pas un doublon de ligne : dans les cas observés, il peut être associé à plusieurs commandes. Réciproquement, une commande possédant plusieurs avis représente une multiplicité métier, pas nécessairement une duplication erronée.

La relation `orders` → `reviews` doit donc être traitée comme une relation **1:N**. Une jointure directe avec d'autres tables de faits peut multiplier les lignes et fausser des KPI ; les avis devront être agrégés séparément au niveau commande lorsque ce grain est requis.

<a id="section-8"></a>

## 8. Commandes et cohérence temporelle

### 8.1 Distribution des statuts

| Statut | Commandes |
|---|---:|
| `delivered` | 96 478 |
| `shipped` | 1 107 |
| `canceled` | 625 |
| `unavailable` | 609 |
| `invoiced` | 314 |
| `processing` | 301 |
| `created` | 5 |
| `approved` | 2 |
| **Total** | **99 441** |

### 8.2 Anomalies de livraison

- **8 commandes** ont le statut `delivered` sans date réelle de livraison.
- **6 commandes** ont le statut `canceled` tout en possédant une date réelle de livraison.
- Les autres valeurs nulles de livraison sont majoritairement cohérentes avec des statuts non livrés.

La Phase 1 conserve ces lignes et les signale. Elle ne remplit pas les dates manquantes et ne modifie pas les statuts.

### 8.3 Préparation du cas d'usage Machine Learning

La cible envisagée pour la prédiction des retards est :

- `1` lorsque la date réelle de livraison est postérieure à la date estimée ;
- `0` lorsqu'elle est antérieure ou égale.

La population initiale doit se limiter aux commandes livrées avec une date réelle renseignée. Les informations indisponibles au moment de la prédiction, telles que les avis ou la date réelle de livraison, ne peuvent pas servir de variables explicatives : leur utilisation créerait une fuite de données.

<a id="section-9"></a>

## 9. Synthèse des anomalies et décisions

| Constat | Volume | Décision de Phase 1 | Traitement prévu |
|---|---:|---|---|
| `review_id` non unique | 789 identifiants, 1 603 lignes | Ne pas supprimer | Retenir (`review_id`, `order_id`) comme clé composite |
| Commandes multi-avis | 547 commandes, maximum 3 avis | Conserver la multiplicité | Agréger par commande pour les usages au grain commande |
| Doublons complets de géolocalisation | 261 831 | Ne pas supprimer sans analyse | Construire une représentation par préfixe postal en Phase 2 |
| Commandes livrées sans date réelle | 8 | Conserver et signaler | Ajouter un indicateur de qualité ; exclure des calculs de retard impossibles |
| Commandes annulées avec date réelle | 6 | Conserver et signaler | Ajouter un indicateur métier dédié |
| Catégories produit manquantes | 610 | Ne pas imputer arbitrairement | Utiliser une catégorie technique et conserver un indicateur |
| Commentaires d'avis manquants | 58 247 messages | Considérer le champ comme facultatif | Distinguer absence de commentaire et avis invalide |

<a id="section-10"></a>

## 10. Validations et limites

### 10.1 Validations réalisées

- Chargement et dimension des neuf fichiers.
- Contrôle des clés simples et composites.
- Recherche de valeurs nulles dans les clés.
- Vérification de six relations par comparaison d'identifiants.
- Analyse des valeurs manquantes par colonne.
- Analyse des doublons complets et de clés.
- Distribution des statuts de commande.
- Analyse détaillée des avis multiples.
- Contrôle des incohérences entre statut et date de livraison.

Les sorties sont enregistrées dans le notebook. Il s'agit de contrôles exploratoires reproductibles, pas encore d'une suite de tests automatisée ni de contraintes de base de données.

### 10.2 Limites

- Les résultats correspondent à la version actuelle du dataset Olist.
- Les contrôles d'intégrité couvrent les principales relations, pas toutes les règles métier possibles.
- Les frontières géographiques et valeurs extrêmes ne sont pas encore auditées en détail.
- Les anomalies sont identifiées, mais aucune correction destructive n'est réalisée.
- Le moment exact de la future prédiction et les variables autorisées restent à formaliser avant le Machine Learning.

<a id="section-11"></a>

## 11. Livrables

| Livrable | Emplacement |
|---|---|
| Notebook d'exploration | [`notebooks/01_olist_data_exploration.ipynb`](../notebooks/01_olist_data_exploration.ipynb) |
| Inspection générale des CSV | [`src/inspect_dataset.py`](../src/inspect_dataset.py) |
| Configuration des chemins et fichiers attendus | [`config/paths.py`](../config/paths.py) |
| Rapport de Phase 1 | [`docs/PHASE_1_DATA_EXPLORATION.md`](PHASE_1_DATA_EXPLORATION.md) |

<a id="section-12"></a>

## 12. Conclusion et transition

La Phase 1 établit une connaissance fiable du dataset : volumes, granularités, clés, relations, valeurs manquantes et anomalies. Elle montre surtout qu'une suppression naïve de doublons ou une jointure directe entre faits pourrait dégrader l'information.

Les décisions prises — conservation des sources, clé composite des avis, signalement des anomalies et prévention des fuites de données — servent de contrat d'entrée à la **Phase 2 — ETL & Data Processing**, qui implémente les transformations sans effacer la traçabilité des données.

**Statut final : Phase 1 terminée.**
