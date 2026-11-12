# Phase 1 — Exploration et compréhension des données

## 1. Présentation du projet

**Nom du projet :** E-Commerce Data Intelligence Platform

L'objectif de ce projet est de concevoir une plateforme Data Engineering et Machine Learning permettant d'analyser les performances d'une activité e-commerce et de prédire les retards de livraison.

Le projet s'appuie sur le dataset public **Brazilian E-Commerce Public Dataset by Olist**, disponible sur Kaggle.

Dataset : https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Cette première phase consiste à explorer les données, comprendre leur structure, identifier les relations entre les tables et détecter les éventuelles anomalies avant de construire le pipeline ETL.

## 2. Environnement technique

| Composant | Technologie |
|---|---|
| Système d'exploitation | Windows 11 |
| Éditeur de code | Visual Studio Code |
| Langage | Python 3.12.10 |
| Environnement virtuel | venv |
| Manipulation des données | Pandas |
| Calcul numérique | NumPy |
| Visualisation | Matplotlib, Seaborn |
| Format de stockage | CSV, Parquet (prévu) |
| Notebook | Jupyter |

Les bibliothèques principales ont été installées et testées avec succès.

## 3. Présentation du dataset

Le dataset Olist contient neuf fichiers CSV représentant différentes entités du système e-commerce.

| Fichier | Nombre de lignes | Nombre de colonnes | Description |
|---|---:|---:|---|
| orders | 99 441 | 8 | Informations sur les commandes |
| customers | 99 441 | 5 | Informations sur les clients |
| order_items | 112 650 | 7 | Articles associés aux commandes |
| products | 32 951 | 9 | Informations sur les produits |
| sellers | 3 095 | 4 | Informations sur les vendeurs |
| order_payments | 103 886 | 5 | Informations sur les paiements |
| order_reviews | 99 224 | 7 | Avis et évaluations des clients |
| geolocation | 1 000 163 | 5 | Informations géographiques |
| category_translation | 71 | 2 | Traduction des catégories de produits |

**Volume total : 1 560 862 lignes.**

## 4. Analyse de la structure et des clés

Une première inspection a permis d'examiner les dimensions, les types de données, les valeurs manquantes et les doublons de chaque fichier.

### 4.1 Clés candidates identifiées

Les contrôles réalisés ont donné les résultats suivants :

| Table | Clé candidate | Doublons détectés |
|---|---|---:|
| orders | order_id | 0 |
| customers | customer_id | 0 |
| order_items | order_id + order_item_id | 0 |
| products | product_id | 0 |
| sellers | seller_id | 0 |
| order_payments | order_id + payment_sequential | 0 |
| order_reviews | review_id + order_id | 0 |

La colonne `review_id` seule n'est pas unique : 789 identifiants sont répétés.

La combinaison `review_id + order_id` ne présente aucun doublon et constitue une clé candidate pour cette table.

### 4.2 Relations entre les tables

Des contrôles d'intégrité référentielle ont été réalisés à l'aide des identifiants distincts et de comparaisons entre tables.

| Relation vérifiée | Clés non correspondantes |
|---|---:|
| orders.customer_id → customers.customer_id | 0 |
| order_items.order_id → orders.order_id | 0 |
| order_items.product_id → products.product_id | 0 |
| order_items.seller_id → sellers.seller_id | 0 |
| order_payments.order_id → orders.order_id | 0 |
| order_reviews.order_id → orders.order_id | 0 |

Aucune clé étrangère orpheline n'a été détectée dans les relations testées.

Ces contrôles vérifient l'existence des références, mais ne constituent pas encore une validation complète de toutes les contraintes du futur Data Warehouse.

## 5. Analyse des valeurs manquantes

Les principales valeurs manquantes identifiées sont :

| Table | Colonne | Valeurs manquantes | Pourcentage |
|---|---|---:|---:|
| order_reviews | review_comment_title | 87 656 | 88,34 % |
| order_reviews | review_comment_message | 58 247 | 58,70 % |
| orders | order_delivered_customer_date | 2 965 | 2,98 % |
| orders | order_delivered_carrier_date | 1 783 | 1,79 % |
| orders | order_approved_at | 160 | 0,16 % |
| products | product_category_name | 610 | 1,85 % |
| products | product_name_lenght | 610 | 1,85 % |
| products | product_description_lenght | 610 | 1,85 % |
| products | product_photos_qty | 610 | 1,85 % |
| products | dimensions et poids | 2 par colonne | 0,006 % |

Les commentaires des clients sont facultatifs. Leur absence ne signifie donc pas nécessairement que les enregistrements sont invalides.

Les autres valeurs manquantes devront être traitées selon leur contexte métier pendant la phase de transformation.

## 6. Analyse des doublons

### 6.1 Doublons complets

Les doublons complets détectés sont principalement concentrés dans la table `geolocation`.

- Nombre de doublons complets : 261 831.
- Nombre total de lignes : 1 000 163.

Ces doublons devront être étudiés avant toute suppression afin de préserver les informations géographiques utiles.

### 6.2 Doublons dans les avis clients

L'analyse de `order_reviews` a révélé :

- 1 603 lignes concernées par des identifiants `review_id` répétés.
- 789 identifiants `review_id` distincts répétés.
- Aucun doublon sur la combinaison `review_id + order_id`.
- 547 commandes possédant plusieurs avis.
- 4 commandes possédant trois avis.

La relation entre `orders` et `order_reviews` est donc de type One-to-Many (1:N).

Cette relation devra être prise en compte lors des jointures et des agrégations afin d'éviter la multiplication artificielle des indicateurs commerciaux.

## 7. Analyse des statuts de commandes

La distribution des statuts observée est la suivante :

| Statut | Nombre de commandes |
|---|---:|
| delivered | 96 478 |
| shipped | 1 107 |
| canceled | 625 |
| unavailable | 609 |
| invoiced | 314 |
| processing | 301 |
| created | 5 |
| approved | 2 |
| **Total** | **99 441** |

Le statut `delivered` représente environ 97 % des commandes.

Cette distribution permettra de définir les populations utilisées dans les analyses commerciales et dans le futur modèle de Machine Learning.

## 8. Analyse de la cohérence des dates de livraison

Deux situations particulières ont été identifiées.

### 8.1 Commandes livrées sans date réelle de livraison

Huit commandes possèdent le statut `delivered`, mais leur colonne `order_delivered_customer_date` est vide.

Ces commandes seront conservées dans les données sources, mais exclues de la construction initiale de la variable cible du modèle prédictif.

### 8.2 Commandes annulées avec une date réelle de livraison

Six commandes possèdent le statut `canceled` tout en ayant une date de livraison effective.

Ces cas seront conservés et signalés comme anomalies métier. Ils ne seront pas utilisés dans le jeu d'entraînement initial.

## 9. Premières décisions pour le Machine Learning

L'objectif du futur modèle est de prédire si une commande risque d'être livrée en retard.

La variable cible envisagée est :

- `1` : livraison réelle après la date de livraison estimée.
- `0` : livraison réelle à la date estimée ou avant.

La population initiale sera limitée aux commandes ayant le statut `delivered` et une date réelle de livraison renseignée.

Une attention particulière sera portée au **Data Leakage** : les informations qui ne sont pas disponibles au moment de la prédiction ne devront pas être utilisées comme variables explicatives.

Le moment exact de prédiction et les variables autorisées seront définis avant la construction du dataset Machine Learning.

## 10. Synthèse et conclusion

La première phase a permis de comprendre la structure du dataset Olist, d'identifier les principales relations entre les tables et de détecter plusieurs anomalies liées aux valeurs manquantes, aux doublons et à la cohérence des données.

Les contrôles effectués constituent une première base de connaissance pour la conception du processus ETL.

**Les données brutes seront conservées sans modification.** Les opérations de nettoyage et de transformation seront réalisées dans une couche dédiée afin de garantir la traçabilité et la reproductibilité du traitement.

### Résultats de la phase 1

- Exploration des neuf fichiers CSV.
- Analyse des dimensions et des types de données.
- Identification des clés candidates.
- Vérification des principales relations entre les tables.
- Analyse des valeurs manquantes.
- Détection des doublons et anomalies.
- Analyse des statuts et des dates de livraison.
- Définition des premières règles métier pour le futur modèle prédictif.

**Statut : Phase 1 — Exploration et compréhension des données terminée.**