# Phase 5.1 — Machine Learning Data Audit & Design

> **Statut : audit terminé — conception proposée, aucun modèle entraîné.**
>
> **Date de l'audit : 18 novembre 2026.**

## Table des matières

1. [Résumé exécutif](#section-1)
2. [Objectif ML et définition de la cible](#section-2)
3. [Sources de données réellement disponibles](#section-3)
4. [Moment de prédiction et disponibilité des données](#section-4)
5. [Granularité et règles de jointure](#section-5)
6. [Inventaire des features candidates](#section-6)
7. [Risques de data leakage](#section-7)
8. [Règles de filtrage et de qualité](#section-8)
9. [Stratégie de validation temporelle](#section-9)
10. [Métriques et modèles envisagés](#section-10)
11. [Compatibilité avec FastAPI, React et la simulation](#section-11)
12. [Questions techniques non résolues](#section-12)
13. [Recommandations pour la Phase 5.2](#section-13)
14. [Sources examinées et méthode d'audit](#section-14)

<a id="section-1"></a>

## 1. Résumé exécutif

L'objectif de la Phase 5 est de prédire, **au moment de la création d'une commande**, le risque que sa livraison intervienne après la date promise au client. Cet audit établit le contrat de données nécessaire avant toute construction de pipeline ou entraînement.

Les constats principaux sont les suivants :

- le Data Warehouse contient **99 441 commandes** au grain `order_id` ;
- la cible `is_delayed` est observable pour **96 470 commandes livrées**, soit **96,01 %** des commandes ;
- la population éligible comprend **6 534 retards** et **89 936 livraisons à temps**, soit un taux de retard de **6,77 %** ;
- **2 971 commandes** ont une cible inconnue et ne doivent être ni imputées à `0`, ni utilisées dans l'apprentissage supervisé ;
- la règle validée compare les **dates calendaires** de livraison réelle et estimée, et non les timestamps complets ;
- le Data Warehouse fournit déjà une ligne par commande, mais les détails articles, vendeurs, produits et paiements nécessitent des agrégations explicites avant jointure ;
- la disponibilité opérationnelle des articles, vendeurs, prix, fret, produits et paiements à l'instant exact de création n'est pas démontrée par un journal d'événements ;
- les dates réelles de traitement et de livraison, le statut final, les avis, la cible et tous leurs dérivés sont interdits comme features ;
- les taux de retard mensuels varient fortement dans le temps, de sorte qu'une séparation aléatoire donnerait une estimation trop optimiste de la généralisation ;
- la première version doit privilégier un socle conservateur : calendrier de création, géographie client et délai promis, ce dernier sous réserve de confirmer qu'il est bien fixé avant l'appel au modèle.

La Phase 5.2 ne doit commencer qu'après validation d'un **contrat d'événement de création de commande** : il doit préciser quels champs sont disponibles, leur horodatage, leur source et leur stabilité entre entraînement et inférence.

<a id="section-2"></a>

## 2. Objectif ML et définition de la cible

### 2.1 Problème supervisé

Le problème proposé est une classification binaire :

> À la création d'une commande, estimer la probabilité qu'elle soit livrée après la date estimée annoncée au client.

L'unité de prédiction est une commande et le score attendu est une probabilité comprise entre `0` et `1`. Le seuil de décision métier sera choisi ultérieurement selon le coût des faux négatifs et des faux positifs ; il ne doit pas être fixé arbitrairement à `0,5`.

### 2.2 Règle cible de référence

La règle proposée dans la mission compare deux timestamps. Le Data Warehouse applique toutefois une règle métier corrigée et déjà validée, qui compare les **jours calendaires** :

```sql
CASE
    WHEN order_status = 'delivered'
         AND delivered_timestamp IS NOT NULL
    THEN (
        delivered_timestamp::date
        > estimated_delivery_timestamp::date
    )
    ELSE NULL
END AS is_delayed
```

Cette définition, implémentée dans [`05_load_fact_orders.sql`](../sql/warehouse/05_load_fact_orders.sql) et tracée par [`06_fix_delivery_delay.sql`](../sql/warehouse/06_fix_delivery_delay.sql), doit rester la définition de référence pour assurer la cohérence avec les Phases 3 et 4.

Une livraison effectuée le jour estimé est donc considérée à temps, quelle que soit son heure. Une comparaison directe des timestamps changerait cette sémantique et créerait une cible incohérente avec le Data Warehouse.

### 2.3 Population éligible vérifiée

Les contrôles PostgreSQL en transaction `READ ONLY` ont produit les résultats suivants :

| Population | Commandes | Part du total |
|---|---:|---:|
| Total `dw.fact_orders` | 99 441 | 100,00 % |
| Cible observable | 96 470 | 97,01 % |
| `is_delayed = TRUE` | 6 534 | 6,77 % des commandes évaluables |
| `is_delayed = FALSE` | 89 936 | 93,23 % des commandes évaluables |
| Cible inconnue | 2 971 | 2,99 % |

La période d'achat des commandes éligibles va du **15 septembre 2016 à 12:16:38** au **29 août 2018 à 15:00:37**.

Les 2 971 cibles inconnues comprennent notamment :

- huit commandes au statut `delivered` sans date réelle de livraison ;
- les commandes `approved`, `created`, `invoiced`, `processing`, `shipped`, `canceled` et `unavailable` ;
- six commandes annulées possédant malgré tout une date de livraison, volontairement exclues parce que leur statut final n'est pas `delivered`.

### 2.4 Portée métier et biais de sélection

Le dataset supervisé répond précisément à la question :

> Quel est le risque de retard **parmi les commandes qui seront finalement livrées et dont la date réelle est observable** ?

Au moment de la création, l'issue finale `delivered`, `canceled` ou `unavailable` n'est pas connue. En production, le modèle recevra donc aussi des commandes qui ne seront jamais livrées. Cette différence crée un biais de sélection à suivre explicitement.

Deux options devront être arbitrées après la première baseline :

1. conserver un modèle de retard conditionnel à la livraison, avec une documentation claire de sa portée ;
2. ajouter ultérieurement un modèle distinct de non-livraison ou une approche multi-états, sans convertir les non-livraisons en livraisons à temps.

<a id="section-3"></a>

## 3. Sources de données réellement disponibles

### 3.1 Couches pertinentes

| Couche | Objets vérifiés | Utilité ML | Limite principale |
|---|---|---|---|
| ETL / Parquet | `orders_clean`, `order_items_clean`, `products_clean`, `customers_clean`, `sellers_clean`, paiements détaillés et agrégés | Reproductibilité et contrôles de transformation | Fichiers batch sans historique d'arrivée des colonnes |
| `staging` | Dix tables issues des Parquet | Colonnes sources détaillées et flags qualité | Couche de chargement, pas un contrat online |
| `dw` | `fact_orders`, `fact_order_items`, `fact_payments`, `fact_reviews`, dimensions client, produit, vendeur et date | Source structurée recommandée pour le dataset historique | Certaines colonnes postérieures à la prédiction coexistent avec les features candidates |
| `analytics` | Sept vues agrégées | Contrôles et compréhension métier | Granularités incompatibles avec une ligne par commande et agrégats calculés sur toute l'histoire |

Les vues `analytics` ne doivent pas être utilisées directement comme matrice d'entraînement. Elles sont adaptées aux dashboards, pas à la reconstruction temporelle d'un snapshot de commande.

### 3.2 Colonnes commande vérifiées

La table `dw.fact_orders` contient notamment :

| Groupe | Colonnes réelles |
|---|---|
| Identité et relations | `order_key`, `order_id`, `customer_key` |
| Dates | `purchase_date_key`, `approved_date_key`, `carrier_date_key`, `delivered_date_key`, `estimated_date_key` |
| Timestamps | `purchase_timestamp`, `delivered_timestamp`, `estimated_delivery_timestamp` |
| Résultat logistique | `order_status`, `delivery_days`, `delivery_delay_days`, `is_delayed` |
| Agrégats articles | `item_count`, `total_item_value`, `total_freight_value`, `total_order_value` |
| Paiement | `total_payment_value` |
| Qualité | `flag_delivered_without_date`, `flag_canceled_with_delivery` |

`staging.orders_clean` conserve en complément `order_approved_at`, `order_delivered_carrier_date` et `flag_delivery_before_purchase`. Ce dernier flag n'est pas propagé dans `dw.fact_orders`.

### 3.3 Tables de détail vérifiées

| Table | Grain | Colonnes ML potentiellement utiles |
|---|---|---|
| `dw.fact_order_items` | Un article d'une commande | `order_key`, `product_key`, `seller_key`, `price`, `freight_value`, `item_total_value`, `shipping_limit_date_key` |
| `dw.dim_product` | Un produit | Catégories, poids, dimensions, longueurs descriptives, photos et flags qualité |
| `dw.dim_customer` | Un `customer_id` | `customer_unique_id`, ville, État, préfixe postal, coordonnées et qualité GPS |
| `dw.dim_seller` | Un vendeur | Ville, État, préfixe postal, coordonnées et qualité GPS |
| `staging.order_payments_clean` | Une séquence de paiement | Type, nombre de versements, montant et flags qualité |
| `staging.order_payments_aggregated` | Une commande payée | Montant, nombre de paiements, méthodes, versements maximums et flags |
| `dw.fact_reviews` | Un avis | Informations postérieures à la livraison, interdites pour la prédiction |

### 3.4 Couverture et qualité utiles au design

Les vérifications en base montrent :

| Contrôle sur les 96 470 commandes éligibles | Résultat |
|---|---:|
| Commandes sans agrégat article | 0 |
| Commandes sans valeur produit ou fret | 0 |
| Commandes sans paiement agrégé | 1 |
| Commandes sans coordonnées client exploitables | 265 |
| Commandes avec géographie client ou au moins un vendeur incomplète | 481 |
| Commandes touchées par une catégorie produit manquante | 1 392 |
| Commandes touchées par des dimensions produit manquantes | 16 |
| Commandes touchées par un poids nul | 6 |
| Commandes touchées par une traduction manquante | 20 |

Les valeurs manquantes doivent être représentées par des indicateurs explicites. Elles ne justifient pas, à elles seules, la suppression d'une commande éligible.

<a id="section-4"></a>

## 4. Moment de prédiction et disponibilité des données

### 4.1 Instant de référence

Le moment de prédiction est défini par :

```text
prediction_time = order_purchase_timestamp
```

Une feature n'est admissible que si sa valeur est connue, ou calculable à partir d'informations connues, **au plus tard à cet instant**. Sa présence dans le dataset final historique ne prouve pas sa disponibilité opérationnelle.

### 4.2 Chronologie des données

| Étape | Exemples de données | Décision initiale |
|---|---|---|
| Création de commande | Timestamp d'achat, adresse client, calendrier | Admissible |
| Promesse client | Date estimée de livraison | Admissible uniquement après confirmation qu'elle est fixée avant le scoring |
| Panier / checkout | Articles, produits, vendeurs, prix, fret, paiement | À vérifier avec le futur contrat d'événement |
| Approbation | `order_approved_at` | Exclu pour une prédiction à la création |
| Préparation / expédition | `shipping_limit_date`, date de remise au transporteur | Exclu ou à vérifier selon la colonne ; jamais utilisé sans preuve d'antériorité |
| Livraison | Date réelle, durée réelle, retard réel, statut final | Interdit, sauf pour construire la cible |
| Après livraison | Avis, note, commentaire, réponse | Interdit |

### 4.3 Socle V1 conservateur

Le premier dataset doit pouvoir être construit en deux niveaux :

- **Baseline stricte :** caractéristiques calendaires de `purchase_timestamp` et géographie du client disponible dans la commande ;
- **Baseline métier conditionnelle :** ajout du délai promis, puis des agrégats panier, produit et vendeur uniquement lorsque leur présence dans le payload de création est formellement confirmée.

Cette séparation permettra de mesurer la valeur ajoutée des données de checkout sans contaminer la baseline avec une hypothèse non vérifiée.

### 4.4 Date estimée et horizon promis

`estimated_delivery_timestamp` est non nul pour les 99 441 commandes. Sur la population éligible, l'horizon calendaire :

```text
promised_lead_days = estimated_delivery_timestamp::date
                     - purchase_timestamp::date
```

présente les valeurs vérifiées suivantes : minimum 3 jours, médiane 24, 95e percentile 39, 99e percentile 51 et maximum 156. Aucun horizon n'est nul ou négatif.

Cette variable est probablement déterminante, mais son **moment de calcul** n'est pas horodaté dans Olist. Elle reste donc conditionnelle à une confirmation du futur système transactionnel.

<a id="section-5"></a>

## 5. Granularité et règles de jointure

### 5.1 Grain imposé

Le dataset ML doit respecter la contrainte :

```text
1 ligne = 1 order_id
```

`dw.fact_orders` respecte déjà ce grain : 99 441 lignes, 99 441 `order_id` distincts. Sur les commandes éligibles, aucun doublon d'`order_id` n'a été détecté.

### 5.2 Multiplicités vérifiées

| Relation | Constat |
|---|---:|
| Commandes ayant au moins un article | 98 666 |
| Commandes multi-articles | 9 803 |
| Commandes multivendeurs | 1 278 |
| Commandes multiproduits | 3 236 |
| Commandes multicatégories | 786 |
| Maximum d'articles par commande | 21 |
| Maximum de vendeurs par commande | 5 |
| Maximum de catégories par commande | 3 |

Parmi les seules commandes éligibles, 9 635 sont multi-articles, 1 275 multivendeurs, 3 197 multiproduits et 780 multicatégories.

Une jointure directe entre commandes, articles, paiements et avis produirait un produit cartésien. Chaque source multiple doit être agrégée séparément au grain commande avant toute combinaison.

### 5.3 Agrégations proposées

| Domaine | Agrégations au grain commande |
|---|---|
| Articles | Nombre de lignes, nombre de produits distincts, sommes des prix et du fret |
| Vendeurs | Nombre distinct, nombre d'États, indicateur multivendeur, indicateur inter-État |
| Catégories | Nombre distinct, catégorie dominante déterminée par une règle stable, éventuellement représentation multi-hot |
| Produit physique | Somme, moyenne et maximum du poids et du volume ; compteurs de valeurs manquantes |
| Distance | Minimum, moyenne pondérée et maximum des distances client–vendeur ; flag de couverture incomplète |
| Paiements | Nombre de paiements et méthodes, versements maximums et montant total, si disponibles au scoring |
| Historique | Agrégats strictement antérieurs à `purchase_timestamp`, avec une date d'observation explicite |

La catégorie dominante et la distance moyenne doivent utiliser une règle de pondération documentée, par exemple le nombre d'articles ou la valeur produit. La Phase 5.2 devra tester la stabilité de cette règle, pas la choisir implicitement.

<a id="section-6"></a>

## 6. Inventaire des features candidates

La colonne « disponibilité » distingue les données observées dans l'historique de celles dont la disponibilité online est prouvée. « Vérifier » signifie que la feature est exclue de la baseline stricte tant que le contrat d'événement n'est pas validé.

| Feature proposée | Source SQL réelle | Transformation | Disponibilité à la création | Risque de fuite | Décision |
|---|---|---|---|---|---|
| `purchase_hour` | `dw.fact_orders.purchase_timestamp` | Extraire l'heure | Oui | Faible | Conserver |
| `purchase_day_of_week` | `purchase_timestamp` | Jour de semaine, éventuellement encodage cyclique | Oui | Faible | Conserver |
| `purchase_month` | `purchase_timestamp` | Mois ou saison, sans utiliser le futur | Oui | Faible | Conserver |
| `purchase_is_weekend` | `purchase_timestamp` ou `dw.dim_date` | Booléen | Oui | Faible | Conserver |
| `customer_state` | `dw.dim_customer.customer_state` | Catégorie à 27 modalités | Oui si adresse capturée à la création | Faible | Conserver |
| `customer_city` | `dw.dim_customer.customer_city` | Regroupement des modalités rares | Probable | Surapprentissage géographique | Vérifier |
| `customer_zip_prefix` | `dw.dim_customer.customer_zip_code_prefix` | Catégoriel ou géographique | Probable | Forte cardinalité, quasi-identification | Exclure de V1 |
| `customer_latitude`, `customer_longitude` | `dw.dim_customer` | Valeurs numériques avec flag manquant | Oui si géocodage déterministe du préfixe | Faible à modéré | Conserver avec flag |
| `promised_lead_days` | `estimated_delivery_timestamp`, `purchase_timestamp` | Différence en jours calendaires | Non démontrée, mais attendue métier | Très faible si fixé avant score ; fuite sinon | Vérifier puis conserver |
| `estimated_delivery_month` | `estimated_delivery_timestamp` | Mois de promesse | Même condition que ci-dessus | Faible sous condition | Vérifier |
| `item_count` | `dw.fact_orders.item_count` | Valeur déjà agrégée | Non horodatée dans Olist | Fuite de disponibilité possible | Vérifier |
| `product_value_total` | `dw.fact_orders.total_item_value` | Somme des prix | Non horodatée | Fuite de disponibilité possible | Vérifier |
| `freight_value_total` | `dw.fact_orders.total_freight_value` | Somme du fret | Non horodatée | Fuite de disponibilité possible | Vérifier |
| `freight_ratio` | Valeurs produit et fret | `freight / NULLIF(product + freight, 0)` | Non horodatée | Même risque que ses sources | Vérifier |
| `seller_count` | `dw.fact_order_items.seller_key` | `COUNT(DISTINCT seller_key)` | Non démontrée | Faible si panier final connu | Vérifier |
| `seller_state_count` | `fact_order_items` + `dim_seller` | Nombre d'États distincts | Non démontrée | Faible si vendeurs connus | Vérifier |
| `cross_state_shipping` | `dim_customer`, `dim_seller` | Client et au moins un vendeur dans des États différents | Non démontrée | Faible si vendeurs connus | Vérifier |
| `distance_km_mean`, `distance_km_max` | Coordonnées client et vendeurs | Haversine, agrégation au grain commande | Non démontrée | Faible si vendeurs connus ; biais GPS | Vérifier |
| `category_count` | `fact_order_items` + `dim_product` | Nombre de catégories distinctes | Non démontrée | Faible si produits connus | Vérifier |
| `dominant_category` | Même source | Règle déterministe et valeur `unknown` | Non démontrée | Cardinalité et catégories futures | Vérifier |
| `product_weight_total` | `dim_product.product_weight_g` | Somme par article, flag manquant | Non démontrée | Faible si catalogue versionné | Vérifier |
| `product_volume_total` | Longueur × hauteur × largeur | Somme et flag dimension manquante | Non démontrée | Faible si catalogue versionné | Vérifier |
| `payment_type`, `max_installments` | `staging.order_payments_*` | Agrégation commande | Peut être postérieure à la création | Forte indisponibilité potentielle | Exclure de V1 |
| `customer_prior_orders` | `customer_unique_id` + commandes antérieures | Fenêtre `purchase_timestamp` strictement passée | Calculable online avec feature store | Fuite temporelle si calcul global | Réserver à V2 |
| `seller_prior_order_count` | Vendeur + commandes antérieures | Agrégat as-of | Vendeur d'abord requis | Fuite temporelle si calcul global | Réserver à V2 |
| `seller_prior_delay_rate` | Historique vendeur | Sommes retard/évaluable sur labels déjà matures, avec lissage | Vendeur et historique requis | Très élevé si calcul full-history | Réserver à V2 |
| IDs bruts | `order_id`, `customer_id`, `customer_unique_id`, `seller_id`, `product_id` | Aucun | Variable | Mémorisation et forte cardinalité | Exclure comme features |

### 6.1 Cardinalités vérifiées

Sur la population éligible et ses articles :

- 27 États client ;
- 22 États vendeur ;
- 74 catégories produit traduites ou normalisées ;
- 2 970 vendeurs actifs ;
- 93 350 personnes distinctes via `customer_unique_id` ;
- 2 801 personnes ont plusieurs commandes éligibles, représentant 5 921 commandes.

La répétition de clients permet des historiques as-of, mais les IDs ne doivent pas devenir des variables catégorielles brutes. Un rapport séparé « clients déjà vus » / « nouveaux clients » sera utile pour mesurer le comportement en cold start.

### 6.2 Distance géographique

La distance client–vendeur est techniquement calculable pour **95 989 des 96 470 commandes éligibles** lorsque toutes les coordonnées nécessaires sont valides. Pour 481 commandes, la géographie client ou celle d'au moins un vendeur est incomplète.

La feature doit donc inclure :

- des statistiques de distance pour les vendeurs couverts ;
- `distance_missing_count` ou `distance_coverage_ratio` ;
- un indicateur signalant qu'au moins un trajet n'est pas calculable ;
- aucune imputation fondée sur l'ensemble train + validation + test.

<a id="section-7"></a>

## 7. Risques de data leakage

| Colonne ou famille | Source | Type de fuite | Règle |
|---|---|---|---|
| `delivered_timestamp` / `order_delivered_customer_date` | Commandes | Résultat réel | Interdit comme feature ; cible uniquement |
| `delivered_date_key` | `dw.fact_orders` | Résultat réel encodé | Interdit |
| `delivery_days` | `dw.fact_orders` | Calcul après livraison | Interdit |
| `delivery_delay_days` | `dw.fact_orders` | Dérivé direct de la cible | Interdit |
| `is_delayed` | `dw.fact_orders` | Cible | Jamais dans `X` |
| `order_status` historique | Commandes | Statut final connu après création | Interdit |
| `carrier_date_key`, date transporteur | Commandes | Événement post-création | Interdit |
| `approved_date_key`, `order_approved_at` | Commandes | Événement post-création | Interdit pour ce moment de prédiction |
| `shipping_limit_date` | Articles | Disponibilité non démontrée et potentiellement générée après achat | Exclure tant que le contrat n'est pas prouvé |
| Flags de livraison | ETL / commandes | Dérivés d'événements futurs | Interdits |
| Avis, notes, messages, temps de réponse | `fact_reviews` / staging | Événements post-livraison | Interdits |
| Vues de satisfaction et livraison | `analytics.*` | Agrégats incluant la cible et le futur | Interdites comme features |
| `vw_seller_performance.delay_rate` | Vue vendeur | Historique calculé sur toute la période | Interdit directement |
| Statistiques catégorie ou vendeur globales | Vues analytiques | Futur incorporé dans les agrégats | Recalcul as-of obligatoire |
| Encodages target mean | Dataset ML | Utilisation potentielle de la cible validation/test | Fit uniquement dans les folds d'entraînement |
| Imputation, scaling, sélection de variables | Pipeline ML | Apprentissage sur validation/test | Fit uniquement sur train |

### 7.1 Fuite par jointure

Une jointure n'est pas sûre parce qu'elle est valide au niveau des clés. Elle doit aussi respecter :

- le grain final `order_id` ;
- une borne temporelle `event_time <= prediction_time` ;
- une date de disponibilité du label pour les historiques de retard ;
- la version du produit, du vendeur ou du référentiel géographique applicable à la date de commande.

Les dimensions actuelles ne sont pas des Slowly Changing Dimensions. Elles représentent l'état chargé du dataset et ne conservent pas de version temporelle. Pour Olist, ce risque doit être documenté ; pour la future simulation, les snapshots d'inférence doivent être persistés.

### 7.2 Fuite temporelle dans les historiques

Pour une commande achetée à `t`, une statistique vendeur ne peut utiliser que les commandes :

1. créées avant `t` ;
2. dont le résultat était déjà observable avant `t` ;
3. agrégées sans inclure la commande courante.

Un simple `GROUP BY seller_id` calculé sur tout le dataset est une fuite, même si la colonne obtenue semble être une feature historique.

<a id="section-8"></a>

## 8. Règles de filtrage et de qualité

### 8.1 Inclusion et exclusion de la cible

Inclure dans l'apprentissage supervisé uniquement les lignes satisfaisant :

```sql
order_status = 'delivered'
AND delivered_timestamp IS NOT NULL
AND estimated_delivery_timestamp IS NOT NULL
AND is_delayed IS NOT NULL
```

Les contrôles ont confirmé, sur les 96 470 lignes éligibles :

- aucune date réelle ou estimée manquante ;
- aucune livraison antérieure à l'achat ;
- aucun flag `delivered_without_date` ou `canceled_with_delivery` ;
- aucun doublon d'`order_id`.

Les 2 971 lignes à cible inconnue doivent être conservées dans un jeu de suivi séparé pour analyser la couverture opérationnelle, mais exclues de `X` et `y`.

### 8.2 Valeurs manquantes

- Ne pas supprimer automatiquement les 265 commandes sans GPS client ni les 481 distances incomplètes.
- Ajouter des flags de présence et apprendre les imputations uniquement sur le train.
- Utiliser une catégorie `unknown` stable pour les catégories absentes ou nouvelles.
- Conserver les flags qualité produit si les features produit sont validées.
- Traiter séparément l'unique paiement agrégé manquant si les paiements deviennent admissibles.

### 8.3 Valeurs extrêmes

La durée réelle de livraison a une médiane de 10,22 jours, un 95e percentile de 29,27 jours, un 99e percentile de 46,05 jours et un maximum de 209,63 jours. Ces valeurs ne doivent pas devenir des features, mais elles renseignent la latence d'observation des labels.

Les règles de winsorisation ou de transformation logarithmique des montants, poids, volumes et distances devront être apprises sur le train. Aucune limite ne doit être choisie après observation du test.

### 8.4 Déséquilibre

Le taux positif de 6,77 % implique :

- aucun sur-échantillonnage avant la séparation temporelle ;
- tout rééchantillonnage réalisé uniquement dans le fold d'entraînement ;
- une comparaison avec et sans pondération des classes ;
- une évaluation en probabilité avant le choix d'un seuil métier.

<a id="section-9"></a>

## 9. Stratégie de validation temporelle

### 9.1 Holdout chronologique proposé

Un premier découpage reproductible par `purchase_timestamp` est proposé :

| Partition | Fenêtre d'achat | Commandes | Retards | Taux vérifié |
|---|---|---:|---:|---:|
| Train | Avant le 1er janvier 2018 | 43 693 | 2 456 | 5,62 % |
| Validation | Du 1er janvier au 30 avril 2018 | 27 425 | 2 963 | 10,80 % |
| Test final | Du 1er mai au 29 août 2018 | 25 352 | 1 115 | 4,40 % |

Ce découpage conserve les quatre derniers mois comme test final et expose volontairement le modèle à une distribution différente. Les taux mensuels varient fortement : février 2018 atteint 14,13 %, mars 18,96 % et juin 1,16 %. Cette dérive est une propriété à mesurer, pas un défaut à masquer par un mélange aléatoire.

Le test final doit rester scellé jusqu'à la fin de la sélection des features, du modèle et du seuil.

### 9.2 Maturité des labels

Une commande achetée avant une coupure peut être livrée après celle-ci. Dans un backtest strict, son label n'aurait pas été disponible au moment du réentraînement.

La Phase 5.2 doit donc définir pour chaque fold :

- `prediction_time` : date de la commande scorée ;
- `as_of_time` : instant auquel le dataset d'entraînement est supposé être construit ;
- `label_observed_at = delivered_timestamp` ;
- inclusion dans l'historique uniquement si `label_observed_at < as_of_time`.

Le 99e percentile de la durée de livraison est de 46,05 jours, mais le maximum atteint 209,63 jours. Une fenêtre de maturité de 60 jours couvrirait la grande majorité des cas sans couvrir tous les extrêmes. Son impact sur la couverture et le biais de censure doit être quantifié avant adoption.

### 9.3 Backtests complémentaires

En plus du holdout final, utiliser des folds à origine glissante ou croissante :

```text
Fold 1 : passé ancien      → mois suivant(s)
Fold 2 : passé étendu      → mois suivant(s)
Fold 3 : passé plus récent → mois suivant(s)
```

Pour chaque fold :

- ajuster imputation, encodage, scaling et sélection uniquement sur le train ;
- recalculer les historiques client et vendeur en as-of ;
- conserver l'ordre temporel ;
- mesurer la performance globale, mensuelle et par sous-groupes ;
- mesurer séparément les clients, vendeurs et catégories jamais vus dans le train.

Une séparation aléatoire peut être utilisée uniquement comme diagnostic secondaire, jamais comme estimation principale de performance.

<a id="section-10"></a>

## 10. Métriques et modèles envisagés

### 10.1 Métriques

| Métrique | Rôle |
|---|---|
| PR-AUC / Average Precision | Métrique principale adaptée à la classe positive rare |
| Recall retard | Part des futurs retards effectivement détectés |
| Precision retard | Part des alertes qui correspondent à un retard réel |
| F1 | Compromis precision–recall pour un seuil donné |
| ROC-AUC | Mesure secondaire de classement, à ne pas interpréter seule |
| Matrice de confusion | Volumes de faux positifs et faux négatifs au seuil métier |
| Courbe de calibration et Brier score | Fiabilité des probabilités prédites |
| Lift / recall dans les déciles supérieurs | Utilité pour prioriser une capacité opérationnelle limitée |

Les métriques doivent être accompagnées d'intervalles de confiance, au minimum par bootstrap respectant les périodes ou par résultats fold par fold.

### 10.2 Baselines

1. **Baseline de prévalence :** probabilité constante égale au taux de retard du train.
2. **Règle métier simple :** seuil sur l'horizon promis, à définir uniquement avec le train.
3. **Régression logistique :** modèle interprétable avec encodage des catégories et pondération optionnelle des classes.

### 10.3 Modèles candidats

- régression logistique régularisée ;
- arbres de décision et forêts comme comparaisons interprétables ;
- gradient boosting sur données tabulaires ;
- bibliothèque spécialisée pour catégories uniquement si son ajout est décidé et documenté en Phase 5.2.

Aucune bibliothèque ML n'est actuellement utilisée par le code du projet. Les dépendances et leur version devront être choisies explicitement ; le présent audit ne suppose pas que scikit-learn, XGBoost, LightGBM ou CatBoost sont déjà intégrés.

### 10.4 Sélection du seuil

Le seuil doit être choisi sur la validation selon une contrainte métier, par exemple :

- recall minimal sur les retards ;
- capacité maximale d'alertes quotidiennes ;
- coût relatif d'une intervention inutile et d'un retard manqué.

Le seuil et la calibration doivent être gelés avant l'évaluation sur le test final.

<a id="section-11"></a>

## 11. Compatibilité avec FastAPI, React et la simulation

### 11.1 Contrat de features commun

Le batch d'entraînement et l'API doivent partager un contrat versionné plutôt que réimplémenter les transformations séparément.

| Champ de contrat | Rôle |
|---|---|
| `feature_schema_version` | Version de la structure et des transformations |
| `order_id` | Identifiant de traçabilité, jamais feature brute |
| `event_time` | Timestamp de création de la commande |
| `prediction_time` | Timestamp effectif du scoring |
| Champs source autorisés | Snapshot immuable des informations connues au scoring |
| Features calculées | Valeurs produites par le même code offline et online |
| Flags de qualité | GPS, catégorie, produit ou distance manquants |

Les transformations déterministes — calendrier, Haversine, agrégations panier, catégories inconnues — doivent vivre dans un composant Python partagé entre construction historique et inférence.

### 11.2 Sortie de prédiction

La réponse FastAPI future devrait au minimum contenir :

```json
{
  "order_id": "...",
  "delay_probability": 0.0,
  "risk_class": "low|medium|high",
  "threshold": 0.0,
  "model_version": "...",
  "feature_schema_version": "...",
  "predicted_at": "..."
}
```

Les valeurs ci-dessus décrivent un schéma, pas un résultat de modèle. Les bornes de classes et le seuil restent à définir.

### 11.3 Persistance pour la simulation

Pour chaque nouvelle commande, la plateforme devra conserver :

- le snapshot source reçu au scoring ;
- les features finales ;
- la probabilité, le seuil et la version du modèle ;
- l'instant de prédiction ;
- la date estimée utilisée ;
- l'issue réelle lorsqu'elle devient observable ;
- un état `pending`, `evaluated`, `canceled` ou `unavailable` sans forcer une cible binaire.

Cette persistance permettra à Power BI de suivre la couverture, la calibration, le drift et la performance après maturation des labels. React doit afficher le score et son contexte, mais ne doit pas recalculer les features métier.

### 11.4 Prévention du training-serving skew

- Un seul schéma de features versionné.
- Un seul code de transformation déterministe.
- Même gestion des valeurs absentes et catégories inconnues.
- Timestamps et fuseau horaire explicités.
- Référentiels géographiques et produit versionnés ou inclus dans le snapshot.
- Tests de parité entre une commande historique construite offline et la même commande envoyée à l'API.

<a id="section-12"></a>

## 12. Questions techniques non résolues

| Question | Impact | Caractère bloquant pour la Phase 5.2 |
|---|---|---|
| La date estimée est-elle définitivement fixée avant l'appel de prédiction ? | Conditionne la feature la plus directement liée à la promesse | Oui |
| Articles, vendeurs, prix et fret sont-ils présents dans l'événement de création ? | Conditionne la majorité des features métier | Oui pour la baseline enrichie |
| Le paiement est-il capturé avant ou après la création / approbation ? | Détermine si les features paiement sont admissibles | Non pour la baseline stricte |
| Quel fuseau horaire doit être appliqué aux timestamps sans timezone ? | Affecte heure, jour et coupures temporelles | Oui |
| Quelle règle d'agrégation utiliser pour les commandes multivendeurs ? | Affecte distance, géographie et historique vendeur | Oui pour les features vendeurs |
| Quelle fenêtre de maturité des labels accepter ? | Contrôle censure, cadence de réentraînement et volume | Oui |
| Quel coût métier attribuer aux faux négatifs et faux positifs ? | Détermine seuil et métrique de sélection | Oui avant mise en production |
| Comment traiter les commandes annulées ou indisponibles en production ? | Définit la portée réelle du score et le monitoring | Oui avant simulation end-to-end |
| Les dimensions client, produit et vendeur seront-elles historisées ? | Évite qu'un état futur modifie un snapshot passé | Oui pour un système durable |
| Quel volume d'alertes opérationnelles peut être traité ? | Conditionne le seuil et le lift utile | Non pour l'entraînement initial |

<a id="section-13"></a>

## 13. Recommandations pour la Phase 5.2

1. **Valider le contrat d'événement** avec la liste exacte des champs disponibles à `order_purchase_timestamp`.
2. **Créer une spécification de dataset**, pas encore un pipeline de production, avec grain, types, contraintes et règles d'as-of.
3. **Construire deux jeux de features** : baseline stricte et baseline enrichie conditionnelle au checkout.
4. **Centraliser la cible** sur la règle calendaire du Data Warehouse et ajouter des tests de non-régression sur les 96 470 lignes éligibles.
5. **Implémenter les agrégations au grain commande** avant toute jointure de faits.
6. **Conserver le test final temporel scellé** et utiliser des backtests à origine croissante pour la sélection.
7. **Définir la maturité des labels** et mesurer le biais introduit par les commandes non encore livrées.
8. **Commencer par les baselines simples** avant tout modèle de boosting.
9. **Évaluer la calibration et la stabilité mensuelle**, pas seulement une métrique globale.
10. **Versionner le schéma de features et le snapshot d'inférence** dès la première expérimentation reproductible.
11. **Tester les sous-groupes** : États, GPS manquant, multivendeur, clients ou vendeurs nouveaux et périodes de forte dérive.
12. **Ne pas intégrer FastAPI ou React avant de figer le contrat offline/online** et les tests de parité.

La sortie attendue de la Phase 5.2 devrait être un dataset reproductible, un dictionnaire de features versionné, un protocole de split exécutable et une baseline évaluée — sans encore confondre expérimentation et service de production.

<a id="section-14"></a>

## 14. Sources examinées et méthode d'audit

### 14.1 Documentation

- [`PHASE_1_DATA_EXPLORATION.md`](PHASE_1_DATA_EXPLORATION.md)
- [`PHASE_2_DATA_CLEANING_ETL.md`](PHASE_2_DATA_CLEANING_ETL.md)
- [`PHASE_3_DATA_WAREHOUSE.md`](PHASE_3_DATA_WAREHOUSE.md)
- [`PHASE_4_SQL_ANALYTICS_BI.md`](PHASE_4_SQL_ANALYTICS_BI.md)
- [`README.md`](../README.md)

### 14.2 Python, configuration et tests

- [`config/paths.py`](../config/paths.py)
- [`src/etl/extract.py`](../src/etl/extract.py)
- [`src/etl/transform.py`](../src/etl/transform.py)
- [`src/etl/pipeline.py`](../src/etl/pipeline.py)
- [`src/etl/load.py`](../src/etl/load.py)
- [`tests/test_etl_integration.py`](../tests/test_etl_integration.py)
- tests ciblés sur commandes, articles, paiements, produits, clients, vendeurs et géolocalisation dans [`tests/`](../tests/)

### 14.3 SQL

- définitions staging dans [`sql/staging/`](../sql/staging/)
- dimensions et faits dans [`sql/warehouse/`](../sql/warehouse/)
- vues analytiques dans [`sql/analytics/`](../sql/analytics/)
- validations dans [`sql/validation/`](../sql/validation/)

### 14.4 Vérifications PostgreSQL

La base locale a été interrogée uniquement par des `SELECT` dans des transactions déclarées `READ ONLY`. Les contrôles ont porté sur :

- les colonnes et types de `information_schema.columns` ;
- les volumes et la distribution de la cible ;
- la distribution par statut et par mois ;
- les plages de dates et les délais ;
- les valeurs manquantes ;
- les cardinalités client, vendeur et catégorie ;
- les multiplicités articles, produits, vendeurs et catégories ;
- la couverture GPS et les conditions nécessaires au calcul de distance ;
- les anomalies de cible et l'unicité des commandes.

Aucune commande d'écriture, migration, création, mise à jour ou suppression n'a été exécutée. Aucun secret de connexion n'a été affiché ou enregistré.

## Conclusion

Le projet dispose d'une cible cohérente et d'une population supervisée substantielle, mais la difficulté principale n'est pas le choix du modèle : c'est la reconstruction fidèle de l'information disponible à la création de la commande.

La Phase 5.2 devra donc commencer par le contrat temporel des données et un dataset une-ligne-par-commande contrôlé. Les features postérieures, les agrégats full-history et les vues analytiques globales doivent rester hors du pipeline prédictif. Cette discipline est indispensable pour obtenir une performance offline crédible et une intégration FastAPI cohérente avec la future simulation.

**Statut final : Phase 5.1 documentée ; aucun entraînement ni pipeline ML créé.**
