# Phase 5.4 — Advanced Feature Engineering, Probability Calibration & Temporal Backtesting

> **Statut : prototype V2 conditionnel, calibration et backtests pré-test exécutés.**
>
> **Candidat expérimental recommandé : régression logistique V2 + calibration sigmoid.**
>
> **Le candidat n'est pas prêt pour la production. Le test final n'a été ni lu, ni prédit, ni évalué.**
>
> **Date d'exécution documentée : 18 novembre 2026.**

## Table des matières

1. [Objectifs et périmètre](#section-1)
2. [Baseline V1](#section-2)
3. [Protection du test final](#section-3)
4. [Audit de disponibilité](#section-4)
5. [Contrat enrichi V2 conditionnel](#section-5)
6. [Feature engineering](#section-6)
7. [Stratégie temporelle et calibration](#section-7)
8. [Folds réellement exécutés](#section-8)
9. [Comparaison des modèles et calibrations](#section-9)
10. [Seuils et charge d'alertes](#section-10)
11. [Robustesse mensuelle et géographique](#section-11)
12. [Drift](#section-12)
13. [Candidat recommandé](#section-13)
14. [Artefacts et reproductibilité](#section-14)
15. [Tests et contrôles](#section-15)
16. [Limites et risques](#section-16)
17. [Recommandations avant la Phase 5.5](#section-17)

<a id="section-1"></a>

## 1. Objectifs et périmètre

Cette phase cherche à améliorer la baseline de Phase 5.3 sans ouvrir le test final. Elle réalise :

- un audit de disponibilité opérationnelle des variables logistiques ;
- un contrat V2 séparé, versionné et explicitement conditionnel ;
- un prototype une-ligne-par-commande enrichi avant mai 2018 ;
- une comparaison V1/V2 avec les modèles de Phase 5.3 ;
- des calibrations sigmoid et isotonic sur des périodes postérieures au train ;
- quatre backtests mensuels à origine croissante ;
- une analyse des seuils, alertes, sous-groupes géographiques et indicateurs de drift ;
- la préparation d'un candidat pré-test, sans aucune évaluation finale.

La Phase 5.4 ne modifie ni PostgreSQL, ni les datasets sources, ni les artefacts historiques 5.3. Elle ne développe aucun service, dashboard ou composant d'interface.

<a id="section-2"></a>

## 2. Baseline V1

La référence historique Phase 5.3 reste :

| Élément | Valeur historique |
|---|---:|
| Contrat | `strict_v1.0.0` |
| Modèle | Régression logistique pondérée |
| Train mature au 01/01/2018 | 41 197 commandes |
| Validation janvier-avril | 27 425 commandes |
| Average Precision | 0,153258 |
| ROC-AUC | 0,586930 |
| Seuil candidat | 0,459566 |
| Précision au seuil | 0,1963 |
| Rappel au seuil | 0,5231 |
| F1 au seuil | 0,2855 |

Ces nombres ne sont pas injectés dans les nouvelles expériences. Les résultats 5.4 sont recalculés à partir des probabilités de chaque fold. La comparaison directe avec la valeur globale 5.3 doit rester prudente, car le protocole à trois fenêtres de la Phase 5.4 est plus strict.

<a id="section-3"></a>

## 3. Protection du test final

La frontière de sécurité est fixée à :

```text
purchase_timestamp < 2018-05-01 00:00:00
```

Les protections appliquées sont :

- le chargeur Phase 5.4 lit uniquement `strict_v1_train.parquet` et `strict_v1_validation.parquet` ;
- `strict_v1_test.parquet` est rejeté avant tout appel à `pandas.read_parquet` ;
- les requêtes PostgreSQL contiennent un filtre exclusif avant le 1er mai 2018 ;
- le cutoff n'est pas exposé comme paramètre CLI modifiable ;
- chaque dataset et chaque fold rejette une commande créée à partir du 1er mai ;
- les artefacts enregistrent `test_accessed: false` et `test_evaluated: false` ;
- aucun hash, volume ou profil du Parquet test n'est calculé par le pipeline 5.4.

Les mois d'évaluation sont janvier, février, mars et avril 2018. Leurs labels sont consultés rétrospectivement après observation ; aucune commande achetée de mai à août 2018 n'est chargée ou scorée.

<a id="section-4"></a>

## 4. Audit de disponibilité

La présence d'une valeur dans Olist ou le DW ne fournit pas la date à laquelle cette valeur était disponible dans le système opérationnel. En conséquence, aucune variable enrichie n'est déclarée définitivement disponible à la création.

### 4.1 Variables autorisées sous les hypothèses existantes

Les neuf variables de `strict_v1.0.0` restent autorisées selon les décisions validées précédemment : calendrier d'achat, État client, coordonnées client et flag GPS.

### 4.2 Variables conditionnelles

| Domaine | Source PostgreSQL exacte | Grain source | Transformation | Preuve historique | Décision |
|---|---|---|---|---|---|
| Promesse | `dw.fact_orders.purchase_timestamp`, `estimated_delivery_timestamp` | commande | horizon en jours calendaires | aucune date de fixation de la promesse | Conditionnelle |
| Articles et montants | `dw.fact_order_items.order_item_id`, `price`, `freight_value` | article | compte, sommes, fret/prix | contenu final non horodaté à la création | Conditionnelle |
| Vendeurs | `fact_order_items.seller_key`, `dim_seller.seller_state` | article/vendeur | vendeurs et États distincts | allocation vendeur non horodatée | Conditionnelle |
| Distance | coordonnées `dim_customer` et `dim_seller` | trajet article | Haversine, moyenne, maximum, moyenne pondérée | référentiels non historisés | Conditionnelle |
| Catégorie | `fact_order_items.product_key`, `dim_product` | article/produit | nombre et catégorie dominante | catalogue non historisé | Conditionnelle |
| Poids/volume | poids et dimensions de `dw.dim_product` | article/produit | sommes en grammes et cm³ | catalogue non historisé | Conditionnelle |

### 4.3 Variables exclues

Restent interdites : livraison réelle, remise au transporteur, statut final, approbation postérieure, avis, notes, durées réelles, cible ou dérivés, identifiants bruts et statistiques vendeur/catégorie calculées sur l'historique complet.

Les paiements ne sont pas intégrés : leur disponibilité avant l'approbation n'est pas démontrée et ils ne sont pas nécessaires au prototype demandé.

<a id="section-5"></a>

## 5. Contrat enrichi V2 conditionnel

Le nouveau contrat est :

```text
feature_schema_version = enriched_v2.0.0
operational_status = conditional_not_production_validated
```

Il ajoute 19 variables aux neuf variables V1, soit 28 features :

- `promised_delivery_days` ;
- `item_count`, `total_item_value`, `total_freight_value`, `freight_ratio` ;
- `seller_count`, `seller_state_count`, `is_multi_seller` ;
- `distance_km_mean`, `distance_km_max`, `distance_km_weighted_by_item_value` ;
- `distance_coverage_ratio`, `distance_incomplete` ;
- `category_count`, `dominant_category` ;
- `total_product_weight_g`, `total_product_volume_cm3` ;
- `product_physical_coverage_ratio`, `product_physical_incomplete`.

Le contrat V1 demeure inchangé. V2 l'étend sans le remplacer et conserve la même cible calendaire.

### 5.1 Effectifs du prototype

| Élément | Effectif |
|---|---:|
| Commandes PostgreSQL lues avant cutoff | 73 577 |
| Lignes article lues avant cutoff | 83 306 |
| Commandes supervisées éligibles V1 | 71 118 |
| Commandes supervisées V2 | 71 118 |
| Livraisons à temps | 65 699 |
| Retards | 5 419 |
| Dernier achat | 30 avril 2018 à 23:47:26 |

V1 et V2 couvrent exactement les mêmes `order_id` et les mêmes cibles.

### 5.2 Données manquantes

- coordonnées client : 0,2728 % ;
- distances : 0,5132 % ;
- autres variables finales : pas de valeur nulle, grâce aux flags et ratios de couverture explicites ;
- 367 commandes ont une couverture de distance incomplète ;
- 16 commandes ont une couverture physique produit incomplète.

<a id="section-6"></a>

## 6. Feature engineering

### 6.1 Prévention des produits cartésiens

La requête commandes et la requête articles sont séparées. Les articles sont agrégés au grain `order_id`, validés uniques, puis joints une seule fois aux commandes. Les paiements et avis ne sont jamais joints. Une clé `(order_id, order_item_id)` dupliquée provoque un arrêt.

### 6.2 Commandes multivendeurs

- `seller_count` compte les `seller_id` distincts ;
- `seller_state_count` compte les États distincts ;
- `is_multi_seller` signale plus d'un vendeur ;
- 851 commandes éligibles pré-test sont multivendeurs.

Aucun « premier vendeur » arbitraire n'est choisi.

### 6.3 Distances

La distance client–vendeur utilise Haversine avec un rayon terrestre moyen de `6 371,0088 km`. Les distances article sont agrégées par :

- moyenne simple ;
- maximum ;
- moyenne pondérée par le prix de l'article.

Les routes sans deux coordonnées valides restent absentes. Leur couverture est exposée par un ratio et un flag, sans imputation géographique globale.

### 6.4 Catégorie et caractéristiques physiques

La catégorie dominante est celle dont la somme des prix est la plus élevée. Les égalités sont départagées par nombre d'articles puis ordre lexical. Les catégories manquantes utilisent `unknown`.

Le poids total est exprimé en grammes. Le volume total additionne `longueur × hauteur × largeur` en cm³. Les sommes partielles sont accompagnées d'un ratio de couverture et d'un flag d'incomplétude.

<a id="section-7"></a>

## 7. Stratégie temporelle et calibration

Chaque fold contient trois fenêtres disjointes :

1. **Entraînement du modèle de base** : achats antérieurs au début de calibration et labels observés avant cette même date.
2. **Calibration** : achats des deux mois précédant l'évaluation, avec labels observés avant le mois évalué.
3. **Évaluation** : commandes du mois suivant, jamais utilisées pour entraîner le modèle, le calibrateur ou le seuil.

Les calibrateurs utilisent `FrozenEstimator` : le pipeline de base est déjà ajusté et ne peut pas être réentraîné pendant la calibration.

Méthodes comparées :

- score brut ;
- sigmoid / Platt scaling ;
- isotonic.

L'ECE est défini comme la moyenne pondérée par effectif de `|probabilité moyenne - taux observé|` dans dix bins de fréquence égale. Les autres métriques probabilistes sont Brier, log-loss, Average Precision et ROC-AUC.

Le seuil spécifique à chaque fold est choisi sur la calibration antérieure en maximisant le F1. Il n'est jamais optimisé sur le mois évalué.

<a id="section-8"></a>

## 8. Folds réellement exécutés

| Évaluation | Train mature : lignes / retards | Calibration mature : lignes / retards | Évaluation : lignes / retards |
|---|---:|---:|---:|
| Janvier 2018 | 29 322 / 944 | 10 309 / 628 | 7 069 / 403 |
| Février 2018 | 33 992 / 1 134 | 9 629 / 357 | 6 555 / 926 |
| Mars 2018 | 41 197 / 1 765 | 9 970 / 273 | 7 003 / 1 328 |
| Avril 2018 | 47 794 / 2 387 | 9 735 / 777 | 6 798 / 306 |

Dates exactes :

| Fold | Train as-of exclusif | Calibration achats | Labels calibration observés avant | Évaluation achats |
|---|---|---|---|---|
| Janvier | 01/11/2017 | novembre–décembre 2017 | 01/01/2018 | janvier 2018 |
| Février | 01/12/2017 | décembre 2017–janvier 2018 | 01/02/2018 | février 2018 |
| Mars | 01/01/2018 | janvier–février 2018 | 01/03/2018 | mars 2018 |
| Avril | 01/02/2018 | février–mars 2018 | 01/04/2018 | avril 2018 |

Les évaluations sont réellement hors échantillon pour les features et probabilités. Elles sont rétrospectives pour la cible : certaines commandes pré-mai ont été livrées plusieurs mois après leur achat. Cette observation tardive n'introduit pas de commande du test final, mais empêche de prétendre que la métrique était disponible à l'origine simulée.

<a id="section-9"></a>

## 9. Comparaison des modèles et calibrations

Moyennes non pondérées des quatre folds :

| Contrat / modèle | Calibration | AP | ROC-AUC | Brier | Log-loss | ECE |
|---|---|---:|---:|---:|---:|---:|
| V1 Logistic | Brut | 0,175967 | 0,643123 | 0,227493 | 0,647032 | 0,342672 |
| V1 Logistic | Sigmoid | 0,152103 | 0,581036 | 0,102617 | **0,411455** | **0,086642** |
| V1 Logistic | Isotonic | 0,141622 | 0,590441 | 0,102370 | 0,502932 | 0,084301 |
| V2 Logistic | Brut | **0,185148** | **0,659230** | 0,253645 | 0,708379 | 0,368412 |
| V2 Logistic | Sigmoid | **0,159052** | **0,596088** | 0,104582 | 0,413851 | 0,103270 |
| V2 Logistic | Isotonic | 0,154698 | 0,619252 | **0,102319** | 0,417179 | 0,092754 |
| V2 Random Forest | Brut | 0,177317 | 0,634115 | 0,169889 | 0,525970 | 0,271466 |
| V2 Random Forest | Sigmoid | 0,155117 | 0,579492 | 0,102952 | 0,421011 | 0,091041 |
| V2 Random Forest | Isotonic | 0,139278 | 0,595508 | 0,204975 | 4,086863 | 0,205688 |
| V2 HistGradientBoosting | Brut | 0,183899 | 0,641457 | 0,197547 | 0,580785 | 0,294967 |
| V2 HistGradientBoosting | Sigmoid | 0,157030 | 0,578731 | 0,103104 | 0,424424 | 0,091205 |
| V2 HistGradientBoosting | Isotonic | 0,147907 | 0,601977 | 0,116507 | 0,841664 | 0,109317 |

### 9.1 Lecture V1/V2

À score brut, V2 Logistic améliore l'AP moyenne de `0,009181` et la ROC-AUC de `0,016107` par rapport à V1 Logistic. Il dégrade cependant Brier et log-loss à cause des scores pondérés non calibrés.

Avec sigmoid, V2 gagne `0,006949` d'AP par rapport à V1 mais possède un Brier, une log-loss et une ECE légèrement moins bons. Le contrat enrichi apporte donc un gain de classement modeste, sans domination uniforme sur la calibration.

### 9.2 Sigmoid contre isotonic

Isotonic atteint un Brier moyen légèrement meilleur sur V2 Logistic, mais produit des probabilités exactement `0` ou `1` lors de la préparation initiale du candidat et présente des log-loss extrêmes sur certains modèles. Sigmoid est retenu pour sa meilleure AP calibrée, sa meilleure stabilité paramétrique et l'absence de probabilités extrêmes sur la préparation finale.

<a id="section-10"></a>

## 10. Seuils et charge d'alertes

### 10.1 Seuils choisis avant chaque évaluation

| Mois | Seuil antérieur | Alertes | Part signalée | Précision | Rappel | F1 | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Janvier | 0,081972 | 6 199 | 87,69 % | 0,0616 | 0,9479 | 0,1157 | 0,8726 |
| Février | 0,034348 | 1 227 | 18,72 % | 0,0750 | 0,0994 | 0,0855 | 0,2016 |
| Mars | 0,050559 | 821 | 11,72 % | 0,3739 | 0,2312 | 0,2857 | 0,0906 |
| Avril | 0,111789 | 1 279 | 18,81 % | 0,1384 | 0,5784 | 0,2233 | 0,1697 |

Moyenne : 2 381,5 alertes par mois, soit 34,24 % des commandes. Cette moyenne masque une instabilité majeure, de 11,72 % à 87,69 %.

### 10.2 Seuils fixes

Sur les probabilités calibrées, les seuils `0,5` et `0,459566` ne sont pas comparables au seuil historique appliqué aux scores bruts. Ils signalent respectivement en moyenne seulement 5,75 et 9 commandes par mois, avec moins de 1 % de rappel. Ils sont conservés comme diagnostics mais ne doivent pas être réutilisés comme règles métier après calibration.

Faute de capacité opérationnelle et de coût métier, aucun seuil n'est déclaré optimal pour la production. Les scénarios à conserver sont :

- seuil probabiliste fixe à définir avec une capacité d'alertes réelle ;
- seuil par quantile ou capacité maximale ;
- seuil adapté temporellement, uniquement sur données antérieures et avec garde-fous de variation.

<a id="section-11"></a>

## 11. Robustesse mensuelle et géographique

### 11.1 Candidat sigmoid par mois

| Mois | AP | ROC-AUC | Brier | Log-loss | ECE |
|---|---:|---:|---:|---:|---:|
| Janvier | 0,098998 | 0,626452 | 0,065120 | 0,261968 | 0,099657 |
| Février | 0,105962 | 0,373715 | 0,134015 | 0,527925 | 0,110272 |
| Mars | 0,285818 | 0,620743 | 0,176614 | 0,683691 | 0,159637 |
| Avril | 0,145428 | 0,763441 | 0,042580 | 0,181819 | 0,043512 |

Février présente une ROC-AUC inférieure à `0,5`, alors que mars obtient la meilleure AP et avril la meilleure ROC-AUC. Une moyenne globale ne suffit donc pas à caractériser le risque.

### 11.2 Géographie

Les États avec au moins 200 commandes par mois ont été évalués. Les résultats sont très hétérogènes :

- en janvier, les taux d'alertes de plusieurs grands États dépassent 90 % ;
- en février, BA et SC ont un rappel nul, tandis que MG et PR dépassent 60 % ;
- en mars, BA atteint un F1 de 0,526, tandis que MG, PR et SP restent très faibles ;
- en avril, BA atteint 97,5 % de rappel, mais MG et PR ont un rappel nul.

Ces écarts reflètent à la fois la dérive temporelle, la prévalence régionale et les interactions du modèle. Ils ne prouvent aucune relation causale et interdisent un déploiement uniforme sans monitoring et règles d'équité définies.

<a id="section-12"></a>

## 12. Drift

Le PSI utilise des déciles définis sur le train pour les variables numériques et le support complet pour les catégories. Les seuils `<0,10`, `0,10–0,25` et `>=0,25` sont seulement des hypothèses configurables de monitoring, pas des normes métier validées.

### 12.1 Score brut du candidat

| Fold | PSI score |
|---|---:|
| Janvier | 0,7944 |
| Février | 0,5704 |
| Mars | 0,0074 |
| Avril | 0,6409 |

Le score est stable en mars uniquement. Les autres mois montrent une variation forte.

### 12.2 Features

`purchase_month` et `purchase_quarter` dominent mécaniquement le PSI, car chaque mois évalué contient des valeurs calendaires absentes ou rares dans le passé. Ce drift est structurel et attendu, mais il révèle la sensibilité du modèle aux variables de calendrier.

Hors calendrier :

- `dominant_category` atteint environ `0,16–0,25` selon le mois ;
- `promised_delivery_days` atteint environ `0,045–0,129` ;
- `total_freight_value` atteint `0,180` en mars et `0,265` en avril ;
- les taux de valeurs manquantes sont enregistrés par feature et par fold.

<a id="section-13"></a>

## 13. Candidat recommandé

Le candidat expérimental est :

```text
Contrat       : enriched_v2.0.0 — conditionnel
Modèle        : LogisticRegression class_weight="balanced"
Calibration   : sigmoid / Platt scaling
Statut        : non production-ready
```

La sélection utilise exclusivement les quatre folds pré-test. Parmi les variantes sigmoid/isotonic, elle maximise l'AP moyenne ; dans une tolérance de `0,005`, elle préfère sigmoid si son Brier reste à moins de `0,005` du meilleur, afin d'éviter les extrêmes isotonic observés.

### 13.1 Préparation as-of 1er mai 2018

Le pipeline sauvegardé est préparé sans test :

| Fenêtre | Lignes | Retards |
|---|---:|---:|
| Modèle de base, mature avant le 01/03/2018 | 53 644 | 2 710 |
| Calibration mars-avril, labels observés avant le 01/05/2018 | 11 040 | 1 201 |

Le seuil expérimental calculé sur cette calibration est `0,145733`. Il produit sur la calibration : précision `0,2318`, rappel `0,3622`, F1 `0,2827`, 1 877 alertes et FPR `0,1466`.

Ces métriques sont des diagnostics in-sample pour le calibrateur et le seuil final, pas une nouvelle estimation de généralisation. La performance crédible reste celle des folds antérieurs.

<a id="section-14"></a>

## 14. Artefacts et reproductibilité

Les artefacts séparés de la Phase 5.3 sont sous :

```text
models/delivery_delay/phase_5_4/conditional_v2_backtest_v1/
```

Le dossier est exclu de Git. Il contient :

- contrat V2 et audit de disponibilité ;
- configuration expérimentale ;
- audit des datasets et empreintes ;
- définition exacte des folds ;
- résultats complets des backtests ;
- comparaison modèles/calibrations ;
- robustesse géographique ;
- analyse du drift ;
- candidat sélectionné ;
- pipeline complet `candidate_pipeline.joblib` ;
- manifeste SHA-256.

Commande :

```powershell
.\.venv\Scripts\python.exe -m src.ml.improve_models
```

Une destination existante est refusée sans `--overwrite` explicite.

<a id="section-15"></a>

## 15. Tests et contrôles

La suite ML couvre :

- V1 inchangé et V2 versionné ;
- absence de colonnes interdites ;
- agrégations multivendeurs ;
- Haversine et unités kilométriques ;
- coordonnées et caractéristiques produit manquantes ;
- rejet d'une duplication article simulant un produit cartésien ;
- maturité des labels ;
- disjonction train/calibration/évaluation ;
- reproductibilité des folds, seuils, métriques et PSI ;
- rejet du Parquet test avant lecture ;
- calibration sur une période distincte du train ;
- seuil choisi avant l'évaluation ;
- sauvegarde et rechargement d'un calibrateur fiable.

Contrôles réels :

- V1 et V2 : 71 118 commandes identiques et cibles identiques ;
- maximum de date d'achat : 30 avril 2018 ;
- quatre folds, quatre configurations et trois variantes de calibration évalués ;
- toutes les empreintes et tailles du manifeste vérifiées ;
- pipeline candidat rechargé et métriques de calibration reproduites sur 11 040 lignes ;
- probabilités candidates comprises entre `0,0000324` et `0,617566` ;
- zéro prédiction sur le test final.

<a id="section-16"></a>

## 16. Limites et risques

1. **V2 n'est pas prouvé opérationnellement.** Ses 19 variables restent conditionnelles.
2. **Calibration non stationnaire.** La relation score–cible change fortement entre les périodes.
3. **Seuil instable.** Les seuils antérieurs varient de `0,0343` à `0,1118` et la charge de 11,7 % à 87,7 %.
4. **Février inverse partiellement le classement.** La ROC-AUC candidate tombe à `0,3737`.
5. **Drift géographique et temporel.** Les résultats par État et par mois sont très hétérogènes.
6. **Dimensions non historisées.** Les coordonnées, catégories, poids et dimensions peuvent refléter un état de référentiel ultérieur.
7. **Population conditionnelle à la livraison.** Les commandes finalement annulées ou indisponibles restent hors cible.
8. **Évaluations rétrospectives.** Les labels des mois évalués ne sont pas tous disponibles immédiatement après le mois.
9. **Pas de coût métier ni capacité d'alertes.** Aucun seuil de production ne peut être choisi.
10. **Gain V2 modeste après calibration.** L'amélioration d'AP ne s'accompagne pas d'une amélioration uniforme de Brier, ECE et log-loss.
11. **Aucune évaluation finale.** La généralisation de mai à août 2018 reste inconnue.

<a id="section-17"></a>

## 17. Recommandations avant la Phase 5.5

Avant toute ouverture du test final :

1. confirmer avec le système cible la disponibilité de la promesse, du panier, du vendeur et du catalogue au scoring ;
2. décider si V2 peut devenir un contrat opérationnel ou doit rester exploratoire ;
3. définir une capacité d'alertes ou des coûts relatifs FP/FN ;
4. figer le candidat sigmoid et les scénarios de seuil sans nouveau tuning ;
5. documenter une politique de recalibration et un garde-fou de dérive ;
6. décider si la forte instabilité de février justifie l'abandon ou l'enrichissement avant le test ;
7. n'ouvrir le test final qu'une seule fois, après validation explicite de la présente phase.

La Phase 5.5 n'est pas commencée dans ce travail.
