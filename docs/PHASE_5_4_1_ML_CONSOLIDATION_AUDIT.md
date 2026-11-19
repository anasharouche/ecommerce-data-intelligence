# Phase 5.4.1 — Audit de consolidation ML avant évaluation finale

> **Décision recommandée : NO-GO pour la Phase 5.5.**
>
> Le holdout final n'a été ni lu, ni scoré, ni évalué. Aucun artefact historique n'a été modifié.
>
> Date de l'audit : 18 novembre 2026.

## 1. Objectifs et périmètre

Cet audit vérifie, sans nouvelle recherche de modèle :

- la calibration des probabilités et la conservation du classement ;
- le recalcul des métriques enregistrées ;
- la séparation temporelle des quatre backtests ;
- la maturité des labels et l'origine des seuils ;
- la reproductibilité du candidat pré-test au 1er mai 2018 ;
- l'intégrité des artefacts des Phases 5.3 et 5.4 ;
- la protection du holdout final ;
- la modification Power BI signalée par Git.

L'audit utilise uniquement les Parquets `strict_v1_train.parquet` et
`strict_v1_validation.parquet`, ainsi que des requêtes PostgreSQL en transaction
`READ ONLY` avec le cutoff strict `purchase_timestamp < 2018-05-01`. Le fichier
`strict_v1_test.parquet` n'a pas été ouvert et aucune commande créée à partir du
1er mai 2018 n'a été chargée.

## 2. Méthodologie

Les contrôles ont été exécutés à partir du code et des artefacts présents dans le
repository :

1. reconstruction des mêmes populations V1 et V2 pré-test ;
2. reproduction des quatre folds et des quatre expériences sauvegardées ;
3. comparaison, commande par commande, des probabilités brutes et calibrées ;
4. recalcul de tous les blocs de métriques et comparaison récursive avec
   `backtest_results.json` ;
5. rechargement de `candidate_pipeline.joblib` et reproduction des probabilités
   autorisées de calibration ;
6. recalcul de tous les SHA-256 déclarés par les deux manifestes ;
7. inspection du diff Git Power BI sans modification du fichier.

La tolérance de comparaison des nombres est :

```text
atol = 1e-12
rtol = 1e-10
```

## 3. Audit de calibration

### 3.1 Implémentation réelle

La Phase 5.4 entraîne le pipeline de base sur le train puis utilise :

```python
CalibratedClassifierCV(
    FrozenEstimator(base_pipeline),
    method="sigmoid",
    ensemble=False,
)
```

Le modèle de base est donc bien le même avant et après calibration. Il n'est pas
réentraîné par le calibrateur. Les probabilités comparées appartiennent aux mêmes
commandes, dans le même ordre. La classe positive est correctement encodée `1`
pour `is_delayed`, et `predict_proba(... )[:, 1]` est utilisé partout.

### 3.2 Cause exacte de l'inversion de février

La calibration sigmoid de scikit-learn n'impose pas une pente croissante. Sa
transformation est équivalente à :

```text
p_calibrée = expit(-(a × score_brut + b))
```

Elle est croissante si `a < 0`, mais décroissante si `a > 0`. Sur la fenêtre de
calibration du fold de février, le score brut est anti-corrélé à la cible :

| Contrat | AP brute calibration | ROC-AUC brute calibration | Coefficient `a` | Sens appris |
|---|---:|---:|---:|---|
| V1 Logistic | 0,031176 | 0,449224 | +0,297389 | décroissant |
| V2 Logistic | 0,027203 | 0,384224 | +0,183576 | décroissant |

Le calibrateur apprend donc légitimement, au sens de son optimisation interne,
une fonction décroissante. Cela inverse le classement sur l'évaluation de février :

| Contrat | Spearman | Kendall | Paires inversées | Paires liées | Total paires |
|---|---:|---:|---:|---:|---:|
| V1 Logistic | -1 | -1 | 21 480 565 | 170 | 21 480 735 |
| V2 Logistic | -1 | -1 | 21 480 727 | 8 | 21 480 735 |

La preuve supplémentaire est `ROC_brute + ROC_sigmoid = 1` à la précision
numérique près :

| Contrat | AP brute février | AP sigmoid février | ROC brute février | ROC sigmoid février |
|---|---:|---:|---:|---:|
| V1 Logistic | 0,206428 | 0,110972 | 0,624174 | 0,375826 |
| V2 Logistic | 0,210345 | 0,105962 | 0,626285 | 0,373715 |

Sur janvier, mars et avril, `a < 0`, Spearman et Kendall valent `+1`, et AP/ROC
sont exactement conservées. L'écart des moyennes Phase 5.4 ne vient donc ni
d'une classe positive inversée, ni de modèles différents, ni de populations
différentes : il provient entièrement du fold de février et de la pente sigmoid
non contrainte.

Les calibrations sigmoid des deux autres familles V2 reproduisent le même sens
décroissant en février. La calibration isotonic reste non décroissante ; elle peut
créer beaucoup d'égalités et donc modifier AP/ROC. Pour le HistGradientBoosting
de février, sa sortie isotonic est constante, ce qui rend la corrélation de rang
indéfinie. Ce comportement est cohérent avec une relation de calibration locale
opposée au classement brut, mais il révèle une forte instabilité temporelle.

### 3.3 Correction locale de fiabilité

Une garde explicite a été ajoutée au code de backtest :

- vérification que les classes du pipeline sont exactement `[0, 1]` ;
- comparaison des scores bruts et calibrés sur la même calibration ;
- levée de `BacktestError` dès qu'une calibration inverse deux niveaux de score
  brut distincts.

Cette garde ne modifie aucun artefact historique et ne fabrique pas de nouvelle
calibration. Elle empêche seulement qu'une inversion soit silencieusement utilisée
dans un futur rapport. Un test synthétique couvre les transformations croissante
et décroissante. Une régénération corrigée des résultats devra être versionnée
dans un répertoire distinct après validation ; les artefacts actuels ne doivent
pas être écrasés.

## 4. Audit des métriques

L'audit a recalculé les 192 blocs de métriques contenus dans les backtests :

- Average Precision ;
- ROC-AUC ;
- Brier score ;
- log-loss ;
- ECE ;
- précision, rappel, F1 et balanced accuracy ;
- faux positifs et matrices de confusion ;
- nombre et taux d'alertes ;
- seuil `0,5`, seuil V1 historique et seuil sélectionné sur calibration.

Tous les blocs des régressions logistiques V1/V2 et du
HistGradientBoosting V2 sont identiques aux valeurs sauvegardées à la tolérance
annoncée. Les métriques d'évaluation du Random Forest sont également reproduites.

Onze écarts numériques ont été observés uniquement dans les métriques de la
partition de calibration du Random Forest isotonic : AP, ROC-AUC et ECE, selon le
fold. Ils restent faibles (ROC-AUC jusqu'à environ `6,64e-4`, AP jusqu'à environ
`1,59e-4`, ECE jusqu'à environ `1,60e-5`) et ne touchent ni les artefacts stockés,
ni le candidat sélectionné, ni les métriques d'évaluation sauvegardées. Ils montrent
toutefois que la reproductibilité bit à bit des diagnostics Random Forest/isotonic
n'est pas garantie malgré `random_state=42` et doit être améliorée avant une
publication définitive.

### Convention ECE

L'ECE sauvegardée est calculée avec :

- 10 bins de fréquences égales ;
- `pandas.qcut(..., duplicates="drop")` ;
- au plus autant de bins que de probabilités uniques ;
- pondération de l'écart absolu calibration/observation par le nombre de lignes ;
- suppression implicite des bins vides ;
- un seul bin si toutes les probabilités sont identiques.

Cette convention a été utilisée à l'identique pendant le recalcul.

## 5. Audit temporel

V1 et V2 couvrent exactement les mêmes 71 118 `order_id`, avec les mêmes cibles.
Chaque partition est unique par commande, V1 et V2 utilisent les mêmes commandes,
et aucun chevauchement train/calibration/évaluation n'a été trouvé.

| Fold | Train (retards) | Calibration (retards) | Évaluation (retards) | Origine d'évaluation |
|---|---:|---:|---:|---|
| Janvier | 29 322 (944) | 10 309 (628) | 7 069 (403) | 2018-01-01 |
| Février | 33 992 (1 134) | 9 629 (357) | 6 555 (926) | 2018-02-01 |
| Mars | 41 197 (1 765) | 9 970 (273) | 7 003 (1 328) | 2018-03-01 |
| Avril | 47 794 (2 387) | 9 735 (777) | 6 798 (306) | 2018-04-01 |

Les fenêtres sont :

| Fold | Train, achat et label observé avant | Calibration, achat | Label calibration observé avant | Évaluation |
|---|---|---|---|---|
| Janvier | 2017-11-01 | [2017-11-01, 2018-01-01) | 2018-01-01 | [2018-01-01, 2018-02-01) |
| Février | 2017-12-01 | [2017-12-01, 2018-02-01) | 2018-02-01 | [2018-02-01, 2018-03-01) |
| Mars | 2018-01-01 | [2018-01-01, 2018-03-01) | 2018-03-01 | [2018-03-01, 2018-04-01) |
| Avril | 2018-02-01 | [2018-02-01, 2018-04-01) | 2018-04-01 | [2018-04-01, 2018-05-01) |

Les maxima de `label_observed_at` du train et de la calibration sont toujours
strictement antérieurs à leur date `as_of`. Le prétraitement est inclus dans le
pipeline entraîné uniquement sur le train. Les seuils candidats sont calculés
exclusivement sur la fenêtre de calibration antérieure, puis appliqués sans
réoptimisation à l'évaluation.

Les labels des fenêtres d'évaluation sont utilisés rétrospectivement et peuvent
être observés plusieurs mois plus tard, jusqu'au 19 septembre 2018 pour février.
Ce n'est pas une fuite dans le fit ou le choix du seuil : ces lignes ne servent
qu'à mesurer le résultat après maturité. Ce point doit rester explicite pour ne
pas présenter les métriques comme disponibles à l'origine simulée.

## 6. Audit du candidat préparé au 1er mai 2018

Le pipeline sauvegardé a été rechargé, sans réentraînement :

| Élément | Résultat vérifié |
|---|---|
| Contrat | `enriched_v2.0.0`, conditionnel |
| Features | 28, liste et ordre exacts |
| Train | 53 644 commandes, 2 710 retards |
| Achats train | 2016-09-15 12:16:38 à 2018-02-27 09:53:16 |
| Label train maximal | 2018-02-28 23:58:37 |
| Calibration | 11 040 commandes, 1 201 retards |
| Achats calibration | 2018-03-01 00:00:00 à 2018-04-27 13:06:08 |
| Label calibration maximal | 2018-04-30 23:58:51 |
| Chevauchement | 0 commande |
| Classes | `[0, 1]` |
| Random state du modèle de base | 42 |
| Coefficient sigmoid `a` | -0,731344, donc croissant |
| Seuil reproduit | 0,14573297927220022 |

Les probabilités rechargées sont finies, comprises entre `0,0000324` et
`0,6175664`. Le seuil, les métriques et les bins de calibration sont reproduits
exactement ; le delta numérique maximal est nul.

Les valeurs annoncées sur ces 11 040 lignes — AP `0,186499`, ROC-AUC `0,625356`,
Brier `0,094337`, log-loss `0,332974`, ECE `0,020380`, précision `0,231753`,
rappel `0,362198`, F1 `0,282651`, 1 877 alertes et matrice
`tn=8397, fp=1442, fn=766, tp=435` — sont des diagnostics sur l'ensemble qui a
servi à ajuster le calibrateur et le seuil. Elles ne constituent pas une
performance hors échantillon du candidat final.

Le candidat est donc techniquement cohérent et reproductible, mais sa sélection
repose sur une comparaison de backtest contaminée par l'inversion de février. Sa
validation doit être suspendue avant tout accès au holdout final.

## 7. Comparaison V1 versus V2

La comparaison est appariée correctement : mêmes commandes, mêmes cibles, mêmes
dates, mêmes folds et mêmes définitions de métriques. Les moyennes brutes sont :

| Variante | AP moyenne | ROC-AUC moyenne |
|---|---:|---:|
| V1 Logistic brute | 0,175967 | 0,643123 |
| V2 Logistic brute | 0,185148 | 0,659230 |
| V1 Logistic sigmoid sauvegardée | 0,152103 | 0,581036 |
| V2 Logistic sigmoid sauvegardée | 0,159052 | 0,596088 |

Le gain brut V2 est modeste : `+0,009180` en AP et `+0,016107` en ROC-AUC. Il
n'autorise pas une déclaration de supériorité robuste car :

- la comparaison calibrée utilise une pente décroissante sur février ;
- les taux de retard mensuels changent fortement, de 2,74 % sur une calibration
  à 18,96 % sur une évaluation ;
- les 19 features supplémentaires restent conditionnelles tant que leur
  disponibilité opérationnelle au moment du scoring n'est pas prouvée ;
- aucun résultat de holdout final n'est disponible, volontairement.

Une calibration croissante devrait conserver AP et ROC-AUC de chaque score brut ;
la sélection doit donc être recalculée dans une version distincte avant de
conclure.

## 8. Vérification des artefacts et de la reproductibilité

| Artefacts | Fichiers contrôlés | SHA-256 et taille |
|---|---:|---|
| Phase 5.3 historique | 10 | 10/10 conformes |
| Phase 5.4 conditionnelle | 11 | 11/11 conformes |

Les versions de l'environnement courant correspondent à
`experiment_config.json` : Python 3.12.10, NumPy 2.5.4, pandas 3.1.1,
scikit-learn 1.9.1, SciPy 1.18.1 et joblib 1.6.0. Les contrats sont
`strict_v1.0.0` et `enriched_v2.0.0`. La liste et l'ordre des 28 features du
candidat correspondent au contrat V2.

Les artefacts historiques n'ont pas été écrasés. Si la Phase 5.4 est recalculée,
elle doit produire un répertoire versionné distinct, avec manifeste propre et
traçabilité de la correction.

## 9. Protection du test final

Les protections vérifiées sont :

- rejet de tout chemin contenant `test` avant `pandas.read_parquet` ;
- chargeur 5.4 limité à `strict_v1_train.parquet` et
  `strict_v1_validation.parquet` ;
- cutoff PostgreSQL codé en dur et exclusif au 1er mai 2018 ;
- transaction d'extraction en lecture seule ;
- rejet par les validateurs de toute ligne postérieure au cutoff ;
- tests de garde sur données synthétiques ;
- champs `test_accessed: false` et `test_evaluated: false` dans les artefacts.

Cet audit n'a lu aucune statistique, aucun identifiant, aucun hash et aucune ligne
du holdout final.

## 10. Vérification Power BI

Le diff exact de
`powerbi/ecommerce_data_intelligence.Report/definition/pages/pages.json` est :

```diff
-  "activePageName": "4d8e1f6a9b2c7350e4d7"
+  "activePageName": "afccc91401495a570048"
```

`pageOrder` n'est pas modifié. Le dossier de page
`afccc91401495a570048` existe. Il s'agit uniquement d'un changement de page
active du rapport, indépendant du code et des artefacts ML de Phase 5.4. Le
fichier n'a été ni restauré, ni modifié par cet audit.

## 11. Anomalies et corrections

### Anomalies confirmées

1. La calibration sigmoid de février est décroissante pour toutes les familles
   auditées et inverse le classement.
2. La politique de sélection utilise l'AP moyenne après calibration ; elle est
   donc sensible à cette inversion et ne permet pas de valider le candidat.
3. Les métriques du candidat préparé sont des métriques de calibration, non une
   évaluation hors échantillon.
4. Onze petites divergences de reproductibilité touchent les diagnostics de
   calibration Random Forest/isotonic.
5. Les variables V2 sont toujours conditionnelles du point de vue opérationnel.

### Correction appliquée

Seule une garde locale de fiabilité a été ajoutée à `src/ml/backtesting.py`, avec
un test de non-régression synthétique dans `tests/ml/test_backtesting.py`. Aucun
modèle, seuil, contrat, dataset ou artefact n'a été régénéré.

### Correction recommandée après validation

1. définir explicitement que la calibration admise doit être non décroissante ;
2. rejeter une sigmoid à pente inversée ou utiliser une méthode contrainte/fallback
   documenté, sans choisir ce fallback sur le holdout ;
3. recalculer les quatre backtests sous une nouvelle version d'artefacts ;
4. séparer les métriques de classement, évaluées sur le score brut ou une
   transformation croissante, des métriques de calibration ;
5. enregistrer les identifiants et probabilités de fold dans un artefact d'audit
   versionné afin d'éviter un réentraînement lors des futurs recalculs ;
6. valider la reproductibilité Random Forest avec exécution déterministe avant
   de publier ses diagnostics.

## 12. Tests et contrôles exécutés

Les contrôles suivants ont réussi pendant l'audit :

- reproduction commande par commande V1/V2 ;
- contrôle des quatre folds, maturité des labels et absence de chevauchement ;
- recalcul des 192 blocs de métriques ;
- rechargement et reproduction du candidat ;
- 21/21 hashes d'artefacts conformes ;
- versions de dépendances conformes ;
- test ciblé `tests.ml.test_backtesting` : 5 tests réussis ;
- suite ML complète `unittest` : 26 tests réussis ;
- compilation de `src/ml` et `tests/ml` réussie ;
- `pip check` : aucune dépendance cassée ;
- `git diff --check` réussi.

Le runner `pytest` n'est pas installé dans l'environnement et n'a pas été ajouté,
conformément à l'interdiction d'installer un outil. La suite utilise
`unittest` et doit être exécutée ainsi :

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/ml -t . -v
```

## 13. Risques résiduels

- Le candidat final n'a pas encore de mesure véritablement hors échantillon.
- La stratégie corrigée de calibration n'est pas encore choisie ni backtestée.
- La faible robustesse mensuelle et le changement de prévalence restent élevés.
- La disponibilité à l'instant de scoring des 19 features V2 n'est pas prouvée.
- Le Random Forest/isotonic n'est pas reproductible au niveau de la tolérance
  numérique stricte sur tous ses diagnostics de calibration.
- Le changement Power BI demeure dans le worktree et doit être géré séparément.

## 14. Décision recommandée

**NO-GO pour la Phase 5.5.**

Cette décision ne signifie pas que le candidat sauvegardé est corrompu : il se
recharge et reproduit exactement sa préparation. Elle signifie que la procédure
qui l'a sélectionné n'est pas suffisamment fiable pour justifier l'ouverture
irréversible du holdout final.

Le GO pourra être reconsidéré lorsque :

1. la règle de calibration non décroissante aura été validée ;
2. les quatre backtests auront été recalculés dans des artefacts versionnés sans
   écraser la Phase 5.4 historique ;
3. le choix V1/V2 et du candidat aura été refait sans inversion de rang ;
4. les 19 features V2 auront reçu une décision explicite de disponibilité ;
5. les contrôles automatisés et les manifestes de la version corrigée seront
   entièrement verts.
