# Phase 5.4.2 — Correction de calibration et backtests temporels versionnés

> **Décision recommandée : NO-GO pour la Phase 5.5.**
>
> L'inversion de classement est corrigée, mais les sous-fenêtres historiques
> indépendantes contiennent trop peu de retards pour figer une calibration et un
> seuil stables. Aucun nouveau candidat n'est conservé.
>
> Le holdout final n'a été ni lu, ni scoré, ni évalué.

## 1. Cause de l'anomalie

La Phase 5.4 utilisait `CalibratedClassifierCV` avec une calibration sigmoid. La
transformation scikit-learn est équivalente à :

```text
p_calibrée = expit(-(a × score_brut + b))
```

Elle est croissante pour `a < 0`, mais décroissante pour `a > 0`. Sur le fold de
février, la fenêtre de calibration Phase 5.4 avait produit `a=+0,297389` en V1 et
`a=+0,183576` en V2. Le classement avait donc été intégralement inversé.

La classe positive était correcte (`is_delayed=1`) et le modèle de base était le
même. L'erreur méthodologique consistait à supposer qu'une calibration sigmoid
scikit-learn serait toujours croissante.

## 2. Politique corrigée

La politique `monotone_temporal_v1.0.0` est appliquée dans cet ordre :

1. ajuster une sigmoid sur la sous-fenêtre de calibration ;
2. l'accepter uniquement si `a < 0` et si aucune inversion de rang n'est détectée ;
3. sinon, essayer une isotonic non décroissante ;
4. accepter l'isotonic uniquement si elle préserve le rang, produit au moins deux
   niveaux et ne place pas plus de 95 % des lignes dans un même niveau ;
5. sinon, conserver le score brut avec le statut explicite
   `uncalibrated_raw_fallback`.

Le score brut de repli n'est jamais présenté comme calibré. La classe positive
doit être exactement `[0, 1]`. Le garde-fou Phase 5.4.1 est conservé.

### Classement, calibration et discrimination

- La **préservation du classement** interdit qu'une commande ayant un score brut
  supérieur obtienne un score calibré inférieur.
- La **qualité de calibration** mesure la cohérence entre probabilités annoncées
  et fréquences observées, via Brier, log-loss, ECE et bins de calibration.
- Le **pouvoir discriminant** mesure l'ordre relatif des commandes via AP et
  ROC-AUC.

Une transformation strictement croissante préserve AP et ROC-AUC sans garantir
une bonne calibration. Une isotonic non décroissante peut créer des égalités et
réduire AP/ROC sans inverser de paire.

## 3. Protocole temporel corrigé

La fenêtre historique de deux mois est désormais divisée avant l'évaluation :

- premier mois : ajustement du calibrateur ;
- second mois : sélection du seuil ;
- mois suivant : évaluation rétrospective.

Chaque sous-fenêtre exige que `label_observed_at` soit strictement antérieur à sa
date `as_of`. Le prétraitement et le modèle de base restent ajustés uniquement sur
le train antérieur.

| Fold | Train (retards) | Calibration fit (retards) | Seuil (retards) | Évaluation (retards) |
|---|---:|---:|---:|---:|
| Janvier | 29 322 (944) | 3 140 (29) | 3 332 (18) | 7 069 (403) |
| Février | 33 992 (1 134) | 3 332 (18) | 4 185 (10) | 6 555 (926) |
| Mars | 41 197 (1 765) | 4 185 (10) | 3 080 (14) | 7 003 (1 328) |
| Avril | 47 794 (2 387) | 3 080 (14) | 3 504 (167) | 6 798 (306) |

Aucun doublon ou chevauchement n'est présent. V1 et V2 utilisent exactement les
mêmes commandes et cibles. Aucun achat à partir du 1er mai 2018 n'est chargé.

La séparation supplémentaire évite d'utiliser les mêmes observations pour
ajuster simultanément le calibrateur et le seuil. Elle révèle cependant que les
trois premiers folds ont seulement 10 à 29 retards par sous-fenêtre. Une garde
conservatrice de 30 événements positifs minimum par sous-fenêtre est enregistrée
comme critère de fiabilité statistique, et non comme coût métier.

## 4. Modifications du code

### `src/ml/calibration_policy.py`

- partition temporelle train/calibration/seuil/évaluation ;
- politique sigmoid/isotonic/raw versionnée ;
- contrôle de la classe positive ;
- diagnostic Spearman, Kendall, inversions, niveaux uniques et égalités ;
- métadonnées exactes des quatre sous-partitions.

### `src/ml/correct_calibration.py`

- exécution des quatre expériences sans hyperparameter tuning ;
- métriques complètes par fold ;
- comparaison V1/V2 ;
- diagnostic de drift et des volumes d'alertes ;
- sélection de candidat bloquée en cas de support historique insuffisant ;
- écriture atomique dans un nouveau dossier Phase 5.4.2 ;
- manifeste SHA-256 et versions de dépendances.

### Tests

`tests/ml/test_calibration_policy.py` couvre :

- sigmoid croissante acceptée ;
- sigmoid décroissante rejetée ;
- repli brut explicitement non calibré ;
- isotonic non décroissante et égalités ;
- classe positive ;
- séparation temporelle et maturité des labels ;
- reproductibilité ;
- rechargement joblib ;
- refus du candidat lorsque le support temporel est insuffisant.

## 5. Résultats des nouveaux backtests

### 5.1 Classement par modèle et par fold

Les valeurs suivantes sont calculées sur les scores bruts, donc comparables entre
les modèles et non affectées par les égalités isotonic.

| Modèle | Jan AP / ROC | Fév AP / ROC | Mar AP / ROC | Avr AP / ROC |
|---|---:|---:|---:|---:|
| V1 Logistic | 0,095438 / 0,628630 | 0,206428 / 0,624174 | 0,271829 / 0,595718 | 0,130175 / 0,723971 |
| V2 Logistic | 0,098998 / 0,626452 | 0,210345 / 0,626285 | 0,285818 / 0,620743 | 0,145428 / 0,763441 |
| V2 Random Forest | 0,104465 / 0,614262 | 0,199242 / 0,609246 | 0,262997 / 0,587967 | 0,142562 / 0,724984 |
| V2 HistGradientBoosting | 0,090842 / 0,594815 | 0,214536 / 0,625451 | 0,275086 / 0,622729 | 0,155133 / 0,722833 |

Toutes les ROC-AUC brutes sont supérieures à 0,5. La ROC-AUC de février n'est
plus inversée après calibration.

### 5.2 Résultats détaillés du leader diagnostique V2 Logistic

La sigmoid est croissante et retenue sur les quatre folds. AP et ROC-AUC sont
donc exactement préservées.

| Fold | AP | ROC | Brier | Log-loss | ECE | Seuil antérieur | Précision | Rappel | F1 | Balanced accuracy | Alertes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Jan | 0,098998 | 0,626452 | 0,054573 | 0,234229 | 0,036029 | 0,010506 | 0,0610 | 0,9479 | 0,1146 | 0,5328 | 6 263 (88,60 %) |
| Fév | 0,210345 | 0,626285 | 0,135876 | 0,583102 | 0,124454 | 0,027404 | 0,2401 | 0,1641 | 0,1950 | 0,5393 | 633 (9,66 %) |
| Mar | 0,285818 | 0,620743 | 0,188634 | 1,130645 | 0,187165 | 0,003347 | 0,3834 | 0,0941 | 0,1511 | 0,5294 | 326 (4,66 %) |
| Avr | 0,145428 | 0,763441 | 0,044346 | 0,226668 | 0,039445 | 0,008364 | 0,1834 | 0,3824 | 0,2479 | 0,6511 | 638 (9,39 %) |

Matrices de confusion au seuil sélectionné antérieurement :

| Fold | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Jan | 785 | 5 881 | 21 | 382 |
| Fév | 5 148 | 481 | 774 | 152 |
| Mar | 5 474 | 201 | 1 203 | 125 |
| Avr | 5 971 | 521 | 189 | 117 |

Les artefacts contiennent les mêmes métriques complètes pour les quatre modèles,
ainsi que les résultats aux seuils `0,5`, V1 historique et sélectionné sur la
sous-fenêtre antérieure.

## 6. Comparaison avec la Phase 5.4

| Variante V2 Logistic | AP moyenne | ROC moyenne | Brier | Log-loss | ECE |
|---|---:|---:|---:|---:|---:|
| Phase 5.4 sigmoid | 0,159052 | 0,596088 | 0,104582 | 0,413851 | 0,103270 |
| Phase 5.4.2 monotone | 0,185148 | 0,659230 | 0,105857 | 0,543661 | 0,096773 |

La correction restaure exactement le classement brut pour V2 Logistic. Elle ne
garantit pas une meilleure log-loss : le calibrateur ne dispose plus que d'un
mois et de très peu d'événements. Le meilleur chiffre de classement ne doit donc
pas masquer la dégradation de stabilité probabiliste.

Pour V1, la sigmoid est admissible en janvier et février, puis rejetée en mars et
avril. Le repli isotonic préserve l'ordre sans inversion, mais crée suffisamment
d'égalités pour réduire l'AP moyenne de `0,175967` brute à `0,151087` après
politique. Cela confirme que l'isotonic est mathématiquement monotone sans être
nécessairement stable sur de petits échantillons.

## 7. Comparaison V1/V2

| Expérience | AP brute moyenne | ROC brute moyenne | AP politique | Brier politique | ECE politique |
|---|---:|---:|---:|---:|---:|
| V2 Logistic | 0,185148 | 0,659230 | 0,185148 | 0,105857 | 0,096773 |
| V2 HistGradientBoosting | 0,183899 | 0,641457 | 0,183899 | 0,104224 | 0,087828 |
| V2 Random Forest | 0,177317 | 0,634115 | 0,177317 | 0,105086 | 0,093388 |
| V1 Logistic | 0,175967 | 0,643123 | 0,151087 | 0,107166 | 0,101772 |

V2 Logistic reste le leader diagnostique en AP brute, avec un gain modeste de
`+0,009180` contre V1. HistGradientBoosting est à moins de `0,005` d'AP, mais la
régression logistique est moins complexe. Cette hiérarchie n'autorise pas la
préparation d'un candidat car les sous-fenêtres temporelles sont insuffisantes.

Les 19 features V2 restent conditionnelles. Aucune disponibilité opérationnelle
supplémentaire n'a été prouvée et V2 n'est pas déclaré prêt pour la production.

## 8. Analyse des alertes et robustesse temporelle

| Expérience | Taux moyen | Minimum | Maximum | Écart-type |
|---|---:|---:|---:|---:|
| V2 Logistic | 28,07 % | 4,66 % | 88,60 % | 35,00 pp |
| V2 HistGradientBoosting | 23,20 % | 3,20 % | 40,58 % | 13,57 pp |
| V2 Random Forest | 37,14 % | 14,39 % | 69,30 % | 21,68 pp |
| V1 Logistic | 77,68 % | 37,28 % | 100,00 % | 24,50 pp |

Les seuils F1 ne sont pas opérationnellement stables. Aucun coût métier ni aucune
capacité d'alertes n'a été inventé. Le seuil V1 historique et le seuil fixe 0,5
sont conservés dans les artefacts à titre de comparaison, mais aucun n'est choisi
comme nouveau seuil final.

Le PSI des scores V2 Logistic est `0,7944`, `0,5704`, `0,0074` et `0,6409` de
janvier à avril. Les variables calendaires `purchase_month` et
`purchase_quarter` produisent mécaniquement les plus forts PSI de features dans
un backtest mensuel ; ces valeurs doivent être interprétées comme un changement
temporel attendu, non comme une preuve autonome de dérive opérationnelle.

## 9. Résultats de calibration

- Aucune inversion de rang n'est présente dans les résultats 5.4.2.
- V2 Logistic, Random Forest et HistGradientBoosting utilisent une sigmoid
  croissante sur les quatre folds.
- V1 Logistic utilise la sigmoid sur janvier/février et l'isotonic sur
  mars/avril, après rejet des pentes sigmoid positives.
- Les diagnostics enregistrent coefficients, corrélations, inversions, niveaux
  uniques et taille du plus grand groupe lié.
- La sélection de méthode n'utilise aucun label du fold d'évaluation.
- Le seuil utilise uniquement la sous-fenêtre antérieure distincte.

La qualité probabiliste reste variable. Les résultats de calibration ne sont pas
présentés comme une performance indépendante du candidat.

## 10. Choix du candidat

**Aucun nouveau candidat n'est préparé ou conservé.**

Le leader diagnostique serait V2 Logistic, mais la garde de support échoue sur
tous les modèles : seulement 10 retards au minimum dans la calibration et 10 au
minimum dans la sélection de seuil, contre 30 requis par sous-fenêtre. La
variabilité des alertes confirme l'instabilité.

Le fichier final `candidate_pipeline.joblib` n'existe pas et le manifeste porte :

```json
{
  "candidate_prepared": false,
  "test_accessed": false,
  "test_evaluated": false
}
```

## 11. Artefacts

Les nouveaux artefacts sont isolés dans :

```text
models/delivery_delay/phase_5_4_2/monotone_temporal_v1/
```

Contenu :

- `calibration_policy.json` ;
- `experiment_config.json` ;
- `dataset_audit.json` ;
- `fold_definitions.json` ;
- `backtest_results.json` ;
- `model_comparison.json` ;
- `calibration_diagnostics.json` ;
- `threshold_analysis.json` ;
- `drift_analysis.json` ;
- `selected_candidate.json` ;
- `candidate_contract.json`, avec `candidate_prepared=false` ;
- `artifact_manifest.json`.

Les 11 artefacts référencés par le manifeste passent leurs contrôles SHA-256 et
de taille. Deux exécutions diagnostiques ont reproduit toutes les valeurs communes
des backtests avec zéro divergence à `atol=1e-12, rtol=1e-10`.

Les artefacts historiques Phase 5.3 et Phase 5.4 n'ont pas été modifiés.

## 12. Tests de non-régression

Commandes de validation :

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/ml -t . -v
.\.venv\Scripts\python.exe -m compileall -q src/ml tests/ml
.\.venv\Scripts\python.exe -m pip check
git diff --check
```

Le runner `pytest` n'est pas installé et aucun outil supplémentaire n'a été
ajouté.

## 13. Limites restantes

- Support positif insuffisant après séparation correcte calibration/seuil.
- Seuils F1 et volumes d'alertes très instables selon le mois.
- Qualité de calibration variable et log-loss élevée en mars.
- Fort changement de prévalence mensuelle.
- V2 toujours conditionnel du point de vue opérationnel.
- Aucun coût FP/FN ni budget d'alertes validé.
- Aucun résultat de holdout final, volontairement.
- Aucun candidat indépendant disponible pour une évaluation unique et finale.

## 14. Décision GO / NO-GO

**NO-GO pour la Phase 5.5.**

La correction technique de l'inversion est validée, mais elle expose le manque de
support historique nécessaire pour séparer proprement calibration et sélection du
seuil. Ouvrir le holdout final maintenant figerait un candidat dont le seuil et la
calibration ne sont pas suffisamment robustes.

Avant de reconsidérer le GO, il faut valider une stratégie qui ne réutilise pas le
holdout : par exemple une fenêtre historique de calibration plus large ou un seuil
fixe préenregistré, puis rejouer des backtests versionnés. Cette décision doit être
prise avant tout nouvel accès aux données finales.
