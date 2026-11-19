# Phase 5.4.3 — Calibration historique, politiques d'alertes stables et gate expérimental final

> **Décision recommandée : GO sous validation humaine pour une unique évaluation expérimentale du holdout.**
>
> **Le candidat n'est pas prêt pour la production. Ses scores sont explicitement non calibrés.**
>
> Le holdout mai–août 2018 n'a été ni lu, ni scoré, ni évalué.

## 1. Contexte

Les Phases 5.4 à 5.4.2 ont montré que :

- la calibration sigmoid peut apprendre une transformation décroissante ;
- une séparation mensuelle calibration/seuil élimine la fuite mais laisse parfois
  seulement 10 retards par sous-fenêtre ;
- l'optimisation mensuelle du seuil F1 produit des volumes d'alertes instables ;
- V2 améliore modestement le classement, mais ses 19 features enrichies restent
  conditionnelles.

La Phase 5.4.3 constitue l'unique et dernière campagne de développement sur les
folds janvier–avril avant la décision de clôture.

## 2. Problème résiduel de la Phase 5.4.2

La procédure 5.4.2 était temporellement correcte, mais trop fragmentée : les
premiers folds contenaient 10 à 29 retards pour ajuster la calibration ou le
seuil. Le taux d'alertes V2 Logistic variait alors de 4,66 % à 88,60 %.

La présente campagne remplace le seuil F1 mensuel par des capacités top-k
préenregistrées et réunit trois mois historiques pour l'étude de calibration. Il
n'y a aucune recherche de durée de fenêtre.

## 3. Configuration préenregistrée

Le fichier `experiment_config.json` a été écrit avant l'exécution des nouveaux
backtests. Son SHA-256 est :

```text
3c20ab70f9b434f6698f41a1ba2bbc0491f7f420aa4fad5b59b0c84c90eb8c39
```

Choix figés :

- une seule campagne principale ;
- modèles : V1 Logistic et V2 Logistic uniquement ;
- fenêtre historique de calibration : trois mois ;
- stratégies : A0 brute, A1 sigmoid, A2 isotonic ;
- minimum de calibration : 30 retards ;
- politiques top-k : 5 %, 10 % et 20 % ;
- `k = ceil(n × capacité)`, au moins une alerte pour un lot non vide ;
- tri par score décroissant puis `order_id` croissant ;
- pas d'extension du budget en cas d'égalité ;
- politique candidate : Top 10 % par lot mensuel ;
- seul le contrat V1 peut devenir candidat ; V2 reste diagnostique ;
- sigmoid candidate uniquement si admissible sur chaque fold, sinon baseline brute ;
- isotonic diagnostique uniquement.

La configuration n'a pas été modifiée après consultation des résultats.

## 4. Faisabilité des fenêtres historiques

Chaque fold utilise :

- train : achat et label observé avant `origine - 3 mois` ;
- calibration : achat dans les trois mois précédents et label observé avant
  l'origine ;
- évaluation : mois suivant, avec labels consultés rétrospectivement.

| Fold | Train (retards) | Calibration (retards) | Évaluation (retards) | Calibration faisable |
|---|---:|---:|---:|---|
| Janvier | 24 828 (761) | 14 784 (812) | 7 069 (403) | Oui |
| Février | 29 322 (944) | 16 904 (1 248) | 6 555 (926) | Oui |
| Mars | 33 992 (1 134) | 15 472 (673) | 7 003 (1 328) | Oui |
| Avril | 41 197 (1 765) | 16 782 (1 158) | 6 798 (306) | Oui |

Toutes les calibrations dépassent largement le minimum préenregistré de 30
retards. V1 et V2 utilisent les mêmes commandes et cibles. Les partitions sont
uniques, disjointes et leurs labels sont matures avant usage.

## 5. Politique de calibration

### A0 — baseline brute

Le score `predict_proba` de la régression logistique pondérée est conservé avec
le statut :

```text
uncalibrated_raw
```

Il sert au classement, mais n'est pas présenté comme une probabilité fiable.

### A1 — sigmoid historique élargie

La sigmoid n'est admissible que si :

- la classe positive est `1` ;
- le coefficient `a` est négatif ;
- aucune inversion n'est détectée ;
- toutes les valeurs sont finies et comprises entre 0 et 1 ;
- la calibration contient au moins 30 retards.

| Contrat | Jan `a` | Fév `a` | Mar `a` | Avr `a` | Admissible sur tous les folds |
|---|---:|---:|---:|---:|---|
| V1 | -0,405871 | +0,239673 | -0,095587 | -0,884413 | Non |
| V2 | -0,481981 | +0,015189 | +0,015294 | -0,689855 | Non |

La fenêtre élargie fournit suffisamment de retards, mais ne supprime pas
l'instabilité du sens sigmoid. Aucune inversion n'est silencieusement acceptée.

### A2 — isotonic historique élargie

L'isotonic est structurellement admissible sur les huit couples contrat/fold :
monotonie non décroissante, 9 à 30 niveaux selon les cas et plus grand plateau
inférieur à 95 %. Elle crée néanmoins de nombreux scores identiques et réduit le
classement sur plusieurs mois.

Conformément à la configuration préenregistrée, elle reste diagnostique et n'est
pas utilisée dans le candidat.

## 6. Baseline brute

### Résultats V1 par fold

| Fold | Prévalence | AP | ROC-AUC | Brier | Log-loss | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Janvier | 5,70 % | 0,092010 | 0,613688 | 0,435760 | 1,106953 | 0,612432 |
| Février | 14,13 % | 0,193527 | 0,607511 | 0,305990 | 0,816496 | 0,422662 |
| Mars | 18,96 % | 0,248548 | 0,571132 | 0,268346 | 0,734510 | 0,328321 |
| Avril | 4,50 % | 0,137635 | 0,723296 | 0,215664 | 0,623814 | 0,411648 |

La ROC-AUC est supérieure à 0,5 et l'AP dépasse la prévalence sur chaque fold.
Les Brier, log-loss et ECE confirment cependant que les scores bruts pondérés ne
sont pas calibrés.

Agrégations V1 brutes :

| Agrégation | AP | ROC-AUC | Brier | Log-loss | ECE |
|---|---:|---:|---:|---:|---:|
| Macro folds | 0,167930 | 0,628907 | 0,306440 | 0,820443 | 0,443766 |
| Pondérée par lignes | 0,167555 | 0,628514 | 0,307437 | 0,822667 | 0,444757 |
| Prédictions OOS concaténées | 0,130742 | 0,566093 | 0,307437 | 0,822667 | 0,444757 |

Les métriques concaténées diffèrent des moyennes mensuelles car les quatre
modèles sont ajustés à des dates différentes et les prévalences changent. Elles
ne doivent pas être confondues avec une moyenne de folds.

## 7. Comparaison V1 / V2

### Baselines brutes par fold

| Fold | V1 AP / ROC | V2 AP / ROC |
|---|---:|---:|
| Janvier | 0,092010 / 0,613688 | 0,099118 / 0,618351 |
| Février | 0,193527 / 0,607511 | 0,201400 / 0,615919 |
| Mars | 0,248548 / 0,571132 | 0,252486 / 0,584783 |
| Avril | 0,137635 / 0,723296 | 0,149769 / 0,753971 |

| Contrat | AP macro | ROC macro | AP OOS concaténée | ROC OOS concaténée |
|---|---:|---:|---:|---:|
| V1 brut | 0,167930 | 0,628907 | 0,130742 | 0,566093 |
| V2 brut | 0,175693 | 0,643256 | 0,148928 | 0,603668 |

V2 améliore le classement sur les quatre folds, mais l'écart macro reste modeste
(`+0,007764` AP et `+0,014349` ROC). V2 ne devient pas candidat : ses 19 features
supplémentaires ne sont pas certifiées disponibles au scoring.

Les résultats sigmoid agrégés ne couvrent que les folds où la méthode est
admissible : trois folds pour V1 et deux pour V2. Ils ne constituent donc pas une
comparaison complète. L'isotonic couvre les quatre folds, mais ses plateaux
réduisent AP/ROC. Les probabilités issues de calibrateurs différents ne sont pas
nécessairement comparables entre mois ; les métriques concaténées doivent être
interprétées avec cette limite.

## 8. Résultats top-k

### V1 brut — résultats mensuels

| Fold | P@5 / R@5 / Lift@5 | P@10 / R@10 / Lift@10 | P@20 / R@20 / Lift@20 |
|---|---:|---:|---:|
| Janvier | 0,1045 / 0,0918 / 1,833 | 0,0962 / 0,1687 / 1,687 | 0,0870 / 0,3052 / 1,526 |
| Février | 0,1646 / 0,0583 / 1,165 | 0,2012 / 0,1425 / 1,424 | 0,2265 / 0,3207 / 1,604 |
| Mars | 0,3134 / 0,0828 / 1,653 | 0,2839 / 0,1498 / 1,497 | 0,2898 / 0,3057 / 1,528 |
| Avril | 0,2029 / 0,2255 / 4,508 | 0,1647 / 0,3660 / 3,659 | 0,1096 / 0,4869 / 2,434 |

Le lift Top 10 % est supérieur à 1 sur chaque fold, conformément au critère
préenregistré.

### V1 brut — agrégations des lots mensuels

| Capacité | Précision macro | Rappel macro | Lift macro | Alertes | Retards détectés | Fausses alertes | Retards manqués |
|---|---:|---:|---:|---:|---:|---:|---:|
| Top 5 % | 0,1964 | 0,1146 | 2,290 | 1 373 | 270 | 1 103 | 2 693 |
| Top 10 % | 0,1865 | 0,2068 | 2,067 | 2 744 | 511 | 2 233 | 2 452 |
| Top 20 % | 0,1782 | 0,3546 | 1,773 | 5 486 | 975 | 4 511 | 1 988 |

### V2 brut — agrégations des lots mensuels

| Capacité | Précision macro | Rappel macro | Lift macro | Retards détectés |
|---|---:|---:|---:|---:|
| Top 5 % | 0,2310 | 0,1291 | 2,579 | 317 |
| Top 10 % | 0,2030 | 0,2271 | 2,270 | 556 |
| Top 20 % | 0,1710 | 0,3584 | 1,792 | 936 |

V2 est meilleur à 5 % et 10 %, mais pas uniformément à 20 %. Cette observation
ne change pas la décision contractuelle préenregistrée en faveur de V1.

Le top-k mensuel contrôle effectivement le volume : à l'arrondi `ceil` près, le
taux vaut 5 %, 10 % ou 20 % sur chaque fold. Les égalités sont départagées par
`order_id` sans augmenter le budget.

## 9. Portée opérationnelle du top-k

Le Top 10 % candidat est une simulation rétrospective par lot mensuel. Il exige de
disposer de toutes les commandes du lot pour les classer. Il ne permet pas de
prendre une décision indépendante à l'arrivée d'une seule commande.

Une future interface devra choisir explicitement : batch quotidien, batch
horaire, file bornée de commandes ou seuil fixe préenregistré. Aucun de ces modes
temps réel n'est validé par la présente campagne.

## 10. Robustesse temporelle et drift

Le classement V1 reste supérieur au hasard sur chaque mois, mais il est plus
faible en mars (`ROC=0,571132`). La précision et le rappel top-k varient avec la
prévalence et le changement de distribution.

| Fold | PSI score V1 | PSI score V2 | PSI État client | Couverture GPS évaluation |
|---|---:|---:|---:|---:|
| Janvier | 2,7836 | 2,0067 | 0,0132 | 99,80 % |
| Février | 0,9293 | 0,8172 | 0,0102 | 99,65 % |
| Mars | 0,4301 | 0,5089 | 0,0090 | 99,69 % |
| Avril | 0,4097 | 0,0321 | 0,0118 | 99,69 % |

La couverture moyenne des distances V2 reste proche de 99,5 %. Le drift
géographique par État est faible dans ces diagnostics. Les PSI maximaux de
features proviennent de `purchase_month`, ce qui est attendu dans des folds
mensuels en expansion et ne constitue pas une explication causale de la variation
des performances.

Il faut distinguer :

- la prévalence, de 4,50 % à 18,96 % ;
- le drift de features et de scores ;
- la qualité de classement ;
- la calibration, très mauvaise pour les scores bruts.

## 11. Candidat expérimental figé

La campagne prépare :

| Élément | Valeur |
|---|---|
| Version | `strict_v1_logistic_top10_v1` |
| Contrat | `strict_v1.0.0` |
| Modèle | LogisticRegression, `class_weight=balanced` |
| Random state | 42 |
| Features | 9 features V1, ordre contractuel |
| Train | 47 794 commandes, 2 387 retards |
| Cutoff train | achat et label avant le 1er février 2018 |
| Calibration auditée | 17 563 commandes, 2 095 retards |
| Calibration retenue | aucune, score brut `uncalibrated_raw` |
| Politique d'alertes | Top 10 % d'un lot mensuel |
| Arrondi | `ceil` |
| Égalités | `order_id` croissant |
| Référence | avant le 1er mai 2018 |

La sigmoid finale préparée sur février–avril est croissante, mais elle n'est pas
retenue car la sigmoid n'était pas admissible sur tous les folds de développement.
Le candidat sauvegardé est donc le pipeline V1 brut ajusté uniquement sur le
train. Ses scores ne sont pas des probabilités calibrées.

Le pipeline rechargé reproduit exactement les prédictions autorisées et expose
les classes `[0, 1]` ainsi que les neuf features dans le bon ordre.

## 12. Critères GO / NO-GO

| Critère préenregistré | Résultat |
|---|---|
| Aucun accès holdout | Réussi |
| Partitions valides et disjointes | Réussi |
| Calibration élargie faisable | Réussi |
| ROC V1 brute > 0,5 sur chaque fold | Réussi |
| AP V1 brute > prévalence sur chaque fold | Réussi |
| Lift V1 Top 10 % > 1 sur chaque fold | Réussi |
| Aucune sigmoid inversée acceptée | Réussi |
| Candidat et politique figés | Réussi |
| Artefacts reproductibles et manifestés | Réussi |

La décision est donc :

**GO sous validation humaine pour une unique évaluation expérimentale du
holdout en Phase 5.5.**

Ce GO ne signifie ni qualité suffisante pour la production, ni calibration
probabiliste, ni déployabilité temps réel de la politique top-k.

## 13. Incertitudes et limites

- Population rétrospective conditionnelle aux commandes finalement livrées.
- Score brut fortement mal calibré.
- Performance mensuelle variable et ROC faible en mars.
- Faible rappel : Top 10 % détecte en moyenne environ 20,7 % des retards.
- Top-k mensuel non utilisable tel quel pour une commande isolée.
- V2 non certifié opérationnel malgré un meilleur classement.
- Drift de score élevé sur plusieurs folds.
- Aucun coût métier FP/FN ou budget opérationnel réel validé.
- Le holdout reste inconnu ; aucun résultat final n'est anticipé.

## 14. Règle d'arrêt

La campagne principale a été exécutée une seule fois. Les fenêtres, modèles,
capacités top-k et critères n'ont pas été modifiés après les résultats.

Il n'y aura pas de Phase 5.4.4 ni de nouvelle recherche de métriques sur les folds
janvier–avril. Après validation humaine, les seules issues autorisées sont :

- ouvrir une seule fois le holdout avec le candidat figé ;
- ou refuser la Phase 5.5 et clôturer la recherche sur ces données.

Le holdout ne doit jamais être ouvert pour résoudre un éventuel NO-GO.

## 15. Tests exécutés

Les validations exécutées sont :

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/ml -t . -v
.\.venv\Scripts\python.exe -m compileall -q src/ml tests/ml
.\.venv\Scripts\python.exe -m pip check
git diff --check
```

Elles couvrent fenêtres historiques, maturité, disjonction, monotonie, fallback,
classe positive, top-k, égalités, budgets, sérialisation, holdout et règle d'arrêt.

- 38 tests ML réussis ;
- compilation de `src/ml` et `tests/ml` réussie ;
- `pip check` sans dépendance cassée ;
- `git diff --check` réussi ;
- rechargement du candidat avec prédictions identiques ;
- manifeste Phase 5.4.3 vérifié intégralement.

## 16. Artefacts et manifestes

Les artefacts sont isolés dans :

```text
models/delivery_delay/phase_5_4_3/historical_calibration_alert_policy_v1/
```

Contenu :

- `experiment_config.json` ;
- `feasibility_report.json` ;
- `temporal_folds.json` ;
- `calibration_comparison.json` ;
- `ranking_metrics.json` ;
- `topk_policy_results.json` ;
- `drift_analysis.json` ;
- `decision_report.json` ;
- `dependency_versions.json` ;
- `candidate_pipeline.joblib` ;
- `candidate_metadata.json` ;
- `artifact_manifest.json`.

Le manifeste référence 11 artefacts avec taille et SHA-256. Les artefacts Phase
5.3, 5.4 et 5.4.2 restent séparés et n'ont pas été écrasés.

## 17. Décision finale recommandée

**GO expérimental pour la Phase 5.5, sous réserve d'une validation humaine
explicite.**

Le candidat et la politique Top 10 % sont désormais figés. La Phase 5.5 ne doit
pas commencer automatiquement et toute évaluation finale devra être unique,
strictement limitée au candidat `strict_v1_logistic_top10_v1`.
