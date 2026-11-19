# Phase 5.5 — Évaluation finale du holdout

## 1. Objectif

Cette phase réalise l'unique évaluation finale du candidat figé en Phase 5.4.3 sur le holdout temporel de mai à août 2018. Elle mesure la généralisation sans réentraînement, recalibration, modification des features ou optimisation de la politique d'alerte.

L'évaluation est rétrospective. Elle ne transforme pas le holdout en nouvelle partition de validation et n'autorise aucune seconde campagne.

## 2. Candidat figé

Le candidat évalué est `strict_v1_logistic_top10_v1` :

- contrat : `strict_v1.0.0` ;
- modèle : `LogisticRegression` avec `class_weight="balanced"` ;
- `random_state` : 42 ;
- features : les 9 features V1 dans leur ordre contractuel ;
- calibration : aucune ;
- statut des scores : `uncalibrated_raw` ;
- politique : Top 10 % par lot mensuel, avec `ceil(n × 0,10)` ;
- départage : score décroissant puis `order_id` croissant ;
- entraînement : 47 794 commandes, dont 2 387 retards.

Le hash SHA-256 du pipeline est :

`92e0d37821dc16c16063d7600ed6ee4488642c50e2496e139482ac83b45a6e48`

Le candidat historique n'a été ni modifié ni resauvegardé.

## 3. Protocole d'évaluation

Le protocole a été écrit avant l'ouverture du holdout dans `evaluation_protocol.json`. Son hash SHA-256 est :

`435bb744b9f8d2a409ca3889e2c44ec78ab995fb1689849643d20996998c8e22`

Il fixe la population, la classe positive `is_delayed = 1`, les métriques, la politique Top 10 %, les règles d'égalité, la gestion des valeurs manquantes, les agrégations mensuelles, l'analyse de drift, le bootstrap et la règle d'arrêt. Le hash a été vérifié avant et après l'évaluation.

Le fichier `evaluation_opened.json` constitue le marqueur irréversible d'ouverture. Le code refuse toute nouvelle exécution dès que ce marqueur existe.

## 4. Vérifications préflight

Tous les contrôles ont réussi avant la lecture du holdout :

- manifeste Phase 5.4.3 et tous ses hashes valides ;
- hash de configuration conforme à `3c20ab70f9b434f6698f41a1ba2bbc0491f7f420aa4fad5b59b0c84c90eb8c39` ;
- hash du pipeline conforme au manifeste ;
- version du candidat, contrat et ordre des 9 features conformes ;
- classe positive et classes `[0, 1]` conformes ;
- pipeline, `random_state=42` et `class_weight="balanced"` conformes ;
- absence de calibration confirmée ;
- capacité, arrondi et départage Top 10 % conformes ;
- rechargement reproductible ;
- scoring avec valeurs GPS manquantes supporté ;
- absence de `fit` et de modification de l'état du pipeline pendant le scoring.

Le holdout n'a pas été utilisé par le préflight ou les tests automatisés.

## 5. Population du holdout

Le holdout a été lu une seule fois depuis `strict_v1_test.parquet` :

| Indicateur | Valeur |
|---|---:|
| Commandes | 25 352 |
| Retards | 1 115 |
| Commandes à temps | 24 237 |
| Prévalence | 4,3981 % |
| Première création | 2018-05-01 00:02:11 |
| Dernière création | 2018-08-29 15:00:37 |
| Dernier label observé | 2018-10-17 13:22:46 |

Le nombre réel correspond à l'attente diagnostique de 25 352 sans avoir été imposé. Les `order_id` sont uniques, le chevauchement avec les partitions de développement est nul, l'ordre des colonnes est conforme et tous les types persistés respectent le contrat. Les coordonnées sont manquantes pour 0,2801 % des lignes ; cette absence est cohérente avec `customer_has_gps` et est traitée par le pipeline figé.

## 6. Métriques globales

| Métrique | Résultat |
|---|---:|
| Average Precision | 0,052363 |
| ROC-AUC | 0,561720 |
| Brier score | 0,184457 |
| Log-loss | 0,556274 |
| ECE | 0,360628 |
| Prévalence | 0,043981 |

L'Average Precision ne dépasse que modestement la prévalence et le ROC-AUC indique une discrimination faible. Le score moyen vaut 0,4046 alors que la prévalence vaut 0,0440. Cette divergence, l'ECE élevée et le Brier score confirment que les sorties brutes du modèle équilibré ne doivent pas être interprétées comme des probabilités fiables. Ces trois métriques de calibration sont uniquement diagnostiques.

Aucun seuil à 0,5 n'est utilisé comme politique opérationnelle.

## 7. Résultats mensuels

| Mois | Lignes | Retards | Prévalence | AP | ROC-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mai | 6 749 | 443 | 6,5639 % | 0,088622 | 0,547960 | 0,155639 | 0,292858 |
| Juin | 6 096 | 71 | 1,1647 % | 0,034320 | 0,730855 | 0,171621 | 0,385928 |
| Juillet | 6 156 | 208 | 3,3788 % | 0,062679 | 0,590563 | 0,191479 | 0,383559 |
| Août | 6 351 | 393 | 6,1880 % | 0,059729 | 0,519629 | 0,220594 | 0,386132 |

Les quatre mois contiennent les deux classes. La prévalence et la discrimination varient fortement, avec une performance proche de l'aléatoire en août.

## 8. Politique Top 10 %

L'agrégation globale concatène les alertes sélectionnées indépendamment dans chaque mois. Elle ne doit pas être confondue avec une moyenne macro mensuelle.

| Indicateur global sur alertes concaténées | Résultat |
|---|---:|
| Alertes | 2 537 |
| Taux d'alerte | 10,0071 % |
| Retards détectés | 165 |
| Faux positifs | 2 372 |
| Retards manqués | 950 |
| Precision@10 % | 0,065037 |
| Recall@10 % | 0,147982 |
| Lift@10 % | 1,478771 |

| Mois | Alertes | Vrais positifs | Precision@10 % | Recall@10 % | Lift@10 % |
|---|---:|---:|---:|---:|---:|
| Mai | 675 | 75 | 0,111111 | 0,169300 | 1,692751 |
| Juin | 610 | 26 | 0,042623 | 0,366197 | 3,659571 |
| Juillet | 616 | 35 | 0,056818 | 0,168269 | 1,681600 |
| Août | 636 | 29 | 0,045597 | 0,073791 | 0,736869 |

Le budget est respecté exactement, sans expansion en cas d'égalité. Le lift global reste supérieur à 1, mais l'instabilité est importante et le lift d'août inférieur à 1 signifie que la sélection mensuelle est alors moins efficace qu'une sélection aléatoire de même taille.

Les moyennes macro mensuelles sont : Precision@10 % 0,064037, Recall@10 % 0,194389 et Lift@10 % 1,942698.

## 9. Analyse du drift

Le PSI des scores vaut 0,8411, ce qui constitue un signal descriptif de drift élevé selon les seuils de monitoring retenus. Le PSI maximal de feature concerne `purchase_month` (9,0533), suivi de `purchase_quarter` (4,8731). Ces valeurs extrêmes proviennent en grande partie de la comparaison structurelle entre une référence d'entraînement antérieure à février et un holdout limité à mai–août ; elles reflètent le calendrier et la saisonnalité autant qu'un changement de population.

Les indicateurs géographiques sont beaucoup plus stables :

- PSI de `customer_state` : 0,02015 ;
- PSI latitude : 0,01700 ;
- PSI longitude : 0,01631 ;
- couverture GPS de référence : 99,7510 % ;
- couverture GPS du holdout : 99,7199 %.

La part de São Paulo augmente de 40,25 % à 46,34 %, mais le PSI global des États reste faible. La variation de performance ne peut donc pas être attribuée uniquement à un drift géographique. Les variations mensuelles de prévalence et le drift des scores sont les signaux principaux.

La référence de drift est exclusivement la population d'entraînement figée de 47 794 lignes. Le holdout n'a jamais servi de référence.

## 10. Comparaison avec le développement

| Métrique | Phase 5.4.3, macro folds | Holdout final | Écart descriptif |
|---|---:|---:|---:|
| AP | 0,167930 | 0,052363 | -0,115567 |
| ROC-AUC | 0,628907 | 0,561720 | -0,067187 |
| Precision@10 % | 0,186500 | 0,065037 | -0,121463 |
| Recall@10 % | 0,206800 | 0,147982 | -0,058818 |
| Lift@10 % | 2,067000 | 1,478771 | -0,588229 |

Les références de développement sont des moyennes macro sur les folds janvier–avril. L'AP et le ROC-AUC du holdout sont calculés sur les lignes concaténées, tandis que les métriques Top 10 % globales sont calculées sur les alertes mensuelles concaténées. Les écarts ne sont donc pas des comparaisons strictement identiques, mais tous vont dans le sens d'une dégradation temporelle. La comparaison macro Top 10 % du holdout donne un lift de 1,9427, plus proche de la référence 2,067, tout en masquant l'échec d'août.

La généralisation est jugée faible et temporellement instable. Cette interprétation post-holdout ne constitue ni un nouveau critère de sélection ni une autorisation d'adapter le candidat.

## 11. Incertitude statistique

Un bootstrap par blocs de mois a été exécuté avec 1 000 réplications, tirage avec remise de quatre blocs calendaires et `random_state=42`.

| Métrique | IC descriptif 95 % |
|---|---:|
| AP | [0,039947 ; 0,071813] |
| ROC-AUC | [0,525195 ; 0,672485] |
| Brier score | [0,163224 ; 0,208723] |
| Precision@10 % | [0,044141 ; 0,095453] |
| Recall@10 % | [0,090400 ; 0,233232] |
| Lift@10 % | [0,902887 ; 2,331107] |

Ces intervalles sont descriptifs et grossiers : quatre blocs seulement ne permettent pas une estimation temporelle très précise. L'intervalle du lift inclut 1, ce qui renforce l'incertitude sur le gain opérationnel stable.

## 12. Limites

- Scores bruts non calibrés et impropres à une interprétation probabiliste.
- Politique Top 10 % définie pour des lots mensuels rétrospectifs, pas pour une décision temps réel par commande.
- Aucun coût métier ni budget opérationnel validé.
- Population conditionnelle aux commandes finalement livrées.
- Forte variation de prévalence et seulement quatre mois de holdout.
- PSI de calendrier structurellement amplifié par les fenêtres temporelles comparées.
- Aucun fichier de prédictions individuelles n'a été conservé.

## 13. Reproductibilité

L'environnement enregistré est CPython 3.12.10, NumPy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, SciPy 1.18.1 et joblib 1.6.0.

Le manifeste contient le nom, la taille et le SHA-256 de chaque artefact Phase 5.5. Sa validation réussit. Le protocole et le candidat conservent leurs hashes initiaux. Le marqueur d'ouverture interdit une seconde évaluation.

## 14. Décision finale

Trois décisions distinctes sont retenues :

1. **Validité expérimentale — GO.** Le préflight, le gel du protocole, la lecture unique, le scoring figé et les contrôles d'intégrité sont valides.
2. **Qualité de généralisation — NO-GO.** La discrimination est faible, les performances reculent par rapport au développement et la politique Top 10 % est instable, avec un lift inférieur à 1 en août.
3. **Aptitude à la production — NO-GO.** Les scores ne sont pas calibrés, la politique est mensuelle, les coûts métier ne sont pas validés et la population est conditionnelle.

Le candidat reste un résultat expérimental documenté. Aucun réentraînement, recalibrage, changement de seuil, changement de features ou nouveau modèle n'est autorisé sur ce holdout.

## 15. Tests et manifestes

La suite complète `tests/ml` contient 47 tests réussis. Les tests Phase 5.5 couvrent notamment :

- validation réelle du manifeste et du candidat figé ;
- contrat, classe positive et ordre des features ;
- absence de `fit` et stabilité de l'état au scoring ;
- valeurs manquantes ;
- arrondi, égalités et budget Top 10 % ;
- agrégation mensuelle ;
- métriques avec une seule classe ou aucun positif ;
- reproductibilité ;
- détection d'une modification du protocole ;
- refus d'une seconde évaluation avant toute lecture.

Les artefacts Phase 5.5 se trouvent dans `models/delivery_delay/phase_5_5/final_holdout_v1/`. Ce dossier est local et exclu de Git par la règle existante sur `models/`.
