# Phase 5 — Audit de consolidation documentaire

## 1. Objet et périmètre

Cet audit contrôle la consolidation des Phases 5.1 à 5.5 dans [`PHASE_5_MACHINE_LEARNING.md`](PHASE_5_MACHINE_LEARNING.md). Il porte sur les populations, contrats, protocoles temporels, métriques, agrégations, statuts, chemins, liens et hashes.

La mission est exclusivement documentaire. Aucun dataset n'a été chargé, aucun modèle n'a été désérialisé, aucun pipeline n'a été exécuté et aucune prédiction, évaluation, calibration ou campagne de tests ML n'a été lancée. Les artefacts historiques, le candidat et le protocole final sont inchangés.

## 2. Sources consultées

### Sources principales lues intégralement

1. [`PHASE_5_1_ML_DATA_AUDIT.md`](PHASE_5_1_ML_DATA_AUDIT.md)
2. [`PHASE_5_2_ML_DATASET.md`](PHASE_5_2_ML_DATASET.md)
3. [`PHASE_5_3_MODEL_TRAINING.md`](PHASE_5_3_MODEL_TRAINING.md)
4. [`PHASE_5_4_MODEL_IMPROVEMENT.md`](PHASE_5_4_MODEL_IMPROVEMENT.md)
5. [`PHASE_5_4_1_ML_CONSOLIDATION_AUDIT.md`](PHASE_5_4_1_ML_CONSOLIDATION_AUDIT.md)
6. [`PHASE_5_4_2_CALIBRATION_CORRECTION.md`](PHASE_5_4_2_CALIBRATION_CORRECTION.md)
7. [`PHASE_5_4_3_HISTORICAL_CALIBRATION_ALERT_POLICY.md`](PHASE_5_4_3_HISTORICAL_CALIBRATION_ALERT_POLICY.md)
8. [`PHASE_5_5_FINAL_HOLDOUT_EVALUATION.md`](PHASE_5_5_FINAL_HOLDOUT_EVALUATION.md)

### Sources de contexte

- [`README.md`](../README.md)
- [`AGENTS.md`](../AGENTS.md)
- [`requirements.txt`](../requirements.txt)
- [`.gitignore`](../.gitignore)

### Vérifications statiques ciblées

- [`src/ml/contract.py`](../src/ml/contract.py), pour le contrat et les types V1 ;
- [`src/ml/contract_v2.py`](../src/ml/contract_v2.py), pour les dix-neuf enrichissements, types et groupes V2 ;
- [`src/ml/training.py`](../src/ml/training.py), pour les prétraitements ;
- inventaire statique de [`src/ml/`](../src/ml/) et [`tests/ml/`](../tests/ml/) ;
- manifestes JSON locaux des campagnes 5.3, 5.4, 5.4.2, 5.4.3 et 5.5 ;
- empreintes SHA-256 statiques du pipeline candidat, de la configuration 5.4.3 et du protocole 5.5.

Aucun fichier Parquet, fichier de prédictions ou objet joblib n'a été ouvert par une bibliothèque ML.

## 3. Matrice de traçabilité par sous-phase

| Phase | Objectif et sources | Méthodologie / contrat | Résultats et métriques | Artefacts / tests documentés | Décision et limites |
|---|---|---|---|---|---|
| 5.1 | Auditer `dw`, staging, analytics et la faisabilité du cas retard | Grain commande, cible calendaire, analyse de disponibilité et fuite | 99 441 historiques ; 96 470 éligibles ; 6 534 retards | Rapport d'audit ; contrôles PostgreSQL read-only | Conception validée ; disponibilité opérationnelle à prouver |
| 5.2 | Construire le dataset supervisé | `strict_v1.0.0`, 9 features, splits chronologiques, `label_observed_at` | 43 693 train descriptif ; 27 425 validation ; 25 352 test ; 41 197 labels matures au 01/01 | Parquet, contrat et rapports JSON ; 8 tests rapportés | Dataset V1 validé ; aucun modèle |
| 5.3 | Comparer quatre familles de modèles | Train mature 41 197 ; validation jan–avr ; pipelines sklearn | Logistic V1 : AP 0,153258, ROC 0,586930 ; seuil F1 0,459566 | 4 joblib et 6 rapports, manifeste 10 artefacts ; 16 tests rapportés | Candidat expérimental ; calibration et stabilité insuffisantes |
| 5.4 | Tester V2, calibration et backtests | `enriched_v2.0.0`, 28 features ; 4 folds train/calibration/évaluation | V2 Logistic brut macro AP 0,185148 ; sigmoid AP 0,159052 | Campagne isolée, manifeste 11 artefacts | Recommandation V2 sigmoid ensuite invalidée par l'audit 5.4.1 |
| 5.4.1 | Auditer calibration, populations, métriques et artefacts | Reproduction des folds et contrôle du rang | Sigmoid février : Spearman/Kendall -1 ; ROC brute + calibrée = 1 | 21/21 hashes historiques ; garde et tests ajoutés | NO-GO ; inversion due à une pente décroissante, pas à une fuite |
| 5.4.2 | Corriger la politique de calibration | Sigmoid croissante, isotonic admissible, fallback brut ; fenêtres calibration/seuil séparées | Inversion supprimée ; seulement 10 à 29 positifs dans plusieurs sous-fenêtres | Nouveau dossier, manifeste 11 artefacts ; aucun candidat | NO-GO ; support et seuils instables |
| 5.4.3 | Dernière campagne de développement préenregistrée | Calibration historique 3 mois ; V1/V2 Logistic ; Top 5/10/20 % | V1 brut macro AP 0,167930, ROC 0,628907 ; Top10 P 0,1865, R 0,2068, lift 2,067 | Manifeste 11 artefacts ; 38 tests rapportés | GO humain pour une évaluation unique ; candidat V1 brut figé |
| 5.5 | Évaluer une fois le candidat sur mai–août | Protocole figé ; aucune adaptation ; Top10 mensuel | 25 352 lignes ; AP 0,052363 ; ROC 0,561720 ; lift 1,4788 ; août 0,7369 | Manifeste 12 artefacts ; 47 tests rapportés | GO technique, NO-GO généralisation, NO-GO production |

## 4. Contrôles de cohérence

### 4.1 Populations

| Contrôle | Conclusion |
|---|---|
| 99 441 = 96 470 + 2 971 | Conforme |
| 96 470 = 89 936 + 6 534 | Conforme |
| 96 470 = 43 693 + 27 425 + 25 352 | Conforme pour le split descriptif 5.2 |
| 41 197 vs 43 693 | Conforme : labels matures au 01/01 vs train descriptif |
| 71 118 V1/V2 pré-test en 5.4 | Conforme : population avant mai, distincte du train mature et des splits complets |
| 47 794 / 2 387 du candidat | Conforme : train figé avant février, non additionnable aux calibrations |
| 25 352 / 1 115 du holdout | Conforme aux rapports 5.2 et 5.5 |

La consolidation n'additionne jamais les fenêtres de folds, calibrations ou évaluations comme si elles formaient de nouvelles partitions indépendantes du dataset complet.

### 4.2 Contrats

| Contrôle | Conclusion |
|---|---|
| V1 | `strict_v1.0.0`, 9 features, ordre et types conformes au code |
| V2 | `enriched_v2.0.0`, 28 features = V1 + 19 enrichissements |
| Statut V2 | `conditional_not_production_validated`, conservé dans le rapport |
| Candidat final | V1 uniquement, sans calibration, conforme à 5.4.3 et 5.5 |

### 4.3 Métriques et agrégations

- `PR-AUC` est présenté comme `average_precision_score`, conformément à la Phase 5.3.
- AP, ROC-AUC, Brier, log-loss et ECE ne sont pas confondus.
- Precision@k, Recall@k et Lift@k sont décrits comme métriques d'une capacité de lot.
- Les valeurs 5.4.3 sont identifiées comme moyennes macro de folds.
- Les métriques globales 5.5 sont distinguées des moyennes macro mensuelles.
- Les métriques Top 10 % globales 5.5 agrègent les alertes sélectionnées séparément par mois.
- Aucune comparaison entre agrégations différentes n'est présentée comme un test statistique apparié.

### 4.4 Statuts

La chronologie documentaire distingue correctement :

1. modèles expérimentaux 5.3/5.4 ;
2. invalidation méthodologique 5.4.1 ;
3. correction sans candidat 5.4.2 ;
4. candidat V1 figé 5.4.3 ;
5. évaluation finale techniquement valide 5.5 ;
6. généralisation insuffisante ;
7. absence d'aptitude production.

Le rapport consolidé ne présente jamais le candidat comme prêt pour la production.

## 5. Contradictions et nuances identifiées

### 5.1 Pourcentage de population dans la Phase 5.1

Le résumé exécutif de `PHASE_5_1_ML_DATA_AUDIT.md` indique que 96 470 commandes représentent **96,01 %** des 99 441 commandes. Le tableau détaillé du même rapport indique **97,01 %**, valeur cohérente avec les effectifs (`96 470 / 99 441 ≈ 97,01 %`).

Décision de présentation : conserver les effectifs, considérés comme autoritatifs, et utiliser 97,01 % si le ratio est mentionné. Le rapport source n'est pas modifié.

### 5.2 README et AGENTS.md obsolètes

Le README présente la Phase 5 comme planifiée et non implémentée. `AGENTS.md` décrit encore la Phase 4 comme en cours. Ces statuts contredisent les modules, tests, rapports et artefacts des Phases 5.1 à 5.5.

Décision de présentation : le rapport consolidé reflète l'état réel de la Phase 5. Le périmètre interdit toute mise à jour du README ou d'`AGENTS.md` ; leur alignement est donc une action documentaire future distincte.

### 5.3 Sens de « holdout non ouvert »

La Phase 5.3 documente le chargement du fichier test pour vérifier son schéma, ses bornes temporelles et l'absence de chevauchement, sans aucune prédiction. Les phases suivantes indiquent que le holdout n'a pas été lu dans leurs propres campagnes ou n'a pas été évalué. La Phase 5.5 parle d'une lecture unique dans le processus final contrôlé.

Décision de présentation : utiliser la formulation précise « aucun scoring ni aucune évaluation du holdout avant la Phase 5.5 » et « une seule évaluation finale autorisée ». Ne pas affirmer que le fichier n'avait littéralement jamais été lu pour un contrôle de schéma.

### 5.4 Résultats Phase 5.4 supersédés

La recommandation V2 sigmoid de la Phase 5.4 est historiquement exacte, puis invalidée par l'audit 5.4.1. Il ne s'agit pas d'une contradiction non résolue : la chronologie explique l'anomalie, la correction 5.4.2 et la sélection V1 brute 5.4.3. Le rapport consolidé conserve les deux états sans présenter la recommandation 5.4 comme finale.

### 5.5 Comptages de tests évolutifs

Les rapports annoncent 8, 16, 26, 38 puis 47 tests selon l'état de la suite à chaque sous-phase. Ces nombres sont cumulatifs et datés, pas contradictoires. Aucun test n'a été réexécuté pendant la consolidation.

Aucune contradiction majeure n'empêche la consolidation scientifique. Les écarts relevés sont documentaires ou terminologiques et n'altèrent pas les résultats finaux.

## 6. Décisions de présentation

- Les nombres du rapport 5.5 sont repris sans recalcul.
- Les valeurs précises priment sur les arrondis des résumés.
- V1 et V2 sont présentés comme deux contrats distincts, non comme deux versions opérationnelles équivalentes.
- Les protocoles temporels sont décrits séparément afin de ne pas mélanger leurs populations.
- La correction de calibration est expliquée mathématiquement sans l'attribuer à une fuite.
- Les métriques calibrées, de classement et de politique Top-k sont séparées.
- Le drift calendaire est qualifié de partiellement structurel ; aucun lien causal n'est affirmé.
- Les recommandations FastAPI, React et Airflow sont explicitement futures.
- La clôture concerne la phase expérimentale et documentaire, pas une validation de production.

## 7. Limites des sources

- Les artefacts modèles sont locaux et exclus de Git ; leur présence ne garantit pas leur disponibilité dans un clone neuf.
- Les rapports sont les sources autoritatives des exécutions historiques ; le repository ne contient pas de journal CI horodaté indépendant.
- Les expériences ont été réalisées sur un historique Olist fini, avec quatre mois de holdout.
- Les métriques de périodes et agrégations différentes ne permettent pas une inférence de significativité directe.
- Les dix-neuf features V2 restent conditionnelles, faute de journal opérationnel prouvant leur disponibilité.
- Le README et les instructions de statut n'ont pas encore été alignés sur la Phase 5 terminée.

## 8. Vérification des hashes, chemins et liens

### 8.1 Hashes statiquement vérifiés

| Élément | Hash attendu et observé | Statut |
|---|---|---|
| Pipeline candidat | `92e0d37821dc16c16063d7600ed6ee4488642c50e2496e139482ac83b45a6e48` | Conforme |
| Configuration 5.4.3 | `3c20ab70f9b434f6698f41a1ba2bbc0491f7f420aa4fad5b59b0c84c90eb8c39` | Conforme |
| Protocole 5.5 | `435bb744b9f8d2a409ca3889e2c44ec78ab995fb1689849643d20996998c8e22` | Conforme |

La vérification a calculé les empreintes de fichiers statiques. Le pipeline joblib n'a pas été désérialisé.

### 8.2 Manifestes

Les manifestes locaux annoncent 10 artefacts en 5.3, 11 en 5.4, 11 en 5.4.2, 11 en 5.4.3 et 12 en 5.5. Ils ont été consultés comme JSON ; aucune validation exécutable de pipeline ou de métrique n'a été relancée.

### 8.3 Liens et Mermaid

Les liens relatifs des deux nouveaux documents ont été conçus depuis `docs/` vers des fichiers ou dossiers présents. Le diagramme Mermaid utilise uniquement `flowchart`, `subgraph`, des nœuds et des arêtes compatibles avec GitHub Markdown. Le résultat du contrôle automatisé de présence des cibles est consigné dans le rapport final Codex.

## 9. Fichiers créés et statut de consolidation

Fichiers créés :

- [`PHASE_5_MACHINE_LEARNING.md`](PHASE_5_MACHINE_LEARNING.md)
- [`PHASE_5_DOCUMENTATION_AUDIT.md`](PHASE_5_DOCUMENTATION_AUDIT.md)

Aucun rapport source, code, test, modèle, dataset, manifeste, protocole, configuration, README, fichier Power BI ou SQL n'a été modifié par cette mission.

Le rapport principal contient les 22 sections obligatoires, un diagramme d'architecture, les deux contrats, la chronologie des expériences, les résultats finaux, les limites et les décisions d'architecture.

## 10. Recommandation de clôture

**Décision documentaire recommandée : GO.**

Les huit sous-phases sont représentées, les contradictions non bloquantes sont explicites, les agrégations sont distinguées et le statut final ne revendique aucune aptitude production. La Phase 5 peut être déclarée clôturée après validation humaine de ces deux documents.

Cette clôture n'autorise ni une nouvelle évaluation du holdout, ni une optimisation du candidat, ni le démarrage automatique de la Phase 6.
