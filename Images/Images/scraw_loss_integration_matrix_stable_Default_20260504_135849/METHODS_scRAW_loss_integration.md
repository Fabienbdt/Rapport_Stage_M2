# Document d'accompagnement - intégration de la loss scRAW

Campagne : `scraw_loss_integration_matrix_stable_Default_20260428_104305`

Date de fin : `2026-05-03 01:35:15`

Ce document accompagne les résultats produits dans :

`/data2/fbidet/scRAW_EXPERIMENTAL/results/scraw_loss_integration_matrix_stable_Default_20260428_104305`

Il décrit comment les composantes de la loss scRAW ont été ajoutées à `scMAE`, `scDeepCluster` et `DESC`, quelles variantes ont été testées, et pourquoi ces choix ont été faits.

## Objectif de l'expérience

L'objectif était de tester si les idées de la loss scRAW peuvent améliorer des algorithmes d'intégration/clustering existants, sans réentraîner ou réoptimiser tous les hyperparamètres.

La campagne compare donc :

- les algorithmes de base, sans ajout scRAW ;
- la loss scRAW complète, avec pseudo-clustering Leiden ou KMeans ;
- une version `density_only`, qui ne garde que la composante de densité KNN ;
- les mêmes variantes avec une triplet loss rare-cell-aware ajoutée.

Le point important est que tous les runs utilisent le même preset verrouillé `stable_Default`, afin que les écarts observés soient principalement attribuables au type de loss ajouté, et non à une nouvelle recherche d'hyperparamètres.

## Preset utilisé

Preset : `stable_Default`

Alias code : `stable_default`

Source unique :

`/data2/fbidet/scRAW_EXPERIMENTAL/results/trial206_default_v6_search_20260415_161134/phase1/stable_generalist/stage1/trials/stable_generalist_trial_0017/trial_config.json`

Paramètres principaux utilisés dans cette campagne :

| Paramètre | Valeur |
|---|---:|
| `epochs` | `120` |
| `batch_size` | `192` |
| `learning_rate` | `0.00164076083297036` |
| `warmup_epochs` | `55` |
| `dynamic_weight_update_interval` | `20` |
| `dynamic_weight_momentum` | `0.6884621079434989` |
| `density_knn_k` | `15` |
| `density_weight_exponent` | `1.0` |
| `density_weight_clip` | `3.0` |
| `weight_exponent` | `0.2` |
| `weight_fusion_mode` | `multiplicative` |
| `min_cell_weight` | `0.3845423008053828` |
| `max_cell_weight` | `10.0` |
| `rare_triplet_weight` | `0.05007581780188212` |
| `rare_triplet_start_epoch` | `60` |
| `rare_triplet_margin` | `0.4` |
| `rare_triplet_min_weight` | `1.2` |
| `max_triplet_anchors_per_batch` | `64` |

Les variantes sans triplet forcent `rare_triplet_weight=0.0`.

Les variantes avec triplet utilisent les paramètres triplet du preset `stable_Default`.

## Datasets et répétitions

Le smoke test a été lancé sur :

- `paul15_bone_marrow`, seed `42`

La campagne complète a été lancée sur :

- `paul15_bone_marrow`
- `bbag094_zeisel`
- `pancreas`
- `bbag094_spleen`
- `kang_pbmc`

Seeds de la campagne complète :

- `42`
- `43`
- `44`
- `45`
- `46`

Les métriques scIB ont été désactivées pour cette campagne (`no_scib_metrics=1`) afin de concentrer le coût de calcul sur les variantes de loss et les métriques principales déjà produites par les wrappers.

## Variantes testées

| Variante | Reconstruction | Pseudo-labels reconstruction | Triplet | Pseudo-labels triplet |
|---|---|---|---|---|
| `baseline` | loss native de l'algorithme | aucun | non | aucun |
| `full_leiden` | pondération scRAW complète | Leiden | non | aucun |
| `full_kmeans` | pondération scRAW complète | KMeans | non | aucun |
| `density_only` | densité KNN uniquement | aucun pseudo-clustering utile | non | aucun |
| `full_leiden_triplet` | pondération scRAW complète | Leiden | oui | Leiden |
| `full_kmeans_triplet` | pondération scRAW complète | KMeans | oui | KMeans |
| `density_only_triplet_kmeans` | densité KNN uniquement | aucun pseudo-clustering reconstruction | oui | KMeans seulement pour la triplet |

La variante `density_only_triplet_kmeans` est volontairement asymétrique : la reconstruction reste strictement `density_only`, tandis que KMeans n'est utilisé que pour définir les positifs et négatifs de la triplet loss.

## Principe général de la pondération scRAW

La loss scRAW ajoutée aux algorithmes repose sur des poids par cellule calculés dans l'espace latent courant. Ces poids modifient la contribution de chaque cellule dans la reconstruction.

Deux composantes peuvent intervenir :

1. Une composante pseudo-cluster rareté.
2. Une composante densité locale KNN.

### Composante pseudo-cluster rareté

Des pseudo-labels sont construits sur l'embedding latent courant, avec Leiden ou KMeans selon la variante.

Chaque pseudo-cluster reçoit ensuite un poids inversement proportionnel à sa fréquence :

`w_cluster(c) = (1 / freq(cluster(c))) ** weight_exponent`

avec `weight_exponent=0.2`.

Cette composante vise à redonner de l'importance aux groupes peu représentés dans l'embedding courant, sans utiliser les labels biologiques comme supervision directe.

### Composante densité KNN

Pour chaque cellule, on calcule la distance au k-ième plus proche voisin dans l'espace latent :

`d_k(c)`

Cette distance est divisée par la médiane des distances `d_k`, puis transformée en poids :

`w_density(c) = (d_k(c) / median(d_k)) ** density_weight_exponent`

avec :

- `density_knn_k=15`
- `density_weight_exponent=1.0`
- clipping local à `density_weight_clip=3.0`

L'intuition est simple : une cellule située dans une zone peu dense a un `d_k` plus grand et reçoit donc un poids plus élevé. Cette composante peut capturer des cellules rares ou isolées sans calculer de pseudo-clusters.

### Fusion des composantes

Dans les variantes `full_*`, les deux composantes sont fusionnées de façon multiplicative :

`w(c) = w_cluster(c) * w_density(c)`

Le résultat est normalisé, puis limité entre `min_cell_weight` et `max_cell_weight`.

Le mode multiplicatif a été choisi parce qu'il favorise surtout les cellules qui sont à la fois dans un pseudo-groupe rare et dans une zone peu dense. Cela évite de surpondérer trop fortement une cellule qui ne serait rare que selon un seul signal.

Dans les variantes `density_only`, la composante pseudo-cluster est complètement court-circuitée. Le code ne calcule pas les poids de fréquence de pseudo-clusters pour la reconstruction, ce qui évite le coût de Leiden/KMeans dans cette partie.

## Ajout aux algorithmes

### scMAE

Version pondérée : `sc_mae_scraw_weighted`

scMAE est un autoencodeur masqué. La version pondérée conserve la logique native de scMAE :

- corruption/masquage de l'entrée ;
- reconstruction ;
- loss de reconstruction et loss liée au masque ;
- clustering final sur l'embedding.

L'ajout scRAW intervient dans la loss d'entraînement. Après une période de warm-up, les poids par cellule sont calculés sur l'embedding latent et injectés comme multiplicateurs des termes de reconstruction/masquage par cellule.

Schématiquement :

`L_scMAE_weighted = mean_i w_i * L_scMAE_i`

Quand la triplet est activée :

`L_total = L_scMAE_weighted + ramp(epoch) * rare_triplet_weight * L_triplet`

La triplet est appliquée sur le latent scMAE, pas sur l'espace d'entrée.

### scDeepCluster

Version pondérée : `scdeepcluster_scraw_weighted`

scDeepCluster combine :

- un autoencodeur de type ZINB ;
- une phase de pré-entraînement ;
- une phase de clustering avec une loss de clustering.

L'ajout scRAW est appliqué à la reconstruction ZINB, sous forme de poids par cellule. La loss de clustering native est conservée.

Pendant le pré-entraînement :

`L_pretrain = weighted_ZINB_reconstruction`

Pendant la phase clustering :

`L_cluster = weighted_ZINB_reconstruction + gamma * L_cluster_native`

Quand la triplet est activée, elle est ajoutée au latent `z` dans les deux phases où les poids sont disponibles :

`L_total = L_base + ramp(epoch) * rare_triplet_weight * L_triplet`

Ce choix permet de tester l'effet de la loss scRAW sans remplacer la mécanique centrale de scDeepCluster.

### DESC

Version pondérée : `desc_scraw_weighted`

DESC repose sur un autoencodeur et un raffinement de clustering à plusieurs résolutions. L'intégration scRAW modifie la phase autoencodeur utilisée par DESC :

- sans triplet, les poids par cellule sont passés comme `sample_weight` lors de l'entraînement de l'autoencodeur ;
- avec triplet, une boucle `GradientTape` TensorFlow calcule explicitement la reconstruction pondérée et ajoute la triplet loss sur l'embedding latent.

La logique multi-résolution de DESC est conservée. L'objectif est de rendre l'espace latent fourni à DESC plus sensible aux cellules rares ou peu denses, sans réécrire l'algorithme DESC complet.

## Triplet loss rare-cell-aware

La triplet loss ajoutée est une triplet semi-hard construite sur les embeddings latents.

Elle utilise :

- des ancres sélectionnées parmi les cellules dont le poids scRAW est élevé ;
- un seuil `rare_triplet_min_weight=1.2` ;
- au maximum `64` ancres par batch ;
- une marge `0.4`.

Pour une ancre donnée :

- les positifs sont les cellules du même pseudo-label triplet ;
- les négatifs sont les cellules d'un pseudo-label différent ;
- le négatif semi-hard est préféré quand il existe.

La forme générale est :

`L_triplet = max(0, d(anchor, positive) - d(anchor, negative) + margin)`

La triplet ne démarre pas immédiatement. Elle commence à `rare_triplet_start_epoch=60`, après le warm-up des poids (`warmup_epochs=55`). Une rampe limite son impact au démarrage afin d'éviter d'imposer une contrainte géométrique trop forte sur un embedding encore instable.

## Séparation entre pseudo-labels reconstruction et triplet

Un choix important de cette campagne est de séparer :

- les pseudo-labels utilisés pour calculer les poids de reconstruction ;
- les pseudo-labels utilisés pour construire les triplets.

Cette séparation évite plusieurs confusions expérimentales :

- `density_only` doit vraiment mesurer la composante densité seule, sans coût ni signal de pseudo-clustering dans la reconstruction ;
- `density_only_triplet_kmeans` doit permettre d'ajouter une structure triplet sans réintroduire KMeans dans la pondération de reconstruction ;
- les variantes `full_leiden_triplet` et `full_kmeans_triplet` peuvent réutiliser le même type de pseudo-labels pour reconstruction et triplet, ce qui teste une version cohérente "full + triplet".

En pratique, cela rend les comparaisons plus propres : on peut distinguer l'effet de la pondération de reconstruction de l'effet de la contrainte géométrique triplet.

## Pourquoi ces choix expérimentaux ?

### Verrouiller `stable_Default`

Le preset `stable_Default` a été verrouillé pour éviter une comparaison biaisée par des hyperparamètres différents. La question posée ici n'est pas "quel est le meilleur réglage possible ?", mais plutôt :

Est-ce que l'ajout de telle composante de la loss scRAW améliore ou dégrade l'intégration, à réglages comparables ?

### Tester Leiden et KMeans

Leiden est souvent mieux adapté aux structures graphes et aux formes de clusters non sphériques, mais il est plus coûteux et peut devenir un goulot d'étranglement.

KMeans est plus simple, souvent plus rapide, et plus facile à contrôler, mais impose une géométrie plus sphérique.

Tester les deux permet d'identifier si les gains viennent de la pondération rareté elle-même ou d'un choix particulier de pseudo-clustering.

### Isoler `density_only`

La variante `density_only` répond à une question pratique très importante :

Peut-on obtenir une partie des bénéfices de scRAW sans payer le coût du pseudo-clustering Leiden/KMeans pendant l'entraînement ?

Cette variante est aussi plus simple à interpréter : elle ne dépend que de la géométrie locale KNN de l'embedding.

### Ajouter la triplet loss

La pondération de reconstruction dit au modèle quelles cellules il doit mieux reconstruire. La triplet loss ajoute une contrainte différente : elle agit directement sur la géométrie du latent.

L'hypothèse testée est donc :

Les cellules rares ou peu denses ne doivent pas seulement être mieux reconstruites ; elles doivent aussi être mieux organisées dans l'espace latent.

### Cibler les ancres à poids élevé

La triplet n'est pas appliquée uniformément à toutes les cellules. Elle cible les cellules dont le poids scRAW est élevé, c'est-à-dire celles que la loss considère comme potentiellement rares, isolées ou sous-représentées.

Ce choix évite de transformer la triplet en contrainte globale trop forte, qui pourrait dégrader les grands types cellulaires bien représentés.

## Organisation des résultats

Le dossier est organisé comme suit :

```text
scraw_loss_integration_matrix_stable_Default_20260428_104305/
  00_manifest/
    matrix_config.json
    aggregation_summary.json
    smoke_screen.log
    full_screen.log
  01_smoke_test/
    runs/
  02_full_runs/
    runs/
  03_imported_existing/
    historical_sources.json
  04_tables/
    all_seed_metrics.csv
    summary_by_dataset_algorithm_variant.csv
    delta_vs_baseline.csv
    source_manifest.csv
  05_plots/
    <dataset>__<algorithm>__boxplots.png
    <dataset>__<algorithm>__boxplots.svg
```

Les tables principales sont :

- `04_tables/all_seed_metrics.csv` : toutes les métriques au niveau seed/run.
- `04_tables/summary_by_dataset_algorithm_variant.csv` : agrégation par dataset, algorithme et variante.
- `04_tables/delta_vs_baseline.csv` : différences par rapport au baseline correspondant.
- `04_tables/source_manifest.csv` : provenance des résultats nouveaux et historiques.

Les figures sont organisées par dataset et algorithme dans `05_plots`.

## Résumé d'exécution

La campagne complète s'est terminée sans crash bloquant détecté.

Résumé de l'agrégation finale :

| Élément | Valeur |
|---|---:|
| lignes totales agrégées | `6265` |
| nouvelles lignes de cette campagne | `4368` |
| lignes historiques importées | `1897` |
| configs full produites | `35` |
| fichiers `analysis_results.csv` full | `210` |
| figures produites | `30` |

Les anciens résultats importés l'ont été en lecture seule depuis :

- `/data2/fbidet/scRAW_EXPERIMENTAL/results/scraw_weighted_worst_datasets_complete_boxplots_20260427_100842`
- `/data2/fbidet/scRAW_EXPERIMENTAL/results/scraw_weighted_three_algos_4way_boxplots_20260423_141758`

## Limites à garder en tête

Cette campagne compare l'effet des variantes de loss sous un preset fixé. Elle ne prouve pas que chaque variante est optimale pour chaque algorithme.

Certaines étapes restent coûteuses ou partiellement CPU :

- Leiden/KMeans ;
- calcul des voisins KNN ;
- métriques et génération des plots ;
- phases internes multi-résolution de DESC.

Les métriques scIB n'ont pas été calculées dans cette campagne. Les conclusions doivent donc s'appuyer sur les métriques présentes dans les tables générées, et une campagne scIB séparée peut être envisagée si une variante candidate doit être validée plus largement.

## Fichiers de code concernés

Les principales modifications de code correspondant à cette campagne sont dans :

- `/data2/fbidet/SCRBenchmark/src/scrbenchmark/algorithms/sc_mae_scraw_weighted.py`
- `/data2/fbidet/SCRBenchmark/src/scrbenchmark/algorithms/scdeepcluster_scraw_weighted.py`
- `/data2/fbidet/scRAW_EXPERIMENTAL/external/desc/desc/models/SAE.py`
- `/data2/fbidet/scRAW_EXPERIMENTAL/external/desc/desc/models/network.py`
- `/data2/fbidet/scRAW_EXPERIMENTAL/external/desc/desc/models/desc.py`
- `/data2/fbidet/scRAW_EXPERIMENTAL/scripts/run_desc_benchmark.py`
- `/data2/fbidet/scRAW_EXPERIMENTAL/scripts/run_scraw_loss_integration_matrix_stable_default.sh`
- `/data2/fbidet/scRAW_EXPERIMENTAL/scripts/aggregate_scraw_loss_integration_matrix.py`

Les tests ciblés ajoutés pour vérifier les invariants `density_only` sont dans :

- `/data2/fbidet/SCRBenchmark/tests/unit_tests/test_scraw_weighted_loss_components.py`

