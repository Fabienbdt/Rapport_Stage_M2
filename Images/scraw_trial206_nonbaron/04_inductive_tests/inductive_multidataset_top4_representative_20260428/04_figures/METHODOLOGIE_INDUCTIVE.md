# Methodologie inductive, version simple

Ce document explique comment les resultats des figures inductives ont ete
obtenus pour les 4 algorithmes:

- scRAW
- scNAME
- scMAE
- scDeepCluster

Le dossier concerne les figures et tables de:

`/data2/fbidet/scRAW_Inductif/results/inductive_multidataset_top4_representative_20260428/04_figures`

## Idee generale

Le but etait de tester une vraie situation inductive:

1. On cache un ou plusieurs donneurs, batches ou individus.
2. On entraine l'algorithme seulement sur les groupes de train.
3. On applique ensuite l'algorithme entraine aux groupes de test.
4. On compare les predictions aux vrais labels de cellules.

Important: les labels du test servent seulement a calculer les scores. Ils ne
servent pas a entrainer le modele sur le test.

Les metriques calculees sont:

- `ACC`
- `BalancedACC`
- `ARI`
- `NMI`
- `RareACC`
- `UltraRareACC`

## Vocabulaire

- `fit`: etape d'apprentissage. L'algorithme voit les cellules de train et
  apprend ses parametres.
- `predict`: etape de prediction. L'algorithme deja entraine recoit les
  cellules de test et leur attribue un cluster ou une classe predite.
- `train_groups`: groupes utilises pour le fit.
- `test_group` ou `test_groups`: groupes gardes de cote pour la prediction.

La question principale est donc:

Est-ce que l'algorithme a ete refitte sur le test ?

Reponse generale: non pour scRAW, scNAME, scDeepCluster, et pour scMAE quand
son clustering train est KMeans. Nuance importante: pour les lignes scMAE ou le
clustering train est Leiden, la prediction du test utilise ensuite un KMeans
ajuste sur les embeddings test. Ces lignes scMAE sont donc a lire avec
prudence.

## Resume tres court par algorithme

| Algorithme | Fit fait sur | Prediction faite sur | Comment la prediction test est produite |
| --- | --- | --- | --- |
| scRAW | train uniquement | test uniquement | encodage par le modele scRAW gele, puis centroide de cluster train le plus proche |
| scNAME | train uniquement | test uniquement | encodage par scNAME, puis centre de cluster scNAME le plus proche |
| scMAE | train uniquement | test uniquement | encodage par scMAE, puis clusterer appris sur train quand KMeans est utilise |
| scDeepCluster | train uniquement | test uniquement | encodage par scDeepCluster, puis assignation aux centres appris par le modele |

Point important: il ne faut pas comprendre ce tableau comme "seul scRAW
utilise des centres". Plusieurs methodes utilisent aussi des centres ou
prototypes en interne. La difference est que, pour scRAW, la tete de prediction
par centroide est l'etape utilisee ici pour transferer les clusters du train
vers les cellules de test.

## Deroule commun d'un split

Pour chaque dataset, on definit d'abord un split:

- cellules de train: groupes dans `train_groups`;
- cellules de test: groupes dans `test_groups`.

Ensuite, pour ce split:

1. Les cellules de train sont extraites.
2. Les cellules de test sont extraites separement.
3. Le preprocessing est appris sur le train.
4. Le test est transforme avec les choix appris sur le train.
5. Chaque algorithme est entraine sur le train.
6. Chaque algorithme predit les cellules de test.
7. Les scores sont calcules entre les labels predits et les vrais labels du
   test.

Le test ne sert pas a ajuster le modele neuronal. Il ne sert pas non plus a
choisir les genes. La seule exception reperee concerne scMAE avec clustering
train Leiden: un KMeans est alors ajuste sur les embeddings test pour produire
des labels, faute de transfert direct des labels Leiden train vers le test.

## Splits utilises

Les nouveaux datasets du comparatif principal utilisent ces splits:

| Dataset | Train | Test | Interpretation |
| --- | --- | --- | --- |
| Macaque retina | `M1,M2` | `M3,M4` | transfert vers des individus non vus |
| Human testis | donneurs 1 et 2 | donneur 3 | transfert vers un donneur non vu |
| Pancreas 4 batches | `smartseq2,celseq2` | `celseq,fluidigmc1` | transfert entre technologies |
| BBAG094 spleen | `3-F-56` | `3-M-8` | transfert entre deux donneurs |

Les autres splits inclus dans la comparaison sont:

- Kang PBMC: train sur les donneurs train, test sur `1039` et `107`.
- Baron pancreas: leave-one-human-out, donc un humain est garde en test a
  chaque split.

Les details exacts sont dans:

`standalone_tables/dataset_split_manifest.csv`

La table finale utilisee pour les figures est:

`standalone_tables/combined_summary.csv`

Note 2026-05-11: dans la version actuelle des figures finales, les lignes
affichees sous le label `scRAW` correspondent a la relance exacte
`trial0017`, et non plus a l'ancien run `scRAW default`. Le label reste `scRAW`
pour conserver la comparaison a 4 algorithmes.

Les boxplots globaux utilisent ensuite une table derivee:

`standalone_tables/dataset_level_metric_summary.csv`

Dans cette table, chaque valeur est la moyenne des splits disponibles pour un
couple dataset / algorithme / metrique.

## Ce qui se passe pour scRAW

Pour scRAW, la prediction des cellules de test est construite comme une etape
de transfert dans l'espace latent. Elle utilise les clusters appris sur le train
comme reference.

Pour un split donne:

1. Le preprocessing scRAW est appris sur le train.
2. Le modele scRAW est entraine sur le train.
3. scRAW produit des embeddings et des clusters pour les cellules de train.
4. On calcule un centroide dans l'espace latent pour chaque cluster de train.
5. Les cellules de test sont transformees avec le preprocessing gele.
6. Le modele scRAW gele encode les cellules de test.
7. Chaque cellule de test est assignee au cluster du centroide de train le plus
   proche.

Donc, pour scRAW:

- fit: oui, sur le train;
- predict: oui, sur le test, via centroide le plus proche;
- refit sur test: non.

Les objets appris sur le train sont donc:

- le modele scRAW;
- le preprocessing;
- les centroides des clusters train.

Cas particulier: si le train ne contient qu'un seul batch, la branche adversariale
batch de scRAW est desactivee, car il n'y a pas de contraste batch a apprendre.

## Ce qui se passe pour scNAME

Pour scNAME, la prediction du test repose sur les centres de clusters appris
dans l'espace latent de scNAME.

Pour un split donne:

1. Les donnees de train sont donnees a scNAME.
2. scNAME est entraine sur ces cellules de train.
3. Pendant le fit, scNAME apprend un encodeur et des centres de clusters dans
   son espace latent.
4. Les donnees de test sont ensuite donnees au modele deja entraine.
5. Les cellules de test sont encodees avec l'encodeur scNAME appris sur le
   train.
6. Il calcule la distance entre chaque embedding test et les centres de
   clusters scNAME appris sur le train.
7. Chaque cellule test recoit le label du centre scNAME le plus proche.
8. Les scores sont calcules sur ces predictions.

Donc, pour scNAME:

- fit: oui, sur le train;
- predict: oui, sur le test, via centres de clusters scNAME appris sur train;
- refit sur test: non.

Parametres importants:

- `map_class=False`
- `n_clusters` fixe a partir du train
- `input_type=raw_filtered`
- `use_raw_data=True`

## Ce qui se passe pour scMAE

Pour scMAE, il faut distinguer deux cas selon la methode de clustering utilisee
apres l'apprentissage du modele.

Pour un split donne:

1. Les donnees de train sont donnees a scMAE.
2. scMAE est entraine sur ces cellules de train.
3. scMAE garde son propre preprocessing interne, appris pendant le fit.
4. Apres entrainement, scMAE clusterise les embeddings du train.
5. Les cellules de test sont ensuite passees au modele entraine.
6. Les cellules de test sont encodees avec le modele scMAE appris sur le train.
7. Quand le clustering train de scMAE est KMeans, ce KMeans deja appris sur le
   train est applique aux embeddings test.
8. Quand le clustering train de scMAE est Leiden, il n'y a pas ici de modele
   Leiden directement transferable au test; un KMeans est alors ajuste sur les
   embeddings test. Ces lignes sont donc moins strictement inductives que les
   lignes KMeans.
9. Les scores sont calcules sur ces predictions.

Donc, pour scMAE:

- fit: oui, sur le train;
- predict: oui, sur le test;
- refit du modele neuronal sur test: non;
- refit d'un clusterer sur test: non pour les lignes KMeans, mais oui pour le
  KMeans ajuste sur test des lignes ou scMAE avait utilise Leiden sur le train.

Nuance importante: dans ces resultats, scMAE a utilise KMeans pour Human
testis, Pancreas 4 batches, BBAG094 spleen et Baron pancreas. Pour Macaque
retina et Kang PBMC, scMAE a utilise Leiden sur le train, puis un KMeans a ete
ajuste sur les embeddings test. Si on veut une comparaison
inductive plus stricte pour scMAE sur ces deux datasets, il faudrait rerun scMAE
en forcant `clustering_method="kmeans"` ou implementer un transfert depuis les
clusters Leiden train.

Parametres importants:

- `use_own_preprocessing=True`
- `n_hvg=1000`
- `input_type=raw_filtered`
- `use_raw_data=True`

## Ce qui se passe pour scDeepCluster

Pour scDeepCluster, la prediction test repose sur les centres appris par le
modele pendant l'entrainement.

Pour un split donne:

1. Les donnees de train sont donnees a scDeepCluster.
2. scDeepCluster est entraine sur ces cellules de train.
3. Pendant le fit, scDeepCluster apprend des centres de clusters dans son espace
   latent.
4. Les cellules de test sont ensuite passees au modele entraine.
5. Les cellules de test sont encodees avec scDeepCluster.
6. Il applique `soft_assign` entre les embeddings test et les centres appris.
7. Le label predit est le centre avec la plus forte assignation.
8. Les scores sont calcules sur ces predictions.

Donc, pour scDeepCluster:

- fit: oui, sur le train;
- predict: oui, sur le test, via centres appris par scDeepCluster;
- refit sur test: non.

Parametres importants:

- `select_genes=0`
- `use_ground_truth_k=False`
- `input_type=raw_filtered`
- `use_raw_data=True`

`select_genes=0` indique que la selection de genes specifique a scDeepCluster
est desactivee ici, car l'espace de genes a deja ete defini a partir du train.

## Difference importante entre scRAW et les autres

Les 4 algorithmes suivent la meme separation train/test, mais ils ne produisent
pas les predictions test exactement de la meme maniere:

- scNAME encode le test puis assigne chaque cellule au centre scNAME appris sur
  le train le plus proche.
- scDeepCluster encode le test puis applique `soft_assign` vers les centres
  appris par le modele.
- scMAE encode le test puis applique le KMeans appris sur train quand le fit a
  choisi KMeans. Si le clustering train est Leiden, un KMeans est ajuste sur
  les embeddings test.

La difference de scRAW est donc plus precise:

Dans cette evaluation, scRAW predit le test avec une tete de transfert par
centroides:

```text
entrainer scRAW sur train
calculer les centroides des clusters train
encoder test avec le modele scRAW gele
assigner chaque cellule test au centroide train le plus proche
```

Pour scNAME et scDeepCluster, les centres font partie de la logique de
l'algorithme. Pour scRAW, les centroides sont une etape de transfert ajoutee
apres l'entrainement pour pouvoir predire le test.

## Baron et Kang PBMC

Les datasets Kang PBMC et Baron pancreas suivent le meme principe train/test,
avec des splits deja definis.

Pour Kang PBMC:

- fit sur les donneurs train;
- prediction sur les donneurs test `1039` et `107`;
- scRAW, scNAME et scDeepCluster suivent le principe train puis test sans refit
  sur test;
- scMAE a utilise Leiden sur le train, puis un KMeans ajuste sur les embeddings
  test.

Pour Baron pancreas:

- chaque split garde un humain en test;
- les autres humains servent au train;
- scNAME encode le test et assigne les cellules aux centres scNAME appris sur
  les humains train;
- scMAE encode le test et, dans ces splits Baron, utilise le KMeans appris sur
  les embeddings train;
- scDeepCluster encode le test et utilise `soft_assign` vers les centres appris
  sur les humains train;
- scRAW utilise le modele entraine sur train puis la prediction par centroide
  train le plus proche.

Note `UltraRareACC`: les anciennes sorties Baron scRAW ne stockaient pas cette
metrique dans `results.json`, mais elles stockaient les vrais labels test et les
labels predits. La valeur `UltraRareACC` a donc ete completee apres coup depuis
ces fichiers, sans refaire le fit.

Kang PBMC donor `1039` reste `nan` pour `UltraRareACC` chez les 4 algorithmes,
car aucune classe vraie n'est sous le seuil ultra-rare de 1 % apres
preprocessing. Ce n'est donc pas un run manquant.

## Comment lire les figures

Les figures comparent les scores obtenus apres prediction du test.

Une ligne de `combined_summary.csv` correspond en general a:

- un dataset;
- un algorithme;
- un split train/test;
- les scores obtenus sur le test.

Pour les boxplots:

- un point = la moyenne d'un dataset pour un algorithme;
- la boite resume la distribution de ces moyennes entre datasets;
- pour `UltraRareACC`, les points `nan` ne sont pas affiches.

Cette aggregation evite de donner plus de poids aux datasets qui ont davantage
de splits. Par exemple, Baron pancreas a 4 splits, BBAG094 spleen en a 1, et
les autres datasets en ont 2. Avec les boxplots actuels, chacun de ces datasets
compte une seule fois par algorithme.

Apres aggregation par dataset:

- chaque boxplot global a 6 points par algorithme, un pour chaque dataset;
- pour `UltraRareACC`, Kang PBMC est moyenne sur les splits non-`nan`
  disponibles, car le donor `1039` reste `nan` pour les 4 algorithmes.

## Pourquoi le dossier est autosuffisant

Le dossier `04_figures` contient les PNG, mais aussi les tables necessaires pour
comprendre les figures:

- `standalone_tables/combined_summary.csv`
- `standalone_tables/balanced_acc_per_split.csv`
- `standalone_tables/dataset_level_metric_summary.csv`
- `standalone_tables/dataset_level_counts_by_metric_algorithm.csv`
- `standalone_tables/mean_std_by_dataset_algorithm.csv`
- `standalone_tables/dataset_split_manifest.csv`
- `standalone_tables/per_dataset_summaries/`
- `standalone_tables/standalone_metadata.json`
- `standalone_tables/validation_warnings.txt`

Cela permet de garder seulement `04_figures` tout en conservant les donnees
utiles pour interpreter les graphiques.

## Validation

Au moment de generation du bundle final:

- 52 lignes sont presentes dans `combined_summary.csv`.
- 52 lignes sont en statut `ok`.
- Les metriques principales `ACC`, `ARI`, `NMI` ne manquent pas.
- `validation_warnings.txt` est vide.
