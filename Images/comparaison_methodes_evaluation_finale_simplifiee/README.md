# Trial 206 non-Baron figures

- Figure profile: `simplified`
- Results table: `/data2/fbidet/scRAW_EXPERIMENTAL/results/presentation_trial206_nonbaron_20260324/00_source_tables/trial206_all_results_table.csv`
- Dataset table: `/data2/fbidet/scRAW_EXPERIMENTAL/results/presentation_trial206_nonbaron_20260324/00_source_tables/trial206_dataset_table.csv`
- Predictions table: `/data2/fbidet/scRAW_EXPERIMENTAL/results/presentation_trial206_nonbaron_20260324/00_source_tables/trial206_predictions_with_ground_truth.csv.gz`
- Output root: `/data2/fbidet/scRAW_EXPERIMENTAL/figures/02_stage_m2_report_trial206_nonbaron_simplified`
- Inductive bundle copied from: `/data2/fbidet/scRAW_Inductif/results/inductive_multidataset_top4_representative_20260428/04_figures`
- Inductive bundle target: `/data2/fbidet/scRAW_EXPERIMENTAL/figures/02_stage_m2_report_trial206_nonbaron_simplified/04_inductive_tests/inductive_multidataset_top4_representative_20260428/04_figures`

## Included methods

- `scRAW` (`scRAW (trial_0017)`)
- `scRAW finetune` (`scRAW (best per dataset)`)
- `scNAME` (`scNAME`)
- `scMAE` (`scMAE`)
- `Harmony` (`Harmony`)
- `ComBat` (`ComBat`)
- `Scanorama` (`Scanorama`)
- `DESC` (`DESC`)
- `PCA+Leiden` (`pca_leiden`)
- `scVI` (`scvi`)

## Contents

- `01_barplots_metriques_par_dataset/`: one 2x2 barplot figure per dataset for `ARI`, `Balanced Acc`, `Rare Acc`, and `Batch correction`.
- `02_heatmaps_erreurs_types_cellulaires/`: per-dataset class-prediction error heatmaps.
- `03_resume_metriques_8_datasets/`: global barplots restricted to the 8 report-selected datasets.
- `04_tests_inductifs/`: copied inductive test figures and their standalone tables.
- `05_tables_qc/`: support CSVs for the chosen rows, the per-cell/classwise aggregation, and augmentation status.

## Notes

- `scRAW (best per dataset)` is restricted to the 8 report-selected dataset/trial pairs listed in `05_tables_qc/evaluation_finale_sources_meilleures_methodes_par_dataset.csv`.
- In the simplified profile, `scRAW` denotes `scRAW (trial_0017)` and `scRAW finetune` denotes `scRAW (best per dataset)` when available.
- `scRAW finetune` is included only for datasets where its mean delta versus `scRAW` is positive over available target metrics.
- The finetune/scRAW comparison and inclusion decision are written to `05_tables_qc/evaluation_finale_filtre_scraw_finetune_vs_scraw.csv`.
- The cell-type heatmaps use error rate (`1 - accuracy`).
- scMAE/scNAME heatmaps are rebuilt from the original SCRBenchmark label files because the consolidated per-cell table stores unaligned numeric labels for these methods.
- Missing per-cell rows are backfilled from matching `final_clustering_labels.csv`, original label files, or `ClassWise` summaries when available.
- Grey cells mean that no compatible prediction/classwise source was available for that method/dataset.
- Cell types are ordered by increasing support, so the rarest labels appear at the top.
