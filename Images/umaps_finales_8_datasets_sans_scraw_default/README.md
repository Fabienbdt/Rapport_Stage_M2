# Common-8 final UMAPs without scRAW default

Source table: `/data2/fbidet/scRAW_EXPERIMENTAL/results/presentation_trial206_nonbaron_20260324/00_source_tables/trial206_all_results_table.csv`.
Scope: 8 report datasets and report8 methods, excluding `scRAW (default)`.

Methods: scRAW, scRAW finetune, scNAME, scMAE, Harmony, ComBat, Scanorama, DESC, PCA + Leiden, scVI.

Statuses:

- `regenerated_from_embeddings`: UMAP recomputed from saved embeddings and labels.
- `regenerated_from_per_cell`: UMAP read from saved per-cell scRAW output and replotted.
- `copied_stage_final_umap`: only a stage-final scRAW trial_0017 UMAP image was available.
- `copied_existing_scrbenchmark_umap`: SCRBenchmark UMAP image was available but embeddings were not saved.
- `missing_or_failed:*`: no usable artifact was found.
