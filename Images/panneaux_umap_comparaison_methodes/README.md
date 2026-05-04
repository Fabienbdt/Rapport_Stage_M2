# UMAP visual panels for the M2 report

Source: `/data2/fbidet/scRAW_EXPERIMENTAL/results/baron_crossalgo_seed42_metrics/comparative_figures_report_refresh_20260429`.

This folder contains a report-ready reconstruction of the cross-algorithm UMAP figure:

- `baron_crossalgo_umap_panel.png`
- `baron_crossalgo_umap_panel.png`
- `panneaux_sources/`: copied source panels used to build the figure
- `source_manifest.json`: copied metadata from the refreshed comparative UMAP run

Rows: scRAW, PCA + Leiden, PCA + K-means, scMAE, scDeepCluster, scNAME.
Columns: ground truth, predicted clusters, misclassifications, cell rarity / scRAW reconstruction weights.

## Error counts

| Method | Correct | Error normal | Error rare | Error ultrarare |
|---|---:|---:|---:|---:|
| scRAW | 8105 | 427 | 19 | 18 |
| PCA + Leiden | 7757 | 314 | 435 | 63 |
| PCA + K-means | 6879 | 1161 | 493 | 36 |
| scMAE | 6262 | 1891 | 380 | 36 |
| scDeepCluster | 4876 | 3314 | 327 | 52 |
| scNAME | 6330 | 2123 | 53 | 63 |
