# Self-contained figure bundle

This directory is intended to remain interpretable even if the other folders in
`/data2/fbidet/scRAW_Inductif/results` are deleted.

## Contents

- `*.png`: final figures.
- `standalone_tables/combined_summary.csv`: normalized split-level metrics
  from the main aggregation.
- `standalone_tables/balanced_acc_per_split.csv`: per-split metric table used
  to derive the dataset-level boxplot values, including `BalancedACC`.
- `standalone_tables/balanced_acc_dataset_algorithm_summary.csv`: `BalancedACC`
  summary by dataset and algorithm.
- `standalone_tables/dataset_level_metric_summary.csv`: dataset-level values
  used for the global boxplots. Each row is the mean of one dataset, algorithm
  and metric over the available inductive splits.
- `standalone_tables/dataset_level_counts_by_metric_algorithm.csv`: dataset
  counts and summary means/medians used in the boxplot legends.
- `standalone_tables/mean_std_by_dataset_algorithm.csv`: summary statistics.
- `standalone_tables/dataset_split_manifest.csv`: dataset choices and splits.
- `standalone_tables/per_dataset_summaries/`: copied source summaries where available.
- `standalone_tables/standalone_metadata.json`: provenance and copied-file manifest.

Rows in current combined table: 52.
Validation warnings: 0.
BalancedACC split-level values available: 52, with 13 split-level values per algorithm.

Global `*_by_algorithm_boxplot.png` figures now use dataset-level means rather
than pooled split-level scores. In those plots, one point is one dataset mean
for one algorithm, so Baron pancreas no longer has extra weight from its four
splits compared with datasets that have one or two splits.
