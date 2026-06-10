"""
Recompute BalancedRareACC per algorithm from raw result files
for the loss transfer experiment (scraw_loss_integration_matrix_stable_default_20260428_104305).

This script:
1. Walks 02_full_runs/runs/<dataset>/<variant>/scrbenchmark/results/analysis_results.csv
   for scMAE and scDeepCluster (ClassWise in CSV)
2. Walks 02_full_runs/runs/<dataset>/<variant>/desc/runs/seed_*/results/results.json
   for DESC (ClassWise in JSON)
3. Computes BalancedRareACC per-algorithm per-seed (from ClassWise metric)
4. Aggregates by (dataset, algorithm, variant) -> mean, std, n
5. Replaces/adds BalancedRareACC rows in loss_transfer_by_dataset_algorithm_variant.csv
6. Regenerates the all_configurations boxplot (4-metric 2x2 grid)
7. Prints a summary table for updating the LaTeX tables
"""

import ast
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────────
MATRIX_ROOT = Path(
    "/data2/fbidet/scRAW_EXPERIMENTAL/results/"
    "scraw_loss_integration_matrix_stable_default_20260428_104305"
)
FULL_RUNS_ROOT = MATRIX_ROOT / "02_full_runs" / "runs"
IMPORTED_ROOT = MATRIX_ROOT / "03_imported_existing"

SYNTH_DIR = Path(
    "/data2/fbidet/Rapport_Stage_M2_git/Images/scraw_loss_transfer_synthesis"
)
CSV_PATH = SYNTH_DIR / "loss_transfer_by_dataset_algorithm_variant.csv"

ALGORITHM_LABELS = {
    "sc_mae": "scMAE",
    "scmae": "scMAE",
    "sc_mae_scraw_weighted": "scMAE",
    "scdeepcluster": "scDeepCluster",
    "scdeepcluster_scraw_weighted": "scDeepCluster",
    "desc": "DESC",
    "desc_scraw_weighted": "DESC",
}

DATASET_LABELS = {
    "baron_human_pancreas": "baron_human_pancreas",
    "pancreas": "baron_human_pancreas",
    "bbag094_spleen": "bbag094_spleen",
    "bbag094_zeisel": "bbag094_zeisel",
    "kang_pbmc": "kang_pbmc",
    "paul15_bone_marrow": "paul15_bone_marrow",
}

VARIANT_ORDER = [
    "baseline",
    "density_only",
    "full_leiden",
    "full_kmeans",
    "full_leiden_triplet",
    "full_kmeans_triplet",
    "density_only_triplet_kmeans",
]

VARIANT_LABELS = {
    "baseline": "baseline",
    "density_only": "density",
    "full_kmeans": "weighted+kmeans",
    "full_leiden": "weighted+leiden",
    "density_only_triplet_kmeans": "density+triplet",
    "full_kmeans_triplet": "weighted+kmeans+triplet",
    "full_leiden_triplet": "weighted+leiden+triplet",
}

PALETTE = {
    "baseline": "#6b7280",
    "full_leiden": "#2563eb",
    "full_kmeans": "#7c3aed",
    "density_only": "#0f766e",
    "full_leiden_triplet": "#dc2626",
    "full_kmeans_triplet": "#ea580c",
    "density_only_triplet_kmeans": "#0891b2",
}


def compute_bra(row, total_cells_col="n_samples_evaluated"):
    """Compute BalancedRareACC for a single row using ClassWise metric."""
    cw = row.get("ClassWise", None)
    total = row.get(total_cells_col, None)
    if pd.isna(cw) if isinstance(cw, float) else not cw:
        return float("nan")
    if total is None or pd.isna(total) or total <= 0:
        return float("nan")
    try:
        cw_dict = ast.literal_eval(str(cw))
    except Exception:
        return float("nan")
    rare_recalls = [
        vals.get("Recall", 0.0)
        for vals in cw_dict.values()
        if vals.get("Support", 0) / total < 0.05
    ]
    return float(np.mean(rare_recalls)) if rare_recalls else float("nan")


def extract_dataset_variant_from_full_runs(path: Path):
    """Infer dataset and variant from a path under 02_full_runs/runs/<dataset>/<variant>/."""
    parts = path.parts
    try:
        # Find the 'runs' directory under FULL_RUNS_ROOT
        # FULL_RUNS_ROOT = .../02_full_runs/runs
        # path structure: .../runs/<dataset>/<variant>/...
        full_runs_str = str(FULL_RUNS_ROOT)
        path_str = str(path)
        if not path_str.startswith(full_runs_str):
            return None, None
        rel = Path(path_str[len(full_runs_str):].lstrip("/"))
        rel_parts = rel.parts
        if len(rel_parts) < 2:
            return None, None
        dataset = rel_parts[0]
        variant = rel_parts[1]
        return DATASET_LABELS.get(dataset, dataset), variant
    except Exception:
        return None, None


# ── Walk full runs: scMAE/scDeepCluster from scrbenchmark/analysis_results.csv ──
records = []
scrbenchmark_files = list(FULL_RUNS_ROOT.rglob("scrbenchmark/results/analysis_results.csv"))
print(f"Found {len(scrbenchmark_files)} scrbenchmark analysis_results.csv files.")

for csv_file in scrbenchmark_files:
    dataset, variant = extract_dataset_variant_from_full_runs(csv_file)
    if dataset is None or variant is None:
        continue
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"  Error reading {csv_file}: {e}")
        continue

    if "ClassWise" not in df.columns:
        continue

    for _, row in df.iterrows():
        raw_algo = str(row.get("algorithm", "")).strip().lower()
        algo_label = ALGORITHM_LABELS.get(raw_algo, None)
        if algo_label is None:
            continue
        bra = compute_bra(row)
        if not np.isnan(bra):
            records.append({
                "dataset": dataset,
                "algorithm": algo_label,
                "variant": variant,
                "BalancedRareACC": bra,
            })

# ── Walk full runs: DESC from desc/runs/seed_*/results/results.json ─────────
desc_json_files = list(FULL_RUNS_ROOT.rglob("desc/runs/*/results/results.json"))
print(f"Found {len(desc_json_files)} DESC results.json files.")

for json_file in desc_json_files:
    dataset, variant = extract_dataset_variant_from_full_runs(json_file)
    if dataset is None or variant is None:
        continue
    try:
        with open(json_file) as f:
            data = json.load(f)
        results_list = data.get("results", [])
        if not results_list:
            continue
        item = results_list[0]  # one result per JSON
        metrics = item.get("metrics", {})
        cw = metrics.get("ClassWise", None)
        total = metrics.get("n_samples_evaluated", None)
        if cw is None or total is None or total <= 0:
            continue
        rare_recalls = [
            vals.get("Recall", 0.0)
            for vals in cw.values()
            if vals.get("Support", 0) / total < 0.05
        ]
        bra = float(np.mean(rare_recalls)) if rare_recalls else float("nan")
        if not np.isnan(bra):
            records.append({
                "dataset": dataset,
                "algorithm": "DESC",
                "variant": variant,
                "BalancedRareACC": bra,
            })
    except Exception as e:
        print(f"  Error reading {json_file}: {e}")
        continue

print(f"Total per-seed BRA records: {len(records)}")
bra_df = pd.DataFrame(records)
print("Unique algorithms:", bra_df["algorithm"].unique())
print("Unique datasets:", bra_df["dataset"].unique())

# ── Aggregate by (dataset, algorithm, variant) ────────────────────────────────
agg = bra_df.groupby(["dataset", "algorithm", "variant"])["BalancedRareACC"].agg(
    mean="mean", std="std", n="count"
).reset_index()
agg.columns = ["dataset", "algorithm", "variant", "mean", "std", "n"]
agg["metric"] = "BalancedRareACC"
agg = agg[["dataset", "algorithm", "variant", "metric", "mean", "std", "n"]]

print(f"\nAggregated rows: {len(agg)}")

# ── Update loss_transfer CSV ──────────────────────────────────────────────────
lt = pd.read_csv(CSV_PATH)
lt_clean = lt[lt["metric"] != "BalancedRareACC"]
lt_updated = pd.concat([lt_clean, agg], ignore_index=True)
lt_updated = lt_updated.sort_values(
    ["dataset", "algorithm", "variant", "metric"]
).reset_index(drop=True)
lt_updated.to_csv(CSV_PATH, index=False)
print(f"Updated CSV: {CSV_PATH} ({len(lt_updated)} rows, {lt_updated['metric'].unique()})")

# ── Print table values for LaTeX ──────────────────────────────────────────────
metrics = ["ARI", "RareACC", "UltraRareACC", "BalancedRareACC"]
lt_sub = lt_updated[lt_updated["metric"].isin(metrics)]
pivot = lt_sub.groupby(["algorithm", "variant", "metric"])["mean"].mean().reset_index()
pivot = pivot.pivot_table(
    index=["algorithm", "variant"], columns="metric", values="mean"
).reset_index()

print("\n" + "=" * 80)
print("LaTeX table values (mean across datasets)")
print("=" * 80)

for algo in ["scMAE", "scDeepCluster", "DESC"]:
    print(f"\n=== {algo} ===")
    sub = pivot[pivot["algorithm"] == algo].copy()
    sub["order"] = sub["variant"].map({v: i for i, v in enumerate(VARIANT_ORDER)})
    sub = sub.sort_values("order")
    for _, row in sub.iterrows():
        name = VARIANT_LABELS.get(row["variant"], row["variant"])
        ari = row.get("ARI", float("nan"))
        rare = row.get("RareACC", float("nan"))
        ultra = row.get("UltraRareACC", float("nan"))
        bra = row.get("BalancedRareACC", float("nan"))
        score = np.nanmean([ari, rare, ultra, bra])
        print(
            f"  {name}: ARI={ari:.3f}, RareACC={rare:.3f}, "
            f"UltraRareACC={ultra:.3f}, BalancedRareACC={bra:.3f}, Score(4-metric)={score:.3f}"
        )

# ── Regenerate boxplot (2x2: ARI, RareACC, UltraRareACC, BalancedRareACC) ────
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
axes = axes.flatten()

metric_titles = {
    "ARI": "ARI",
    "RareACC": "RareACC",
    "UltraRareACC": "UltraRareACC",
    "BalancedRareACC": "BalancedRareACC",
}

for ax, metric in zip(axes, ["ARI", "RareACC", "UltraRareACC", "BalancedRareACC"]):
    df_m = lt_updated[lt_updated["metric"] == metric].copy()
    df_m["variant"] = pd.Categorical(
        df_m["variant"], categories=VARIANT_ORDER, ordered=True
    )
    df_m = df_m.sort_values("variant")

    # Aggregate mean across datasets for each algorithm/variant
    df_agg = df_m.groupby(["dataset", "algorithm", "variant"])["mean"].mean().reset_index()

    sns.boxplot(
        data=df_agg,
        x="algorithm",
        y="mean",
        hue="variant",
        hue_order=VARIANT_ORDER,
        palette=PALETTE,
        showfliers=False,
        width=0.7,
        linewidth=1.2,
        ax=ax,
    )
    sns.stripplot(
        data=df_agg,
        x="algorithm",
        y="mean",
        hue="variant",
        hue_order=VARIANT_ORDER,
        dodge=True,
        jitter=0.15,
        alpha=0.6,
        size=5,
        color="black",
        ax=ax,
        legend=False,
    )
    ax.set_title(metric_titles[metric], fontweight="bold", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("Score moyen", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.get_legend().remove()

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
new_labels = [VARIANT_LABELS.get(l, l) for l in labels]
fig.legend(
    handles,
    new_labels,
    title="Variante scRAW",
    fontsize=9,
    title_fontsize=10,
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.04),
    frameon=True,
)

fig.suptitle(
    "Distribution des scores de transfert de loss par algorithme, variante et métrique",
    fontsize=13,
    fontweight="bold",
    y=1.01,
)
plt.tight_layout()

output_pdf = SYNTH_DIR / "loss_transfer_all_configurations_boxplot_4metrics.pdf"
output_png = SYNTH_DIR / "loss_transfer_all_configurations_boxplot_4metrics.png"
fig.savefig(output_pdf, bbox_inches="tight")
fig.savefig(output_png, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"\nSaved: {output_pdf}")
print(f"Saved: {output_png}")
print("\nDone.")
