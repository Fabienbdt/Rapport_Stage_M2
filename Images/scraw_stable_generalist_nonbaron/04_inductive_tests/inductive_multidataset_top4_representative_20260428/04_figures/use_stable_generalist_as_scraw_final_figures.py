#!/usr/bin/env python3
"""Rebuild final inductive figures with scRAW values replaced by stable_generalist.

The public label remains "scRAW"; only the underlying scRAW rows are swapped
from the original default run to the exact stable_generalist rerun.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


FIGURES_DIR = Path(__file__).resolve().parent
TABLES_DIR = FIGURES_DIR / "standalone_tables"
COMPARISON_TABLE = (
    Path("/data2/fbidet/scRAW_EXPERIMENTAL/results/00_inductif_comparaison_algorithmes_scraw")
    / "15_scraw_stable_generalist_exact_comparison"
    / "tables"
    / "per_split_values_with_stable_generalist.csv"
)
TRIAL_SOURCE_RUN = (
    "/data2/fbidet/scRAW_Inductif/results/"
    "inductive_scraw_stable_generalist_exact_all_datasets_20260507_145430"
)

METRICS = ["ACC", "BalancedACC", "ARI", "NMI", "RareACC", "UltraRareACC"]
METRIC_LABELS = {
    "ACC": "ACC",
    "BalancedACC": "Balanced ACC",
    "ARI": "ARI",
    "NMI": "NMI",
    "RareACC": "Rare ACC",
    "UltraRareACC": "Ultra Rare ACC",
}
ALGORITHM_ORDER = ["scraw", "scname", "sc_mae", "scdeepcluster"]
ALGORITHM_LABELS = {
    "scraw": "scRAW",
    "scname": "scNAME",
    "sc_mae": "scMAE",
    "scdeepcluster": "scDeepCluster",
}
DATASET_ORDER = [
    "baron_human_pancreas",
    "bbag094_spleen",
    "gse112013_human_testis_raw_counts",
    "kang_pbmc_gse96583_singlets_raw_counts",
    "macaque_retina_gse118480_bipolar_raw_counts",
    "pancreas_raw_counts_four_batches_celseq_celseq2_fluidigmc1_smartseq2",
]
DATASET_LABELS = {
    "baron_human_pancreas": "Baron pancreas",
    "bbag094_spleen": "BBAG094 spleen",
    "gse112013_human_testis_raw_counts": "Human testis",
    "kang_pbmc_gse96583_singlets_raw_counts": "Kang PBMC",
    "macaque_retina_gse118480_bipolar_raw_counts": "Macaque retina",
    "pancreas_raw_counts_four_batches_celseq_celseq2_fluidigmc1_smartseq2": "Pancreas 4 batches",
}
DATASET_FIGURE_NAMES = {
    "baron_human_pancreas": "baron_pancreas",
    "bbag094_spleen": "bbag094_spleen",
    "gse112013_human_testis_raw_counts": "human_testis",
    "kang_pbmc_gse96583_singlets_raw_counts": "kang_pbmc",
    "macaque_retina_gse118480_bipolar_raw_counts": "macaque_retina",
    "pancreas_raw_counts_four_batches_celseq_celseq2_fluidigmc1_smartseq2": "pancreas_4_batches",
}
PALETTE = {
    "scRAW": "#9fc9c9",
    "scNAME": "#9bb7f0",
    "scMAE": "#f3c994",
    "scDeepCluster": "#c7a8f5",
}


def configure_matplotlib() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 16,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [float("nan")] * len(p_values)
    valid = [(idx, value) for idx, value in enumerate(p_values) if np.isfinite(value)]
    valid.sort(key=lambda item: item[1])
    running_max = 0.0
    m = len(valid)
    for rank, (idx, value) in enumerate(valid):
        corrected = min(1.0, (m - rank) * value)
        running_max = max(running_max, corrected)
        adjusted[idx] = running_max
    return adjusted


def paired_greater(first: pd.Series, second: pd.Series) -> tuple[float, float]:
    paired = pd.concat([first, second], axis=1).dropna()
    if paired.shape[0] < 2:
        return float("nan"), float("nan")
    differences = paired.iloc[:, 0].to_numpy(dtype=float) - paired.iloc[:, 1].to_numpy(dtype=float)
    if np.allclose(differences, 0.0):
        return 0.0, 0.5
    if np.isclose(float(np.std(differences, ddof=1)), 0.0):
        mean_diff = float(np.mean(differences))
        if mean_diff > 0:
            return float("inf"), 0.0
        if mean_diff < 0:
            return float("-inf"), 1.0
        return 0.0, 0.5
    result = stats.ttest_rel(paired.iloc[:, 0], paired.iloc[:, 1], alternative="greater")
    return float(result.statistic), float(result.pvalue)


def load_final_per_split() -> pd.DataFrame:
    source = pd.read_csv(COMPARISON_TABLE)
    base = source[~source["algorithm"].isin(["scraw", "scraw_stable_generalist"])].copy()
    trial = source[source["algorithm"] == "scraw_stable_generalist"].copy()
    trial["algorithm"] = "scraw"
    trial["algorithm_label"] = "scRAW"
    trial["source"] = "scraw_stable_generalist_exact_used_as_scraw_final"
    final = pd.concat([base, trial], ignore_index=True, sort=False)
    for metric in METRICS:
        final[metric] = pd.to_numeric(final[metric], errors="coerce")
    final["algorithm"] = pd.Categorical(final["algorithm"], ALGORITHM_ORDER, ordered=True)
    final["dataset_key"] = pd.Categorical(final["dataset_key"], DATASET_ORDER, ordered=True)
    return final.sort_values(["dataset_key", "algorithm", "test_group"]).reset_index(drop=True)


def build_dataset_summary(per_split: pd.DataFrame) -> pd.DataFrame:
    long_df = per_split.melt(
        id_vars=["dataset_key", "dataset_label", "algorithm", "algorithm_label"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    summary = (
        long_df.groupby(
            ["dataset_key", "dataset_label", "algorithm", "algorithm_label", "metric"],
            observed=True,
        )
        .agg(
            n_splits=("value", "count"),
            mean=("value", "mean"),
            std=("value", "std"),
            min=("value", "min"),
            max=("value", "max"),
        )
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    summary["metric_label"] = summary["metric"].map(METRIC_LABELS)
    summary["algorithm"] = pd.Categorical(summary["algorithm"], ALGORITHM_ORDER, ordered=True)
    summary["dataset_key"] = pd.Categorical(summary["dataset_key"], DATASET_ORDER, ordered=True)
    return summary[
        [
            "dataset_key",
            "dataset_label",
            "algorithm",
            "algorithm_label",
            "metric",
            "metric_label",
            "n_splits",
            "mean",
            "std",
            "min",
            "max",
        ]
    ].sort_values(["dataset_key", "algorithm", "metric"]).reset_index(drop=True)


def write_tables(per_split: pd.DataFrame, summary: pd.DataFrame) -> None:
    per_split.to_csv(TABLES_DIR / "combined_summary.csv", index=False)
    per_split.to_csv(TABLES_DIR / "balanced_acc_per_split.csv", index=False)
    summary.to_csv(TABLES_DIR / "dataset_level_metric_summary.csv", index=False)

    balanced = summary[summary["metric"] == "BalancedACC"].copy()
    balanced = balanced.rename(columns={"n_splits": "n"})
    balanced[
        ["dataset_key", "dataset_label", "algorithm", "algorithm_label", "metric", "n", "mean", "std", "min", "max"]
    ].to_csv(TABLES_DIR / "balanced_acc_dataset_algorithm_summary.csv", index=False)

    counts = (
        per_split[["algorithm", "algorithm_label", "BalancedACC"]]
        .dropna(subset=["BalancedACC"])
        .groupby(["algorithm", "algorithm_label"], observed=True)
        .size()
        .reset_index(name="n_balanced_acc")
    )
    counts.to_csv(TABLES_DIR / "balanced_acc_counts_by_algorithm.csv", index=False)

    dataset_counts = (
        summary.groupby(["metric", "algorithm", "algorithm_label"], observed=True)
        .agg(
            n_datasets=("mean", "count"),
            dataset_level_mean=("mean", "mean"),
            dataset_level_median=("mean", "median"),
        )
        .reset_index()
    )
    dataset_counts.to_csv(TABLES_DIR / "dataset_level_counts_by_metric_algorithm.csv", index=False)

    wide_rows: list[dict[str, object]] = []
    for (dataset_key, algorithm), group in summary.groupby(["dataset_key", "algorithm"], observed=True):
        row: dict[str, object] = {
            "dataset_key": dataset_key,
            "algorithm": algorithm,
            "n_splits": int(group["n_splits"].max()),
        }
        for metric in ["ACC", "ARI", "NMI", "RareACC", "UltraRareACC"]:
            metric_row = group[group["metric"] == metric]
            if metric_row.empty:
                row[f"{metric}_mean"] = math.nan
                row[f"{metric}_std"] = math.nan
                row[f"{metric}_count"] = 0
            else:
                row[f"{metric}_mean"] = float(metric_row["mean"].iloc[0])
                row[f"{metric}_std"] = float(metric_row["std"].iloc[0])
                row[f"{metric}_count"] = int(metric_row["n_splits"].iloc[0])
        wide_rows.append(row)
    pd.DataFrame(wide_rows).sort_values(["dataset_key", "algorithm"]).to_csv(
        TABLES_DIR / "mean_std_by_dataset_algorithm.csv",
        index=False,
    )

    metadata = {
        "description": "Self-contained final figure bundle with scRAW values replaced by exact stable_generalist results.",
        "figures_dir": str(FIGURES_DIR),
        "scraw_label_source": "scRAW points use stable_generalist exact rerun, not the older default scRAW run.",
        "trial_source_run": TRIAL_SOURCE_RUN,
        "source_table": str(COMPARISON_TABLE),
        "n_rows": int(len(per_split)),
        "datasets": DATASET_ORDER,
        "metrics": METRICS,
        "algorithms": ALGORITHM_ORDER,
    }
    (TABLES_DIR / "standalone_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def metric_data(summary: pd.DataFrame, metric: str) -> list[pd.Series]:
    metric_df = summary[summary["metric"] == metric]
    return [
        pd.to_numeric(
            metric_df.loc[metric_df["algorithm_label"] == ALGORITHM_LABELS[algorithm], "mean"],
            errors="coerce",
        ).dropna()
        for algorithm in ALGORITHM_ORDER
    ]


def plot_global_without_significance(summary: pd.DataFrame, metric: str, suffix: str) -> None:
    metric_df = summary[summary["metric"] == metric]
    metric_label = METRIC_LABELS[metric]
    data = metric_data(summary, metric)
    positions = np.arange(1, len(ALGORITHM_ORDER) + 1)

    fig, ax = plt.subplots(figsize=(11.6, 6.6))
    box = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True, showmeans=True)
    for patch, algorithm in zip(box["boxes"], ALGORITHM_ORDER):
        patch.set_facecolor(PALETTE[ALGORITHM_LABELS[algorithm]])
        patch.set_alpha(0.55)
        patch.set_edgecolor("#6b7280")

    for x, values in zip(positions, data):
        values_np = values.to_numpy(dtype=float)
        offsets = np.linspace(-0.06, 0.06, len(values_np)) if len(values_np) > 1 else np.array([0.0])
        ax.scatter(x + offsets, values_np, s=34, color="#111827", alpha=0.9, zorder=3)

    if suffix == "__scraw_brackets":
        wide = metric_df.pivot_table(
            index="dataset_key",
            columns="algorithm_label",
            values="mean",
            aggfunc="mean",
            observed=True,
        )
        p_rows: list[dict[str, object]] = []
        for comparator in ["scNAME", "scMAE", "scDeepCluster"]:
            t_stat, p_value = paired_greater(wide["scRAW"], wide[comparator])
            p_rows.append({"comparator": comparator, "t": t_stat, "p": p_value})
        p_adj = holm_adjust([row["p"] for row in p_rows])
        y0 = min(1.14, max(1.02, float(metric_df["mean"].max()) + 0.055))
        for i, (row, adj) in enumerate(zip(p_rows, p_adj)):
            x1, x2 = 1, ALGORITHM_ORDER.index({v: k for k, v in ALGORITHM_LABELS.items()}[row["comparator"]]) + 1
            y = y0 + i * 0.045
            ax.plot([x1, x1, x2, x2], [y, y + 0.012, y + 0.012, y], color="#374151", lw=1.0)
            ax.text((x1 + x2) / 2, y + 0.014, stars(adj), ha="center", va="bottom", fontsize=9)

    counts = [len(values.dropna()) for values in data]
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{ALGORITHM_LABELS[a]}\nn={count}" for a, count in zip(ALGORITHM_ORDER, counts)])
    ax.set_ylabel(metric_label)
    ax.set_ylim(-0.02, 1.22)
    ax.grid(axis="y", color="#d1d5db", linewidth=0.85, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{metric_label} by algorithm - dataset-level means", loc="left", pad=20)
    ax.text(
        0.0,
        1.03,
        "scRAW values are from exact stable_generalist; each point is one dataset mean.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#4b5563",
    )
    fig.savefig(FIGURES_DIR / f"{metric}_by_algorithm_boxplot{suffix}.png", bbox_inches="tight")
    plt.close(fig)


def plot_mean_by_dataset(summary: pd.DataFrame, metric: str) -> None:
    metric_df = summary[summary["metric"] == metric]
    pivot = (
        metric_df.pivot_table(index="dataset_label", columns="algorithm_label", values="mean", observed=True)
        .reindex(index=[DATASET_LABELS[key] for key in DATASET_ORDER], columns=[ALGORITHM_LABELS[a] for a in ALGORITHM_ORDER])
    )
    x = np.arange(len(pivot.index))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13.6, 6.2))
    for idx, algorithm in enumerate(ALGORITHM_ORDER):
        label = ALGORITHM_LABELS[algorithm]
        ax.bar(
            x + (idx - 1.5) * width,
            pivot[label].to_numpy(dtype=float),
            width=width,
            color=PALETTE[label],
            edgecolor="#6b7280",
            linewidth=0.7,
            label=label,
        )
    ax.set_title(f"Mean {METRIC_LABELS[metric]} by dataset and algorithm", loc="left")
    ax.text(
        0.0,
        1.02,
        "scRAW bars use exact stable_generalist values.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#4b5563",
    )
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=22, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.savefig(FIGURES_DIR / f"mean_{metric}_by_dataset_algorithm.png", bbox_inches="tight")
    plt.close(fig)


def plot_dataset_balanced(per_split: pd.DataFrame, dataset_key: str) -> None:
    dataset_df = per_split[per_split["dataset_key"] == dataset_key]
    x = np.arange(len(ALGORITHM_ORDER))
    means = []
    values_by_algorithm = []
    for algorithm in ALGORITHM_ORDER:
        values = dataset_df[dataset_df["algorithm"] == algorithm]["BalancedACC"].dropna().to_numpy(dtype=float)
        values_by_algorithm.append(values)
        means.append(float(np.mean(values)) if len(values) else math.nan)

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    ax.bar(
        x,
        means,
        color=[PALETTE[ALGORITHM_LABELS[a]] for a in ALGORITHM_ORDER],
        edgecolor="#6b7280",
        linewidth=0.8,
        alpha=0.72,
    )
    for idx, values in enumerate(values_by_algorithm):
        offsets = np.linspace(-0.07, 0.07, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(np.full(len(values), idx) + offsets, values, color="#111827", s=28, zorder=3)
        ax.text(idx, 0.02, f"n={len(values)}", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"{DATASET_LABELS[dataset_key]} - Balanced ACC", loc="left")
    ax.text(
        0.0,
        1.02,
        "scRAW uses exact stable_generalist values.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#4b5563",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([ALGORITHM_LABELS[a] for a in ALGORITHM_ORDER], rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIGURES_DIR / f"{DATASET_FIGURE_NAMES[dataset_key]}_BalancedACC_4algorithms.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    configure_matplotlib()
    per_split = load_final_per_split()
    summary = build_dataset_summary(per_split)
    write_tables(per_split, summary)

    for metric in METRICS:
        plot_global_without_significance(summary, metric, "__without_significance")
        plot_global_without_significance(summary, metric, "__scraw_brackets")
    for metric in ["ACC", "ARI", "BalancedACC", "NMI"]:
        plot_mean_by_dataset(summary, metric)
    for dataset_key in DATASET_ORDER:
        plot_dataset_balanced(per_split, dataset_key)

    print(f"wrote_tables={TABLES_DIR}")
    print(f"wrote_figures={FIGURES_DIR}")
    print("scRAW label now uses exact stable_generalist values")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
