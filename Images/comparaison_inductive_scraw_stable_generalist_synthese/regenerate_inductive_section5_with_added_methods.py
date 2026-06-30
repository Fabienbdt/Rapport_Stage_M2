#!/usr/bin/env python3
"""Regenerate section 5 inductive plots after adding inductive baseline methods."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_ADDITIONAL_ROOT = Path(
    "/data2/fbidet/scRAW_Inductif/results/inductive_section5_added_methods_20260529"
)

ADDED_ALGORITHMS = ["scaide", "pca_harmony", "scvi"]
ALGORITHM_ORDER = ["scraw", "scname", "sc_mae", "scdeepcluster", "scaide", "scvi"]
ALGORITHM_LABELS = {
    "scraw": "scRAW",
    "scname": "scNAME",
    "sc_mae": "scMAE",
    "scdeepcluster": "scDeepCluster",
    "scaide": "scAIDE",
    "scvi": "scVI",
}
PALETTE = {
    "scRAW": "#9fc9c9",
    "scNAME": "#9bb7f0",
    "scMAE": "#f3c994",
    "scDeepCluster": "#c7a8f5",
    "scAIDE": "#7fbc8c",
    "PCA+Harmony": "#f08a6c",
    "scVI": "#d6c84f",
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
METRICS = ["ARI", "ACC", "BalancedACC", "RareACC", "BalancedRareACC", "UltraRareACC"]
ALL_METRICS = ["ACC", "BalancedACC", "ARI", "NMI", "RareACC", "UltraRareACC", "BalancedRareACC"]
METRIC_LABELS = {
    "ACC": "ACC",
    "BalancedACC": "Balanced ACC",
    "ARI": "ARI",
    "NMI": "NMI",
    "RareACC": "Rare ACC",
    "BalancedRareACC": "Balanced Rare ACC",
    "UltraRareACC": "Ultra Rare ACC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--additional-root", default=str(DEFAULT_ADDITIONAL_ROOT))
    parser.add_argument("--allow-empty-added", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def coerce_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def metric_from_json(output_dir: str, metric: str) -> float:
    if not output_dir:
        return float("nan")
    path = Path(output_dir) / "metrics.json"
    if not path.exists():
        path = Path(output_dir) / "results.json"
        if not path.exists():
            path = Path(output_dir) / "results/results.json"
            if not path.exists():
                path = Path(output_dir) / "results/metrics.json"
                if not path.exists():
                    return float("nan")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return float("nan")
    metrics = payload.get("metrics", payload.get("test_metrics", payload))
    if metric == "BalancedRareACC" and "ClassWise" in metrics:
        classwise = metrics["ClassWise"]
        total_cells = sum(item["Support"] for item in classwise.values())
        if total_cells == 0:
            return float("nan")
        rare_classes = {name: item for name, item in classwise.items() if (item["Support"] / total_cells) < 0.05}
        if not rare_classes:
            return float("nan")
        recalls = [item["Recall"] for item in rare_classes.values()]
        return float(np.mean(recalls))
    return coerce_float(metrics.get(metric))


def dataset_key_from_name(name: str) -> str:
    value = str(name)
    aliases = {
        "kang_pbmc_gse96583_shared_sample_train_donor_test": "kang_pbmc_gse96583_singlets_raw_counts",
    }
    if value in aliases:
        return aliases[value]
    if value in DATASET_LABELS:
        return value
    for key in DATASET_LABELS:
        if value.startswith(key):
            return key
    return value


def load_added_rows(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted((root / "01_runs").glob("**/summary.csv")):
        summary = read_csv(summary_path)
        if summary.empty:
            continue
        for row in summary.to_dict(orient="records"):
            algorithm = str(row.get("algorithm", ""))
            if algorithm not in ADDED_ALGORITHMS or str(row.get("status", "")) != "ok":
                continue
            dataset_key = dataset_key_from_name(str(row.get("dataset_name", "")))
            output_dir = str(row.get("output_dir", ""))
            out: dict[str, Any] = {
                "dataset_key": dataset_key,
                "dataset_label": DATASET_LABELS.get(dataset_key, dataset_key),
                "algorithm": algorithm,
                "algorithm_label": ALGORITHM_LABELS[algorithm],
                "split_key": row.get("split_key", ""),
                "train_groups": row.get("train_batches", ""),
                "test_group": row.get("test_batch", ""),
                "source": root.name,
                "output_dir": output_dir,
                "BalancedACC_source_file": str(Path(output_dir) / "metrics.json") if output_dir else "",
                "BalancedACC_source_key": "metrics.BalancedACC",
            }
            for metric in ALL_METRICS:
                if metric in row and not pd.isna(row.get(metric)):
                    out[metric] = coerce_float(row.get(metric))
                else:
                    out[metric] = metric_from_json(output_dir, metric)
            rows.append(out)
    return pd.DataFrame(rows)


def build_dataset_summary(per_split: pd.DataFrame) -> pd.DataFrame:
    for metric in ALL_METRICS:
        per_split[metric] = pd.to_numeric(per_split.get(metric), errors="coerce")
    long_df = per_split.melt(
        id_vars=["dataset_key", "dataset_label", "algorithm", "algorithm_label"],
        value_vars=ALL_METRICS,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    summary = (
        long_df.groupby(["dataset_key", "dataset_label", "algorithm", "algorithm_label", "metric"], observed=True)
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
    summary["dataset_short"] = summary["dataset_key"].map(DATASET_LABELS)
    summary["algorithm"] = pd.Categorical(summary["algorithm"], ALGORITHM_ORDER, ordered=True)
    summary["dataset_key"] = pd.Categorical(summary["dataset_key"], DATASET_ORDER, ordered=True)
    return summary.sort_values(["dataset_key", "algorithm", "metric"]).reset_index(drop=True)


def fmt_fr(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    return f"{value:.3f}".replace(".", ",")


def tex_escape(value: Any) -> str:
    return str(value).replace("&", "\\&").replace("_", "\\_")


def write_metric_value_table(summary: pd.DataFrame) -> None:
    rows: list[dict[str, Any]] = []
    selected = summary[summary["metric"].isin(METRICS)].copy()
    for metric in METRICS:
        metric_df = selected[selected["metric"] == metric]
        for dataset_key in DATASET_ORDER:
            row: dict[str, Any] = {
                "metric": METRIC_LABELS[metric],
                "dataset": DATASET_LABELS[dataset_key],
            }
            for algorithm in ALGORITHM_ORDER:
                match = metric_df[
                    (metric_df["dataset_key"].astype(str) == dataset_key)
                    & (metric_df["algorithm"].astype(str) == algorithm)
                ]
                row[algorithm] = float(match["mean"].iloc[0]) if not match.empty else math.nan
            rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(HERE / "inductive_dataset_metric_values.csv", index=False)

    align = "l l " + " ".join(["c"] * len(ALGORITHM_ORDER))
    header = " & ".join(
        ["\\textbf{Métrique}", "\\textbf{Jeu de données}"]
        + [f"\\textbf{{{tex_escape(ALGORITHM_LABELS[a])}}}" for a in ALGORITHM_ORDER]
    )
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\scriptsize",
        "\\renewcommand{\\arraystretch}{1.06}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]
    for metric in METRICS:
        metric_rows = table[table["metric"] == METRIC_LABELS[metric]].reset_index(drop=True)
        for idx, row in metric_rows.iterrows():
            values = [coerce_float(row[a]) for a in ALGORITHM_ORDER]
            finite = [v for v in values if np.isfinite(v)]
            best = max(finite) if finite else float("nan")
            metric_cell = tex_escape(row["metric"]) if idx == 0 else ""
            cells = [metric_cell, tex_escape(row["dataset"])]
            for value in values:
                text = fmt_fr(value)
                if np.isfinite(value) and np.isclose(value, best):
                    text = f"\\textbf{{{text}}}"
                cells.append(text)
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\addlinespace[0.25em]")
    if lines[-1] == "\\addlinespace[0.25em]":
        lines.pop()
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            (
                "\\caption{Moyennes par jeu de données des expériences inductives pour les "
                "métriques principales. Chaque valeur correspond à la moyenne des splits "
                "inductifs disponibles pour un couple jeu de données--algorithme ; la meilleure "
                "valeur de chaque ligne est en gras.}"
            ),
            "\\label{tab:inductive_dataset_metric_values}",
            "\\end{table}",
        ]
    )
    (HERE / "inductive_dataset_metric_values_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_transductive_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()
    
    # Filter by target datasets
    df = df[df["dataset_key"].isin(DATASET_ORDER)].copy()
    
    TRANS_METHOD_MAP = {
        "scraw": "scRAW",
        "scname": "scNAME",
        "sc_mae": "scMAE",
        "scdeepcluster": "DESC",
        "scaide": "scAIDE",
        "scvi": "scvi"
    }
    
    # Filter by mapped methods
    df = df[df["method"].isin(TRANS_METHOD_MAP.values())].copy()
    
    # Map back to algorithm names
    inv_map = {v: k for k, v in TRANS_METHOD_MAP.items()}
    df["algorithm"] = df["method"].map(inv_map)
    
    # Coerce metric columns to numeric
    for metric in ALL_METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        
    return df


def plot_boxplots(summary: pd.DataFrame, trans_df: pd.DataFrame, metrics_list: list[str], prefix: str) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 170,
            "savefig.dpi": 240,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
        }
    )
    n_metrics = len(metrics_list)
    if n_metrics == 6:
        fig, axes = plt.subplots(2, 3, figsize=(22.5, 10.4), sharey=True)
    elif n_metrics == 5:
        fig, axes = plt.subplots(2, 3, figsize=(22.5, 10.4), sharey=True)
    elif n_metrics == 4:
        fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.4), sharey=True)
    else:
        raise ValueError("Unsupported number of metrics")
        
    axes = axes.ravel()
    for idx, metric in enumerate(metrics_list):
        ax = axes[idx]
        metric_df = summary[summary["metric"] == metric]
        
        ind_data = []
        trans_data = []
        for algorithm in ALGORITHM_ORDER:
            # Inductive results (averaged over splits per dataset, then distribution over datasets)
            ind_values = pd.to_numeric(
                metric_df.loc[metric_df["algorithm"].astype(str) == algorithm, "mean"],
                errors="coerce",
            ).dropna()
            ind_data.append(ind_values.to_numpy(dtype=float))
            
            # Transductive results (complete dataset run per dataset, distribution over datasets)
            trans_values = pd.to_numeric(
                trans_df.loc[trans_df["algorithm"] == algorithm, metric],
                errors="coerce",
            ).dropna()
            trans_data.append(trans_values.to_numpy(dtype=float))
            
        positions = np.arange(1, len(ALGORITHM_ORDER) + 1)
        
        # 1. Inductive boxplots (left, colored)
        box_ind = ax.boxplot(
            ind_data,
            positions=positions - 0.18,
            widths=0.3,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "#111827",
                "markersize": 4.5,
            },
            medianprops={"color": "#111827", "linewidth": 1.2},
        )
        
        # Color inductive boxes
        for patch, algorithm in zip(box_ind["boxes"], ALGORITHM_ORDER):
            patch.set_facecolor(PALETTE[ALGORITHM_LABELS[algorithm]])
            patch.set_alpha(0.72)
            patch.set_edgecolor("#4b5563")
            patch.set_linewidth(0.9)
            
        # 2. Transductive boxplots (right, greyed out baseline)
        box_trans = ax.boxplot(
            trans_data,
            positions=positions + 0.18,
            widths=0.3,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "#111827",
                "markersize": 4.5,
            },
            medianprops={"color": "#111827", "linewidth": 1.2},
        )
        
        # Color transductive boxes
        for patch in box_trans["boxes"]:
            patch.set_facecolor("#cbd5e1")
            patch.set_alpha(0.6)
            patch.set_edgecolor("#64748b")
            patch.set_linewidth(0.9)
            
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [f"{ALGORITHM_LABELS[a]}" for a in ALGORITHM_ORDER],
            rotation=26,
            ha="right",
        )
        for label in ax.get_xticklabels():
            if label.get_text() == "scRAW":
                label.set_color("#dc2626")  # nice red color (tailwindcss red-600)
                label.set_fontweight("bold")
        ax.set_title(METRIC_LABELS[metric], loc="left", fontweight="bold")
        ax.set_ylim(-0.02, 1.08)
        ax.grid(axis="y", color="#d1d5db", linewidth=0.8, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        
    if n_metrics == 5:
        axes[-1].axis("off")
        axes[0].set_ylabel("Score")
        axes[3].set_ylabel("Score")
    elif n_metrics == 6:
        axes[0].set_ylabel("Score")
        axes[3].set_ylabel("Score")
    else:
        axes[0].set_ylabel("Score")
        axes[2].set_ylabel("Score")
        
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="white", edgecolor="#4b5563", label="Inductif (données nouvelles, 6 jeux de données)"),
        Patch(facecolor="#cbd5e1", edgecolor="#64748b", alpha=0.6, label="Baseline (jeu de données complet, transductif - 6 jeux de données)")
    ]
    
    fig.suptitle("Comparaison inductive par moyennes dataset", y=0.98, fontsize=15, fontweight="bold")
    fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, fontsize=11, frameon=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    
    for ext in ["png", "pdf"]:
        fig.savefig(HERE / f"{prefix}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(23.0, 9.8))
    axes = axes.ravel()
    for ax, metric in zip(axes, METRICS):
        metric_df = summary[summary["metric"] == metric]
        pivot = (
            metric_df.pivot_table(
                index="dataset_key",
                columns="algorithm",
                values="mean",
                aggfunc="mean",
                observed=True,
            )
            .reindex(index=DATASET_ORDER, columns=ALGORITHM_ORDER)
            .astype(float)
        )
        image = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="viridis")
        ax.set_title(METRIC_LABELS[metric], loc="left", fontweight="bold")
        ax.set_xticks(np.arange(len(ALGORITHM_ORDER)))
        ax.set_xticklabels([ALGORITHM_LABELS[a] for a in ALGORITHM_ORDER], rotation=28, ha="right")
        ax.set_yticks(np.arange(len(DATASET_ORDER)))
        ax.set_yticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
        for row_idx in range(pivot.shape[0]):
            for col_idx in range(pivot.shape[1]):
                value = pivot.iloc[row_idx, col_idx]
                if np.isfinite(value):
                    color = "white" if value < 0.55 else "#111827"
                    ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    axes[-1].axis("off")
    fig.suptitle("Moyennes inductives par dataset", y=0.995, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    for ext in ["png", "pdf"]:
        fig.savefig(HERE / f"inductive_metrics_by_dataset_heatmap.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    source_path = HERE / "inductive_per_split_combined_summary_source.csv"

    final = read_csv(source_path)
    if final.empty:
        raise SystemExit(f"Missing base per-split table: {source_path}")

    # Exclude pca_harmony
    final = final[final["algorithm"] != "pca_harmony"].copy()

    final["algorithm"] = pd.Categorical(final["algorithm"], ALGORITHM_ORDER, ordered=True)
    final["dataset_key"] = pd.Categorical(final["dataset_key"], DATASET_ORDER, ordered=True)
    final = final.sort_values(["dataset_key", "algorithm", "test_group"]).reset_index(drop=True)
    final.to_csv(source_path, index=False)

    summary = build_dataset_summary(final)
    selected = summary[summary["metric"].isin(METRICS)].copy()
    selected.to_csv(HERE / "inductive_dataset_level_metric_summary_selected.csv", index=False)
    write_metric_value_table(summary)
    
    # Load transductive results
    trans_csv_path = HERE.parent.parent.parent / "presentation_stable_generalist_nonbaron_20260324/00_source_tables/stable_generalist_all_results_table.csv"
    trans_df = load_transductive_data(trans_csv_path)
    
    plot_boxplots(selected, trans_df, METRICS, "inductive_metrics_boxplots_ari_acc_rare_balancedrare_ultrarare")
    plot_boxplots(selected, trans_df, ["ARI", "ACC", "BalancedACC", "RareACC", "UltraRareACC"], "inductive_metrics_boxplots_ari_acc_rare_ultrarare")
    plot_heatmap(selected)

    manifest = {
        "source_table": str(source_path),
        "additional_root": "",
        "n_base_rows": int(len(final)),
        "n_added_rows": 0,
        "algorithms": ALGORITHM_ORDER,
        "datasets": DATASET_ORDER,
        "metrics": METRICS,
    }
    (HERE / "manifest_inductive_synthesis.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote={HERE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
