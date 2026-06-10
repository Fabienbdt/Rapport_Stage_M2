#!/usr/bin/env python3
"""Generate existing-method analysis figures for the report."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR.parents[1] / "SCRBenchmark"

BARON_PAPER_LOGIC_ROOT = Path(
    "/data2/fbidet/SCRBenchmark/results/"
    "baron_paper_logic_existing_algorithms_3seeds_20260527"
)
BARON_INDUCTIVE_SCIB_ROOT = Path(
    "/data2/fbidet/SCRBenchmark/results/"
    "baron_paper_logic_existing_algorithms_3seeds_inductive_scibe_20260528_100508"
)
SCCDCG_SCIB_ROOT = Path(
    "/data2/fbidet/SCRBenchmark/results/"
    "baron_sccdcg_scib_3seeds_20260609_143857"
)
SCCDCG_INDUCTIVE_ROOT = Path(
    "/data2/fbidet/SCRBenchmark/results/"
    "baron_split_70_10_20_existing_algorithms_5seeds_20260526_103715/"
    "gpu1_deep_core"
)
SCCDCG_TRANSDUCTIVE_ROOT = Path(
    "/data2/fbidet/SCRBenchmark/results/"
    "baron_full_existing_algorithms_5seeds_20260522_162936/"
    "gpu1_deep_core"
)
N_RUNS = 3

ALGORITHMS = [
    ("pca_kmeans", "PCA+KMeans", BARON_PAPER_LOGIC_ROOT / "core"),
    ("pca_leiden", "PCA+Leiden", BARON_PAPER_LOGIC_ROOT / "core"),
    ("sc_mae", "scMAE", BARON_PAPER_LOGIC_ROOT / "core"),
    ("scdeepcluster", "scDeepCluster", BARON_PAPER_LOGIC_ROOT / "core"),
    ("scname", "scNAME", BARON_PAPER_LOGIC_ROOT / "scname"),
    ("sccdcg", "scCDCG", SCCDCG_TRANSDUCTIVE_ROOT),
]

INDUCTIVE_ALGORITHMS = [
    ("pca_kmeans", "PCA+KMeans", BARON_PAPER_LOGIC_ROOT / "inductive_core"),
    ("pca_leiden", "PCA+Leiden", BARON_PAPER_LOGIC_ROOT / "inductive_core"),
    ("sc_mae", "scMAE", BARON_PAPER_LOGIC_ROOT / "inductive_core"),
    ("scdeepcluster", "scDeepCluster", BARON_PAPER_LOGIC_ROOT / "inductive_core"),
    ("scname", "scNAME", BARON_PAPER_LOGIC_ROOT / "inductive_scname"),
    ("sccdcg", "scCDCG", SCCDCG_INDUCTIVE_ROOT),
]

TRANSDUCTIVE_SCIB_ROOTS = {
    "sccdcg": [SCCDCG_SCIB_ROOT / "transductive"],
}

INDUCTIVE_SCIB_ROOTS = {
    "pca_kmeans": [BARON_INDUCTIVE_SCIB_ROOT / "inductive_core"],
    "pca_leiden": [BARON_INDUCTIVE_SCIB_ROOT / "inductive_core"],
    "sc_mae": [BARON_INDUCTIVE_SCIB_ROOT / "inductive_core"],
    "scdeepcluster": [BARON_INDUCTIVE_SCIB_ROOT / "inductive_core"],
    "scname": [BARON_INDUCTIVE_SCIB_ROOT / "inductive_scname"],
    "sccdcg": [SCCDCG_SCIB_ROOT / "inductive"],
}

CELL_ORDER = [
    "t_cell",
    "schwann",
    "epsilon",
    "mast",
    "macrophage",
    "quiescent_stellate",
    "endothelial",
    "gamma",
    "activated_stellate",
    "delta",
    "acinar",
    "ductal",
    "alpha",
    "beta",
]

DISPLAY_LABELS = {
    "activated_stellate": "activated\nstellate",
    "quiescent_stellate": "quiescent\nstellate",
    "t_cell": "T cell",
}

METRIC_COLUMNS = [
    "NMI",
    "ARI",
    "ACC",
    "BalancedACC",
    "F1Macro",
    "RareACC",
    "BalancedRareACC",
    "UltraRareACC",
    "BatchCorrection",
    "n_eval",
    "n_rare_cells",
    "n_ultrarare_cells",
    "min_class_support",
]


def display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label)


def format_metric(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def align_cluster_labels(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    contingency = pd.crosstab(y_pred.astype(str), y_true.astype(str))
    if contingency.empty:
        return y_pred.astype(str)
    rows, cols = linear_sum_assignment(-contingency.to_numpy())
    mapping = {
        str(contingency.index[row]): str(contingency.columns[col])
        for row, col in zip(rows, cols)
    }
    return y_pred.astype(str).map(lambda label: mapping.get(str(label), f"unmatched_{label}"))


def class_recalls(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    recalls: dict[str, float] = {}
    for label in sorted(y_true.unique()):
        mask = y_true == label
        recalls[label] = float((y_pred[mask].to_numpy() == y_true[mask].to_numpy()).mean())
    return recalls


def restricted_accuracy(y_true: pd.Series, y_pred: pd.Series, classes: set[str]) -> float:
    mask = y_true.isin(classes)
    if not bool(mask.any()):
        return float("nan")
    return float((y_true[mask].to_numpy() == y_pred[mask].to_numpy()).mean())


def load_baron_support() -> pd.Series:
    path = BARON_PAPER_LOGIC_ROOT / "core" / "results" / "labels" / "labels_sc_mae_run0.csv"
    labels = pd.read_csv(path)
    return labels["true_label"].astype(str).value_counts().reindex(CELL_ORDER)


def rare_sets(full_support: pd.Series) -> tuple[set[str], set[str]]:
    total = float(full_support.sum())
    rare = set(full_support[full_support / total < 0.05].index)
    ultra = set(full_support[full_support / total < 0.01].index)
    return rare, ultra


def transductive_label_path(algorithm_key: str, run_id: int) -> Path:
    root = {key: path for key, _, path in ALGORITHMS}[algorithm_key]
    return root / "results" / "labels" / f"labels_{algorithm_key}_run{run_id}.csv"


def inductive_label_path(algorithm_key: str, run_id: int) -> Path:
    root = {key: path for key, _, path in INDUCTIVE_ALGORITHMS}[algorithm_key]
    return root / "results" / "labels" / f"benchmark_{algorithm_key}_run{run_id}_test.csv"


def evaluate_label_file(
    path: Path,
    rare: set[str],
    ultra: set[str],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    labels = pd.read_csv(path)
    y_true = labels["true_label"].astype(str)
    y_raw = labels["predicted_label"].astype(str)
    y_aligned = align_cluster_labels(y_true, y_raw)

    recalls = class_recalls(y_true, y_aligned)
    counts = y_true.value_counts()
    
    rare_recalls = [recalls[c] for c in recalls if c in rare]
    balanced_rare_acc = float(np.mean(rare_recalls)) if rare_recalls else float("nan")

    metrics = {
        "NMI": normalized_mutual_info_score(y_true, y_raw),
        "ARI": adjusted_rand_score(y_true, y_raw),
        "ACC": accuracy_score(y_true, y_aligned),
        "BalancedACC": float(np.mean(list(recalls.values()))),
        "BalancedRareACC": balanced_rare_acc,
        "F1Macro": f1_score(
            y_true,
            y_aligned,
            labels=sorted(y_true.unique()),
            average="macro",
            zero_division=0,
        ),
        "RareACC": restricted_accuracy(y_true, y_aligned, rare),
        "UltraRareACC": restricted_accuracy(y_true, y_aligned, ultra),
        "BatchCorrection": float("nan"),
        "n_eval": float(len(y_true)),
        "n_rare_cells": float(y_true.isin(rare).sum()),
        "n_ultrarare_cells": float(y_true.isin(ultra).sum()),
        "min_class_support": float(counts.min()),
    }

    errors = []
    for cell_type in CELL_ORDER:
        support = float(counts.get(cell_type, 0))
        recall = recalls.get(cell_type, float("nan"))
        errors.append(
            {
                "cell_type": cell_type,
                "support": support,
                "error_rate": 1.0 - recall if not pd.isna(recall) else float("nan"),
            }
        )
    return metrics, errors


def _candidate_roots(extra_roots: dict[str, list[Path]], root: Path, algorithm_key: str) -> list[Path]:
    candidates = [candidate for candidate in extra_roots.get(algorithm_key, []) if candidate != root]
    candidates.append(root)
    return candidates


def load_transductive_extra_metrics(root: Path, algorithm_key: str, run_id: int) -> dict[str, float]:
    for candidate_root in _candidate_roots(TRANSDUCTIVE_SCIB_ROOTS, root, algorithm_key):
        path = candidate_root / "results" / "results.csv"
        if not path.exists():
            continue
        results = pd.read_csv(path)
        if "Batch correction" not in results.columns:
            continue
        row = results[(results["algorithm"] == algorithm_key) & (results["run_id"] == run_id)]
        if row.empty:
            continue
        value = row.iloc[0]["Batch correction"]
        if not pd.isna(value):
            return {"BatchCorrection": float(value)}
    return {}


def load_inductive_extra_metrics(root: Path, algorithm_key: str, run_id: int) -> dict[str, float]:
    for candidate_root in _candidate_roots(INDUCTIVE_SCIB_ROOTS, root, algorithm_key):
        path = candidate_root / "results" / "benchmark_results.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for row in payload.get("results", []):
            if row.get("algorithm_name") == algorithm_key and int(row.get("run_id", -1)) == run_id:
                value = row.get("test_metrics", {}).get("Batch correction", float("nan"))
                if not pd.isna(value):
                    return {"BatchCorrection": float(value)}
    return {}


def collect_transductive(full_support: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    rare, ultra = rare_sets(full_support)
    metric_rows: list[dict[str, float | int | str]] = []
    error_rows: list[dict[str, float | int | str]] = []
    for algorithm_key, method_name, _ in ALGORITHMS:
        for run_id in range(N_RUNS):
            root = {key: path for key, _, path in ALGORITHMS}[algorithm_key]
            metrics, errors = evaluate_label_file(
                transductive_label_path(algorithm_key, run_id),
                rare,
                ultra,
            )
            metrics.update(load_transductive_extra_metrics(root, algorithm_key, run_id))
            metric_rows.append(
                {
                    "mode_key": "transductive",
                    "mode": "Transductif",
                    "algorithm_key": algorithm_key,
                    "Methode": method_name,
                    "seed": run_id,
                    **metrics,
                }
            )
            for row in errors:
                error_rows.append(
                    {
                        "mode_key": "transductive",
                        "mode": "Transductif",
                        "algorithm_key": algorithm_key,
                        "Methode": method_name,
                        "seed": run_id,
                        **row,
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(error_rows)


def collect_inductive(full_support: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    rare, ultra = rare_sets(full_support)
    metric_rows: list[dict[str, float | int | str]] = []
    error_rows: list[dict[str, float | int | str]] = []
    for algorithm_key, method_name, _ in INDUCTIVE_ALGORITHMS:
        for run_id in range(N_RUNS):
            root = {key: path for key, _, path in INDUCTIVE_ALGORITHMS}[algorithm_key]
            metrics, errors = evaluate_label_file(
                inductive_label_path(algorithm_key, run_id),
                rare,
                ultra,
            )
            metrics.update(load_inductive_extra_metrics(root, algorithm_key, run_id))
            metric_rows.append(
                {
                    "mode_key": "inductive",
                    "mode": "Inductif",
                    "algorithm_key": algorithm_key,
                    "Methode": method_name,
                    "seed": run_id,
                    **metrics,
                }
            )
            for row in errors:
                error_rows.append(
                    {
                        "mode_key": "inductive",
                        "mode": "Inductif",
                        "algorithm_key": algorithm_key,
                        "Methode": method_name,
                        "seed": run_id,
                        **row,
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(error_rows)


def collect_baron_panel_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    full_support = load_baron_support()
    inductive_metrics, inductive_errors = collect_inductive(full_support)
    transductive_metrics, transductive_errors = collect_transductive(full_support)
    per_run_metrics = pd.concat([inductive_metrics, transductive_metrics], ignore_index=True)
    per_run_errors = pd.concat([inductive_errors, transductive_errors], ignore_index=True)

    metrics = (
        per_run_metrics.groupby(["mode_key", "mode", "algorithm_key", "Methode"], sort=False)[METRIC_COLUMNS]
        .mean()
        .reset_index()
    )
    errors = (
        per_run_errors.groupby(["mode_key", "mode", "algorithm_key", "Methode", "cell_type"], sort=False)
        .agg(
            error_rate=("error_rate", "mean"),
            support_mean=("support", "mean"),
            support_min=("support", "min"),
            support_max=("support", "max"),
        )
        .reset_index()
    )

    mode_order = {"inductive": 0, "transductive": 1}
    algorithm_order = {key: order for order, (key, _, _) in enumerate(ALGORITHMS)}
    for frame in (per_run_metrics, metrics, errors):
        frame["_mode_order"] = frame["mode_key"].map(mode_order)
        frame["_algorithm_order"] = frame["algorithm_key"].map(algorithm_order)
        frame.sort_values(["_mode_order", "_algorithm_order"], inplace=True)
        frame.drop(columns=["_mode_order", "_algorithm_order"], inplace=True)

    per_run_metrics.to_csv(OUT_DIR / "baron_inductive_transductive_metrics_per_run.csv", index=False)
    metrics.to_csv(OUT_DIR / "baron_inductive_transductive_metrics_table.csv", index=False)
    errors.to_csv(OUT_DIR / "baron_inductive_transductive_error_rates.csv", index=False)
    return metrics, errors, full_support


def score_color(value: float) -> str:
    if np.isnan(value):
        return "#f1f3f5"
    value = min(max(float(value), 0.0), 1.0)
    return plt.get_cmap("RdYlGn")(value)


def draw_metric_table(ax: plt.Axes, metrics: pd.DataFrame, mode_name: str, title: str) -> None:
    ax.axis("off")
    data = metrics[metrics["mode"] == mode_name].copy()
    data["_algorithm_order"] = data["algorithm_key"].map(
        {key: order for order, (key, _, _) in enumerate(ALGORITHMS)}
    )
    data.sort_values("_algorithm_order", inplace=True)
    table_cols = [
        "Methode",
        "NMI",
        "ARI",
        "ACC",
        "BalancedACC",
        "RareACC",
        "BalancedRareACC",
        "UltraRareACC",
        "BatchCorrection",
        "n_eval",
    ]
    headers = ["Methode", "NMI", "ARI", "ACC", "Bal.\nACC", "Rare\nACC", "Bal.\nRare", "Ultra\nRare", "Batch\ncorr.", "n"]
    display = data[table_cols].copy()
    for col in table_cols[1:-1]:
        display[col] = display[col].map(format_metric)
    display["n_eval"] = display["n_eval"].round().astype(int).astype(str)

    ax.set_title(title, loc="left", fontsize=10.4, fontweight="bold", pad=4)
    table = ax.table(
        cellText=display.values,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.17, 0.085, 0.085, 0.085, 0.085, 0.085, 0.085, 0.085, 0.11, 0.075],
        bbox=[0.0, 0.0, 1.0, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.2)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d9dee3")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#22313f")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.set_height(0.095)
        else:
            if col == 0:
                cell.set_facecolor("#f8f9fa")
                cell.get_text().set_fontweight("bold")
            elif col == len(headers) - 1:
                cell.set_facecolor("#f1f3f5")
            else:
                value = float(data.iloc[row - 1][table_cols[col]])
                cell.set_facecolor(score_color(value))


def draw_error_heatmap(
    ax: plt.Axes,
    errors: pd.DataFrame,
    mode_name: str,
    method_order: list[str],
    full_support: pd.Series,
    title: str,
    show_ylabels: bool,
) -> plt.AxesImage:
    matrix = (
        errors[errors["mode"] == mode_name]
        .pivot(index="cell_type", columns="Methode", values="error_rate")
        .reindex(index=CELL_ORDER, columns=method_order)
    )
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("#f1f3f5")
    image = ax.imshow(np.ma.masked_invalid(matrix.to_numpy(dtype=float)), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_title(title, loc="left", fontsize=10.8, fontweight="bold", pad=6)
    ax.set_xticks(np.arange(len(method_order)))
    ax.set_xticklabels(method_order, rotation=35, ha="right", fontsize=7.7)
    ax.set_yticks(np.arange(len(CELL_ORDER)))
    if show_ylabels:
        labels = [
            f"{display_label(label).replace(chr(10), ' ')} (n={int(full_support[label])})"
            for label in CELL_ORDER
        ]
        ax.set_yticklabels(labels, fontsize=7.3)
        ax.set_ylabel("Type cellulaire (effectif Baron complet)", fontsize=8.3)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="both", length=0)

    for i, _ in enumerate(CELL_ORDER):
        for j, _ in enumerate(method_order):
            value = matrix.iloc[i, j]
            if pd.isna(value):
                continue
            color = "white" if value <= 0.16 or value >= 0.72 else "#343a40"
            ax.text(j, i, f"{float(value):.2f}", ha="center", va="center", fontsize=6.1, color=color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def plot_baron_inductive_transductive_error_panel() -> None:
    metrics, errors, full_support = collect_baron_panel_data()
    inductive_methods = [name for _, name, _ in INDUCTIVE_ALGORITHMS]
    transductive_methods = [name for _, name, _ in ALGORITHMS]

    fig = plt.figure(figsize=(15.0, 9.8), facecolor="white")
    grid = GridSpec(
        3,
        2,
        height_ratios=[0.11, 0.57, 0.32],
        hspace=0.43,
        wspace=0.12,
        figure=fig,
    )

    ax_title = fig.add_subplot(grid[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.0,
        0.72,
        "Baron - taux d'erreur par type cellulaire et scores globaux",
        transform=ax_title.transAxes,
        fontsize=15,
        fontweight="bold",
        color="#1f2933",
    )
    ax_title.text(
        0.0,
        0.25,
        "Erreur = 1 - rappel apres appariement hongrois ; resultats moyennes sur 3 seeds. "
        "Meme pretraitement que le papier scRAW : filtrage 200/3, normalisation 20000, 2000 HVG Seurat.",
        transform=ax_title.transAxes,
        fontsize=9.2,
        color="#495057",
    )

    ax_inductive = fig.add_subplot(grid[1, 0])
    ax_transductive = fig.add_subplot(grid[1, 1])
    image = draw_error_heatmap(
        ax_inductive,
        errors,
        "Inductif",
        inductive_methods,
        full_support,
        "A. Inductif - split 70/10/20, test 20%",
        show_ylabels=True,
    )
    draw_error_heatmap(
        ax_transductive,
        errors,
        "Transductif",
        transductive_methods,
        full_support,
        "B. Transductif - Baron complet",
        show_ylabels=False,
    )
    cbar = fig.colorbar(image, ax=[ax_inductive, ax_transductive], fraction=0.023, pad=0.012)
    cbar.set_label("Taux d'erreur", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)

    ax_table_inductive = fig.add_subplot(grid[2, 0])
    ax_table_transductive = fig.add_subplot(grid[2, 1])
    draw_metric_table(
        ax_table_inductive,
        metrics,
        "Inductif",
        "C. Metriques inductives (test 20%, 3 seeds)",
    )
    draw_metric_table(
        ax_table_transductive,
        metrics,
        "Transductif",
        "D. Metriques transductives (Baron complet, 3 seeds)",
    )

    fig.text(
        0.015,
        0.012,
        "Note : PCA+Leiden reprend la selection de resolution par silhouette du papier, sans forcer 14 clusters. "
        "La metrique Batch corr. correspond au score global scIB de correction batch lorsque les embeddings sont disponibles.",
        fontsize=8.3,
        color="#495057",
    )
    fig.savefig(OUT_DIR / "baron_inductive_transductive_error_panel.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_baron_inductive_transductive_error_panel()


if __name__ == "__main__":
    main()
