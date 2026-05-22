#!/usr/bin/env python3
"""Generate existing-method analysis figures for the report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    normalized_mutual_info_score,
)


OUT_DIR = Path(__file__).resolve().parent
IMAGE_ROOT = OUT_DIR.parent
SOURCE_ROOT = IMAGE_ROOT / "panneaux_umap_comparaison_methodes" / "panneaux_sources"
INDUCTIVE_METRICS = (
    IMAGE_ROOT
    / "16_baron_h123_to_h4_train_test_loss_all_algorithms"
    / "tables"
    / "train_test_metric_delta_h123_to_h4.csv"
)

ALGORITHMS = [
    ("pca_kmeans", "PCA+KMeans"),
    ("pca_leiden", "PCA+Leiden"),
    ("sc_mae", "scMAE"),
    ("scdeepcluster", "scDeepCluster"),
    ("scname", "scNAME"),
]

CELL_ORDER = [
    "beta",
    "alpha",
    "ductal",
    "acinar",
    "delta",
    "activated_stellate",
    "gamma",
    "endothelial",
    "quiescent_stellate",
    "macrophage",
    "mast",
    "epsilon",
    "schwann",
    "t_cell",
]

DISPLAY_LABELS = {
    "activated_stellate": "activated\nstellate",
    "quiescent_stellate": "quiescent\nstellate",
    "t_cell": "T cell",
}


def display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label)


def load_algorithm_table(key: str) -> pd.DataFrame:
    path = SOURCE_ROOT / key / "per_cell_with_scraw_weights.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def rare_classes(labels: pd.Series, threshold: float) -> list[str]:
    frequencies = labels.value_counts(normalize=True)
    return [label for label, freq in frequencies.items() if freq < threshold]


def restricted_accuracy(y_true: pd.Series, y_pred: pd.Series, classes: list[str]) -> float:
    mask = y_true.isin(classes)
    if not bool(mask.any()):
        return float("nan")
    return float((y_true[mask].to_numpy() == y_pred[mask].to_numpy()).mean())


def class_recalls(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    recalls: dict[str, float] = {}
    for label in sorted(y_true.unique()):
        mask = y_true == label
        recalls[label] = float((y_pred[mask].to_numpy() == y_true[mask].to_numpy()).mean())
    return recalls


def collect_metrics() -> pd.DataFrame:
    rows = []
    for key, name in ALGORITHMS:
        df = load_algorithm_table(key)
        y_true = df["true_label"].astype(str)
        raw_pred = df["predicted_label"].astype(str)
        aligned_pred = df["aligned_predicted_label"].astype(str)
        rare = rare_classes(y_true, 0.05)
        ultra = rare_classes(y_true, 0.01)
        recalls = class_recalls(y_true, aligned_pred)
        missed_rare = sum(1 for label in rare if recalls.get(label, 0.0) < 0.05)
        rows.append(
            {
                "method_key": key,
                "Methode": name,
                "NMI": normalized_mutual_info_score(y_true, raw_pred),
                "ARI": adjusted_rand_score(y_true, raw_pred),
                "ACC": accuracy_score(y_true, aligned_pred),
                "BalancedACC": balanced_accuracy_score(y_true, aligned_pred),
                "RareACC": restricted_accuracy(y_true, aligned_pred, rare),
                "UltraRareACC": restricted_accuracy(y_true, aligned_pred, ultra),
                "Ecart ARI-Rare": adjusted_rand_score(y_true, raw_pred)
                - restricted_accuracy(y_true, aligned_pred, rare),
                "Types rares manques": missed_rare,
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT_DIR / "existant_baron_metrics_table.csv", index=False)
    return metrics


def score_color(value: float, reverse: bool = False) -> str:
    if np.isnan(value):
        return "#f1f3f5"
    if reverse:
        value = 1.0 - min(max(value / 9.0, 0.0), 1.0)
    value = min(max(value, 0.0), 1.0)
    cmap = plt.get_cmap("RdYlGn")
    return cmap(value)


def plot_metrics_table(metrics: pd.DataFrame) -> None:
    table_cols = [
        "Methode",
        "NMI",
        "ARI",
        "ACC",
        "BalancedACC",
        "RareACC",
        "UltraRareACC",
        "Types rares manques",
    ]
    headers = [
        "Methode",
        "NMI",
        "ARI",
        "ACC",
        "Balanced\nACC",
        "RareACC\n<5%",
        "UltraRareACC\n<1%",
        "Types rares\nnon retrouves",
    ]
    display = metrics[table_cols].copy()
    for col in table_cols[1:-1]:
        display[col] = display[col].map(lambda value: f"{value:.3f}")
    display["Types rares manques"] = display["Types rares manques"].astype(int).astype(str)

    fig, ax = plt.subplots(figsize=(12.2, 4.6))
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_title(
        "Baron human pancreas - scores globaux et fragilite des types rares",
        fontsize=15,
        fontweight="bold",
        pad=18,
    )
    ax.text(
        0.5,
        0.965,
        "Analyse standard sur les methodes executees dans SCRBenchmark",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#495057",
    )

    table = ax.table(
        cellText=display.values,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.18, 0.96, 0.66],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d9dee3")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#22313f")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.set_height(0.12)
        else:
            metric_row = metrics.iloc[row - 1]
            if col == 0:
                cell.set_facecolor("#f8f9fa")
                cell.get_text().set_fontweight("bold")
            elif col == len(headers) - 1:
                cell.set_facecolor(score_color(float(metric_row["Types rares manques"]), reverse=True))
            else:
                score = float(metric_row[table_cols[col]])
                cell.set_facecolor(score_color(score))
            if metric_row["method_key"] == "pca_leiden":
                cell.set_linewidth(1.5)
                cell.set_edgecolor("#d9480f")

    ax.text(
        0.02,
        0.08,
        "Lecture : PCA+Leiden conserve des scores globaux tres eleves, mais plusieurs classes rares ont un rappel nul.",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#343a40",
    )
    ax.text(
        0.02,
        0.035,
        "RareACC et UltraRareACC sont calculees seulement sur les cellules des classes <5% et <1% du jeu Baron.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#6c757d",
    )
    fig.savefig(OUT_DIR / "existant_baron_metrics_table.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(metrics: pd.DataFrame) -> None:
    df = load_algorithm_table("pca_leiden")
    y_true = df["true_label"].astype(str)
    y_pred = df["aligned_predicted_label"].astype(str)
    counts = pd.crosstab(y_true, y_pred).reindex(index=CELL_ORDER, columns=CELL_ORDER, fill_value=0)
    row_sums = counts.sum(axis=1).replace(0, np.nan)
    norm = counts.div(row_sums, axis=0).fillna(0.0)

    metric_row = metrics.loc[metrics["method_key"] == "pca_leiden"].iloc[0]
    class_counts = y_true.value_counts()
    rare = set(rare_classes(y_true, 0.05))
    ultra = set(rare_classes(y_true, 0.01))

    fig, ax = plt.subplots(figsize=(13.2, 10.8))
    image = ax.imshow(norm.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")

    xlabels = [display_label(label) for label in CELL_ORDER]
    ylabels = []
    for label in CELL_ORDER:
        suffix = "UR" if label in ultra else "R" if label in rare else "F"
        ylabels.append(f"{display_label(label).replace(chr(10), ' ')} ({suffix}, n={class_counts[label]})")

    ax.set_xticks(np.arange(len(CELL_ORDER)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(len(CELL_ORDER)))
    ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.set_xlabel("Type cellulaire predit apres appariement hongrois")
    ax.set_ylabel("Type cellulaire annote")
    fig.suptitle(
        "PCA+Leiden sur Baron : une bonne ARI/NMI peut masquer les classes rares",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    ax.set_title(
        (
            f"NMI={metric_row['NMI']:.3f} | ARI={metric_row['ARI']:.3f} | "
            f"BalancedACC={metric_row['BalancedACC']:.3f} | RareACC={metric_row['RareACC']:.3f}"
        ),
        fontsize=10.5,
        color="#495057",
        pad=18,
    )

    for i, label in enumerate(CELL_ORDER):
        for j, _ in enumerate(CELL_ORDER):
            value = float(norm.iloc[i, j])
            count = int(counts.iloc[i, j])
            if count == 0:
                continue
            should_annotate = value >= 0.08 or label in rare
            if not should_annotate:
                continue
            color = "white" if value >= 0.52 else "#1b1e23"
            ax.text(
                j,
                i,
                f"{100 * value:.0f}%\n({count})",
                ha="center",
                va="center",
                fontsize=6.4,
                color=color,
            )

    for tick, label in zip(ax.get_yticklabels(), CELL_ORDER):
        if label in ultra:
            tick.set_color("#b00020")
            tick.set_fontweight("bold")
        elif label in rare:
            tick.set_color("#b15f00")
            tick.set_fontweight("bold")

    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Proportion par ligne")
    ax.text(
        0,
        -0.18,
        "F: frequent | R: rare (<5%) | UR: ultra-rare (<1%). Les lignes sont normalisees a 100%.",
        transform=ax.transAxes,
        fontsize=9,
        color="#495057",
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "existant_baron_pca_leiden_confusion_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def collect_inductive_metrics() -> pd.DataFrame:
    if not INDUCTIVE_METRICS.exists():
        raise FileNotFoundError(INDUCTIVE_METRICS)
    raw = pd.read_csv(INDUCTIVE_METRICS)
    keep_algorithms = ["scname", "sc_mae", "scdeepcluster"]
    keep_metrics = ["NMI", "ARI", "ACC", "BalancedACC", "RareACC", "UltraRareACC"]
    sub = raw[
        raw["algorithm_key"].isin(keep_algorithms)
        & raw["metric"].isin(keep_metrics)
    ].copy()
    pivot = sub.pivot(index="algorithm", columns="metric", values="test")
    pivot = pivot.reindex(["scNAME", "scMAE", "scDeepCluster"])
    pivot = pivot[keep_metrics]
    pivot.insert(0, "Methode", pivot.index)
    return pivot.reset_index(drop=True)


def draw_metric_table(
    ax: plt.Axes,
    data: pd.DataFrame,
    title: str,
    highlight_method: str | None = None,
) -> None:
    ax.axis("off")
    table_cols = ["Methode", "NMI", "ARI", "ACC", "BalancedACC", "RareACC", "UltraRareACC"]
    headers = ["Methode", "NMI", "ARI", "ACC", "Balanced\nACC", "RareACC\n<5%", "UltraRareACC\n<1%"]
    display = data[table_cols].copy()
    for col in table_cols[1:]:
        display[col] = display[col].map(lambda value: f"{float(value):.3f}")

    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=6)
    table = ax.table(
        cellText=display.values,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.03, 1.0, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d9dee3")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#22313f")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            method = str(data.iloc[row - 1]["Methode"])
            if col == 0:
                cell.set_facecolor("#f8f9fa")
                cell.get_text().set_fontweight("bold")
            else:
                value = float(data.iloc[row - 1][table_cols[col]])
                cell.set_facecolor(score_color(value))
            if highlight_method and method == highlight_method:
                cell.set_edgecolor("#d9480f")
                cell.set_linewidth(1.5)


def plot_inductive_transductive_panel() -> None:
    inductive = collect_inductive_metrics()
    transductive = collect_metrics()

    df = load_algorithm_table("pca_leiden")
    y_true = df["true_label"].astype(str)
    y_pred = df["aligned_predicted_label"].astype(str)
    class_counts = y_true.value_counts()
    rare = set(rare_classes(y_true, 0.05))
    ultra = set(rare_classes(y_true, 0.01))
    rare_rows = [label for label in CELL_ORDER if label in rare]
    raw_counts = pd.crosstab(y_true, y_pred).reindex(index=rare_rows, columns=CELL_ORDER, fill_value=0)
    nonzero_cols = [col for col in CELL_ORDER if int(raw_counts[col].sum()) > 0]
    selected_cols = [col for col in CELL_ORDER if col in rare or col in nonzero_cols]
    counts = raw_counts.reindex(columns=selected_cols, fill_value=0)
    norm = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    pca_leiden = transductive.loc[transductive["method_key"] == "pca_leiden"].iloc[0]

    fig = plt.figure(figsize=(13.4, 8.8), facecolor="white")
    grid = GridSpec(2, 1, height_ratios=[0.34, 0.66], hspace=0.34, figure=fig)

    ax_table = fig.add_subplot(grid[0])
    draw_metric_table(
        ax_table,
        inductive,
        "A. Inductif - Baron H123->H4 : evaluation sur un donneur jamais vu",
        highlight_method="scMAE",
    )
    ax_table.text(
        0.0,
        -0.02,
        "Le cas scMAE garde des scores globaux eleves sur H4 (NMI/ARI), mais chute nettement sur RareACC.",
        transform=ax_table.transAxes,
        fontsize=9,
        color="#495057",
    )

    ax_matrix = fig.add_subplot(grid[1])
    image = ax_matrix.imshow(norm.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax_matrix.set_title(
        "B. Transductif - Baron complet : confusion des classes rares pour PCA+Leiden",
        loc="left",
        fontsize=12,
        fontweight="bold",
        pad=22,
    )
    ax_matrix.text(
        0.0,
        1.02,
        (
            f"NMI={pca_leiden['NMI']:.3f} | ARI={pca_leiden['ARI']:.3f} | "
            f"BalancedACC={pca_leiden['BalancedACC']:.3f} | RareACC={pca_leiden['RareACC']:.3f}"
        ),
        transform=ax_matrix.transAxes,
        fontsize=9.5,
        color="#495057",
    )
    ax_matrix.set_xticks(np.arange(len(selected_cols)))
    ax_matrix.set_xticklabels([display_label(label).replace("\n", " ") for label in selected_cols], rotation=45, ha="right", fontsize=8)
    ylabels = []
    for label in rare_rows:
        status = "UR" if label in ultra else "R"
        ylabels.append(f"{display_label(label).replace(chr(10), ' ')} ({status}, n={class_counts[label]})")
    ax_matrix.set_yticks(np.arange(len(rare_rows)))
    ax_matrix.set_yticklabels(ylabels, fontsize=8.2)
    ax_matrix.set_xlabel("")
    ax_matrix.set_ylabel("Type annote rare ou ultra-rare", fontsize=9)

    for tick, label in zip(ax_matrix.get_yticklabels(), rare_rows):
        if label in ultra:
            tick.set_color("#b00020")
            tick.set_fontweight("bold")
        else:
            tick.set_color("#b15f00")
            tick.set_fontweight("bold")

    for i, label in enumerate(rare_rows):
        for j, _ in enumerate(selected_cols):
            count = int(counts.iloc[i, j])
            if count == 0:
                continue
            value = float(norm.iloc[i, j])
            color = "white" if value >= 0.52 else "#1b1e23"
            ax_matrix.text(j, i, f"{100 * value:.0f}%\n({count})", ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(image, ax=ax_matrix, fraction=0.025, pad=0.015)
    cbar.set_label("Proportion par ligne", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)
    ax_matrix.text(
        0.0,
        -0.24,
        "Colonnes: types predits apres appariement hongrois. R: rare (<5%) | UR: ultra-rare (<1%). Les lignes sont normalisees a 100%.",
        transform=ax_matrix.transAxes,
        fontsize=8.5,
        color="#495057",
    )
    fig.suptitle(
        "Comparaison de l'existant : scores globaux eleves, identification rare fragile",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.savefig(OUT_DIR / "existant_inductive_transductive_panel.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metrics = collect_metrics()
    plot_metrics_table(metrics)
    plot_confusion_matrix(metrics)
    plot_inductive_transductive_panel()


if __name__ == "__main__":
    main()
