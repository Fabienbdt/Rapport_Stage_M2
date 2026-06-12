#!/usr/bin/env python3
"""Generate cropped and std-enhanced existing-method analysis figures from precalculated CSV files."""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR.parents[1] / "SCRBenchmark"

ALGORITHMS = [
    ("pca_kmeans", "PCA+KMeans", None),
    ("pca_leiden", "PCA+Leiden", None),
    ("sc_mae", "scMAE", None),
    ("scdeepcluster", "scDeepCluster", None),
    ("scname", "scNAME", None),
    ("sccdcg", "scCDCG", None),
]

INDUCTIVE_ALGORITHMS = [
    ("pca_kmeans", "PCA+KMeans", None),
    ("pca_leiden", "PCA+Leiden", None),
    ("sc_mae", "scMAE", None),
    ("scdeepcluster", "scDeepCluster", None),
    ("scname", "scNAME", None),
    ("sccdcg", "scCDCG", None),
]

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
]

def display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label)

def collect_baron_panel_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    per_run_metrics = pd.read_csv(OUT_DIR / "baron_inductive_transductive_metrics_per_run.csv")
    errors = pd.read_csv(OUT_DIR / "baron_inductive_transductive_error_rates.csv")
    
    trans_errors = errors[errors["mode_key"] == "transductive"]
    full_support_dict = {}
    for cell_type in CELL_ORDER:
        val = trans_errors[trans_errors["cell_type"] == cell_type]["support_mean"].mean()
        full_support_dict[cell_type] = int(val)
    full_support = pd.Series(full_support_dict)
    
    return per_run_metrics, errors, full_support

def score_color(value: float) -> str:
    if np.isnan(value):
        return "#f1f3f5"
    value = min(max(float(value), 0.0), 1.0)
    return plt.get_cmap("RdYlGn")(value)

def draw_metric_table(ax: plt.Axes, metrics_mean: pd.DataFrame, metrics_std: pd.DataFrame, mode_name: str, title: str) -> None:
    ax.axis("off")
    
    # Filter by mode
    data_mean = metrics_mean[metrics_mean["mode"] == mode_name].copy()
    data_std = metrics_std[metrics_std["mode"] == mode_name].copy()
    
    # Sort by algorithm order
    algo_to_order = {key: order for order, (key, _, _) in enumerate(ALGORITHMS)}
    data_mean["_order"] = data_mean["algorithm_key"].map(algo_to_order)
    data_mean.sort_values("_order", inplace=True)
    data_std["_order"] = data_std["algorithm_key"].map(algo_to_order)
    data_std.sort_values("_order", inplace=True)
    
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
    
    # Build display values
    display_rows = []
    for idx in range(len(data_mean)):
        row_mean = data_mean.iloc[idx]
        row_std = data_std.iloc[idx]
        row_display = []
        
        # Methode
        row_display.append(row_mean["Methode"])
        
        # NMI, ARI, ACC, BalancedACC, RareACC, UltraRareACC, BatchCorrection
        for col in table_cols[1:-1]:
            val_mean = row_mean[col]
            val_std = row_std[col]
            if pd.isna(val_mean):
                row_display.append("NA")
            elif pd.isna(val_std) or val_std == 0:
                row_display.append(f"{val_mean:.3f}")
            else:
                row_display.append(f"{val_mean:.3f}±{val_std:.3f}")
        
        # n_eval
        n_val = row_mean["n_eval"]
        if pd.isna(n_val):
            row_display.append("NA")
        else:
            row_display.append(str(int(round(n_val))))
            
        display_rows.append(row_display)
        
    display_values = np.array(display_rows)
    
    ax.set_title(title, loc="left", fontsize=12.0, fontweight="bold", pad=6)
    
    col_widths = [0.16, 0.097, 0.097, 0.097, 0.097, 0.097, 0.097, 0.097, 0.10, 0.061]
    
    table = ax.table(
        cellText=display_values,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        bbox=[0.0, 0.0, 1.0, 0.93],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.1) # Adjusted font size to fit text perfectly within cell boundaries
    
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d9dee3")
        cell.set_linewidth(0.6)
        cell.PAD = 0.03  # Reduce padding to allow more text space
        if row == 0:
            cell.set_facecolor("#22313f")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.set_height(0.12)
            if col == 0:
                cell._loc = 'left'
                cell.get_text().set_horizontalalignment('left')
            else:
                cell._loc = 'center'
                cell.get_text().set_horizontalalignment('center')
        else:
            cell.set_height(0.14)
            if col == 0:
                cell.set_facecolor("#f8f9fa")
                cell.get_text().set_fontweight("bold")
                cell._loc = 'left'
                cell.get_text().set_horizontalalignment('left')
            elif col == len(headers) - 1:
                cell.set_facecolor("#f1f3f5")
                cell._loc = 'center'
                cell.get_text().set_horizontalalignment('center')
            else:
                val_mean = float(data_mean.iloc[row - 1][table_cols[col]])
                cell.set_facecolor(score_color(val_mean))
                cell._loc = 'center'
                cell.get_text().set_horizontalalignment('center')

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
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", pad=8)
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
    per_run_metrics, errors, full_support = collect_baron_panel_data()
    
    # Group per_run_metrics by (mode_key, mode, algorithm_key, Methode) and compute mean and std
    grouped = per_run_metrics.groupby(["mode_key", "mode", "algorithm_key", "Methode"], sort=False)
    metrics_mean = grouped[METRIC_COLUMNS].mean().reset_index()
    metrics_std = grouped[METRIC_COLUMNS].std().reset_index()
    
    inductive_methods = [name for _, name, _ in INDUCTIVE_ALGORITHMS]
    transductive_methods = [name for _, name, _ in ALGORITHMS]

    # Reduced hspace and modified height_ratios to make tables larger
    fig = plt.figure(figsize=(16.5, 11.5), facecolor="white")
    grid = GridSpec(
        2,
        2,
        height_ratios=[0.50, 0.50],
        hspace=0.35,
        wspace=0.12,
        figure=fig,
    )

    ax_inductive = fig.add_subplot(grid[0, 0])
    ax_transductive = fig.add_subplot(grid[0, 1])
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
    cbar.set_label(r"$1 - \mathrm{rappel}_c$", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)

    ax_table_inductive = fig.add_subplot(grid[1, 0])
    ax_table_transductive = fig.add_subplot(grid[1, 1])
    draw_metric_table(
        ax_table_inductive,
        metrics_mean,
        metrics_std,
        "Inductif",
        "C. Metriques inductives (test 20%, 3 seeds)",
    )
    draw_metric_table(
        ax_table_transductive,
        metrics_mean,
        metrics_std,
        "Transductif",
        "D. Metriques transductives (Baron complet, 3 seeds)",
    )

    fig.savefig(OUT_DIR / "baron_inductive_transductive_error_panel.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    plot_baron_inductive_transductive_error_panel()

if __name__ == "__main__":
    main()
