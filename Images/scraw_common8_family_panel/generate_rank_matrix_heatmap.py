#!/usr/bin/env python3
"""Generate the common-8 rank matrix heatmap used in the report."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fbidet")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "common8_primary_all_methods_means.csv"
OUTPUT_STEM = ROOT / "common8_rank_matrix_heatmap"

METRICS = [
    "ARI",
    "ACC",
    "BalancedACC",
    "RareACC",
    "BalancedRareACC",
    "UltraRareACC",
    "Batch correction",
]
METRIC_LABELS = {
    "ARI": "ARI",
    "ACC": "ACC",
    "BalancedACC": "Bal. ACC",
    "RareACC": "RareACC",
    "BalancedRareACC": "Bal. RareACC",
    "UltraRareACC": "UltraRareACC",
    "Batch correction": "Batch",
}

METHOD_ORDER = [
    "scRAW",
    "scVI",
    "Harmony",
    "scAIDE",
    "DESC",
    "PCA+Leiden",
    "scMAE",
    "scNAME",
    "CellSIUS",
    "ComBat",
    "Scanorama",
    "scCAD",
    "DeepScena",
    "GiniClust",
]


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    df = df.drop_duplicates("method_display", keep="first").set_index("method_display")
    values = df.loc[METHOD_ORDER, METRICS]

    ranks = values.rank(axis=0, ascending=False, method="min").astype(int)
    ranks = ranks.rename(columns=METRIC_LABELS)
    ranks.to_csv(ROOT / "common8_rank_matrix_heatmap_ranks.csv")

    cmap = LinearSegmentedColormap.from_list(
        "rank_purple_to_light_yellow",
        ["#6F1D9B", "#C52CCC", "#EA6D8A", "#F4B49A", "#FFF7D6"],
        N=256,
    )

    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(10.0, 5.7))
    heatmap = sns.heatmap(
        ranks,
        ax=ax,
        cmap=cmap,
        vmin=1,
        vmax=len(METHOD_ORDER),
        annot=True,
        fmt="d",
        linewidths=0.8,
        linecolor="white",
        cbar_kws={
            "label": "Rang (1 = meilleur)",
            "ticks": [1, 3, 5, 7, 9, 11, 14],
            "shrink": 0.88,
        },
        annot_kws={"fontsize": 10, "fontweight": "bold"},
    )

    # ax.set_title("Matrice de rangs par métrique sur les huit jeux communs", pad=12, fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", length=0, labelsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontweight="bold")

    for tick in ax.get_yticklabels():
        if tick.get_text() == "scRAW":
            tick.set_color("#C7222A")
            tick.set_fontweight("bold")

    for text, value in zip(ax.texts, ranks.to_numpy().ravel()):
        text.set_color("white" if value <= 5 else "#222222")

    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            ranks.shape[1],
            1,
            fill=False,
            edgecolor="#C7222A",
            linewidth=2.0,
            clip_on=False,
        )
    )

    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9, length=0)
    cbar.set_label("Rang (1 = meilleur)", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
