#!/usr/bin/env python3
"""Regenerate Figure 4 – common-8 benchmark panel.

Layout: 5 rows × 1 column (one row per metric).
Each subplot contains all methods as vertical boxplots grouped by family:
  - scRAW (trial_0017) — red, leftmost, separated by a dashed line
  - Rare Specific      — blue
  - Méthodes traditionnelles — orange
  - Correction batch   — green

Within each family the methods are sorted by ascending mean for that metric.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CSV_PATH = Path(
    "/data2/fbidet/scRAW_EXPERIMENTAL/results/"
    "presentation_trial206_nonbaron_20260324/00_source_tables/"
    "trial206_all_results_table.csv"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COMMON8 = [
    "BBAG094 Zeisel",
    "BBAG094 spleen",
    "Baron human pancreas",
    "Human testis GSE112013",
    "Kang PBMC",
    "Macaque retina bipolar",
    "Paul15 bone marrow",
    "Tabula Muris liver",
]

SCRAW_METHOD = "scRAW (trial_0017)"

METRICS = [
    "ARI",
    "ACC",
    "RareACC",
    "UltraRareACC",
    "Batch correction",
]

METRIC_LABELS = {
    "ARI": "ARI",
    "ACC": "ACC",
    "RareACC": "Rare ACC",
    "UltraRareACC": "Ultra Rare ACC",
    "Batch correction": "Correction batch",
}

# Families and their methods (excluding scRAW, which is handled separately)
PRIMARY_FAMILIES: dict[str, list[str]] = {
    "Rare Specific": [
        "scAIDE",
        "scCAD",
        "GiniClust",
        "DeepScena",
        "CellSIUS",
    ],
    "Méthodes traditionnelles": [
        "pca_leiden",
        "scMAE",
        "scNAME",
        "scvi",
    ],
    "Correction batch": [
        "Harmony",
        "ComBat",
        "DESC",
        "Scanorama",
        "scvi",
    ],
}

# Keep scVI in only one family to avoid duplication in the plot
# It officially belongs to "Méthodes traditionnelles" (primary) and also
# "Correction batch" — keep it in both to honour the original figure.

METHOD_DISPLAY: dict[str, str] = {
    "pca_leiden": "PCA+Leiden",
    "scvi": "scVI",
    SCRAW_METHOD: "scRAW",
}

# Colours
SCRAW_COLOR = "#dc2626"          # red
FAMILY_COLORS: dict[str, str] = {
    "Rare Specific":             "#2563eb",   # blue
    "Méthodes traditionnelles":  "#ea580c",   # orange
    "Correction batch":          "#16a34a",   # green
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def display(method: str) -> str:
    return METHOD_DISPLAY.get(method, method)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    scraw_ok = (df["method"] == SCRAW_METHOD) & (df["trial_id"] == "trial_0017")
    keep_non_scraw = ~df["is_scraw_method"].fillna(False)
    df = df[df["dataset"].isin(COMMON8) & (keep_non_scraw | scraw_ok)].copy()
    return df


def top3_per_family(df: pd.DataFrame, metric: str) -> dict[str, list[str]]:
    """Return top-3 methods per family sorted ascending (worst→best left→right)."""
    result: dict[str, list[str]] = {}
    for family, methods in PRIMARY_FAMILIES.items():
        sub = df[df["method"].isin(methods)]
        means = (
            sub.groupby("method")[metric]
            .mean()
            .dropna()
            .sort_values(ascending=True)  # worst on left, best on right
        )
        # Keep only the 3 methods with the HIGHEST mean (but display worst→best)
        # First pick the top-3 by mean, then re-sort ascending for display
        top3_methods = means.sort_values(ascending=False).head(3).index.tolist()
        # Re-sort the selected 3 ascending (worst→best)
        top3_sorted = means.loc[means.index.isin(top3_methods)].sort_values(ascending=True).index.tolist()
        result[family] = top3_sorted
    return result


def values_for(df: pd.DataFrame, method: str, metric: str) -> np.ndarray:
    return df.loc[df["method"] == method, metric].dropna().values


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_figure(df: pd.DataFrame) -> None:
    plt.style.use("default")
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(
        nrows=len(METRICS),
        ncols=1,
        figsize=(12.5, 3.8 * len(METRICS)),
        squeeze=True,
    )

    for ax, metric in zip(axes, METRICS):
        # -- build ordered list of (method, family, color) -----------------
        families_top3 = top3_per_family(df, metric)

        ordered: list[tuple[str, str, str]] = []

        # 1. scRAW first
        ordered.append((SCRAW_METHOD, "scRAW", SCRAW_COLOR))

        # 2. Then each family
        family_boundaries: list[float] = []   # x positions where families end
        for family, methods in families_top3.items():
            family_boundaries.append(len(ordered))  # start of this family
            for m in methods:
                ordered.append((m, family, FAMILY_COLORS[family]))

        # -- collect data arrays -------------------------------------------
        data   = [values_for(df, m, metric) for m, _, _ in ordered]
        colors = [c for _, _, c in ordered]
        labels = [display(m) for m, _, _ in ordered]
        positions = np.arange(1, len(ordered) + 1, dtype=float)

        # -- boxplot -------------------------------------------------------
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showmeans=True,
            showfliers=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "#111827",
                "markersize": 4.5,
                "markeredgewidth": 1.0,
            },
            medianprops={"color": "#111827", "linewidth": 1.5},
            whiskerprops={"color": "#374151", "linewidth": 1.1},
            capprops={"color": "#374151", "linewidth": 1.1},
            flierprops={
                "marker": "o",
                "markerfacecolor": "none",
                "markeredgecolor": "#6b7280",
                "markersize": 3,
                "alpha": 0.5,
            },
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.60)
            patch.set_edgecolor("#374151")
            patch.set_linewidth(1.0)

        # -- dashed separators between groups ------------------------------
        # After scRAW (position 1) and between each family
        sep_positions = [1.5]  # after scRAW
        cursor = 1  # scRAW occupies position 1
        for family, methods in families_top3.items():
            cursor += len(methods)
            sep_positions.append(cursor + 0.5)
        # Remove last separator (after last family — not needed)
        sep_positions = sep_positions[:-1]

        for xpos in sep_positions:
            ax.axvline(
                xpos,
                color="#9ca3af",
                linestyle="--",
                linewidth=1.0,
                zorder=1,
                alpha=0.8,
            )

        # -- axes styling --------------------------------------------------
        ax.set_xlim(0.3, len(ordered) + 0.7)
        ax.set_ylim(-0.04, 1.08)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.grid(axis="y", color="#d1d5db", linewidth=0.75, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)

        # Bold scRAW tick label
        for tick, (m, _, _) in zip(ax.get_xticklabels(), ordered):
            if m == SCRAW_METHOD:
                tick.set_fontweight("bold")
                tick.set_color(SCRAW_COLOR)

        # Family name annotations — shown only on the first (top) subplot
        if metric == METRICS[0]:
            cursor_x = 1  # scRAW
            for family, methods in families_top3.items():
                n = len(methods)
                if n == 0:
                    continue
                start_x = cursor_x + 1
                end_x   = cursor_x + n
                mid_x   = (start_x + end_x) / 2
                ax.text(
                    mid_x,
                    1.07,
                    family,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=FAMILY_COLORS[family],
                    fontweight="bold",
                )
                cursor_x += n

    # -- global legend at top of figure -----------------------------------
    legend_handles = [
        mpatches.Patch(facecolor=SCRAW_COLOR,                       alpha=0.60, label="scRAW"),
        mpatches.Patch(facecolor=FAMILY_COLORS["Rare Specific"],    alpha=0.60, label="Rare Specific"),
        mpatches.Patch(facecolor=FAMILY_COLORS["Méthodes traditionnelles"], alpha=0.60, label="Méthodes traditionnelles"),
        mpatches.Patch(facecolor=FAMILY_COLORS["Correction batch"], alpha=0.60, label="Correction batch"),
        plt.Line2D([0], [0], marker="D", linestyle="", markerfacecolor="white",
                   markeredgecolor="#111827", markersize=5, label="Moyenne"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        frameon=False,
        fontsize=9,
    )

    plt.suptitle(
        "Distribution des performances sur les huit jeux de données communs",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # -- save --------------------------------------------------------------
    output_pdf = ROOT / "common8_family_top3_plus_scraw_panel.pdf"
    output_png = ROOT / "common8_family_top3_plus_scraw_panel.png"

    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, bbox_inches="tight", dpi=220)
    plt.close(fig)

    print(f"Saved: {output_pdf}")
    print(f"Saved: {output_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    make_figure(df)
