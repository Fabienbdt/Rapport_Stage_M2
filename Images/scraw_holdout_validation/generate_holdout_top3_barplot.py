#!/usr/bin/env python3
"""Generate holdout validation bar plots for scRAW stable_generalist.

The organization mirrors the common-8 figure: metrics as rows, method families
as columns, top 3 methods per family/metric plus scRAW, ordered by increasing
mean score within each panel.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent
CSV_PATH = Path(
    "/data2/fbidet/scRAW_EXPERIMENTAL/results/"
    "presentation_stable_generalist_nonbaron_20260324/00_source_tables/"
    "stable_generalist_all_results_table.csv"
)

SCRAW_METHOD = "scRAW (stable_generalist)"

DATASETS = {
    "Pancreas 4 batches": "Human Pancreas",
    "SCIB Mouse Pancreas 1": "SCIB Mouse Pancreas 1",
}

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
    "RareACC": "RareACC",
    "UltraRareACC": "UltraRareACC",
    "Batch correction": "Correction batch",
}

FAMILIES = {
    "Rare Specific": [
        "scAIDE",
        "scCAD",
        "GiniClust",
        "DeepScena",
        "CellSIUS",
    ],
    "Methodes traditionnelles": [
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

FAMILY_DISPLAY = {
    "Rare Specific": "Rare Specific",
    "Methodes traditionnelles": "Méthodes\ntraditionnelles",
    "Correction batch": "Correction\nbatch",
}

FAMILY_COLORS = {
    "Rare Specific": "#4C78A8",
    "Methodes traditionnelles": "#F58518",
    "Correction batch": "#54A24B",
}

SCRAW_COLOR = "#E45756"

METHOD_DISPLAY = {
    "pca_leiden": "PCA+Leiden",
    "scvi": "scVI",
    SCRAW_METHOD: "scRAW",
}

DATASET_STYLES = {
    "Human Pancreas": {"offset": -0.17, "alpha": 0.95, "hatch": None},
    "SCIB Mouse Pancreas 1": {"offset": 0.17, "alpha": 0.55, "hatch": "///"},
}


def display_method(method: str) -> str:
    return METHOD_DISPLAY.get(method, method)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df[df["dataset"].isin(DATASETS)].copy()
    df = df.drop_duplicates(subset=["result_row_id", "dataset", "method"])

    is_scraw_stable_generalist = (
        (df["method"] == SCRAW_METHOD)
        & (df["trial_id"].fillna("") == "stable_generalist")
    )
    is_non_scraw = ~df["method"].astype(str).str.startswith("scRAW")
    allowed_methods = {SCRAW_METHOD}
    for methods in FAMILIES.values():
        allowed_methods.update(methods)

    df = df[(is_non_scraw | is_scraw_stable_generalist) & df["method"].isin(allowed_methods)].copy()
    df["dataset_display"] = df["dataset"].map(DATASETS)
    return df


def select_methods(df: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    selected: dict[str, dict[str, list[str]]] = {}
    rows = []
    for family, methods in FAMILIES.items():
        selected[family] = {}
        for metric in METRICS:
            family_scores = (
                df[df["method"].isin(methods)]
                .groupby("method")[metric]
                .mean()
                .dropna()
                .sort_values(ascending=False)
            )
            top3 = family_scores.head(3).index.tolist()
            methods_for_panel = [SCRAW_METHOD] + [
                method for method in top3 if method != SCRAW_METHOD
            ]
            selected[family][metric] = methods_for_panel

            for rank, method in enumerate(methods_for_panel, start=1):
                mean_score = df.loc[df["method"] == method, metric].mean()
                rows.append(
                    {
                        "family": family,
                        "metric": metric,
                        "rank_in_panel": rank,
                        "method": method,
                        "method_display": display_method(method),
                        "mean_score": mean_score,
                    }
                )

    pd.DataFrame(rows).to_csv(ROOT / "holdout_top3_plus_scraw_selection.csv", index=False)
    return selected


def draw_barplot(df: pd.DataFrame, selected: dict[str, dict[str, list[str]]]) -> None:
    family_names = list(FAMILIES)
    fig, axes = plt.subplots(
        nrows=len(METRICS),
        ncols=len(family_names),
        figsize=(12.2, 10.4),
        sharex=True,
        squeeze=False,
    )

    for row, metric in enumerate(METRICS):
        for col, family in enumerate(family_names):
            ax = axes[row, col]
            methods = selected[family][metric]
            means = (
                df[df["method"].isin(methods)]
                .groupby("method")[metric]
                .mean()
                .reindex(methods)
                .sort_values(ascending=True)
            )
            ordered_methods = means.index.tolist()

            for y_base, method in enumerate(ordered_methods):
                method_color = SCRAW_COLOR if method == SCRAW_METHOD else FAMILY_COLORS[family]
                for dataset_display, style in DATASET_STYLES.items():
                    sub = df[
                        (df["dataset_display"] == dataset_display)
                        & (df["method"] == method)
                    ]
                    if sub.empty or pd.isna(sub.iloc[0][metric]):
                        continue
                    value = float(sub.iloc[0][metric])
                    ax.barh(
                        y_base + style["offset"],
                        value,
                        height=0.28,
                        color=method_color,
                        alpha=style["alpha"],
                        edgecolor=method_color,
                        linewidth=0.7,
                        hatch=style["hatch"],
                    )

            ax.set_yticks(range(len(ordered_methods)))
            ax.set_yticklabels([display_method(method) for method in ordered_methods], fontsize=7)
            ax.set_xlim(0, 1.0)
            ax.grid(axis="x", color="#D9D9D9", linewidth=0.5, alpha=0.7)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", labelsize=7)
            ax.tick_params(axis="y", length=0)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color("#BFBFBF")

            if row == 0:
                ax.set_title(FAMILY_DISPLAY[family], fontsize=10, fontweight="bold", pad=10)
            if col == 0:
                ax.set_ylabel(METRIC_LABELS[metric], fontsize=9, fontweight="bold")
            else:
                ax.set_ylabel("")
            if row == len(METRICS) - 1:
                ax.set_xlabel("Score", fontsize=8)

    legend_handles = [
        Patch(facecolor=SCRAW_COLOR, edgecolor=SCRAW_COLOR, label="scRAW"),
        Patch(facecolor=FAMILY_COLORS["Rare Specific"], edgecolor=FAMILY_COLORS["Rare Specific"], label="Rare Specific"),
        Patch(facecolor=FAMILY_COLORS["Methodes traditionnelles"], edgecolor=FAMILY_COLORS["Methodes traditionnelles"], label="Méthodes traditionnelles"),
        Patch(facecolor=FAMILY_COLORS["Correction batch"], edgecolor=FAMILY_COLORS["Correction batch"], label="Correction batch"),
        Patch(facecolor="#808080", edgecolor="#808080", alpha=0.95, label="Human Pancreas"),
        Patch(facecolor="#808080", edgecolor="#808080", alpha=0.55, hatch="///", label="SCIB Mouse Pancreas 1"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.suptitle(
        "Validation externe : top 3 par famille et métrique + scRAW stable_generalist",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.965), h_pad=0.8, w_pad=0.8)

    for suffix in ("pdf", "png"):
        fig.savefig(
            ROOT / f"holdout_top3_plus_scraw_barplot.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    df = load_data()
    selected = select_methods(df)
    draw_barplot(df, selected)


if __name__ == "__main__":
    main()
