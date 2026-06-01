#!/usr/bin/env python3
"""Generate common-8 boxplots for the oral presentation.

For each metric, the figure keeps scRAW plus the top three non-scRAW methods
from each report family. The selected boxplots are then sorted globally by
decreasing mean score, with the boxplots showing the distribution across
the common datasets.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fbidet")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


SOURCE_CSV = (
    ROOT.parent
    / "scraw_common8_family_panel"
    / "common8_primary_raw_by_dataset_long.csv"
)

OUTPUT_STEM = "common8_top3_family_sorted_barplots"
SCRAW_SOURCE_METHOD = "scRAW (trial_0017)"

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

FAMILIES = [
    "Rare Specific",
    "Méthodes traditionnelles",
    "Correction batch",
]

FAMILY_ALLOWED_METHODS = {
    "Correction batch": {"Harmony", "ComBat", "DESC", "Scanorama"},
}

COLORS = {
    "scRAW": "#dc2626",
    "Rare Specific": "#2563eb",
    "Méthodes traditionnelles": "#ea580c",
    "Correction batch": "#16a34a",
}

METHOD_DISPLAY = {
    SCRAW_SOURCE_METHOD: "scRAW",
    "pca_leiden": "PCA+Leiden",
    "scvi": "scVI",
}


def display_method(method: str, source_method: str) -> str:
    return METHOD_DISPLAY.get(source_method, METHOD_DISPLAY.get(method, method))


def load_data() -> pd.DataFrame:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Missing source table: {SOURCE_CSV}")

    df = pd.read_csv(SOURCE_CSV)
    expected = {"family", "dataset", "method", "source_method", "metric", "value"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {SOURCE_CSV}: {sorted(missing)}")

    df = df[df["metric"].isin(METRICS)].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def summarize_values(values: pd.Series) -> tuple[float, float, int]:
    clean = values.dropna()
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    return mean, std, int(len(clean))


def select_metric_rows(df: pd.DataFrame, metric: str) -> list[dict]:
    metric_df = df[df["metric"] == metric].copy()
    rows: list[dict] = []

    scraw_df = (
        metric_df[metric_df["source_method"] == SCRAW_SOURCE_METHOD]
        .drop_duplicates(["dataset", "metric", "source_method"])
        .copy()
    )
    if scraw_df.empty:
        raise ValueError(f"No scRAW values found for {metric}")
    mean, std, n_values = summarize_values(scraw_df["value"])
    rows.append(
        {
            "metric": metric,
            "family": "scRAW",
            "source_method": SCRAW_SOURCE_METHOD,
            "method_display": "scRAW",
            "mean": mean,
            "std": std,
            "n_values": n_values,
            "raw_values": scraw_df["value"].to_numpy(),
            "datasets": "; ".join(sorted(scraw_df["dataset"].unique())),
        }
    )

    for family in FAMILIES:
        family_df = metric_df[
            (metric_df["family"] == family)
            & (metric_df["source_method"] != SCRAW_SOURCE_METHOD)
        ].copy()
        allowed_methods = FAMILY_ALLOWED_METHODS.get(family)
        if allowed_methods is not None:
            family_df = family_df[family_df["source_method"].isin(allowed_methods)].copy()
        if family_df.empty:
            raise ValueError(f"No values found for {family} / {metric}")

        means = family_df.groupby(["source_method", "method"], sort=False)["value"].mean()
        top3 = means.sort_values(ascending=False).head(3)
        for source_method, method in top3.index:
            method_df = (
                family_df[family_df["source_method"] == source_method]
                .drop_duplicates(["dataset", "metric", "source_method"])
                .copy()
            )
            mean, std, n_values = summarize_values(method_df["value"])
            rows.append(
                {
                    "metric": metric,
                    "family": family,
                    "source_method": source_method,
                    "method_display": display_method(method, source_method),
                    "mean": mean,
                    "std": std,
                    "n_values": n_values,
                    "raw_values": method_df["value"].to_numpy(),
                    "datasets": "; ".join(sorted(method_df["dataset"].unique())),
                }
            )

    rows.sort(key=lambda row: row["mean"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank_after_global_sort"] = rank
    return rows


def build_selection(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for metric in METRICS:
        rows.extend(select_metric_rows(df, metric))
    return pd.DataFrame(rows)


def draw_plot(selection: pd.DataFrame) -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
        }
    )

    fig, axes = plt.subplots(
        nrows=len(METRICS),
        ncols=1,
        figsize=(13.0, 15.2),
        squeeze=True,
    )

    for ax, metric in zip(axes, METRICS):
        sub = selection[selection["metric"] == metric].sort_values(
            "rank_after_global_sort"
        )
        positions = np.arange(len(sub), dtype=float)
        
        data = [row["raw_values"] for _, row in sub.iterrows()]
        colors = [COLORS[row["family"]] for _, row in sub.iterrows()]

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
            patch.set_alpha(0.72)
            patch.set_edgecolor("#111827")
            patch.set_linewidth(0.9)

        # Annotate mean value of each boxplot
        for idx, (_, row) in enumerate(sub.iterrows()):
            v = row["raw_values"]
            mean_val = row["mean"]
            max_val = np.nanmax(v) if len(v) > 0 else mean_val
            y_text = min(max_val + 0.035, 1.045)
            ax.text(
                positions[idx],
                y_text,
                f"{mean_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight="bold",
                color=COLORS[row["family"]],
                zorder=4,
            )

        labels = sub["method_display"].tolist()
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_xlim(-0.7, len(sub) - 0.3)
        ax.set_ylim(0.0, 1.1)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.grid(axis="y", color="#d1d5db", linewidth=0.75, alpha=0.7, zorder=0)
        ax.set_axisbelow(True)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        for tick, family in zip(ax.get_xticklabels(), sub["family"]):
            if family == "scRAW":
                tick.set_fontweight("bold")
                tick.set_color(COLORS["scRAW"])

    legend_handles = [
        mpatches.Patch(facecolor=COLORS["scRAW"], alpha=0.72, label="scRAW"),
        mpatches.Patch(
            facecolor=COLORS["Rare Specific"], alpha=0.72, label="Rare Specific"
        ),
        mpatches.Patch(
            facecolor=COLORS["Méthodes traditionnelles"],
            alpha=0.72,
            label="Méthodes traditionnelles",
        ),
        mpatches.Patch(
            facecolor=COLORS["Correction batch"],
            alpha=0.72,
            label="Correction batch",
        ),
        plt.Line2D([0], [0], marker="D", linestyle="", markerfacecolor="white",
                   markeredgecolor="#111827", markersize=5, label="Moyenne"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=5,
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(
        "Top 3 par famille + scRAW, triés par moyenne décroissante",
        fontsize=12.5,
        fontweight="bold",
        y=1.026,
    )
    fig.text(
        0.5,
        0.004,
        "Boîtes : distribution sur les jeux de données communs. Losange blanc : moyenne.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#374151",
    )
    fig.tight_layout(rect=[0, 0.018, 1, 0.988], h_pad=1.05)

    png_path = ROOT / f"{OUTPUT_STEM}.png"
    pdf_path = ROOT / f"{OUTPUT_STEM}.pdf"
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    df = load_data()
    selection = build_selection(df)
    
    selection_csv = selection.drop(columns=["raw_values"])
    selection_path = ROOT / f"{OUTPUT_STEM}_selection.csv"
    selection_csv.to_csv(selection_path, index=False)
    print(f"Saved: {selection_path}")
    
    draw_plot(selection)


if __name__ == "__main__":
    main()
