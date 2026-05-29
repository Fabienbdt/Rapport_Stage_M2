#!/usr/bin/env python3
"""Generate a compact oral-presentation panel.

The plot compares scRAW against the best non-scRAW method of each report
family, selected independently for every metric from the same common-8 data
used in the report figure.
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
from matplotlib.lines import Line2D


SOURCE_CSV = (
    ROOT.parent
    / "scraw_common8_family_panel"
    / "common8_primary_raw_by_dataset_long.csv"
)

OUTPUT_STEM = "common8_scraw_vs_best_family_per_metric"
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

FAMILY_SHORT_LABELS = {
    "scRAW": "scRAW",
    "Rare Specific": "Rare Specific",
    "Méthodes traditionnelles": "Traditionnelles",
    "Correction batch": "Correction batch",
}

COLORS = {
    "scRAW": "#dc2626",
    "Rare Specific": "#2563eb",
    "Méthodes traditionnelles": "#ea580c",
    "Correction batch": "#16a34a",
}


def load_data() -> pd.DataFrame:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Missing source table: {SOURCE_CSV}")

    df = pd.read_csv(SOURCE_CSV)
    expected_cols = {"family", "dataset", "method", "source_method", "metric", "value"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {SOURCE_CSV}: {sorted(missing)}")

    df = df[df["metric"].isin(METRICS)].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def values_for(
    df: pd.DataFrame, *, metric: str, source_method: str, family: str | None = None
) -> pd.DataFrame:
    sub = df[(df["metric"] == metric) & (df["source_method"] == source_method)]
    if family is not None:
        sub = sub[sub["family"] == family]
    return sub.drop_duplicates(["dataset", "metric", "source_method"]).copy()


def select_methods(df: pd.DataFrame) -> tuple[dict[str, list[dict]], pd.DataFrame]:
    by_metric: dict[str, list[dict]] = {}
    selection_rows: list[dict] = []

    for metric in METRICS:
        selected: list[dict] = []

        scraw_values = values_for(df, metric=metric, source_method=SCRAW_SOURCE_METHOD)
        if scraw_values.empty:
            raise ValueError(f"No scRAW values found for metric {metric}")

        selected.append(
            {
                "metric": metric,
                "family": "scRAW",
                "source_method": SCRAW_SOURCE_METHOD,
                "method_display": "scRAW",
                "values": scraw_values["value"].to_numpy(),
                "datasets": sorted(scraw_values["dataset"].unique()),
            }
        )

        for family in FAMILIES:
            family_values = df[
                (df["metric"] == metric)
                & (df["family"] == family)
                & (df["source_method"] != SCRAW_SOURCE_METHOD)
            ].copy()
            if family_values.empty:
                raise ValueError(f"No values found for family {family} and metric {metric}")

            means = (
                family_values.groupby(["source_method", "method"], sort=False)["value"]
                .mean()
                .sort_values(ascending=False)
            )
            best_source_method, best_display = means.index[0]
            best_values = values_for(
                df,
                metric=metric,
                source_method=best_source_method,
                family=family,
            )

            selected.append(
                {
                    "metric": metric,
                    "family": family,
                    "source_method": best_source_method,
                    "method_display": best_display,
                    "values": best_values["value"].to_numpy(),
                    "datasets": sorted(best_values["dataset"].unique()),
                }
            )

        for rank, row in enumerate(selected, start=1):
            values = row["values"]
            selection_rows.append(
                {
                    "metric": metric,
                    "rank_in_plot": rank,
                    "family": row["family"],
                    "source_method": row["source_method"],
                    "method_display": row["method_display"],
                    "mean": float(np.nanmean(values)),
                    "median": float(np.nanmedian(values)),
                    "n_values": int(np.sum(~np.isnan(values))),
                    "datasets": "; ".join(row["datasets"]),
                }
            )

        by_metric[metric] = selected

    return by_metric, pd.DataFrame(selection_rows)


def tick_label(row: dict) -> str:
    family = row["family"]
    if family == "scRAW":
        return "scRAW"
    return f"{row['method_display']}\n{FAMILY_SHORT_LABELS[family]}"


def draw_plot(by_metric: dict[str, list[dict]]) -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
        }
    )

    fig, axes = plt.subplots(
        nrows=len(METRICS),
        ncols=1,
        figsize=(9.4, 12.2),
        squeeze=True,
    )

    for ax, metric in zip(axes, METRICS):
        rows = by_metric[metric]
        data = [row["values"] for row in rows]
        positions = np.arange(1, len(rows) + 1, dtype=float)
        colors = [COLORS[row["family"]] for row in rows]

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.52,
            patch_artist=True,
            showmeans=True,
            showfliers=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "#111827",
                "markersize": 4.7,
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
                "alpha": 0.48,
            },
        )

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.62)
            patch.set_edgecolor("#374151")
            patch.set_linewidth(1.0)

        for xpos in [1.5, 2.5, 3.5]:
            ax.axvline(
                xpos,
                color="#9ca3af",
                linestyle="--",
                linewidth=0.8,
                alpha=0.55,
                zorder=1,
            )

        means = [float(np.nanmean(row["values"])) for row in rows]
        best_mean = max(means)
        for pos, mean_value, row, color in zip(positions, means, rows, colors):
            y_annot = min(float(np.nanmax(row["values"])) + 0.035, 1.035)
            ax.text(
                pos,
                y_annot,
                f"{mean_value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.6,
                fontweight="bold" if mean_value == best_mean else "semibold",
                color=color,
                zorder=5,
            )

        ax.set_xlim(0.45, len(rows) + 0.55)
        ax.set_ylim(-0.04, 1.08)
        ax.set_xticks(positions)
        ax.set_xticklabels([tick_label(row) for row in rows], ha="center")
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
        ax.grid(axis="y", color="#d1d5db", linewidth=0.75, alpha=0.7)
        ax.set_axisbelow(True)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        for tick, row in zip(ax.get_xticklabels(), rows):
            if row["family"] == "scRAW":
                tick.set_fontweight("bold")
                tick.set_color(COLORS["scRAW"])

    legend_handles = [
        mpatches.Patch(facecolor=COLORS["scRAW"], alpha=0.62, label="scRAW"),
        mpatches.Patch(
            facecolor=COLORS["Rare Specific"], alpha=0.62, label="Rare Specific"
        ),
        mpatches.Patch(
            facecolor=COLORS["Méthodes traditionnelles"],
            alpha=0.62,
            label="Méthodes traditionnelles",
        ),
        mpatches.Patch(
            facecolor=COLORS["Correction batch"],
            alpha=0.62,
            label="Correction batch",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="#111827",
            markersize=5,
            label="Moyenne",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.012),
        ncol=5,
        frameon=False,
        fontsize=8.8,
    )

    fig.suptitle(
        "scRAW vs meilleur algorithme de chaque famille",
        fontsize=12.5,
        fontweight="bold",
        y=1.035,
    )
    fig.text(
        0.5,
        0.004,
        "Chaque boîte résume les scores disponibles sur les huit jeux de données communs.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#374151",
    )

    fig.tight_layout(rect=[0, 0.018, 1, 0.985], h_pad=1.05)

    png_path = ROOT / f"{OUTPUT_STEM}.png"
    pdf_path = ROOT / f"{OUTPUT_STEM}.pdf"
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main() -> None:
    df = load_data()
    by_metric, selection = select_methods(df)
    selection_path = ROOT / f"{OUTPUT_STEM}_selection.csv"
    selection.to_csv(selection_path, index=False)
    print(f"Saved: {selection_path}")
    draw_plot(by_metric)


if __name__ == "__main__":
    main()
