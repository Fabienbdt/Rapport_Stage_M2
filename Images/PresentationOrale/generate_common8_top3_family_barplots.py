#!/usr/bin/env python3
"""Generate the report figure 4 common-8 family barplots.

The figure is rebuilt directly from the trial206 consolidated table. scRAW is
the fixed `scRAW (trial_0017)` row, and the batch-correction metric is set to
NA for common-8 datasets without an exploitable batch effect.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fbidet")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE_CSV = Path(
    "/data2/fbidet/scRAW_EXPERIMENTAL/results/"
    "presentation_trial206_nonbaron_20260324/00_source_tables/"
    "trial206_all_results_table.csv"
)

OUTPUT_STEM = "common8_top3_family_sorted_barplots"
SCRAW_METHOD = "scRAW (trial_0017)"

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

NO_BATCH_EFFECT_DATASETS = {
    "Paul15 bone marrow",
    "Tabula Muris liver",
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
    "RareACC": "Rare ACC",
    "UltraRareACC": "Ultra Rare ACC",
    "Batch correction": "Correction batch",
}

FAMILY_METHODS = {
    "Rare Specific": ["scAIDE", "scCAD", "GiniClust", "DeepScena", "CellSIUS"],
    "Méthodes généralistes": ["pca_leiden", "scMAE", "scNAME", "scvi"],
    "Correction batch": ["Harmony", "ComBat", "DESC", "Scanorama"],
}

COLORS = {
    "scRAW": "#dc2626",
    "Rare Specific": "#2563eb",
    "Méthodes généralistes": "#ea580c",
    "Correction batch": "#16a34a",
}

METHOD_DISPLAY = {
    SCRAW_METHOD: "scRAW",
    "pca_leiden": "PCA+Leiden",
    "scvi": "scVI",
}


def display_method(method: str) -> str:
    return METHOD_DISPLAY.get(method, method)


def is_scraw_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1"})


def load_filtered() -> pd.DataFrame:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Missing source table: {SOURCE_CSV}")

    df = pd.read_csv(SOURCE_CSV)
    required = {"dataset", "method", "trial_id", "is_scraw_method", *METRICS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {SOURCE_CSV}: {sorted(missing)}")

    scraw_trial_0017 = (df["method"] == SCRAW_METHOD) & (
        df["trial_id"] == "trial_0017"
    )
    keep_non_scraw = ~is_scraw_series(df["is_scraw_method"])
    df = df[df["dataset"].isin(COMMON8) & (keep_non_scraw | scraw_trial_0017)].copy()

    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df.loc[df["dataset"].isin(NO_BATCH_EFFECT_DATASETS), "Batch correction"] = np.nan
    return df


def build_long_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    all_methods = {SCRAW_METHOD}
    for methods in FAMILY_METHODS.values():
        all_methods.update(methods)

    for method in sorted(all_methods):
        method_df = df[df["method"] == method].copy()
        family = "scRAW"
        for family_name, methods in FAMILY_METHODS.items():
            if method in methods:
                family = family_name
                break

        for dataset in COMMON8:
            dataset_df = method_df[method_df["dataset"] == dataset]
            for metric in METRICS:
                value = dataset_df[metric].mean() if not dataset_df.empty else np.nan
                rows.append(
                    {
                        "metric": metric,
                        "family": family,
                        "source_method": method,
                        "method_display": display_method(method),
                        "dataset": dataset,
                        "value": value,
                    }
                )

    return pd.DataFrame(rows)


def summarize_values(values: pd.Series) -> tuple[float, float, int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
    return mean, std, int(len(clean))


def row_for_method(metric_df: pd.DataFrame, method: str, family: str) -> dict[str, object]:
    method_df = metric_df[metric_df["source_method"] == method].copy()
    mean, std, n_values = summarize_values(method_df["value"])
    datasets = method_df.loc[method_df["value"].notna(), "dataset"].tolist()
    na_datasets = method_df.loc[method_df["value"].isna(), "dataset"].tolist()
    return {
        "metric": method_df["metric"].iloc[0],
        "family": family,
        "source_method": method,
        "method_display": display_method(method),
        "mean": mean,
        "std": std,
        "n_values": n_values,
        "datasets": "; ".join(datasets),
        "na_datasets": "; ".join(na_datasets),
    }


def select_metric_rows(long_df: pd.DataFrame, metric: str) -> list[dict[str, object]]:
    metric_df = long_df[long_df["metric"] == metric].copy()
    rows: list[dict[str, object]] = []

    scraw_df = metric_df[metric_df["source_method"] == SCRAW_METHOD]
    if scraw_df["value"].dropna().empty:
        raise ValueError(f"No scRAW values found for {metric}")
    rows.append(row_for_method(metric_df, SCRAW_METHOD, "scRAW"))

    for family, methods in FAMILY_METHODS.items():
        family_df = metric_df[metric_df["source_method"].isin(methods)].copy()
        means = (
            family_df.groupby("source_method")["value"]
            .mean()
            .dropna()
            .sort_values(ascending=False)
        )
        if means.empty:
            raise ValueError(f"No values found for {family} / {metric}")
        for method in means.head(3).index:
            rows.append(row_for_method(metric_df, method, family))

    rows.sort(key=lambda row: float(row["mean"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank_after_global_sort"] = rank
    return rows


def build_selection(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        rows.extend(select_metric_rows(long_df, metric))
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
        colors = [COLORS[family] for family in sub["family"]]

        bars = ax.bar(
            positions,
            sub["mean"].to_numpy(),
            yerr=sub["std"].to_numpy(),
            width=0.68,
            color=colors,
            alpha=0.72,
            edgecolor="#111827",
            linewidth=0.9,
            capsize=4.5,
            error_kw={
                "elinewidth": 1.15,
                "ecolor": "#111827",
                "capthick": 1.15,
            },
            zorder=3,
        )

        for bar, (_, row) in zip(bars, sub.iterrows()):
            y_text = min(row["mean"] + row["std"] + 0.035, 1.045)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_text,
                f"{row['mean']:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight="bold",
                color=COLORS[row["family"]],
                zorder=4,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(sub["method_display"].tolist(), rotation=35, ha="right")
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
            facecolor=COLORS["Méthodes généralistes"],
            alpha=0.72,
            label="Méthodes généralistes",
        ),
        mpatches.Patch(
            facecolor=COLORS["Correction batch"],
            alpha=0.72,
            label="Correction batch",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=4,
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
        "Barres : moyenne. Traits : écart type. Correction batch : Paul15 bone marrow et Tabula Muris liver = NA.",
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
    df = load_filtered()
    long_df = build_long_table(df)
    selection = build_selection(long_df)
    selection_path = ROOT / f"{OUTPUT_STEM}_selection.csv"
    selection.to_csv(selection_path, index=False)
    print(f"Saved: {selection_path}")
    draw_plot(selection)


if __name__ == "__main__":
    main()
