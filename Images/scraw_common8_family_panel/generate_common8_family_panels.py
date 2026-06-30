#!/usr/bin/env python3
"""Generate common-8 family panels for the M2 report.

The script keeps only scRAW stable_generalist, groups methods into the report
families requested for the M2 manuscript. The main figure shows scRAW
stable_generalist in every family/metric panel plus the top 3 methods for that metric
inside the family. Appendix panels keep every method in each family.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fbidet")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
CSV_PATH = Path(
    "/Users/fabienbidet/Documents/MASTER2/STAGE/"
    "presentation_stable_generalist_nonbaron_20260324/00_source_tables/"
    "stable_generalist_all_results_table.csv"
)
if not CSV_PATH.exists():
    CSV_PATH = Path(
        "/data2/fbidet/scRAW_EXPERIMENTAL/results/"
        "presentation_stable_generalist_nonbaron_20260324/00_source_tables/"
        "stable_generalist_all_results_table.csv"
    )

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

METRICS = [
    "ARI",
    "ACC",
    "BalancedACC",
    "RareACC",
    "BalancedRareACC",
    "UltraRareACC",
    "Batch correction",
]

NO_BATCH_EFFECT_DATASETS = {
    "Paul15 bone marrow",
    "Tabula Muris liver",
}

METRIC_LABELS = {
    "ARI": "ARI",
    "ACC": "ACC",
    "BalancedACC": "BalancedACC",
    "RareACC": "RareACC",
    "BalancedRareACC": "BalancedRareACC",
    "UltraRareACC": "UltraRareACC",
    "Batch correction": "Correction batch",
}

SCRAW_SOURCE_METHOD = "scRAW"
SCRAW_METHOD = "scRAW (stable_generalist)"

PRIMARY_FAMILIES = {
    "Rare Specific": [
        "scAIDE",
        "scCAD",
        "GiniClust",
        "DeepScena",
        "CellSIUS",
    ],
    "Methodes generalistes": [
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

PRIMARY_FAMILIES_WITH_SCRAW = {
    family: ([SCRAW_METHOD] + [m for m in methods if m != SCRAW_METHOD])
    for family, methods in PRIMARY_FAMILIES.items()
}

HARMONY_COMPLEMENT_FAMILIES = {
    "Rare Specific + Harmony": [
        "scAIDE+Harmony",
        "scCAD+Harmony",
        "GiniClust+Harmony",
        "DeepScena+Harmony",
        "CellSIUS+Harmony",
    ],
    "Generalistes + Harmony": [
        "Harmony",
        "scMAE+Harmony",
        "scNAME+Harmony",
        "scvi",
    ],
}

HARMONY_COMPLEMENT_FAMILIES_WITH_SCRAW = {
    family: ([SCRAW_METHOD] + [m for m in methods if m != SCRAW_METHOD])
    for family, methods in HARMONY_COMPLEMENT_FAMILIES.items()
}

FAMILY_DISPLAY = {
    "Rare Specific": "Rare Specific",
    "Methodes generalistes": "Méthodes\ngénéralistes",
    "Correction batch": "Correction\nbatch",
    "Rare Specific + Harmony": "Rare Specific\n+ Harmony",
    "Generalistes + Harmony": "Généralistes\n+ Harmony",
}

FAMILY_TABLE_DISPLAY = {
    "Rare Specific": "Rare Specific",
    "Methodes generalistes": "Méthodes généralistes",
    "Correction batch": "Correction batch",
    "Rare Specific + Harmony": "Rare Specific + Harmony",
    "Generalistes + Harmony": "Méthodes généralistes + Harmony",
}

FAMILY_COLORS = {
    "Rare Specific": "#4C78A8",
    "Methodes generalistes": "#F58518",
    "Correction batch": "#54A24B",
    "Rare Specific + Harmony": "#72B7B2",
    "Generalistes + Harmony": "#B279A2",
}

METHOD_DISPLAY = {
    "pca_leiden": "PCA+Leiden",
    "ComBat": "ComBat",
    "scvi": "scVI",
    SCRAW_METHOD: "scRAW",
}


def display_method(method: str) -> str:
    return METHOD_DISPLAY.get(method, method)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def fmt_fr(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.3f}".replace(".", ",")


def is_scraw_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1"})


def load_filtered() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    scraw_stable_generalist = (
        df["method"].isin({SCRAW_SOURCE_METHOD, SCRAW_METHOD})
        & (df["trial_id"] == "stable_generalist")
    )
    keep_non_scraw = ~is_scraw_series(df["is_scraw_method"])
    df = df[df["dataset"].isin(COMMON8) & (keep_non_scraw | scraw_stable_generalist)].copy()
    df.loc[
        df["method"].isin({SCRAW_SOURCE_METHOD, SCRAW_METHOD})
        & (df["trial_id"] == "stable_generalist"),
        "method",
    ] = SCRAW_METHOD
    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df.loc[df["dataset"].isin(NO_BATCH_EFFECT_DATASETS), "Batch correction"] = pd.NA
    df["method_display"] = df["method"].map(display_method)
    return df


def summarize(df: pd.DataFrame, families: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for family, methods in families.items():
        for method in methods:
            sub = df[df["method"] == method]
            if sub.empty:
                continue
            row = {"family": family, "method": method}
            for metric in METRICS:
                row[metric] = sub[metric].mean()
            row["score_moyen"] = pd.Series([row[m] for m in METRICS]).mean()
            row["n_datasets"] = sub["dataset"].nunique()
            row["method_display"] = display_method(method)
            rows.append(row)
    return pd.DataFrame(rows)


def select_top_n(
    summary: pd.DataFrame, families: dict[str, list[str]], n: int = 4
) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for family in families:
        sub = summary[summary["family"] == family]
        selected[family] = (
            sub.sort_values("score_moyen", ascending=False)["method"].head(n).tolist()
        )
    return selected


def select_top3_per_metric_plus_scraw(
    df: pd.DataFrame, families: dict[str, list[str]]
) -> dict[str, dict[str, list[str]]]:
    selected: dict[str, dict[str, list[str]]] = {}
    for family, methods in families.items():
        selected[family] = {}
        for metric in METRICS:
            sub = df[df["method"].isin(methods)].copy()
            top3 = (
                sub.groupby("method")[metric]
                .mean()
                .dropna()
                .sort_values(ascending=False)
                .head(3)
                .index.tolist()
            )
            selected[family][metric] = [SCRAW_METHOD] + [
                method for method in top3 if method != SCRAW_METHOD
            ]
    return selected


def draw_panel(
    df: pd.DataFrame,
    methods_by_family: dict[str, list[str] | dict[str, list[str]]],
    output_stem: str,
    title: str,
    *,
    figsize: tuple[float, float],
    label_size: float,
) -> None:
    family_names = list(methods_by_family)
    fig, axes = plt.subplots(
        nrows=len(METRICS),
        ncols=len(family_names),
        figsize=figsize,
        sharex=True,
        squeeze=False,
    )

    for row, metric in enumerate(METRICS):
        for col, family in enumerate(family_names):
            ax = axes[row, col]
            methods_spec = methods_by_family[family]
            methods = (
                methods_spec[metric]
                if isinstance(methods_spec, dict)
                else methods_spec
            )
            sub = df[df["method"].isin(methods)].copy()
            means = sub.groupby("method")[metric].mean().sort_values(ascending=True)
            ordered = means.index.tolist()
            values = [sub.loc[sub["method"] == method, metric].dropna().values for method in ordered]
            labels = [display_method(method) for method in ordered]

            bp = ax.boxplot(
                values,
                vert=False,
                patch_artist=True,
                widths=0.58,
                showmeans=True,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "white",
                    "markeredgecolor": "#333333",
                    "markersize": 4.0,
                    "linestyle": "none",
                },
                medianprops={"color": "#9A031E", "linewidth": 1.5},
                boxprops={"linewidth": 1.0, "edgecolor": "#333333"},
                whiskerprops={"linewidth": 0.9, "color": "#333333"},
                capprops={"linewidth": 0.9, "color": "#333333"},
                flierprops={
                    "marker": "o",
                    "markerfacecolor": "white",
                    "markeredgecolor": "#333333",
                    "markersize": 3.2,
                    "linestyle": "none",
                },
            )
            for box in bp["boxes"]:
                box.set_facecolor(FAMILY_COLORS[family])
                box.set_alpha(0.68)

            ax.set_xlim(-0.02, 1.02)
            ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.grid(axis="x", color="#D9D9D9", linewidth=0.55)
            ax.grid(axis="y", color="#EEEEEE", linewidth=0.45)
            ax.set_axisbelow(True)
            ax.tick_params(axis="both", labelsize=label_size)
            ax.set_yticklabels(labels, fontsize=label_size)
            for tick, method in zip(ax.get_yticklabels(), ordered):
                if method == SCRAW_METHOD:
                    tick.set_fontweight("bold")

            if row == 0:
                ax.set_title(FAMILY_DISPLAY[family], fontsize=9.5, fontweight="bold")
            if col == 0:
                ax.set_ylabel(
                    METRIC_LABELS[metric],
                    fontsize=8.5,
                    fontweight="bold",
                    rotation=90,
                    labelpad=34,
                )
            else:
                ax.set_ylabel("")
            if row != len(METRICS) - 1:
                ax.tick_params(axis="x", labelbottom=False)

    legend_handles = [
        Line2D([0], [0], color="#9A031E", lw=2, label="Médiane"),
        Line2D(
            [0],
            [0],
            marker="D",
            markerfacecolor="white",
            markeredgecolor="#333333",
            color="none",
            label="Moyenne",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=True,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.012),
    )
    fig.suptitle(title, fontsize=11.5, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.043,
        "Chaque boîte résume les scores disponibles sur les 8 jeux de données communs.",
        ha="center",
        va="center",
        fontsize=7.5,
    )
    fig.tight_layout(rect=[0.018, 0.065, 0.995, 0.972], h_pad=0.85, w_pad=0.85)
    fig.savefig(ROOT / f"{output_stem}.png", dpi=300)
    fig.savefig(ROOT / f"{output_stem}.pdf")
    plt.close(fig)


def export_summary_tables(
    summary: pd.DataFrame,
    selected: dict[str, list[str] | dict[str, list[str]]],
    *,
    prefix: str,
    caption: str,
    label: str,
) -> None:
    selected_rows = []
    for family, methods_spec in selected.items():
        if isinstance(methods_spec, dict):
            for metric, methods in methods_spec.items():
                for rank, method in enumerate(methods, start=1):
                    selected_rows.append(
                        {
                            "family": family,
                            "metric": metric,
                            "rank": rank,
                            "method": method,
                        }
                    )
        else:
            for rank, method in enumerate(methods_spec, start=1):
                selected_rows.append({"family": family, "rank": rank, "method": method})
    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(ROOT / f"{prefix}_selection.csv", index=False)

    out = summary.sort_values(["family", "score_moyen"], ascending=[True, False])
    out.to_csv(ROOT / f"{prefix}_all_methods_means.csv", index=False)

    lines = [
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{longtable}{p{0.20\textwidth}p{0.17\textwidth}rrrrrrrr}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
        r"\textbf{Famille} & \textbf{Méthode} & \textbf{ARI} & \textbf{ACC} & \textbf{BalACC} & \textbf{RareACC} & \textbf{BalRareACC} & \textbf{UltraRareACC} & \textbf{Batch} & \textbf{Score} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{Famille} & \textbf{Méthode} & \textbf{ARI} & \textbf{ACC} & \textbf{BalACC} & \textbf{RareACC} & \textbf{BalRareACC} & \textbf{UltraRareACC} & \textbf{Batch} & \textbf{Score} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{10}{r}{\textit{Suite page suivante}}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for _, row in out.iterrows():
        family = latex_escape(FAMILY_TABLE_DISPLAY[row["family"]])
        method = row["method_display"]
        if row["method"] == SCRAW_METHOD:
            method = r"\textbf{scRAW}"
        else:
            method = latex_escape(method)
        values = " & ".join(fmt_fr(row[col]) for col in [*METRICS, "score_moyen"])
        lines.append(f"{family} & {method} & {values} \\\\")
    lines.extend([r"\end{longtable}", r"}"])
    (ROOT / f"{prefix}_all_methods_means_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def export_raw_csvs(
    df: pd.DataFrame, families: dict[str, list[str]], prefix: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for family, methods in families.items():
        for method in methods:
            sub = df[df["method"] == method]
            for _, row in sub.sort_values("dataset").iterrows():
                out = {
                    "family": FAMILY_TABLE_DISPLAY[family],
                    "dataset": row["dataset"],
                    "method": display_method(method),
                    "source_method": method,
                }
                for metric in METRICS:
                    out[metric] = row[metric]
                rows.append(out)
    wide = pd.DataFrame(rows)
    wide.to_csv(ROOT / f"{prefix}_raw_by_dataset.csv", index=False)

    long = wide.melt(
        id_vars=["family", "dataset", "method", "source_method"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    ).sort_values(["family", "dataset", "method", "metric"])
    long.to_csv(ROOT / f"{prefix}_raw_by_dataset_long.csv", index=False)
    return wide, long


def export_scraw_first_counts(df: pd.DataFrame) -> pd.DataFrame:
    methods = sorted({method for methods in PRIMARY_FAMILIES.values() for method in methods})
    methods = [SCRAW_METHOD] + [method for method in methods if method != SCRAW_METHOD]
    base = df[df["method"].isin(methods)].copy()

    rows = []
    for metric in METRICS:
        first_datasets = []
        ranks = []
        available = 0
        for dataset in COMMON8:
            sub = base.loc[base["dataset"] == dataset, ["method", metric]].dropna()
            scraw_value = sub.loc[sub["method"] == SCRAW_METHOD, metric]
            if sub.empty or scraw_value.empty:
                ranks.append(
                    {
                        "metric": metric,
                        "dataset": dataset,
                        "scraw_value": pd.NA,
                        "best_value": pd.NA,
                        "scraw_rank": pd.NA,
                        "is_scraw_first": False,
                    }
                )
                continue
            available += 1
            value = scraw_value.iloc[0]
            best_value = sub[metric].max()
            rank = int((sub[metric] > value).sum() + 1)
            is_first = bool(value == best_value)
            if is_first:
                first_datasets.append(dataset)
            ranks.append(
                {
                    "metric": metric,
                    "dataset": dataset,
                    "scraw_value": value,
                    "best_value": best_value,
                    "scraw_rank": rank,
                    "is_scraw_first": is_first,
                }
            )
        rows.append(
            {
                "metric": metric,
                "datasets_available": available,
                "scraw_first_count": len(first_datasets),
                "datasets_where_scraw_is_first": "; ".join(first_datasets),
            }
        )

        pd.DataFrame(ranks).to_csv(ROOT / f"common8_scraw_ranks_{metric.replace(' ', '_')}.csv", index=False)

    counts = pd.DataFrame(rows)
    counts.to_csv(ROOT / "common8_scraw_first_counts.csv", index=False)
    return counts


def main() -> None:
    df = load_filtered()
    primary_summary = summarize(df, PRIMARY_FAMILIES_WITH_SCRAW)
    primary_selected = select_top3_per_metric_plus_scraw(df, PRIMARY_FAMILIES)
    export_raw_csvs(df, PRIMARY_FAMILIES_WITH_SCRAW, "common8_primary")
    export_raw_csvs(df, HARMONY_COMPLEMENT_FAMILIES_WITH_SCRAW, "common8_harmony_complement")
    first_counts = export_scraw_first_counts(df)
    export_summary_tables(
        primary_summary,
        primary_selected,
        prefix="common8_primary",
        caption=(
            r"Moyennes des performances des familles principales sur les huit jeux "
            r"de données communs, avec la configuration \textbf{scRAW} retenue."
        ),
        label="tab:common8_primary_all_methods_means",
    )

    harmony_summary = summarize(df, HARMONY_COMPLEMENT_FAMILIES_WITH_SCRAW)
    harmony_selected = select_top_n(harmony_summary, HARMONY_COMPLEMENT_FAMILIES_WITH_SCRAW)
    export_summary_tables(
        harmony_summary,
        harmony_selected,
        prefix="common8_harmony_complement",
        caption=(
            r"Moyennes des performances complémentaires avec Harmony sur les huit "
            r"jeux de données communs."
        ),
        label="tab:common8_harmony_complement_means",
    )

    draw_panel(
        df,
        primary_selected,
        "common8_family_top3_plus_scraw_panel",
        "Top 3 par métrique et par famille + scRAW",
        figsize=(11.2, 16.2),
        label_size=6.7,
    )
    # Backward-compatible filename kept to avoid stale manually opened figures.
    draw_panel(
        df,
        primary_selected,
        "common8_family_top4_panel",
        "Top 3 par métrique et par famille + scRAW",
        figsize=(11.2, 16.2),
        label_size=6.7,
    )

    draw_panel(
        df,
        PRIMARY_FAMILIES_WITH_SCRAW,
        "common8_family_all_methods_panel",
        "Résultats complets par famille sur les 8 jeux de données communs",
        figsize=(11.8, 17.2),
        label_size=6.1,
    )

    draw_panel(
        df,
        HARMONY_COMPLEMENT_FAMILIES_WITH_SCRAW,
        "common8_harmony_complement_panel",
        "Résultats complémentaires avec Harmony",
        figsize=(8.0, 15.0),
        label_size=6.4,
    )

    print("Top 3 per metric + scRAW selection:")
    for family, by_metric in primary_selected.items():
        for metric, methods in by_metric.items():
            print(f"- {family} / {metric}: {', '.join(methods)}")
    print("Harmony complement:")
    for family, methods in HARMONY_COMPLEMENT_FAMILIES_WITH_SCRAW.items():
        print(f"- {family}: {', '.join(methods)}")
    print("scRAW first counts:")
    print(first_counts.to_string(index=False))
    print(f"Outputs written to {ROOT}")


if __name__ == "__main__":
    main()
