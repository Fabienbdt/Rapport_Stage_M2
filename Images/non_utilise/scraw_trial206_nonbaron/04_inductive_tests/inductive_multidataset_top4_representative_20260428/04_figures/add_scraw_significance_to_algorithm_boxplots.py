#!/usr/bin/env python3
"""Add post-hoc pairwise significance letters to global algorithm boxplots.

The input table contains one mean value per dataset, algorithm and metric.  The
tests therefore pair algorithms by dataset.  For each metric, all algorithm
pairs are compared with two-sided paired t-tests and Holm correction.  Compact
letters mark algorithms that are not significantly different after correction.
"""

from __future__ import annotations

import itertools
import math
import shutil
import string
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


FIGURES_DIR = Path(__file__).resolve().parent
TABLES_DIR = FIGURES_DIR / "standalone_tables"
INPUT_CSV = TABLES_DIR / "dataset_level_metric_summary.csv"
TESTS_CSV = TABLES_DIR / "scraw_superiority_significance_tests.csv"
POSTHOC_TESTS_CSV = TABLES_DIR / "algorithm_pairwise_posthoc_tests.csv"
POSTHOC_LETTERS_CSV = TABLES_DIR / "algorithm_posthoc_significance_letters.csv"

ALGORITHM_ORDER = ["scRAW", "scNAME", "scMAE", "scDeepCluster"]
ALPHA = 0.05

PALETTE = {
    "scRAW": "#9fc9c9",
    "scNAME": "#9bb7f0",
    "scMAE": "#f3c994",
    "scDeepCluster": "#c7a8f5",
}


def _stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _format_p(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p_adj=n/a"
    if p_value < 0.001:
        return "p_adj<0.001"
    return f"p_adj={p_value:.3f}"


def _holm_adjust(p_values: list[float]) -> list[float]:
    adjusted = [float("nan")] * len(p_values)
    valid = [(idx, value) for idx, value in enumerate(p_values) if np.isfinite(value)]
    valid.sort(key=lambda item: item[1])
    running_max = 0.0
    m = len(valid)
    for rank, (idx, value) in enumerate(valid):
        corrected = min(1.0, (m - rank) * value)
        running_max = max(running_max, corrected)
        adjusted[idx] = running_max
    return adjusted


def _paired_ttest(first: pd.Series, second: pd.Series, alternative: str) -> tuple[float, float]:
    paired = pd.concat([first, second], axis=1).dropna()
    if paired.shape[0] < 2:
        return float("nan"), float("nan")

    differences = paired.iloc[:, 0].to_numpy(dtype=float) - paired.iloc[:, 1].to_numpy(dtype=float)
    mean_diff = float(np.mean(differences))
    if np.allclose(differences, 0.0):
        return 0.0, 1.0 if alternative == "two-sided" else 0.5
    std_diff = float(np.std(differences, ddof=1))
    if np.isclose(std_diff, 0.0):
        if alternative == "greater":
            if mean_diff > 0:
                return float("inf"), 0.0
            if mean_diff < 0:
                return float("-inf"), 1.0
            return 0.0, 0.5
        return float("inf") * (1 if mean_diff > 0 else -1), 0.0

    t_stat = mean_diff / (std_diff / math.sqrt(len(differences)))
    df = len(differences) - 1
    if alternative == "greater":
        p_value = float(stats.t.sf(t_stat, df=df))
    elif alternative == "two-sided":
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df=df))
    else:
        raise ValueError(f"Unsupported alternative: {alternative}")
    return float(t_stat), p_value


def _compute_scraw_superiority_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, metric_df in df.groupby("metric", sort=False):
        wide = metric_df.pivot_table(
            index="dataset_key",
            columns="algorithm_label",
            values="mean",
            aggfunc="mean",
            observed=True,
        )
        if "scRAW" not in wide:
            continue

        metric_rows: list[dict[str, object]] = []
        for comparator in [algorithm for algorithm in ALGORITHM_ORDER if algorithm != "scRAW"]:
            if comparator not in wide:
                continue
            paired = wide[["scRAW", comparator]].dropna()
            if paired.empty:
                continue

            t_stat, p_raw = _paired_ttest(
                paired["scRAW"],
                paired[comparator],
                alternative="greater",
            )
            scraw_mean = float(paired["scRAW"].mean())
            comparator_mean = float(paired[comparator].mean())
            metric_rows.append(
                {
                    "metric": metric,
                    "metric_label": str(metric_df["metric_label"].dropna().iloc[0]),
                    "comparison": f"scRAW>{comparator}",
                    "reference_algorithm": "scRAW",
                    "comparator_algorithm": comparator,
                    "test": "one_sided_paired_t_test_dataset_means",
                    "multiple_testing_correction": "Holm within metric across scRAW-vs-comparator tests",
                    "n_paired_datasets": int(len(paired)),
                    "scraw_mean": scraw_mean,
                    "comparator_mean": comparator_mean,
                    "mean_delta_scraw_minus_comparator": scraw_mean - comparator_mean,
                    "t_statistic": t_stat,
                    "p_value_raw": p_raw,
                }
            )

        adjusted = _holm_adjust([float(row["p_value_raw"]) for row in metric_rows])
        for row, p_adj in zip(metric_rows, adjusted):
            row["p_value_holm"] = p_adj
            row["significance"] = _stars(p_adj)
        rows.extend(metric_rows)

    return pd.DataFrame(rows)


def _compute_pairwise_posthoc_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, metric_df in df.groupby("metric", sort=False):
        wide = metric_df.pivot_table(
            index="dataset_key",
            columns="algorithm_label",
            values="mean",
            aggfunc="mean",
            observed=True,
        )

        metric_rows: list[dict[str, object]] = []
        for algorithm_a, algorithm_b in itertools.combinations(ALGORITHM_ORDER, 2):
            if algorithm_a not in wide or algorithm_b not in wide:
                continue
            paired = wide[[algorithm_a, algorithm_b]].dropna()
            if paired.empty:
                continue

            t_stat, p_raw = _paired_ttest(
                paired[algorithm_a],
                paired[algorithm_b],
                alternative="two-sided",
            )
            mean_a = float(paired[algorithm_a].mean())
            mean_b = float(paired[algorithm_b].mean())
            metric_rows.append(
                {
                    "metric": metric,
                    "metric_label": str(metric_df["metric_label"].dropna().iloc[0]),
                    "comparison": f"{algorithm_a} vs {algorithm_b}",
                    "algorithm_a": algorithm_a,
                    "algorithm_b": algorithm_b,
                    "test": "two_sided_paired_t_test_dataset_means",
                    "multiple_testing_correction": "Holm within metric across all algorithm pairs",
                    "n_paired_datasets": int(len(paired)),
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "mean_delta_a_minus_b": mean_a - mean_b,
                    "t_statistic": t_stat,
                    "p_value_raw": p_raw,
                }
            )

        adjusted = _holm_adjust([float(row["p_value_raw"]) for row in metric_rows])
        for row, p_adj in zip(metric_rows, adjusted):
            row["p_value_holm"] = p_adj
            row["significance"] = _stars(p_adj)
            row["significant"] = bool(np.isfinite(p_adj) and p_adj < ALPHA)
        rows.extend(metric_rows)

    return pd.DataFrame(rows)


def _letters_for_metric(metric_df: pd.DataFrame, metric_tests: pd.DataFrame) -> dict[str, str]:
    means = {
        algorithm: float(
            pd.to_numeric(
                metric_df.loc[metric_df["algorithm_label"] == algorithm, "mean"],
                errors="coerce",
            ).dropna().mean()
        )
        for algorithm in ALGORITHM_ORDER
        if not metric_df.loc[metric_df["algorithm_label"] == algorithm, "mean"].dropna().empty
    }
    ordered = sorted(means, key=lambda algorithm: (-means[algorithm], ALGORITHM_ORDER.index(algorithm)))
    if not ordered:
        return {}

    significant_pairs: set[frozenset[str]] = set()
    for _, row in metric_tests.iterrows():
        p_adj = float(row["p_value_holm"])
        if np.isfinite(p_adj) and p_adj < ALPHA:
            significant_pairs.add(frozenset([str(row["algorithm_a"]), str(row["algorithm_b"])]))

    def can_share_letter(group: tuple[str, ...]) -> bool:
        return all(
            frozenset(pair) not in significant_pairs
            for pair in itertools.combinations(group, 2)
        )

    cliques: list[tuple[str, ...]] = []
    for size in range(1, len(ordered) + 1):
        for group in itertools.combinations(ordered, size):
            if can_share_letter(group):
                cliques.append(group)

    maximal_cliques = [
        group
        for group in cliques
        if not any(set(group) < set(other) for other in cliques)
    ]
    maximal_cliques.sort(
        key=lambda group: (
            min(ordered.index(algorithm) for algorithm in group),
            -len(group),
            [ordered.index(algorithm) for algorithm in group],
        )
    )

    alphabet = list(string.ascii_lowercase)
    if len(maximal_cliques) > len(alphabet):
        raise ValueError("Not enough letters for compact letter display")

    letters = {algorithm: "" for algorithm in ordered}
    for idx, group in enumerate(maximal_cliques):
        letter = alphabet[idx]
        for algorithm in group:
            letters[algorithm] += letter
    return letters


def _compute_posthoc_letters(df: pd.DataFrame, posthoc_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric, metric_df in df.groupby("metric", sort=False):
        metric_tests = posthoc_df[posthoc_df["metric"] == metric]
        letters = _letters_for_metric(metric_df, metric_tests)
        metric_label = str(metric_df["metric_label"].dropna().iloc[0])
        for algorithm in ALGORITHM_ORDER:
            values = pd.to_numeric(
                metric_df.loc[metric_df["algorithm_label"] == algorithm, "mean"],
                errors="coerce",
            ).dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "metric": metric,
                    "metric_label": metric_label,
                    "algorithm": algorithm,
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "n_datasets": int(len(values)),
                    "posthoc_letters": letters.get(algorithm, ""),
                }
            )
    return pd.DataFrame(rows)


def _summary_label(values: pd.Series, algorithm: str) -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return f"{algorithm}: mean=NA, med=NA, n=0"
    return (
        f"{algorithm}: mean={values.mean():.3f}, "
        f"med={values.median():.3f}, n={len(values)}"
    )


def _annotate_posthoc_letters(
    ax,
    metric_df: pd.DataFrame,
    letters_df: pd.DataFrame,
    metric: str,
) -> None:
    metric_letters = letters_df[letters_df["metric"] == metric]
    if metric_letters.empty:
        return

    max_value = float(pd.to_numeric(metric_df["mean"], errors="coerce").max())
    label_y = max(1.035, max_value + 0.045)
    for x, algorithm in enumerate(ALGORITHM_ORDER, start=1):
        row = metric_letters.loc[metric_letters["algorithm"] == algorithm]
        if row.empty:
            continue
        ax.text(
            x,
            label_y,
            str(row["posthoc_letters"].iloc[0]),
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="#111827",
            clip_on=False,
        )


def _plot_metric(
    df: pd.DataFrame,
    posthoc_letters_df: pd.DataFrame,
    metric: str,
) -> None:
    metric_df = df[df["metric"] == metric].copy()
    metric_df["algorithm_label"] = pd.Categorical(
        metric_df["algorithm_label"], categories=ALGORITHM_ORDER, ordered=True
    )
    metric_df = metric_df.sort_values("algorithm_label")
    metric_label = str(metric_df["metric_label"].dropna().iloc[0])

    data = [
        pd.to_numeric(
            metric_df.loc[metric_df["algorithm_label"] == algorithm, "mean"],
            errors="coerce",
        ).dropna()
        for algorithm in ALGORITHM_ORDER
    ]
    positions = np.arange(1, len(ALGORITHM_ORDER) + 1)

    fig, ax = plt.subplots(figsize=(11.6, 6.6))
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "#111827",
            "markersize": 6,
            "markeredgewidth": 1.1,
        },
        medianprops={"color": "#111827", "linewidth": 1.6},
        whiskerprops={"color": "#374151", "linewidth": 1.2},
        capprops={"color": "#374151", "linewidth": 1.2},
    )
    for patch, algorithm in zip(box["boxes"], ALGORITHM_ORDER):
        patch.set_facecolor(PALETTE[algorithm])
        patch.set_alpha(0.55)
        patch.set_edgecolor("#6b7280")
        patch.set_linewidth(1.2)

    for x, values in zip(positions, data):
        values = values.to_numpy(dtype=float)
        offsets = np.linspace(-0.06, 0.06, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(
            x + offsets,
            values,
            s=34,
            color="#111827",
            alpha=0.9,
            zorder=3,
            label="_nolegend_",
        )

    y_upper = max(1.12, min(1.22, float(metric_df["mean"].max()) + 0.18))
    ax.set_ylim(-0.02, y_upper)
    _annotate_posthoc_letters(ax, metric_df, posthoc_letters_df, metric)

    counts = [len(values.dropna()) for values in data]
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{algorithm}\nn={count}" for algorithm, count in zip(ALGORITHM_ORDER, counts)]
    )
    ax.set_xlabel("Algorithm")
    ax.set_ylabel(metric_label)
    ax.grid(axis="y", color="#d1d5db", linewidth=0.85, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        f"{metric_label} by algorithm - dataset-level means",
        loc="left",
        fontsize=18,
        pad=24,
    )
    ax.text(
        0.0,
        1.04,
        "Each point is one dataset mean over available inductive splits.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#4b5563",
    )
    ax.text(
        0.0,
        -0.16,
        "Letters: pairwise two-sided paired t-tests on dataset means; Holm-adjusted per metric. Shared letters: not significantly different (alpha=0.05).",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#374151",
    )

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color="#111827", label="dataset mean"),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            linestyle="",
            markerfacecolor="white",
            markeredgecolor="#111827",
            color="#111827",
            label="algorithm mean",
        ),
    ]
    legend_handles.extend(
        plt.Line2D(
            [0],
            [0],
            color="none",
            label=_summary_label(
                metric_df.loc[metric_df["algorithm_label"] == algorithm, "mean"],
                algorithm,
            ),
        )
        for algorithm in ALGORITHM_ORDER
    )
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.72),
        borderaxespad=0.0,
        frameon=True,
        fontsize=9,
    )

    output_path = FIGURES_DIR / f"{metric}_by_algorithm_boxplot.png"
    backup_path = FIGURES_DIR / f"{metric}_by_algorithm_boxplot__without_significance.png"
    scraw_backup_path = FIGURES_DIR / f"{metric}_by_algorithm_boxplot__scraw_brackets.png"
    if output_path.exists() and not scraw_backup_path.exists():
        shutil.copy2(output_path, scraw_backup_path)
    if output_path.exists() and not backup_path.exists():
        shutil.copy2(output_path, backup_path)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing dataset-level table: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df = df.dropna(subset=["mean"])
    df = df[df["algorithm_label"].isin(ALGORITHM_ORDER)].copy()

    scraw_tests_df = _compute_scraw_superiority_tests(df)
    scraw_tests_df.to_csv(TESTS_CSV, index=False)
    posthoc_tests_df = _compute_pairwise_posthoc_tests(df)
    posthoc_tests_df.to_csv(POSTHOC_TESTS_CSV, index=False)
    posthoc_letters_df = _compute_posthoc_letters(df, posthoc_tests_df)
    posthoc_letters_df.to_csv(POSTHOC_LETTERS_CSV, index=False)

    for metric in sorted(df["metric"].unique()):
        _plot_metric(df, posthoc_letters_df, str(metric))

    print(f"wrote={POSTHOC_TESTS_CSV}")
    print(f"wrote={POSTHOC_LETTERS_CSV}")
    print(f"figures_dir={FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
