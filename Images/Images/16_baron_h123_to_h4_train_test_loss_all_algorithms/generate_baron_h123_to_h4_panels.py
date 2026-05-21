#!/usr/bin/env python3
"""Assemble BARON h123_to_h4 inductive train/test and loss panels."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"
SOURCE_DIR = OUT_DIR / "source_files"

SCRAW_RUN = Path(
    "/data2/fbidet/scRAW_Inductif/results/"
    "inductive_baron_trial0017_h123_to_h4_loss_metrics_20260507_143730"
)
BASELINE_RUN = Path(
    "/data2/fbidet/scRAW_Inductif/results/"
    "inductive_baron_h123_to_h4_all_baselines_loss_metrics_20260511"
)

ALGORITHMS = [
    {
        "key": "scraw_trial0017",
        "display": "scRAW trial0017",
        "metrics": SCRAW_RUN / "results" / "results.json",
        "loss_csv": SCRAW_RUN / "results" / "loss_history.csv",
        "loss_png": SCRAW_RUN / "figures" / "loss_history.png",
    },
    {
        "key": "scname",
        "display": "scNAME",
        "metrics": BASELINE_RUN / "scname_h123_to_h4" / "metrics.json",
        "loss_csv": BASELINE_RUN / "scname_h123_to_h4" / "results" / "loss_history.csv",
        "loss_png": BASELINE_RUN / "scname_h123_to_h4" / "figures" / "loss_history.png",
    },
    {
        "key": "sc_mae",
        "display": "scMAE",
        "metrics": BASELINE_RUN / "sc_mae_h123_to_h4" / "metrics.json",
        "loss_csv": BASELINE_RUN / "sc_mae_h123_to_h4" / "results" / "loss_history.csv",
        "loss_png": BASELINE_RUN / "sc_mae_h123_to_h4" / "figures" / "loss_history.png",
    },
    {
        "key": "scdeepcluster",
        "display": "scDeepCluster",
        "metrics": BASELINE_RUN / "scdeepcluster_h123_to_h4" / "metrics.json",
        "loss_csv": BASELINE_RUN / "scdeepcluster_h123_to_h4" / "results" / "loss_history.csv",
        "loss_png": BASELINE_RUN / "scdeepcluster_h123_to_h4" / "figures" / "loss_history.png",
    },
]

METRIC_ORDER = [
    "ACC",
    "ARI",
    "NMI",
    "BalancedACC",
    "RareACC",
    "UltraRareACC",
    "Silhouette",
]

COLORS = {
    "train": "#4c78a8",
    "test": "#f58518",
    "delta_pos": "#54a24b",
    "delta_neg": "#e45756",
    "loss": "#4c78a8",
    "loss_alt": "#e45756",
    "loss_aux": "#72b7b2",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def copy_if_exists(src: Path, dst_dir: Path) -> None:
    if not src.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / src.name)


def collect_metric_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for algo in ALGORITHMS:
        metrics_path = algo["metrics"]
        if not metrics_path.exists():
            print(f"Missing metrics: {metrics_path}")
            continue

        payload = load_json(metrics_path)
        train_metrics = payload.get("train_metrics", {})
        test_metrics = payload.get("test_metrics", {})
        source_dir = SOURCE_DIR / str(algo["key"])
        copy_if_exists(metrics_path, source_dir)
        copy_if_exists(algo["loss_csv"], source_dir)
        copy_if_exists(algo["loss_png"], source_dir)

        for metric in METRIC_ORDER:
            train_value = as_float(train_metrics.get(metric))
            test_value = as_float(test_metrics.get(metric))
            if train_value is None or test_value is None:
                continue
            rows.append(
                {
                    "algorithm": algo["display"],
                    "algorithm_key": algo["key"],
                    "metric": metric,
                    "train": train_value,
                    "test": test_value,
                    "test_minus_train": test_value - train_value,
                    "train_minus_test": train_value - test_value,
                    "abs_delta": abs(test_value - train_value),
                    "metrics_source": str(metrics_path),
                }
            )

    if not rows:
        raise RuntimeError("No metric rows found. Did the runs finish?")
    return pd.DataFrame(rows)


def plot_train_test(metrics: pd.DataFrame) -> None:
    present_metrics = [m for m in METRIC_ORDER if m in set(metrics["metric"])]
    n_cols = 3
    n_rows = math.ceil(len(present_metrics) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.2 * n_rows), squeeze=False)

    for ax, metric in zip(axes.ravel(), present_metrics):
        sub = metrics[metrics["metric"] == metric].copy()
        x = np.arange(len(sub))
        width = 0.36
        ax.bar(x - width / 2, sub["train"], width, label="Train", color=COLORS["train"])
        ax.bar(x + width / 2, sub["test"], width, label="Test", color=COLORS["test"])
        ax.set_title(metric)
        ax.set_ylim(0, min(1.08, max(float(sub[["train", "test"]].max().max()) + 0.12, 0.2)))
        ax.set_xticks(x)
        ax.set_xticklabels(sub["algorithm"], rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)

        for xpos, value in zip(x - width / 2, sub["train"]):
            ax.text(xpos, value + 0.015, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
        for xpos, value in zip(x + width / 2, sub["test"]):
            ax.text(xpos, value + 0.015, f"{value:.2f}", ha="center", va="bottom", fontsize=8)

    for ax in axes.ravel()[len(present_metrics) :]:
        ax.axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("BARON h123_to_h4 - Scores train/test par algorithme", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURE_DIR / "train_vs_test_metrics_all_algorithms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_delta_heatmap(metrics: pd.DataFrame) -> None:
    pivot = (
        metrics.pivot(index="algorithm", columns="metric", values="test_minus_train")
        .reindex(columns=[m for m in METRIC_ORDER if m in set(metrics["metric"])])
        .reindex([algo["display"] for algo in ALGORITHMS])
    )
    fig, ax = plt.subplots(figsize=(12, 4.8))
    values = pivot.to_numpy(dtype=float)
    limit = max(0.05, float(np.nanmax(np.abs(values))))
    im = ax.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Delta de score: test - train")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = values[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("test - train")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "metric_delta_test_minus_train_all_algorithms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def read_loss_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"Missing loss CSV: {path}")
        return None
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        return None
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    return df.dropna(subset=["epoch"])


def plot_one_loss(ax: plt.Axes, algo: dict[str, Any]) -> None:
    df = read_loss_table(algo["loss_csv"])
    ax.set_title(str(algo["display"]))
    ax.grid(axis="y", alpha=0.25)
    if df is None or df.empty:
        ax.text(0.5, 0.5, "loss absente", transform=ax.transAxes, ha="center", va="center")
        return

    if algo["key"] == "scraw_trial0017":
        plotted = False
        if "reconstruction_loss" in df:
            ax.plot(df["epoch"], df["reconstruction_loss"], color=COLORS["loss"], lw=2, label="reconstruction")
            plotted = True
        if "triplet_loss" in df and pd.to_numeric(df["triplet_loss"], errors="coerce").fillna(0).abs().max() > 0:
            ax.plot(df["epoch"], df["triplet_loss"], color=COLORS["loss_alt"], lw=2, label="triplet")
            plotted = True
        if not plotted and "total_loss" in df:
            ax.plot(df["epoch"], df["total_loss"], color=COLORS["loss"], lw=2, label="total")
    else:
        if "train_loss" in df:
            ax.plot(df["epoch"], df["train_loss"], color=COLORS["loss"], lw=2.2, label="train_loss")
        component_cols = [
            c
            for c in df.columns
            if c not in {"phase", "epoch", "train_loss", "val_loss"}
            and pd.to_numeric(df[c], errors="coerce").notna().any()
        ]
        for idx, col in enumerate(component_cols[:3]):
            color = [COLORS["loss_alt"], COLORS["loss_aux"], "#b279a2"][idx % 3]
            ax.plot(df["epoch"], df[col], lw=1.4, alpha=0.9, linestyle="--", color=color, label=col)

    if "phase" in df.columns:
        phase_changes = df.loc[df["phase"].ne(df["phase"].shift())]
        ymin, ymax = ax.get_ylim()
        for _, row in phase_changes.iloc[1:].iterrows():
            ax.axvline(row["epoch"], color="#555555", lw=0.8, alpha=0.35)
        ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False, fontsize=8)


def plot_loss_panel() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), squeeze=False)
    for ax, algo in zip(axes.ravel(), ALGORITHMS):
        plot_one_loss(ax, algo)
    fig.suptitle("BARON h123_to_h4 - Courbes de loss par algorithme", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURE_DIR / "loss_curves_panel_all_algorithms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_overview(metrics: pd.DataFrame) -> None:
    core_metrics = [m for m in ["ACC", "ARI", "NMI", "BalancedACC"] if m in set(metrics["metric"])]
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0], width_ratios=[1.1, 0.9])
    ax_scores = fig.add_subplot(gs[0, 0])
    ax_delta = fig.add_subplot(gs[0, 1])
    loss_axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]

    score_sub = metrics[metrics["metric"].isin(core_metrics)].copy()
    x = np.arange(len(core_metrics))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(ALGORITHMS))
    for offset, algo in zip(offsets, ALGORITHMS):
        sub = score_sub[score_sub["algorithm"] == algo["display"]].set_index("metric")
        vals = [sub.loc[m, "test"] if m in sub.index else np.nan for m in core_metrics]
        ax_scores.bar(x + offset, vals, width, label=algo["display"])
    ax_scores.set_title("Scores test")
    ax_scores.set_xticks(x)
    ax_scores.set_xticklabels(core_metrics)
    ax_scores.set_ylim(0, 1.05)
    ax_scores.grid(axis="y", alpha=0.25)
    ax_scores.legend(frameon=False, fontsize=8)

    delta_sub = metrics[metrics["metric"].isin(core_metrics)]
    pivot = delta_sub.pivot(index="algorithm", columns="metric", values="test_minus_train").reindex(
        [algo["display"] for algo in ALGORITHMS]
    )
    values = pivot.to_numpy(dtype=float)
    limit = max(0.05, float(np.nanmax(np.abs(values))))
    ax_delta.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    ax_delta.set_title("Delta test - train")
    ax_delta.set_xticks(np.arange(len(core_metrics)))
    ax_delta.set_xticklabels(core_metrics, rotation=30, ha="right")
    ax_delta.set_yticks(np.arange(len(pivot.index)))
    ax_delta.set_yticklabels(pivot.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.isnan(values[i, j]):
                ax_delta.text(j, i, f"{values[i, j]:+.2f}", ha="center", va="center", fontsize=9)

    for ax, algo in zip(loss_axes, ALGORITHMS):
        plot_one_loss(ax, algo)

    fig.suptitle("BARON h123_to_h4 - Synthese inductive", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGURE_DIR / "baron_h123_to_h4_overview_panel.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_readme(metrics: pd.DataFrame) -> None:
    lines = [
        "# BARON h123_to_h4 - comparaison inductive",
        "",
        "Ce dossier regroupe un seul split BARON (`human1,human2,human3 -> human4`) pour les algorithmes inductifs.",
        "",
        "Fichiers principaux :",
        "- `tables/train_test_metric_delta_h123_to_h4.csv` : scores train, test, `test_minus_train` et `train_minus_test`.",
        "- `figures/train_vs_test_metrics_all_algorithms.png` : barres train/test par metrique.",
        "- `figures/metric_delta_test_minus_train_all_algorithms.png` : delta `test - train`.",
        "- `figures/loss_curves_panel_all_algorithms.png` : courbes de loss par algorithme.",
        "- `source_files/` : copies des JSON/CSV/PNG sources utilises.",
        "",
        "Lecture du delta : une valeur negative signifie que le score baisse sur le test par rapport au train.",
        "",
        "Sources :",
    ]
    for source in sorted(metrics["metrics_source"].unique()):
        lines.append(f"- `{source}`")
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    metrics = collect_metric_rows()
    metrics.to_csv(TABLE_DIR / "train_test_metric_delta_h123_to_h4.csv", index=False)
    plot_train_test(metrics)
    plot_delta_heatmap(metrics)
    plot_loss_panel()
    plot_overview(metrics)
    write_readme(metrics)
    print(f"Wrote panels to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
