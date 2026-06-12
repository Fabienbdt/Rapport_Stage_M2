#!/usr/bin/env python3
"""Compare annotations (Hungarian vs Marker-overlap vs Predicted vs Ground Truth) on t-SNE.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
TSNE_CSV_PATH = ROOT / "tsne_coordinates.csv"
OUTPUT_PATH = ROOT / "tsne_annotations_comparison.png"

def scatter_categories_with_colormap(
    ax: plt.Axes, 
    coords: np.ndarray, 
    labels: np.ndarray, 
    title: str, 
    color_map: dict[str, tuple], 
    max_legend: int = 18
) -> None:
    vals = np.asarray(labels, dtype=str)
    categories = sorted(np.unique(vals))
    
    for cat in categories:
        mask = vals == cat
        color = color_map.get(cat, (0.5, 0.5, 0.5))
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=5,
            alpha=0.78,
            linewidths=0,
            c=[color],
            label=cat,
        )
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    if len(categories) <= max_legend:
        ax.legend(loc="best", fontsize=7.5, markerscale=2.2, frameon=False)

def main() -> None:
    df = pd.read_csv(TSNE_CSV_PATH)
    coords = df[["tsne_1", "tsne_2"]].values

    # Clean labels (replace underscores with spaces for aesthetic reasons)
    true = np.array([str(t).replace("_", " ") for t in df["true_label"]])
    pred = np.array([f"Cluster {p}" for p in df["predicted_label"]])
    hungarian = np.array([str(h).replace("_", " ") for h in df["hungarian_annotation"]])
    marker = np.array([str(m).replace("_", " ") for m in df["marker_overlap_annotation"]])
    batches = np.array([str(b) for b in df["batch"]])

    # Build consistent cell type categories colormap (union of true, hungarian, marker labels)
    all_cell_types = sorted(list(set(true) | set(hungarian) | set(marker)))
    palette = sns.color_palette("tab20", n_colors=max(1, len(all_cell_types)))
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(all_cell_types)}

    # Build cluster colormap
    all_clusters = sorted(np.unique(pred), key=lambda x: int(x.split()[-1]))
    cluster_palette = sns.color_palette("tab20", n_colors=max(1, len(all_clusters)))
    cluster_color_map = {c: cluster_palette[i % len(cluster_palette)] for i, c in enumerate(all_clusters)}

    # Build batch colormap
    all_batches = sorted(np.unique(batches))
    batch_palette = sns.color_palette("Set2", n_colors=max(1, len(all_batches)))
    batch_color_map = {b: batch_palette[i % len(batch_palette)] for i, b in enumerate(all_batches)}

    plt.style.use("default")
    plt.rcParams.update({
        "figure.dpi": 180,
        "font.size": 9.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(3, 2, figsize=(14, 18))

    # Panel 1: Ground Truth
    scatter_categories_with_colormap(
        axes[0, 0], coords, true, "A. Vérité terrain (Ground Truth)", color_map
    )

    # Panel 2: Predicted Clusters
    scatter_categories_with_colormap(
        axes[0, 1], coords, pred, "B. Clusters prédits", cluster_color_map, max_legend=18
    )

    # Panel 3: Hungarian Annotation
    scatter_categories_with_colormap(
        axes[1, 0], coords, hungarian, "C. Annotation par association optimale (Hongrois)", color_map
    )

    # Panel 4: Marker Annotation
    scatter_categories_with_colormap(
        axes[1, 1], coords, marker, "D. Annotation par recouvrement de marqueurs", color_map
    )

    # Panel 5: Lots (Batches)
    scatter_categories_with_colormap(
        axes[2, 0], coords, batches, "E. Lots du jeu de données (Batches)", batch_color_map
    )

    # Hide the 6th empty subplot
    fig.delaxes(axes[2, 1])

    fig.suptitle("Comparaison des annotations de clustering - scRAW", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
