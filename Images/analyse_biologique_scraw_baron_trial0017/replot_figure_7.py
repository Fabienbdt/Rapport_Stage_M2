#!/usr/bin/env python3
"""Regenerate Figure 7 – t-SNE qualitative panel in French.
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
CSV_PATH = ROOT / "tsne_coordinates.csv"
OUTPUT_PATH = ROOT / "tsne_qualitative_panel.png"

def scatter_categories(ax: plt.Axes, coords: np.ndarray, labels: np.ndarray, title: str, palette_name: str = "tab20", sort_key=None) -> None:
    vals = np.asarray(labels, dtype=str)
    categories = np.unique(vals)
    if sort_key is not None:
        categories = sorted(categories, key=sort_key)
    else:
        categories = sorted(categories)
        
    palette = sns.color_palette(palette_name, n_colors=max(1, len(categories)))
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}
    
    for cat in categories:
        mask = vals == cat
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=5,
            alpha=0.78,
            linewidths=0,
            c=[color_map[cat]],
            label=cat,
        )
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=7.5, markerscale=2.2, frameon=False)

def scatter_status(ax: plt.Axes, coords: np.ndarray, labels: np.ndarray, title: str, color_map: dict[str, str]) -> None:
    vals = np.asarray(labels, dtype=str)
    categories = ["Correct", "Erreur"]
    for cat in categories:
        mask = vals == cat
        if np.any(mask):
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=5,
                alpha=0.78,
                linewidths=0,
                c=[color_map[cat]],
                label=cat,
            )
    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=9, markerscale=2.2, frameon=False)

def make_figure() -> None:
    df = pd.read_csv(CSV_PATH)
    coords = df[["tsne_1", "tsne_2"]].values

    true = df["true_label"].astype(str).to_numpy()
    # Remplacer les tirets bas par des espaces pour une meilleure présentation
    true = np.array([t.replace("_", " ") for t in true])

    pred = df["predicted_label"].astype(str).to_numpy()
    # Formater les clusters en "Cluster X"
    pred = np.array([f"Cluster {p}" for p in pred])

    # Convertir les statuts d'association/recouvrement en français (Correct/Erreur)
    hungarian_status = np.where(
        df["hungarian_annotation"] == df["true_label"],
        "Correct",
        "Erreur"
    )
    marker_status = np.where(
        df["marker_overlap_annotation"] == df["true_label"],
        "Correct",
        "Erreur"
    )

    # Vert pour correct, Rouge pour erreur
    status_colors = {
        "Correct": "#16a34a",
        "Erreur": "#dc2626"
    }

    plt.style.use("default")
    plt.rcParams.update({
        "figure.dpi": 180,
        "font.size": 9.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # 1. Type cellulaire de référence
    scatter_categories(axes[0, 0], coords, true, "Type cellulaire de référence", palette_name="tab20")
    
    # 2. Cluster prédit
    # Trier numériquement par numéro de cluster
    scatter_categories(axes[0, 1], coords, pred, "Cluster prédit", palette_name="tab20", sort_key=lambda x: int(x.split()[-1]))

    # 3. Association optimale
    scatter_status(axes[1, 0], coords, hungarian_status, "Résultat de l'association optimale (Hongrois)", status_colors)

    # 4. Recouvrement des marqueurs
    scatter_status(axes[1, 1], coords, marker_status, "Résultat de l'annotation par marqueurs", status_colors)

    # Titre demandé par l'utilisateur
    fig.suptitle("visualisation tsne - scRAW", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    make_figure()
