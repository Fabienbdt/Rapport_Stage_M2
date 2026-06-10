#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Set plotting style
plt.style.use("default")
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 220,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# Path to the data
ROOT = Path(__file__).resolve().parent
csv_path = ROOT / "loss_transfer_by_dataset_algorithm_variant.csv"
df = pd.read_csv(csv_path)

# Algorithms and their corresponding best variants
algos = ["scMAE", "scDeepCluster", "DESC"]
best_variants = {
    "scMAE": "full_leiden",
    "scDeepCluster": "full_leiden",
    "DESC": "full_kmeans_triplet"
}

metrics = ["ARI", "RareACC", "UltraRareACC", "BalancedRareACC"]
metric_titles = {
    "ARI": "ARI",
    "RareACC": "Rare ACC",
    "UltraRareACC": "Ultra Rare ACC",
    "BalancedRareACC": "Balanced Rare ACC"
}

# Positions for the boxplots:
# For each of the 3 algorithms, we plot baseline and best variant side by side.
positions = [0.8, 1.2, 1.8, 2.2, 2.8, 3.2]
colors = ["#cbd5e1", "#5f9ea0", "#cbd5e1", "#5f9ea0", "#cbd5e1", "#5f9ea0"]

# Create figure (4 panels in a row)
fig, axes = plt.subplots(1, 4, figsize=(18.0, 5.0), sharey=True)

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    
    # Prepare data for the 6 boxes
    data_for_metric = []
    for algo in algos:
        # Baseline data
        base_data = df[(df["algorithm"] == algo) & (df["variant"] == "baseline") & (df["metric"] == metric)]["mean"].dropna()
        # Best variant data
        best_var = best_variants[algo]
        best_data = df[(df["algorithm"] == algo) & (df["variant"] == best_var) & (df["metric"] == metric)]["mean"].dropna()
        
        data_for_metric.append(base_data)
        data_for_metric.append(best_data)
        
    # Plot boxplot
    box = ax.boxplot(
        data_for_metric,
        positions=positions,
        widths=0.3,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "#111827",
            "markersize": 5,
            "markeredgewidth": 1.0,
        },
        medianprops={"color": "#111827", "linewidth": 1.5},
        whiskerprops={"color": "#374151", "linewidth": 1.1},
        capprops={"color": "#374151", "linewidth": 1.1},
        showfliers=False
    )
    
    # Color box patches
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("#6b7280")
        patch.set_linewidth(1.1)
        
    # Plot individual dataset points
    for pos, values in zip(positions, data_for_metric):
        values_np = values.to_numpy(dtype=float)
        # Jitter the points slightly to avoid overlapping
        offsets = np.linspace(-0.04, 0.04, len(values_np)) if len(values_np) > 1 else np.array([0.0])
        ax.scatter(
            np.full_like(values_np, pos) + offsets,
            values_np,
            s=28,
            color="#111827",
            alpha=0.9,
            zorder=3
        )
        
    # Style subplot
    ax.set_title(metric_titles[metric], fontsize=12, pad=10)
    ax.set_xticks([1.0, 2.0, 3.0])
    ax.set_xticklabels(algos, fontsize=10)
    ax.set_xlim(0.4, 3.6)
    ax.set_ylim(-0.05, 1.05)
    
    # Grid and spines
    ax.grid(axis="y", color="#d1d5db", linewidth=0.85, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    
    # Only show y-axis label on the first subplot
    if idx == 0:
        ax.set_ylabel("Score moyen", fontsize=11)
        
# Add global title and subtitle
fig.suptitle("Loss pondérée scRAW intégrée à d'autres algorithmes", fontsize=13, fontweight="bold", y=0.98)
fig.text(
    0.5, 0.91, 
    "Moyennes par dataset ; meilleure variante sélectionnée par la moyenne ARI/RareACC/UltraRareACC/BalancedRareACC.", 
    ha="center", fontsize=9.5, color="#4b5563"
)

# Custom legend
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor="#cbd5e1", edgecolor="#6b7280", alpha=0.55, label="Baseline"),
    plt.Rectangle((0, 0), 1, 1, facecolor="#5f9ea0", edgecolor="#6b7280", alpha=0.55, label="Meilleure variante pondérée"),
    plt.Line2D([0], [0], marker="o", linestyle="", color="#111827", markersize=5, label="moyenne par dataset"),
    plt.Line2D([0], [0], marker="D", linestyle="", markerfacecolor="white", markeredgecolor="#111827", markersize=5, label="moyenne de l'algorithme")
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=4,
    frameon=False,
    fontsize=9.5
)

plt.tight_layout(rect=[0, 0.05, 1, 0.88])

# Save outputs
output_pdf = ROOT / "loss_transfer_baseline_vs_best_weighted.pdf"
output_png = ROOT / "loss_transfer_baseline_vs_best_weighted.png"

plt.savefig(output_pdf, bbox_inches="tight")
plt.savefig(output_png, bbox_inches="tight", dpi=220)

print(f"Successfully generated: {output_pdf}")
print(f"Successfully generated: {output_png}")
