import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# Path to the data
csv_path = "Images/scraw_loss_transfer_synthesis/loss_transfer_by_dataset_algorithm_variant.csv"
df = pd.read_csv(csv_path)

# Filter for metrics of interest
metrics_of_interest = ["ARI", "RareACC", "UltraRareACC"]
df_filtered = df[df["metric"].isin(metrics_of_interest)]

# Group by dataset, algorithm, variant and compute the mean of 'mean' column across the three metrics
df_agg = df_filtered.groupby(["dataset", "algorithm", "variant"])["mean"].mean().reset_index()

# Define order and labels for variants
variant_order = [
    "baseline",
    "density_only",
    "full_kmeans",
    "full_leiden",
    "density_only_triplet_kmeans",
    "full_kmeans_triplet",
    "full_leiden_triplet"
]

variant_labels = {
    "baseline": "baseline",
    "density_only": "density",
    "full_kmeans": "weighted+kmeans",
    "full_leiden": "weighted+leiden",
    "density_only_triplet_kmeans": "density+triplet",
    "full_kmeans_triplet": "weighted+kmeans+triplet",
    "full_leiden_triplet": "weighted+leiden+triplet"
}

# Define colors
palette = {
    "baseline": "#6b7280",
    "full_leiden": "#2563eb",
    "full_kmeans": "#7c3aed",
    "density_only": "#0f766e",
    "full_leiden_triplet": "#dc2626",
    "full_kmeans_triplet": "#ea580c",
    "density_only_triplet_kmeans": "#0891b2"
}

# Apply categorical ordering
df_agg["variant"] = pd.Categorical(df_agg["variant"], categories=variant_order, ordered=True)
df_agg = df_agg.sort_values("variant")

# Create figure
plt.figure(figsize=(10, 6))

# Plot boxplot
ax = sns.boxplot(
    data=df_agg,
    x="algorithm",
    y="mean",
    hue="variant",
    hue_order=variant_order,
    palette=palette,
    showfliers=False,
    width=0.7,
    linewidth=1.2
)

# Plot individual dataset points
sns.stripplot(
    data=df_agg,
    x="algorithm",
    y="mean",
    hue="variant",
    hue_order=variant_order,
    dodge=True,
    jitter=0.15,
    alpha=0.5,
    size=4,
    color="black",
    ax=ax,
    legend=False
)

# Set labels and title
plt.title("Distribution des scores de transfert de loss par algorithme et variante", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Algorithme de base", fontsize=11)
plt.ylabel("Score moyen (ARI, RareACC, UltraRareACC)", fontsize=11)
plt.ylim(0, 1.05)

# Adjust legend labels
handles, labels = ax.get_legend_handles_labels()
new_labels = [variant_labels.get(l, l) for l in labels[:len(variant_order)]]
plt.legend(
    handles[:len(variant_order)],
    new_labels,
    title="Variante scRAW",
    fontsize=8,
    title_fontsize=9,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0
)

plt.tight_layout()

# Save
output_pdf = "Images/scraw_loss_transfer_synthesis/loss_transfer_all_configurations_boxplot.pdf"
output_png = "Images/scraw_loss_transfer_synthesis/loss_transfer_all_configurations_boxplot.png"

plt.savefig(output_pdf, bbox_inches="tight")
plt.savefig(output_png, bbox_inches="tight", dpi=200)

print(f"Saved: {output_pdf}")
print(f"Saved: {output_png}")
