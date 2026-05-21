import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Path to the data
data_path = "/data2/fbidet/scRAW_EXPERIMENTAL/results/scraw_loss_integration_matrix_stable_default_20260428_104305/04_tables/all_seed_metrics.csv"
output_dir = "/data2/fbidet/scRAW_EXPERIMENTAL/results/scraw_loss_integration_matrix_stable_default_20260428_104305/05_plots"

# Load data
df = pd.read_csv(data_path)

# Filter for "full" phase and exclude historical results if possible
# Or just keep the main variants
main_variants = [
    'baseline', 
    'density_only', 
    'full_kmeans', 
    'full_leiden', 
    'full_kmeans_triplet', 
    'full_leiden_triplet'
]
df_filtered = df[df['variant'].isin(main_variants)]

# Filter for main metrics
main_metrics = ['ARI', 'NMI', 'ACC', 'BalancedACC', 'RareACC', 'UltraRareACC', 'F1_Macro']
df_filtered = df_filtered[df_filtered['metric'].isin(main_metrics)]

# --- NEW: Aggregation by mean per dataset ---
print("Aggregating results: calculating mean across seeds for each dataset/condition...")
df_agg = df_filtered.groupby(['dataset', 'algorithm', 'variant', 'metric'])['value'].mean().reset_index()

# Use the aggregated data for plotting
plot_df = df_agg

# Set aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 150

# Generate one plot per metric
for metric in main_metrics:
    plt.figure(figsize=(12, 8))
    data_metric = plot_df[plot_df['metric'] == metric]
    
    # Create the boxplot
    # The distribution is now across datasets (mean of seeds)
    ax = sns.boxplot(
        data=data_metric, 
        x='algorithm', 
        y='value', 
        hue='variant',
        showfliers=False,  # Hide outliers to keep it clean, points will be shown by stripplot
        linewidth=1.5
    )
    
    # Add individual points (stripplot) to see the datasets
    # Using 'dataset' as the color might be too busy if there are many, but let's try
    sns.stripplot(
        data=data_metric,
        x='algorithm',
        y='value',
        hue='variant',
        dodge=True,
        alpha=0.4,
        size=4,
        ax=ax,
        legend=False  # Avoid duplicate legend
    )
    
    plt.title(f"Synthetic Comparison of Algorithms and Variants - {metric}", fontsize=16, pad=20)
    plt.ylabel(f"{metric} Value", fontsize=14)
    plt.xlabel("Base Algorithm", fontsize=14)
    plt.ylim(0, 1.05) if metric != 'runtime' else None
    plt.legend(title="scRAW Variant", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f"synthetic_boxplot_{metric}.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")

# --- NEW: Generate a combined faceted figure ---
print("Generating combined faceted figure...")
plt.figure(figsize=(16, 12))
# Melt or prepare the dataframe for catplot/faceting
# Actually sns.catplot is easier for this
g = sns.catplot(
    data=plot_df,
    x='algorithm',
    y='value',
    hue='variant',
    col='metric',
    kind='box',
    col_wrap=2,
    sharey=False,
    height=5,
    aspect=1.2,
    showfliers=False,
    linewidth=1.2
)

# Add stripplot to each facet
for ax_metric, ax in zip(plot_df['metric'].unique(), g.axes.flat):
    data_sub = plot_df[plot_df['metric'] == ax_metric]
    sns.stripplot(
        data=data_sub,
        x='algorithm',
        y='value',
        hue='variant',
        dodge=True,
        alpha=0.3,
        size=3,
        ax=ax,
        legend=False
    )
    ax.set_title(f"Metric: {ax_metric}", fontsize=14)
    if ax_metric in ['ARI', 'NMI', 'ACC', 'BalancedACC', 'RareACC', 'UltraRareACC', 'F1_Macro']:
        ax.set_ylim(0, 1.05)

g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Synthetic Performance Overview across Datasets and Seeds", fontsize=18)

combined_output_path = os.path.join(output_dir, "synthetic_overview_all_metrics.png")
g.savefig(combined_output_path, bbox_inches='tight')
print(f"Saved {combined_output_path}")

print("Done!")
