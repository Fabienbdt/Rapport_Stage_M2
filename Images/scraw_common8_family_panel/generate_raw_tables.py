#!/usr/bin/env python3
"""Generate raw performance tables for each metric in the report appendix.
This script replaces the global averages tables with detailed, per-dataset raw results.
"""

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
CSV_PATH = Path(
    "/Users/fabienbidet/Documents/MASTER2/STAGE/"
    "presentation_trial206_nonbaron_20260324/00_source_tables/"
    "trial206_all_results_table.csv"
)
if not CSV_PATH.exists():
    CSV_PATH = Path(
        "/data2/fbidet/scRAW_EXPERIMENTAL/results/"
        "presentation_trial206_nonbaron_20260324/00_source_tables/"
        "trial206_all_results_table.csv"
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

NO_BATCH_EFFECT_DATASETS = {
    "Paul15 bone marrow",
    "Tabula Muris liver",
}

DATASET_DISPLAY = {
    "BBAG094 Zeisel": "Zeisel",
    "BBAG094 spleen": "Spleen",
    "Baron human pancreas": "Baron Pancreas",
    "Human testis GSE112013": "Human Testis",
    "Kang PBMC": "Kang PBMC",
    "Macaque retina bipolar": "Macaque Retina",
    "Paul15 bone marrow": "Bone Marrow",
    "Tabula Muris liver": "TM Liver",
}

METRICS = [
    "ARI",
    "ACC",
    "BalancedACC",
    "RareACC",
    "UltraRareACC",
    "Batch correction",
]

METRIC_LABELS = {
    "ARI": "Rand Index Ajusté (ARI)",
    "ACC": "Précision Globale (ACC)",
    "BalancedACC": "Balanced Accuracy (BalancedACC)",
    "RareACC": "Précision sur Classes Rares (RareACC)",
    "UltraRareACC": "Précision sur Classes Ultra-Rares (UltraRareACC)",
    "Batch correction": "Correction de Batch",
}

SCRAW_SOURCE_METHOD = "scRAW"
SCRAW_METHOD = "scRAW (trial_0017)"

# Define the columns (algorithms) in the desired order
PRIMARY_COLUMNS = {
    "scRAW": [SCRAW_METHOD],
    "Rare Specific": ["scAIDE", "CellSIUS", "DeepScena", "scCAD", "GiniClust"],
    "Généralistes": ["scvi", "scMAE", "pca_leiden", "scNAME"],
    "Correction Batch": ["Harmony", "ComBat", "DESC", "Scanorama"]
}

HARMONY_COLUMNS = {
    "scRAW": [SCRAW_METHOD],
    "Rare + Harmony": ["scAIDE+Harmony", "CellSIUS+Harmony", "scCAD+Harmony", "DeepScena+Harmony", "GiniClust+Harmony"],
    "Généralistes + Harmony": ["Harmony", "scvi", "scMAE+Harmony", "scNAME+Harmony"]
}

DISPLAY_MAP = {
    SCRAW_METHOD: "scRAW",
    "pca_leiden": "PCA+Leiden",
    "scvi": "scVI",
    "scAIDE+Harmony": "scAIDE",
    "CellSIUS+Harmony": "CellSIUS",
    "scCAD+Harmony": "scCAD",
    "DeepScena+Harmony": "DeepScena",
    "GiniClust+Harmony": "GiniClust",
    "scMAE+Harmony": "scMAE",
    "scNAME+Harmony": "scNAME",
}

def get_display_name(method: str) -> str:
    return DISPLAY_MAP.get(method, method)

def format_value(val: float) -> str:
    if pd.isna(val) or np.isnan(val):
        return "--"
    return f"{val:.3f}".replace(".", ",")

def format_float_french(val: float, decimals: int = 3) -> str:
    if pd.isna(val) or np.isnan(val):
        return "--"
    return f"{val:.{decimals}f}".replace(".", ",")

def is_scraw_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1"})

def generate_table(df: pd.DataFrame, columns_dict: dict, metric: str, file_prefix: str, caption: str, label: str, font_size: str = "\\scriptsize", tabcolsep: str = "1.5pt"):
    # Flatten the columns dict to get all method keys
    all_methods = []
    for grp, methods in columns_dict.items():
        all_methods.extend(methods)
        
    # Extract data for the metric
    pivot_df = pd.DataFrame(index=COMMON8, columns=all_methods)
    for dataset in COMMON8:
        for method in all_methods:
            sub = df[(df["dataset"] == dataset) & (df["method"] == method)]
            if not sub.empty:
                pivot_df.loc[dataset, method] = sub.iloc[0][metric]
            else:
                pivot_df.loc[dataset, method] = np.nan
                
    # Cast to float
    pivot_df = pivot_df.astype(float)
    
    # Calculate ranks for each row (dataset)
    ranks_df = pd.DataFrame(index=COMMON8, columns=all_methods)
    for dataset in COMMON8:
        row_vals = pivot_df.loc[dataset]
        non_nan_vals = row_vals.dropna()
        if not non_nan_vals.empty:
            # Rank descending (higher is better, best rank = 1)
            # Ranks are calculated using 'min' method (tied values get the same rank)
            sorted_unique = sorted(non_nan_vals.unique(), reverse=True)
            for method in all_methods:
                val = row_vals[method]
                if not pd.isna(val):
                    # Rank is count of elements strictly greater than val, plus 1
                    ranks_df.loc[dataset, method] = sum(1 for x in non_nan_vals if x > val) + 1
                else:
                    ranks_df.loc[dataset, method] = np.nan

    # Calculate AVG and Model Rank AVG
    avg_row = pivot_df.mean(axis=0)
    avg_rank_row = ranks_df.mean(axis=0)
    
    # Start building LaTeX
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"  \centering")
    lines.append(f"  {font_size}")
    lines.append(f"  \\setlength{{\\tabcolsep}}{{{tabcolsep}}}")
    
    # Column specifier
    # e.g., l | c | ccccc | cccc | cccc
    col_spec = ["l"]
    for grp, methods in columns_dict.items():
        col_spec.append("|")
        col_spec.append("c" * len(methods))
    lines.append(f"  \\begin{{tabular}}{{{''.join(col_spec)}}}")
    lines.append(r"    \toprule")
    
    # Group headers (row 1 of header)
    header1 = [r"    \textbf{Jeu de données}"]
    for grp, methods in columns_dict.items():
        header1.append(f"\\multicolumn{{{len(methods)}}}{{c|}}{{\\textbf{{{grp}}}}}" if grp != list(columns_dict.keys())[-1] else f"\\multicolumn{{{len(methods)}}}{{c}}{{\\textbf{{{grp}}}}}")
    lines.append(" & ".join(header1) + r" \\")
    
    # Method headers (row 2 of header)
    header2 = ["   "]
    for grp, methods in columns_dict.items():
        for m in methods:
            disp_name = get_display_name(m)
            # Bold face for scRAW in column headers
            if "scRAW" in disp_name:
                header2.append(f"\\textbf{{{disp_name}}}")
            else:
                header2.append(disp_name)
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"    \midrule")
    
    # Data rows
    for dataset in COMMON8:
        row_vals = pivot_df.loc[dataset]
        non_nan_vals = row_vals.dropna()
        
        # Find max and second max values for bolding / underlining
        if not non_nan_vals.empty:
            max_val = non_nan_vals.max()
            # Second max is max of values strictly less than max_val
            less_than_max = non_nan_vals[non_nan_vals < max_val]
            second_max_val = less_than_max.max() if not less_than_max.empty else np.nan
        else:
            max_val = np.nan
            second_max_val = np.nan
            
        row_cells = [f"    {DATASET_DISPLAY[dataset]}"]
        for method in all_methods:
            val = row_vals[method]
            if pd.isna(val):
                if metric == "Batch correction" and dataset in NO_BATCH_EFFECT_DATASETS:
                    row_cells.append("NA")
                else:
                    row_cells.append("--")
            else:
                formatted = format_value(val)
                # Apply bold to max, underline to second max
                # Standard tolerance check for floats
                if abs(val - max_val) < 1e-9:
                    row_cells.append(f"\\textbf{{{formatted}}}")
                elif not pd.isna(second_max_val) and abs(val - second_max_val) < 1e-9:
                    row_cells.append(f"\\underline{{{formatted}}}")
                else:
                    row_cells.append(formatted)
        lines.append(" & ".join(row_cells) + r" \\")
        
    lines.append(r"    \midrule")
    
    # AVG Row
    avg_cells = [r"    \textbf{Moyenne}"]
    # Find max and second max in averages
    valid_avgs = avg_row.dropna()
    if not valid_avgs.empty:
        max_avg = valid_avgs.max()
        less_than_max_avg = valid_avgs[valid_avgs < max_avg]
        second_max_avg = less_than_max_avg.max() if not less_than_max_avg.empty else np.nan
    else:
        max_avg = np.nan
        second_max_avg = np.nan
        
    for method in all_methods:
        val = avg_row[method]
        if pd.isna(val):
            avg_cells.append("--")
        else:
            formatted = format_float_french(val)
            if abs(val - max_avg) < 1e-9:
                avg_cells.append(f"\\textbf{{{formatted}}}")
            elif not pd.isna(second_max_avg) and abs(val - second_max_avg) < 1e-9:
                avg_cells.append(f"\\underline{{{formatted}}}")
            else:
                avg_cells.append(formatted)
    lines.append(" & ".join(avg_cells) + r" \\")

    
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(r"\end{table}")
    
    # Write file
    out_file = ROOT / f"{file_prefix}_{metric.replace(' ', '_')}_table.tex"
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated {out_file.name}")

def main():
    # Load and filter CSV data
    df = pd.read_csv(CSV_PATH)
    scraw_trial_0017 = (
        df["method"].isin({SCRAW_SOURCE_METHOD, SCRAW_METHOD})
        & (df["trial_id"] == "trial_0017")
    )
    keep_non_scraw = ~is_scraw_series(df["is_scraw_method"])
    filtered_df = df[df["dataset"].isin(COMMON8) & (keep_non_scraw | scraw_trial_0017)].copy()
    filtered_df.loc[
        filtered_df["method"].isin({SCRAW_SOURCE_METHOD, SCRAW_METHOD})
        & (filtered_df["trial_id"] == "trial_0017"),
        "method",
    ] = SCRAW_METHOD
    for metric in METRICS:
        filtered_df[metric] = pd.to_numeric(filtered_df[metric], errors="coerce")
    filtered_df.loc[
        filtered_df["dataset"].isin(NO_BATCH_EFFECT_DATASETS), "Batch correction"
    ] = np.nan
    
    # Generate tables for each metric
    for metric in METRICS:
        metric_label = METRIC_LABELS[metric]
        
        # Primary methods
        batch_note = (
            " Les jeux sans effet de batch exploitable sont indiqués NA et exclus de la moyenne."
            if metric == "Batch correction"
            else ""
        )

        generate_table(
            df=filtered_df,
            columns_dict=PRIMARY_COLUMNS,
            metric=metric,
            file_prefix="common8_primary_raw",
            caption=f"Résultats bruts de {metric_label} pour les méthodes principales sur les 8 jeux de données communs. La meilleure performance par jeu de données est en gras, et la deuxième est soulignée.{batch_note}",
            label=f"tab:common8_primary_raw_{metric.replace(' ', '_')}",
            font_size="\\fontsize{5.5pt}{7.0pt}\\selectfont",
            tabcolsep="1.0pt"
        )
        
        # Harmony complement
        generate_table(
            df=filtered_df,
            columns_dict=HARMONY_COLUMNS,
            metric=metric,
            file_prefix="common8_harmony_complement_raw",
            caption=f"Résultats bruts complémentaires de {metric_label} avec Harmony pour les 8 jeux de données communs. La meilleure performance par jeu de données est en gras, et la deuxième est soulignée.{batch_note}",
            label=f"tab:common8_harmony_complement_raw_{metric.replace(' ', '_')}",
            font_size="\\scriptsize",
            tabcolsep="1.5pt"
        )

if __name__ == "__main__":
    main()
