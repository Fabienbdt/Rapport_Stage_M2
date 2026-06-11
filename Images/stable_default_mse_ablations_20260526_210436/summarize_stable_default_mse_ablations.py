#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


VARIANT_LABELS = {
    "01_plain_mse": "MSE",
    "02_weighted_mse": "Weighted MSE",
    "03_weighted_mse_triplet": "Weighted MSE + triplet",
    "04_weighted_mse_dann": "Weighted MSE + DANN",
    "05_weighted_mse_dann_triplet_full": "Weighted MSE + DANN + triplet",
    "06_weighted_mse_dann_triplet_single_update": "Weighted MSE + DANN + triplet, single update",
    "07_weighted_mse_dann_triplet_density_only": "Weighted MSE + DANN + triplet, density-only",
    "08_weighted_mse_dann_triplet_cluster_only": "Weighted MSE + DANN + triplet, cluster-size-only",
}


METRICS = [
    "ARI",
    "NMI",
    "ACC",
    "F1_Macro",
    "BalancedACC",
    "RareACC",
    "BalancedRareACC",
    "UltraRareACC",
    "Silhouette",
    "RareWeightedSilhouette",
    "scIB-E Total score",
    "Batch correction",
    "Inter cell-type conservation",
    "Intra cell-type conservation",
    "n_clusters_found",
    "runtime",
    "num_parameters",
]


PARAMS = [
    "param_hidden_layers",
    "param_z_dim",
    "param_epochs",
    "param_warmup_epochs",
    "param_reconstruction_distribution",
    "param_rare_triplet_weight",
    "param_adversarial_batch_weight",
    "param_use_batch_conditioning",
    "param_cluster_weight_power",
    "param_density_weight_power",
    "param_dynamic_weight_update_interval",
    "param_final_clustering_method_effective",
    "param_n_batches_effective",
]


def read_first_csv(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def main() -> int:
    run_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    rows: list[dict[str, str]] = []
    for variant, label in VARIANT_LABELS.items():
        analysis = read_first_csv(run_root / "runs" / variant / "results" / "analysis_results.csv")
        if analysis is None:
            rows.append(
                {
                    "variant": variant,
                    "label": label,
                    "status": "missing",
                    "run_path": str(run_root / "runs" / variant),
                }
            )
            continue
        row = {
            "variant": variant,
            "label": label,
            "status": "complete",
            "run_path": str(run_root / "runs" / variant),
        }
        for key in METRICS + PARAMS:
            row[key] = analysis.get(key, "")
        
        # Calculate BalancedRareACC if missing but ClassWise is present
        if not row.get("BalancedRareACC") and "ClassWise" in analysis and analysis["ClassWise"]:
            try:
                import ast
                classwise = ast.literal_eval(analysis["ClassWise"])
                total_cells = sum(item["Support"] for item in classwise.values())
                if total_cells > 0:
                    rare_classes = {name: item for name, item in classwise.items() if (item["Support"] / total_cells) < 0.05}
                    if rare_classes:
                        recalls = [item["Recall"] for item in rare_classes.values()]
                        row["BalancedRareACC"] = sum(recalls) / len(recalls)
            except Exception:
                pass
        rows.append(row)

    summaries = run_root / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    out_csv = summaries / "stable_default_mse_ablation_summary.csv"
    fieldnames = ["variant", "label", "status", "run_path"] + METRICS + PARAMS
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "preset": "stable_default",
        "dataset": "/data2/fbidet/scRAW_EXPERIMENTAL/data/baron_human_pancreas.h5ad",
        "seed": 42,
        "variants": VARIANT_LABELS,
        "summary_csv": str(out_csv),
        "complete_count": sum(1 for r in rows if r["status"] == "complete"),
        "total_count": len(rows),
    }
    (summaries / "stable_default_mse_ablation_summary.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
