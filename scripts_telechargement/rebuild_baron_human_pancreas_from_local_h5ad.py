#!/usr/bin/env python3
"""Rebuild data/baron_human_pancreas.h5ad from local GSE84133 human H5AD files.

This script uses exactly one file per donor (human1..human4) from:
  data/GSE84133_RAW

It avoids accidental duplication when both:
  - GSM2230760_human4_umifm_counts.h5ad
  - GSM2230760_human4_umifm_counts.csv.h5ad
exist (same content in many setups).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR_DEFAULT = ROOT_DIR / "data" / "GSE84133_RAW"
OUTPUT_DEFAULT = ROOT_DIR / "data" / "baron_human_pancreas.h5ad"

DONOR_FILES = {
    "human1": "GSM2230757_human1_umifm_counts.csv.h5ad",
    "human2": "GSM2230758_human2_umifm_counts.csv.h5ad",
    "human3": "GSM2230759_human3_umifm_counts.csv.h5ad",
    "human4": "GSM2230760_human4_umifm_counts.csv.h5ad",
}


def _load_donor(file_path: Path, batch_name: str) -> ad.AnnData:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing donor file: {file_path}")

    adata = ad.read_h5ad(file_path).copy()
    if "assigned_cluster" not in adata.obs.columns:
        raise ValueError(f"'assigned_cluster' missing in {file_path}")

    if "barcode" not in adata.obs.columns:
        adata.obs["barcode"] = adata.obs_names.astype(str)

    adata.obs["batch"] = batch_name
    # Keep explicit aliases used across the codebase/UI.
    adata.obs["Group"] = adata.obs["assigned_cluster"].astype(str)
    adata.obs["label"] = adata.obs["assigned_cluster"].astype(str)
    adata.obs["cell_type"] = adata.obs["assigned_cluster"].astype(str)
    adata.obs["labels"] = adata.obs["assigned_cluster"].astype(str)
    adata.obs = adata.obs[
        [
            "barcode",
            "assigned_cluster",
            "Group",
            "label",
            "cell_type",
            "labels",
            "batch",
        ]
    ]
    return adata


def rebuild(raw_dir: Path, output: Path, backup: bool = True) -> None:
    adatas = []
    for donor in ("human1", "human2", "human3", "human4"):
        donor_path = raw_dir / DONOR_FILES[donor]
        adatas.append(_load_donor(donor_path, donor))

    merged = ad.concat(adatas, join="inner", merge="same", index_unique="-")
    merged.obs_names_make_unique()
    merged.var_names_make_unique()
    merged.uns["dataset_name"] = "Baron Human Pancreas"
    merged.uns["source"] = "GSE84133_RAW (local human donor H5AD files)"
    merged.uns["donors"] = ["human1", "human2", "human3", "human4"]
    merged.uns["n_donors"] = 4

    if backup and output.exists():
        backup_path = output.with_name(output.stem + ".backup_before_rebuild.h5ad")
        if not backup_path.exists():
            output.replace(backup_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    merged.write_h5ad(output, compression="gzip")

    counts = merged.obs["batch"].value_counts().to_dict()
    print(f"[OK] wrote: {output}")
    print(f"[OK] shape: {merged.n_obs} cells x {merged.n_vars} genes")
    print(f"[OK] batch counts: {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild baron_human_pancreas.h5ad from local human donor H5AD files."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR_DEFAULT,
        help=f"Directory with donor H5AD files (default: {RAW_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DEFAULT,
        help=f"Output H5AD path (default: {OUTPUT_DEFAULT})",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create backup of existing output file.",
    )
    args = parser.parse_args()
    rebuild(args.raw_dir, args.output, backup=not args.no_backup)


if __name__ == "__main__":
    main()
