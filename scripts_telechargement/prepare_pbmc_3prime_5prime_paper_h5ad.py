#!/usr/bin/env python3
"""Prepare the Tran et al. 2020 PBMC 3' vs 5' benchmark dataset as one h5ad."""

from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--counts-3p",
        default="/tmp/pbmc8k_filtered_gene_bc_matrices.tar.gz",
        help="Path to 10x PBMC 3' filtered_gene_bc_matrices tar.gz",
    )
    p.add_argument(
        "--counts-5p",
        default="/tmp/vdj_v1_hs_pbmc_5gex_filtered_gene_bc_matrices.tar.gz",
        help="Path to 10x PBMC 5' filtered_gene_bc_matrices tar.gz",
    )
    p.add_argument(
        "--annotations",
        default="/data2/fbidet/scRAW_EXPERIMENTAL/data/pbmc_3prime_5prime_bbknn_annotated.h5ad",
        help="Path to the BBKNN annotation h5ad used by the paper",
    )
    p.add_argument(
        "--output",
        default="/data2/fbidet/scRAW_EXPERIMENTAL/data/pbmc_3prime_5prime_paper_raw_counts.h5ad",
        help="Output h5ad path",
    )
    p.add_argument(
        "--summary-json",
        default="",
        help="Optional summary JSON path (default: alongside output with .summary.json suffix)",
    )
    return p.parse_args()


def _extract_tar(archive_path: Path, workdir: Path) -> Path:
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(workdir)
    base = workdir / "filtered_gene_bc_matrices" / "GRCh38"
    if not base.exists():
        raise FileNotFoundError(f"Expected extracted 10x directory missing: {base}")
    return base


def _read_10x(base_dir: Path, prefix: str, batch_label: str) -> ad.AnnData:
    adata = sc.read_10x_mtx(
        str(base_dir),
        var_names="gene_ids",
        make_unique=False,
        cache=False,
    )
    gene_symbols = None
    genes_tsv = base_dir / "genes.tsv"
    if genes_tsv.exists():
        genes_df = pd.read_csv(genes_tsv, sep="\t", header=None)
        if genes_df.shape[1] >= 2:
            gene_symbols = genes_df.iloc[:, 1].astype(str).to_numpy()

    adata.var["gene_id"] = adata.var_names.astype(str)
    if gene_symbols is not None and len(gene_symbols) == adata.n_vars:
        adata.var["gene_symbol"] = gene_symbols
    else:
        adata.var["gene_symbol"] = adata.var_names.astype(str)

    normalized_barcodes = []
    for barcode in adata.obs_names.astype(str):
        core = barcode.rsplit("-", 1)[0] if "-" in barcode else barcode
        normalized_barcodes.append(f"{prefix}-{core}-{batch_label}")
    adata.obs_names = normalized_barcodes
    adata.obs["batch"] = str(batch_label)
    adata.obs["protocol"] = "3p" if str(batch_label) == "0" else "5p"
    adata.obs["assay"] = (
        "10x_chromium_3prime_v2" if str(batch_label) == "0" else "10x_chromium_5prime"
    )
    return adata


def _annotation_subset(ann: ad.AnnData, batch_label: str) -> ad.AnnData:
    mask = ann.obs["batch"].astype(str) == str(batch_label)
    return ann[mask].copy()


def _prepare_obs_annotations(ann: ad.AnnData) -> pd.DataFrame:
    obs = ann.obs.copy()
    obs["cell_type"] = obs["Cell type"].astype(str)
    obs["batch_code"] = obs["batch"].astype(str)
    obs["batch"] = obs["batch"].astype(str).map({"0": "3p", "1": "5p"}).fillna(obs["batch"].astype(str))
    obs["protocol"] = obs["batch"]
    obs["source_study"] = "Tran2020_PBMC_3p_5p"
    obs["dataset_name"] = "pbmc_3prime_5prime_paper"
    return obs


def _intersect_counts_with_annotations(
    counts: ad.AnnData,
    ann: ad.AnnData,
) -> Tuple[ad.AnnData, Dict[str, int]]:
    ann_names = pd.Index(ann.obs_names.astype(str))
    count_names = pd.Index(counts.obs_names.astype(str))
    matched = ann_names.intersection(count_names)
    if matched.empty:
        raise ValueError("No shared barcodes between raw counts and annotation h5ad.")
    counts = counts[matched].copy()
    ann = ann[matched].copy()
    counts = counts[ann.obs_names].copy()
    return counts, {
        "n_annotation_cells": int(ann.n_obs),
        "n_matched_cells": int(counts.n_obs),
    }


def _genes_detected_mask(X: Any) -> np.ndarray:
    if sparse.issparse(X):
        return np.asarray((X > 0).sum(axis=0)).ravel() > 0
    return (np.asarray(X) > 0).sum(axis=0) > 0


def main() -> int:
    args = parse_args()
    counts_3p_path = Path(args.counts_3p).resolve()
    counts_5p_path = Path(args.counts_5p).resolve()
    ann_path = Path(args.annotations).resolve()
    output_path = Path(args.output).resolve()
    summary_path = (
        Path(args.summary_json).resolve()
        if args.summary_json
        else output_path.with_suffix(".summary.json")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    ann = ad.read_h5ad(ann_path)
    ann_3p = _annotation_subset(ann, "0")
    ann_5p = _annotation_subset(ann, "1")

    with tempfile.TemporaryDirectory(prefix="pbmc_paper_prepare_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        dir_3p = _extract_tar(counts_3p_path, tmpdir_path / "pbmc3p")
        dir_5p = _extract_tar(counts_5p_path, tmpdir_path / "pbmc5p")

        counts_3p = _read_10x(dir_3p, prefix="data_3p", batch_label="0")
        counts_5p = _read_10x(dir_5p, prefix="data_5p", batch_label="1")

    common_genes = pd.Index(counts_3p.var_names.astype(str)).intersection(
        pd.Index(counts_5p.var_names.astype(str))
    )
    if common_genes.empty:
        raise ValueError("No shared genes between PBMC 3' and 5' raw matrices.")

    counts_3p = counts_3p[:, common_genes].copy()
    counts_5p = counts_5p[:, common_genes].copy()

    counts_3p, stats_3p = _intersect_counts_with_annotations(counts_3p, ann_3p)
    counts_5p, stats_5p = _intersect_counts_with_annotations(counts_5p, ann_5p)

    common_detected = _genes_detected_mask(counts_3p.X) & _genes_detected_mask(counts_5p.X)
    counts_3p = counts_3p[:, common_detected].copy()
    counts_5p = counts_5p[:, common_detected].copy()

    merged = ad.concat(
        [counts_3p, counts_5p],
        axis=0,
        join="inner",
        merge="same",
        label=None,
        index_unique=None,
    )
    merged.layers["counts"] = merged.X.copy()
    merged.raw = merged.copy()

    ann_obs = _prepare_obs_annotations(ann[merged.obs_names].copy())
    merged.obs = ann_obs.loc[merged.obs_names].copy()
    merged.var["gene_id"] = merged.var_names.astype(str)
    if "gene_symbol" not in merged.var.columns:
        merged.var["gene_symbol"] = merged.var_names.astype(str)
    merged.var_names_make_unique()

    merged.uns["dataset_name"] = "pbmc_3prime_5prime_paper"
    merged.uns["source_paper"] = "Tran et al. 2020 (PMID 31948481)"
    merged.uns["preparation_script"] = str(Path(__file__).resolve())

    merged.write_h5ad(output_path)

    summary = {
        "output_h5ad": str(output_path),
        "annotations_h5ad": str(ann_path),
        "counts_3p_archive": str(counts_3p_path),
        "counts_5p_archive": str(counts_5p_path),
        "n_cells_total": int(merged.n_obs),
        "n_genes_total": int(merged.n_vars),
        "batch_counts": merged.obs["batch"].astype(str).value_counts().to_dict(),
        "cell_type_counts": merged.obs["cell_type"].astype(str).value_counts().to_dict(),
        "match_stats": {
            "3p": stats_3p,
            "5p": stats_5p,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
