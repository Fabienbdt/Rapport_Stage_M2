#!/usr/bin/env python3
"""Fetch the Paul15 reference dataset via Scanpy and save it as `.h5ad`."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

LOGGER = logging.getLogger("prepare_paul15_h5ad")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        default="data/paul15_bone_marrow_raw_counts.h5ad",
        help="Output `.h5ad` path.",
    )
    p.add_argument(
        "--compression",
        default="gzip",
        choices=["gzip", "lzf", "none"],
        help="Compression used when writing the `.h5ad` file.",
    )
    p.add_argument("--verbose", action="store_true", help="Enable info logs.")
    return p


def _compression_arg(choice: str) -> str | None:
    if choice == "none":
        return None
    return choice


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Fetching Paul15 through scanpy.datasets.paul15()")
    adata_src = sc.datasets.paul15()

    X = adata_src.X
    if not sp.issparse(X):
        X = sp.csr_matrix(np.asarray(X, dtype=np.float32))
    else:
        X = X.tocsr().astype(np.float32)

    obs = adata_src.obs.copy()
    obs.index = obs.index.astype(str)
    obs.index.name = "cell_id"
    obs["cell_type"] = obs["paul15_clusters"].astype(str)
    obs["Group"] = obs["cell_type"]
    obs["label"] = obs["cell_type"]
    obs["labels"] = obs["cell_type"]
    obs["batch"] = "single_batch"
    obs["study"] = "Paul2015"
    obs["organism"] = "Mus musculus"
    obs["tissue"] = "bone_marrow"
    obs["assay"] = "non_log_raw"
    obs["raw_data_accession"] = "scanpy_paul15"

    categorical_cols = [
        "paul15_clusters",
        "cell_type",
        "Group",
        "label",
        "labels",
        "batch",
        "study",
        "organism",
        "tissue",
        "assay",
        "raw_data_accession",
    ]
    for col in categorical_cols:
        obs[col] = pd.Categorical(obs[col].astype(str))

    var = adata_src.var.copy()
    if "gene_symbol" not in var.columns:
        var["gene_symbol"] = var.index.astype(str)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    duplicate_gene_count = int(adata.var_names.duplicated().sum())
    if duplicate_gene_count:
        LOGGER.warning("Detected %d duplicated gene symbols; making var names unique.", duplicate_gene_count)
        adata.var_names_make_unique()

    adata.uns.update(adata_src.uns.copy())
    adata.uns["dataset_name"] = "paul15_bone_marrow"
    adata.uns["source"] = "scanpy.datasets.paul15()"
    adata.uns["cell_types"] = sorted(obs["cell_type"].astype(str).unique().tolist())
    adata.uns["conversion"] = {
        "script": "scripts/prepare_paul15_h5ad.py",
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "nnz": int(adata.X.nnz),
        "duplicate_gene_symbols": duplicate_gene_count,
        "label_column": "label",
        "batch_column": "batch",
    }

    LOGGER.info("Writing %s with shape=%s", output_path, adata.shape)
    adata.write_h5ad(output_path, compression=_compression_arg(args.compression))

    summary = {
        "output": str(output_path),
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "cell_types": adata.uns["cell_types"],
    }
    LOGGER.info("Conversion summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
