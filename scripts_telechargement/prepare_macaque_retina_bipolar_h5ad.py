#!/usr/bin/env python3
"""Standardize the macaque retina bipolar benchmark `.h5ad` into project conventions."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import anndata as ad
import pandas as pd
import scipy.sparse as sp

LOGGER = logging.getLogger("prepare_macaque_retina_bipolar_h5ad")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default="data/macaque_retina_scdml_source.h5ad",
        help="Input benchmark `.h5ad` path.",
    )
    p.add_argument(
        "--output",
        default="data/macaque_retina_gse118480_bipolar_raw_counts.h5ad",
        help="Output standardized `.h5ad` path.",
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

    input_path = _resolve_path(args.input)
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    LOGGER.info("Reading %s", input_path)
    adata_src = ad.read_h5ad(input_path)
    X = adata_src.X.tocsr() if sp.issparse(adata_src.X) else sp.csr_matrix(adata_src.X)

    obs = adata_src.obs.copy()
    obs.index = obs.index.astype(str)
    obs.index.name = "cell_id"
    obs["cell_type"] = obs["cluster"].astype(str)
    obs["Group"] = obs["cell_type"]
    obs["label"] = obs["cell_type"]
    obs["labels"] = obs["cell_type"]
    obs["original_batch"] = obs["batch"].astype(str)
    obs["batch"] = obs["sample"].astype(str)
    obs["study"] = "Peng2019MacaqueRetina"
    obs["organism"] = "Macaca fascicularis"
    obs["tissue"] = "retina_bipolar_cells"
    obs["assay"] = "10x_raw_counts"
    obs["raw_data_accession"] = "GSE118480"

    categorical_cols = [
        "sample",
        "macaque_id",
        "cluster",
        "region",
        "class",
        "celltype",
        "cell_type",
        "Group",
        "label",
        "labels",
        "original_batch",
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
    adata = ad.AnnData(X=X, obs=obs, var=var)
    if adata.var_names.duplicated().any():
        LOGGER.warning("Detected duplicated var names; making them unique.")
        adata.var_names_make_unique()

    adata.uns.update(adata_src.uns.copy())
    adata.uns["dataset_name"] = "macaque_retina_bipolar_cells"
    adata.uns["source"] = "scDML reproduction archive"
    adata.uns["source_input"] = str(input_path)
    adata.uns["n_samples"] = int(obs["sample"].nunique())
    adata.uns["samples"] = sorted(obs["sample"].astype(str).unique().tolist())
    adata.uns["macaque_ids"] = sorted(obs["macaque_id"].astype(str).unique().tolist())
    adata.uns["regions"] = sorted(obs["region"].astype(str).unique().tolist())
    adata.uns["cell_types"] = sorted(obs["cell_type"].astype(str).unique().tolist())
    adata.uns["conversion"] = {
        "script": "scripts/prepare_macaque_retina_bipolar_h5ad.py",
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "nnz": int(adata.X.nnz),
        "label_column": "label",
        "batch_column": "batch",
    }

    LOGGER.info("Writing %s with shape=%s", output_path, adata.shape)
    adata.write_h5ad(output_path, compression=_compression_arg(args.compression))

    summary = {
        "output": str(output_path),
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "n_cell_types": int(obs["cell_type"].nunique()),
        "regions": adata.uns["regions"],
    }
    LOGGER.info("Conversion summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
