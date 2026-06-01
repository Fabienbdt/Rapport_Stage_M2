#!/usr/bin/env python3
"""Convert the GSE146974 monocyte raw archive into an AnnData `.h5ad` file."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread

LOGGER = logging.getLogger("prepare_monocyte_gse146974_h5ad")

SERIES_ACCESSION = "GSE146974"
PREFIX_MEMBERS = {
    "MH001": {
        "matrix": "GSM4411625_matrix_MH001.mtx.gz",
        "meta": "GSM4411625_barcodes_MH001.tsv.gz",
        "genes": "GSM4411625_genes_MH001.tsv.gz",
    },
    "RP002": {
        "matrix": "GSM4411626_matrix_RP002.mtx.gz",
        "meta": "GSM4411626_barcodes_RP002.tsv.gz",
        "genes": "GSM4411626_genes_RP002.tsv.gz",
    },
    "RP009": {
        "matrix": "GSM4411627_matrix_RP009.mtx.gz",
        "meta": "GSM4411627_barcodes_RP009.tsv.gz",
        "genes": "GSM4411627_genes_RP009.tsv.gz",
    },
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw-tar",
        default="data/GSE146974_RAW.tar",
        help="Path to the GEO raw tar archive.",
    )
    p.add_argument(
        "--output",
        default="data/monocyte_gse146974_raw_counts.h5ad",
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


def _load_batch_data(tf: tarfile.TarFile, prefix: str) -> tuple[sp.csr_matrix, pd.DataFrame, pd.DataFrame]:
    members = PREFIX_MEMBERS[prefix]
    matrix = mmread(io.BytesIO(gzip.decompress(tf.extractfile(tf.getmember(members["matrix"])).read()))).tocsr()
    meta = pd.read_csv(io.BytesIO(gzip.decompress(tf.extractfile(tf.getmember(members["meta"])).read())), sep="\t")
    genes = pd.read_csv(io.BytesIO(gzip.decompress(tf.extractfile(tf.getmember(members["genes"])).read())), sep="\t")

    meta["cellname"] = meta["cellname"].astype(str)
    if matrix.shape[1] != len(meta):
        raise ValueError(
            f"Column mismatch for {prefix}: matrix has {matrix.shape[1]} cells, metadata has {len(meta)} rows."
        )
    if matrix.shape[0] != len(genes):
        raise ValueError(
            f"Row mismatch for {prefix}: matrix has {matrix.shape[0]} genes, genes table has {len(genes)} rows."
        )
    return matrix, meta, genes


def _build_obs(metadata: pd.DataFrame) -> pd.DataFrame:
    obs = metadata.copy()
    obs.index = pd.Index(obs["cellname"].astype(str), name="cell_id")
    obs["batch"] = obs["dataset_batch"].astype(str)
    obs["sample"] = obs["dataset_batch"].astype(str)
    obs["collection_day"] = obs["batch_label"].astype(str)
    obs["study"] = "MonocyteHealthy2020"
    obs["organism"] = "Homo sapiens"
    obs["tissue"] = "PBMC_monocytes"
    obs["assay"] = "10x_raw_counts"
    obs["raw_data_accession"] = SERIES_ACCESSION
    obs["n_genes"] = pd.to_numeric(obs["n_genes"], errors="coerce").astype("Int64")
    obs["n_counts"] = pd.to_numeric(obs["n_counts"], errors="coerce").astype(np.float32)
    obs["percent_mito"] = pd.to_numeric(obs["percent_mito"], errors="coerce").astype(np.float32)

    categorical_cols = [
        "batch_label",
        "dataset_batch",
        "dataset_label",
        "status_label",
        "batch",
        "sample",
        "collection_day",
        "study",
        "organism",
        "tissue",
        "assay",
        "raw_data_accession",
    ]
    for col in categorical_cols:
        obs[col] = pd.Categorical(obs[col].astype(str))
    return obs


def _build_var(genes: pd.DataFrame) -> pd.DataFrame:
    var = genes.copy()
    var["genename"] = var["genename"].astype(str)
    var.index = pd.Index(var["genename"], name="gene_symbol")
    return var


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    raw_tar_path = _resolve_path(args.raw_tar)
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_tar_path.exists():
        raise FileNotFoundError(f"Input archive not found: {raw_tar_path}")

    matrices: list[sp.spmatrix] = []
    metadata_parts: list[pd.DataFrame] = []
    gene_reference: pd.DataFrame | None = None

    with tarfile.open(raw_tar_path, "r") as tf:
        for prefix in ["MH001", "RP002", "RP009"]:
            LOGGER.info("Loading batch %s", prefix)
            matrix, meta, genes = _load_batch_data(tf, prefix)
            matrices.append(matrix)
            metadata_parts.append(meta)

            if gene_reference is None:
                gene_reference = genes.copy()
            elif not gene_reference["genename"].astype(str).equals(genes["genename"].astype(str)):
                raise ValueError(f"Gene ordering mismatch detected for batch {prefix}.")

    metadata = pd.concat(metadata_parts, ignore_index=True)
    counts_gene_by_cell = sp.hstack(matrices, format="csr")
    counts_cell_by_gene = counts_gene_by_cell.transpose().tocsr().astype(np.float32)

    obs = _build_obs(metadata)
    var = _build_var(gene_reference)
    adata = ad.AnnData(X=counts_cell_by_gene, obs=obs, var=var)

    duplicate_gene_count = int(adata.var_names.duplicated().sum())
    if duplicate_gene_count:
        LOGGER.warning("Detected %d duplicated gene symbols; making var names unique.", duplicate_gene_count)
        adata.var_names_make_unique()

    adata.uns["dataset_name"] = "healthy_human_monocytes"
    adata.uns["source"] = "GEO raw tar archive"
    adata.uns["series_accession"] = SERIES_ACCESSION
    adata.uns["batches"] = sorted(obs["batch"].astype(str).unique().tolist())
    adata.uns["collection_days"] = sorted(obs["collection_day"].astype(str).unique().tolist())
    adata.uns["input_file"] = str(raw_tar_path)
    adata.uns["conversion"] = {
        "script": "scripts/prepare_monocyte_gse146974_h5ad.py",
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "nnz": int(adata.X.nnz),
        "duplicate_gene_symbols": duplicate_gene_count,
        "batch_column": "batch",
        "ground_truth_labels_present": False,
    }

    LOGGER.info("Writing %s with shape=%s", output_path, adata.shape)
    adata.write_h5ad(output_path, compression=_compression_arg(args.compression))

    summary = {
        "output": str(output_path),
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "batches": adata.uns["batches"],
    }
    LOGGER.info("Conversion summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
