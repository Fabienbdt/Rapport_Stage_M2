#!/usr/bin/env python3
"""Convert the Kang et al. batch2 PBMC dataset into an AnnData `.h5ad` file."""

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

LOGGER = logging.getLogger("prepare_kang_pbmc_gse96583_h5ad")

SERIES_ACCESSION = "GSE96583"
MATRIX_MEMBERS = [
    ("GSM2560248_2.1.mtx.gz", "GSM2560248_barcodes.tsv.gz"),
    ("GSM2560249_2.2.mtx.gz", "GSM2560249_barcodes.tsv.gz"),
]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw-tar",
        default="data/GSE96583_RAW.tar",
        help="Path to the GEO raw tar archive.",
    )
    p.add_argument(
        "--metadata",
        default="data/GSE96583_batch2.total.tsne.df.tsv.gz",
        help="Path to the GEO batch2 metadata table.",
    )
    p.add_argument(
        "--genes",
        default="data/GSE96583_batch2.genes.tsv.gz",
        help="Path to the GEO batch2 gene annotation table.",
    )
    p.add_argument(
        "--output",
        default="data/kang_pbmc_gse96583_singlets_raw_counts.h5ad",
        help="Output `.h5ad` path.",
    )
    p.add_argument(
        "--keep-nonsinglets",
        action="store_true",
        help="Keep rows whose `multiplets` annotation is not `singlet`.",
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


def _read_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    df.index.name = "cell_barcode"
    return df


def _fill_missing_cell_types(metadata: pd.DataFrame) -> pd.DataFrame:
    df = metadata.copy()
    if not df["cell"].isna().any():
        return df

    cluster_majority = (
        df.dropna(subset=["cell"])
        .groupby("cluster")["cell"]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )
    missing_mask = df["cell"].isna()
    df.loc[missing_mask, "cell"] = df.loc[missing_mask, "cluster"].map(cluster_majority)
    unresolved = int(df["cell"].isna().sum())
    if unresolved:
        raise ValueError(f"Unable to infer {unresolved} missing cell labels from cluster majority mapping.")
    LOGGER.info("Imputed %d missing cell labels from cluster majority mapping.", int(missing_mask.sum()))
    return df


def _read_gene_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["gene_id", "gene_symbol"])
    df["gene_id"] = df["gene_id"].astype(str)
    df["gene_symbol"] = df["gene_symbol"].astype(str)
    return df


def _load_raw_counts(raw_tar_path: Path) -> tuple[sp.csr_matrix, list[str], list[str]]:
    matrices: list[sp.spmatrix] = []
    barcodes_all: list[str] = []
    matrix_blocks: list[str] = []

    with tarfile.open(raw_tar_path, "r") as tf:
        for matrix_name, barcode_name in MATRIX_MEMBERS:
            matrix_bytes = gzip.decompress(tf.extractfile(tf.getmember(matrix_name)).read())
            barcode_lines = gzip.decompress(tf.extractfile(tf.getmember(barcode_name)).read()).decode("utf-8")
            barcodes = [line.strip() for line in barcode_lines.splitlines() if line.strip()]
            block_name = matrix_name.replace(".mtx.gz", "")

            matrix = mmread(io.BytesIO(matrix_bytes)).tocsr()
            if matrix.shape[1] != len(barcodes):
                raise ValueError(
                    f"Column mismatch for {matrix_name}: matrix has {matrix.shape[1]} cells, "
                    f"barcode file has {len(barcodes)} entries."
                )

            matrices.append(matrix)
            barcodes_all.extend(barcodes)
            matrix_blocks.extend([block_name] * len(barcodes))

    counts_gene_by_cell = sp.hstack(matrices, format="csr")
    return counts_gene_by_cell, barcodes_all, matrix_blocks


def _build_obs(metadata: pd.DataFrame) -> pd.DataFrame:
    obs = pd.DataFrame(index=metadata.index.copy())
    obs.index.name = "cell_id"
    obs["raw_barcode"] = metadata["raw_barcode"].astype(str)
    obs["matrix_block"] = metadata["matrix_block"].astype(str)
    obs["tsne1"] = metadata["tsne1"].astype(np.float32)
    obs["tsne2"] = metadata["tsne2"].astype(np.float32)
    obs["donor"] = metadata["ind"].astype(str)
    obs["condition"] = metadata["stim"].astype(str)
    obs["cluster_id"] = pd.array(metadata["cluster"], dtype="Int64")
    obs["cell_type"] = metadata["cell"].astype(str)
    obs["Group"] = obs["cell_type"]
    obs["label"] = obs["cell_type"]
    obs["labels"] = obs["cell_type"]
    obs["sample"] = obs["donor"] + "_" + obs["condition"]
    obs["confounded_batch"] = obs["sample"]
    # The original DESC paper analyzed this dataset without explicit batch info,
    # because technical variation is confounded with stimulated biology.
    obs["batch"] = "single_batch"
    obs["multiplets"] = metadata["multiplets"].astype(str)
    obs["study"] = "Kang2018"
    obs["organism"] = "Homo sapiens"
    obs["tissue"] = "PBMC"
    obs["assay"] = "10x_raw_counts"
    obs["raw_data_accession"] = SERIES_ACCESSION

    categorical_cols = [
        "matrix_block",
        "donor",
        "condition",
        "cell_type",
        "Group",
        "label",
        "labels",
        "sample",
        "confounded_batch",
        "batch",
        "multiplets",
        "study",
        "organism",
        "tissue",
        "assay",
        "raw_data_accession",
    ]
    for col in categorical_cols:
        obs[col] = pd.Categorical(obs[col])
    return obs


def _build_var(genes: pd.DataFrame) -> pd.DataFrame:
    var = genes.copy()
    var.index = pd.Index(var["gene_symbol"].astype(str), name="gene")
    return var


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    raw_tar_path = _resolve_path(args.raw_tar)
    metadata_path = _resolve_path(args.metadata)
    genes_path = _resolve_path(args.genes)
    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for path in [raw_tar_path, metadata_path, genes_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    LOGGER.info("Reading metadata from %s", metadata_path)
    metadata = _fill_missing_cell_types(_read_metadata(metadata_path))
    LOGGER.info("Reading genes from %s", genes_path)
    genes = _read_gene_table(genes_path)
    LOGGER.info("Reading raw matrices from %s", raw_tar_path)
    counts_gene_by_cell, barcodes, matrix_blocks = _load_raw_counts(raw_tar_path)

    if counts_gene_by_cell.shape[0] != len(genes):
        raise ValueError(
            f"Gene mismatch: matrix has {counts_gene_by_cell.shape[0]} rows, gene table has {len(genes)} rows."
        )

    if len(metadata) != len(barcodes):
        raise ValueError(
            f"Metadata length mismatch: metadata has {len(metadata)} rows, raw matrices have {len(barcodes)} cells."
        )

    metadata = metadata.copy()
    metadata["raw_barcode"] = barcodes
    metadata["matrix_block"] = matrix_blocks

    keep_mask = np.ones(len(barcodes), dtype=bool)
    if not args.keep_nonsinglets:
        keep_mask = metadata["multiplets"].to_numpy() == "singlet"
        LOGGER.info("Keeping %d/%d singlets", int(keep_mask.sum()), len(keep_mask))

    counts_cell_by_gene = counts_gene_by_cell.transpose().tocsr()[keep_mask]
    metadata = metadata.loc[keep_mask].copy()
    obs = _build_obs(metadata)
    var = _build_var(genes)

    adata = ad.AnnData(X=counts_cell_by_gene.astype(np.float32), obs=obs, var=var)
    duplicate_gene_count = int(adata.var_names.duplicated().sum())
    if duplicate_gene_count:
        LOGGER.warning("Detected %d duplicated gene symbols; making var names unique.", duplicate_gene_count)
        adata.var_names_make_unique()

    adata.uns["dataset_name"] = "kang_pbmc_batch2"
    adata.uns["source"] = "GEO batch2 metadata + raw matrices"
    adata.uns["series_accession"] = SERIES_ACCESSION
    adata.uns["n_donors"] = int(obs["donor"].nunique())
    adata.uns["donors"] = sorted(obs["donor"].astype(str).unique().tolist())
    adata.uns["conditions"] = sorted(obs["condition"].astype(str).unique().tolist())
    adata.uns["cell_types"] = sorted(obs["cell_type"].astype(str).unique().tolist())
    adata.uns["input_files"] = {
        "raw_tar": str(raw_tar_path),
        "metadata": str(metadata_path),
        "genes": str(genes_path),
    }
    adata.uns["conversion"] = {
        "script": "scripts/prepare_kang_pbmc_gse96583_h5ad.py",
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "nnz": int(adata.X.nnz),
        "raw_cells_before_filter": int(len(barcodes)),
        "kept_singlets_only": bool(not args.keep_nonsinglets),
        "duplicate_gene_symbols": duplicate_gene_count,
        "recommended_batch_key_for_paper_setting": "single_batch",
        "available_confounded_batch_key": "confounded_batch",
        "label_column": "label",
    }

    LOGGER.info("Writing %s with shape=%s", output_path, adata.shape)
    adata.write_h5ad(output_path, compression=_compression_arg(args.compression))

    summary = {
        "output": str(output_path),
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "cell_types": adata.uns["cell_types"],
        "conditions": adata.uns["conditions"],
        "kept_singlets_only": bool(not args.keep_nonsinglets),
    }
    LOGGER.info("Conversion summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
