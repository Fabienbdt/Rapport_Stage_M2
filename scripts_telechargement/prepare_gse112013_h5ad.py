#!/usr/bin/env python3
"""Convert GEO GSE112013 processed UMI counts into an AnnData `.h5ad` file."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

LOGGER = logging.getLogger("prepare_gse112013_h5ad")

SERIES_ACCESSION = "GSE112013"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default="data/GSE112013_Combined_UMI_table.txt.gz",
        help="Path to the GEO processed UMI count table (.txt.gz).",
    )
    p.add_argument(
        "--series-matrix",
        default="data/GSE112013_series_matrix.txt.gz",
        help="Optional GEO series matrix file for sample metadata.",
    )
    p.add_argument(
        "--output",
        default="data/gse112013_human_testis_raw_counts.h5ad",
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


def _parse_series_matrix(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        LOGGER.warning("Series matrix not found at %s; continuing without GEO sample accessions.", path)
        return {}

    title_row: List[str] = []
    accession_row: List[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("!Sample_title\t"):
                title_row = [x.strip('"') for x in line.split("\t")[1:]]
            elif line.startswith("!Sample_geo_accession\t"):
                accession_row = [x.strip('"') for x in line.split("\t")[1:]]
            if title_row and accession_row:
                break

    if not title_row or not accession_row or len(title_row) != len(accession_row):
        LOGGER.warning("Unable to recover clean sample metadata from %s.", path)
        return {}

    return {
        title: {"sample_title": title, "sample_geo_accession": accession}
        for title, accession in zip(title_row, accession_row)
    }


def _parse_cell_id(cell_id: str, sample_meta: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    if "-" not in cell_id:
        raise ValueError(f"Unexpected cell identifier without donor prefix: {cell_id}")
    donor, barcode = cell_id.split("-", 1)

    if "-" in barcode:
        _, replicate = barcode.rsplit("-", 1)
    else:
        replicate = "1"

    sample_title = f"{donor}_scRNA-seq_rep{replicate}"
    sample_info = sample_meta.get(sample_title, {})
    return {
        "barcode": barcode,
        "donor": donor,
        "replicate": f"rep{replicate}",
        "sample": sample_title,
        "sample_geo_accession": sample_info.get("sample_geo_accession", ""),
        "batch": sample_title,
        "tissue": "testicle",
        "organism": "Homo sapiens",
        "series_accession": SERIES_ACCESSION,
    }


def _load_sparse_gene_by_cell_matrix(path: Path) -> Tuple[sp.csr_matrix, List[str], List[str]]:
    LOGGER.info("Reading sparse matrix from %s", path)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        header = next(fh).rstrip("\n").split("\t")
        if not header or header[0] != "Gene":
            raise ValueError(f"Unexpected first header column in {path}: {header[:3]}")

        cell_ids = header[1:]
        n_cells = len(cell_ids)
        genes: List[str] = []
        row_indices: List[np.ndarray] = []
        row_values: List[np.ndarray] = []
        indptr = [0]

        for line_no, raw in enumerate(fh, start=2):
            raw = raw.rstrip("\n")
            if not raw:
                continue
            gene, counts_txt = raw.split("\t", 1)
            counts = np.fromstring(counts_txt, sep="\t", dtype=np.int32)
            if counts.size != n_cells:
                raise ValueError(
                    f"Row {line_no} has {counts.size} counts, expected {n_cells} for gene {gene}."
                )

            nz = np.flatnonzero(counts)
            row_indices.append(nz.astype(np.int32, copy=False))
            row_values.append(counts[nz].astype(np.int32, copy=False))
            indptr.append(indptr[-1] + int(nz.size))
            genes.append(gene)

            if line_no % 5000 == 0:
                LOGGER.info("Parsed %d genes", line_no - 1)

    indices = np.concatenate(row_indices) if row_indices else np.array([], dtype=np.int32)
    data = np.concatenate(row_values) if row_values else np.array([], dtype=np.int32)
    gene_by_cell = sp.csr_matrix(
        (data, indices, np.asarray(indptr, dtype=np.int64)),
        shape=(len(genes), len(cell_ids)),
        dtype=np.int32,
    )
    return gene_by_cell, genes, cell_ids


def _build_obs(cell_ids: List[str], sample_meta: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    obs = pd.DataFrame([_parse_cell_id(cell_id, sample_meta) for cell_id in cell_ids], index=cell_ids)
    obs.index.name = "cell_id"
    for col in ["donor", "replicate", "sample", "sample_geo_accession", "batch", "tissue", "organism"]:
        obs[col] = pd.Categorical(obs[col])
    return obs


def _build_var(genes: List[str]) -> pd.DataFrame:
    var = pd.DataFrame(index=pd.Index(genes, name="gene"))
    var["gene_symbol"] = genes
    return var


def _compression_arg(choice: str) -> str | None:
    if choice == "none":
        return None
    return choice


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    input_path = Path(args.input).expanduser().resolve()
    series_path = Path(args.series_matrix).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input count table not found: {input_path}")

    sample_meta = _parse_series_matrix(series_path)
    gene_by_cell, genes, cell_ids = _load_sparse_gene_by_cell_matrix(input_path)
    cell_by_gene = gene_by_cell.transpose().tocsr()

    obs = _build_obs(cell_ids, sample_meta)
    var = _build_var(genes)
    adata = ad.AnnData(X=cell_by_gene, obs=obs, var=var)

    duplicate_gene_count = int(pd.Index(genes).duplicated().sum())
    if duplicate_gene_count:
        LOGGER.warning("Detected %d duplicated gene symbols; making var names unique.", duplicate_gene_count)
        adata.var_names_make_unique()

    adata.uns["dataset_name"] = "human_testis_cell_atlas_healthy_men"
    adata.uns["source"] = "GEO processed UMI table"
    adata.uns["series_accession"] = SERIES_ACCESSION
    adata.uns["sample_titles"] = sorted(obs["sample"].astype(str).unique().tolist())
    adata.uns["sample_geo_accessions"] = sorted(
        [x for x in obs["sample_geo_accession"].astype(str).unique().tolist() if x]
    )
    adata.uns["donors"] = sorted(obs["donor"].astype(str).unique().tolist())
    adata.uns["n_donors"] = int(obs["donor"].nunique())
    adata.uns["n_samples"] = int(obs["sample"].nunique())
    adata.uns["input_file"] = str(input_path)
    adata.uns["conversion"] = {
        "script": "scripts/prepare_gse112013_h5ad.py",
        "counts_shape_gene_by_cell": [int(gene_by_cell.shape[0]), int(gene_by_cell.shape[1])],
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "nnz": int(adata.X.nnz),
        "duplicate_gene_symbols": duplicate_gene_count,
    }

    LOGGER.info(
        "Writing %s with shape=%s, nnz=%d",
        output_path,
        adata.shape,
        int(adata.X.nnz),
    )
    adata.write_h5ad(output_path, compression=_compression_arg(args.compression))

    summary = {
        "output": str(output_path),
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "nnz": int(adata.X.nnz),
        "donors": adata.uns["donors"],
        "samples": adata.uns["sample_titles"],
    }
    LOGGER.info("Conversion summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
