#!/usr/bin/env python3
"""Prepare scCluBench datasets whose names match rows from the benchmark table."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pandas.api.types import is_object_dtype, is_string_dtype

LOGGER = logging.getLogger("prepare_scclubench_table_rows_h5ads")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "scclubench_table_rows"
SCRBENCH_ROOT = Path("/data2/fbidet/SCRBenchmark/data/GSE84133_RAW")


@dataclass(frozen=True)
class DatasetSpec:
    output_name: str
    kind: str
    source: str
    tissue: str
    organism: str
    assay: str
    batch_value: str | None = None


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        output_name="Human_Pancreas_1_raw_counts",
        kind="human_pancreas_split",
        source="data/baron_human_pancreas.h5ad",
        tissue="pancreas",
        organism="Homo sapiens",
        assay="CEL-seq_raw_counts",
        batch_value="human1",
    ),
    DatasetSpec(
        output_name="Human_Pancreas_2_raw_counts",
        kind="human_pancreas_split",
        source="data/baron_human_pancreas.h5ad",
        tissue="pancreas",
        organism="Homo sapiens",
        assay="CEL-seq_raw_counts",
        batch_value="human2",
    ),
    DatasetSpec(
        output_name="Human_Pancreas_3_raw_counts",
        kind="human_pancreas_split",
        source="data/baron_human_pancreas.h5ad",
        tissue="pancreas",
        organism="Homo sapiens",
        assay="CEL-seq_raw_counts",
        batch_value="human3",
    ),
    DatasetSpec(
        output_name="Human_Pancreas_4_raw_counts",
        kind="human_pancreas_split",
        source="data/baron_human_pancreas.h5ad",
        tissue="pancreas",
        organism="Homo sapiens",
        assay="CEL-seq_raw_counts",
        batch_value="human4",
    ),
    DatasetSpec(
        output_name="Mouse_Pancreas_1_raw_counts",
        kind="mouse_pancreas",
        source=str(SCRBENCH_ROOT / "GSM2230761_mouse1_umifm_counts.h5ad"),
        tissue="pancreas",
        organism="Mus musculus",
        assay="CEL-seq_raw_counts",
    ),
    DatasetSpec(
        output_name="Mouse_Pancreas_2_raw_counts",
        kind="mouse_pancreas",
        source=str(SCRBENCH_ROOT / "GSM2230762_mouse2_umifm_counts.h5ad"),
        tissue="pancreas",
        organism="Mus musculus",
        assay="CEL-seq_raw_counts",
    ),
    DatasetSpec(
        output_name="Muris_Brain_raw_counts",
        kind="muris_public",
        source="data/scclubench_sources/tabula_muris_brain_myeloid_facs.h5ad",
        tissue="brain",
        organism="Mus musculus",
        assay="Tabula_Muris_public_counts",
    ),
    DatasetSpec(
        output_name="Muris_Kidney_raw_counts",
        kind="muris_public",
        source="data/scclubench_sources/tabula_muris_kidney_facs.h5ad",
        tissue="kidney",
        organism="Mus musculus",
        assay="Tabula_Muris_public_counts",
    ),
    DatasetSpec(
        output_name="Muris_Limb_Muscle_raw_counts",
        kind="muris_public",
        source="data/scclubench_sources/tabula_muris_limb_muscle_facs.h5ad",
        tissue="limb_muscle",
        organism="Mus musculus",
        assay="Tabula_Muris_public_counts",
    ),
    DatasetSpec(
        output_name="Muris_Liver_raw_counts",
        kind="muris_public",
        source="data/scclubench_sources/tabula_muris_liver_facs.h5ad",
        tissue="liver",
        organism="Mus musculus",
        assay="Tabula_Muris_public_counts",
    ),
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where standardized `.h5ad` files will be written.",
    )
    p.add_argument(
        "--compression",
        default="gzip",
        choices=["gzip", "lzf", "none"],
        help="Compression used when writing `.h5ad` files.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    p.add_argument("--verbose", action="store_true", help="Enable info logs.")
    return p


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _compression_arg(choice: str) -> str | None:
    return None if choice == "none" else choice


def _to_csr(matrix) -> sp.csr_matrix:
    if sp.issparse(matrix):
        return matrix.tocsr().astype(np.float32)
    return sp.csr_matrix(np.asarray(matrix, dtype=np.float32))


def _is_stringish(series: pd.Series) -> bool:
    return (
        is_string_dtype(series)
        or is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _categoricalize_string_columns(frame: pd.DataFrame) -> None:
    for col in frame.columns:
        if _is_stringish(frame[col]):
            frame[col] = pd.Categorical(frame[col].astype(str))


def _set_common_obs_columns(
    obs: pd.DataFrame,
    *,
    label_col: str,
    dataset_name: str,
    organism: str,
    tissue: str,
    assay: str,
    source_tag: str,
    source_batch: str | None = None,
) -> pd.DataFrame:
    obs = obs.copy()
    obs.index = obs.index.astype(str)
    obs.index.name = "cell_id"
    obs["cell_type"] = obs[label_col].astype(str)
    obs["Group"] = obs["cell_type"]
    obs["label"] = obs["cell_type"]
    obs["labels"] = obs["cell_type"]
    if source_batch is None and "batch" in obs.columns:
        source_batch = obs["batch"].astype(str)
    if source_batch is not None:
        obs["source_batch"] = source_batch if isinstance(source_batch, str) else source_batch.astype(str)
    obs["batch"] = "single_batch"
    obs["study"] = dataset_name
    obs["organism"] = organism
    obs["tissue"] = tissue
    obs["assay"] = assay
    obs["source_dataset"] = source_tag
    _categoricalize_string_columns(obs)
    return obs


def _finalize_adata(
    *,
    X,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    spec: DatasetSpec,
    source_path: Path,
    extra_uns: dict | None = None,
) -> ad.AnnData:
    adata = ad.AnnData(X=_to_csr(X), obs=obs, var=var.copy())
    if adata.var_names.duplicated().any():
        LOGGER.warning("Detected duplicated var names in %s; making them unique.", spec.output_name)
        adata.var_names_make_unique()
    if "gene_symbol" not in adata.var.columns:
        adata.var["gene_symbol"] = adata.var_names.astype(str)

    adata.uns["dataset_name"] = spec.output_name
    adata.uns["source"] = str(source_path)
    adata.uns["cell_types"] = sorted(obs["cell_type"].astype(str).unique().tolist())
    adata.uns["conversion"] = {
        "script": "scripts/prepare_scclubench_table_rows_h5ads.py",
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "nnz": int(adata.X.nnz),
        "label_column": "label",
        "batch_column": "batch",
    }
    if extra_uns:
        adata.uns.update(extra_uns)
    return adata


def _prepare_human_pancreas_split(spec: DatasetSpec) -> ad.AnnData:
    source_path = _resolve_path(spec.source)
    adata_src = ad.read_h5ad(source_path)
    if spec.batch_value is None:
        raise ValueError(f"{spec.output_name} requires a batch_value")
    mask = adata_src.obs["batch"].astype(str) == spec.batch_value
    adata_src = adata_src[mask].copy()
    obs = _set_common_obs_columns(
        adata_src.obs,
        label_col="cell_type",
        dataset_name=spec.output_name,
        organism=spec.organism,
        tissue=spec.tissue,
        assay=spec.assay,
        source_tag="baron_human_pancreas",
        source_batch=spec.batch_value,
    )
    extra_uns = {
        "benchmark_family": "scCluBench_table_rows",
        "table_row_name": spec.output_name.replace("_raw_counts", "").replace("_", " "),
        "source_study": "Baron2016",
        "selection": {"batch": spec.batch_value},
    }
    return _finalize_adata(
        X=adata_src.X,
        obs=obs,
        var=adata_src.var.copy(),
        spec=spec,
        source_path=source_path,
        extra_uns=extra_uns,
    )


def _prepare_mouse_pancreas(spec: DatasetSpec) -> ad.AnnData:
    source_path = _resolve_path(spec.source)
    adata_src = ad.read_h5ad(source_path)
    obs = _set_common_obs_columns(
        adata_src.obs,
        label_col="assigned_cluster",
        dataset_name=spec.output_name,
        organism=spec.organism,
        tissue=spec.tissue,
        assay=spec.assay,
        source_tag=source_path.stem,
        source_batch=source_path.stem,
    )
    extra_uns = {
        "benchmark_family": "scCluBench_table_rows",
        "table_row_name": spec.output_name.replace("_raw_counts", "").replace("_", " "),
        "source_study": "Baron2016",
    }
    return _finalize_adata(
        X=adata_src.X,
        obs=obs,
        var=adata_src.var.copy(),
        spec=spec,
        source_path=source_path,
        extra_uns=extra_uns,
    )


def _prepare_muris_public(spec: DatasetSpec) -> ad.AnnData:
    source_path = _resolve_path(spec.source)
    adata_src = ad.read_h5ad(source_path)
    if adata_src.raw is not None:
        X = adata_src.raw.X
        var = adata_src.raw.var.copy()
    else:
        X = adata_src.X
        var = adata_src.var.copy()
    obs = _set_common_obs_columns(
        adata_src.obs,
        label_col="cell_ontology_class",
        dataset_name=spec.output_name,
        organism=spec.organism,
        tissue=spec.tissue,
        assay=spec.assay,
        source_tag=source_path.stem,
        source_batch=source_path.stem,
    )
    extra_uns = {
        "benchmark_family": "scCluBench_table_rows",
        "table_row_name": spec.output_name.replace("_raw_counts", "").replace("_", " "),
        "source_study": "TabulaMuris_public",
    }
    return _finalize_adata(
        X=X,
        obs=obs,
        var=var,
        spec=spec,
        source_path=source_path,
        extra_uns=extra_uns,
    )


def _prepare_dataset(spec: DatasetSpec) -> ad.AnnData:
    if spec.kind == "human_pancreas_split":
        return _prepare_human_pancreas_split(spec)
    if spec.kind == "mouse_pancreas":
        return _prepare_mouse_pancreas(spec)
    if spec.kind == "muris_public":
        return _prepare_muris_public(spec)
    raise ValueError(f"Unsupported dataset kind: {spec.kind}")


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, object]] = []
    for spec in DATASETS:
        out_path = output_dir / f"{spec.output_name}.h5ad"
        if out_path.exists() and not args.force:
            LOGGER.info("Skipping existing file: %s", out_path)
            adata_existing = ad.read_h5ad(out_path, backed="r")
            summary.append(
                {
                    "dataset": spec.output_name,
                    "output": str(out_path),
                    "shape": [int(adata_existing.n_obs), int(adata_existing.n_vars)],
                    "status": "existing",
                }
            )
            adata_existing.file.close()
            continue

        LOGGER.info("Preparing %s", spec.output_name)
        adata = _prepare_dataset(spec)
        LOGGER.info("Writing %s with shape=%s", out_path, adata.shape)
        adata.write_h5ad(out_path, compression=_compression_arg(args.compression))
        summary.append(
            {
                "dataset": spec.output_name,
                "output": str(out_path),
                "shape": [int(adata.n_obs), int(adata.n_vars)],
                "n_cell_types": int(adata.obs["cell_type"].astype(str).nunique()),
                "status": "written",
            }
        )

    summary_path = output_dir / "scclubench_table_rows_preparation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    LOGGER.info("Summary written to %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
