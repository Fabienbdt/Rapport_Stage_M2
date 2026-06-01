#!/usr/bin/env python3
"""Prepare a pragmatic set of additional scCluBench-style `.h5ad` files."""

from __future__ import annotations

import argparse
import json
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pandas.api.types import is_object_dtype, is_string_dtype

LOGGER = logging.getLogger("prepare_scclubench_extra_h5ads")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "scclubench_extra"
SOURCES_DIR = PROJECT_ROOT / "data" / "scclubench_sources"
SCRBENCH_ROOT = Path("/data2/fbidet/SCRBenchmark/data/GSE84133_RAW")


@dataclass(frozen=True)
class DatasetSpec:
    output_name: str
    kind: str
    source: str
    tissue: str
    organism: str
    assay: str
    sample_fraction: float | None = None
    sample_seed: int = 42


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        output_name="Mouse_Pancreas_1_raw_counts",
        kind="mouse_pancreas",
        source=str(SCRBENCH_ROOT / "GSM2230761_mouse1_umifm_counts.h5ad"),
        tissue="pancreas",
        organism="Mus musculus",
        assay="celseq_raw_counts",
    ),
    DatasetSpec(
        output_name="Mouse_Pancreas_2_raw_counts",
        kind="mouse_pancreas",
        source=str(SCRBENCH_ROOT / "GSM2230762_mouse2_umifm_counts.h5ad"),
        tissue="pancreas",
        organism="Mus musculus",
        assay="celseq_raw_counts",
    ),
    DatasetSpec(
        output_name="Mauro_human_Pancreas_cell_raw_counts",
        kind="mauro_pancreas",
        source="data/pancreas_raw_counts.h5ad",
        tissue="pancreas",
        organism="Homo sapiens",
        assay="celseq2_raw_counts",
    ),
    DatasetSpec(
        output_name="Tabula_Muris_brain_filtered_raw_counts",
        kind="tabula_muris",
        source="data/scclubench_sources/tabula_muris_brain_myeloid_facs.h5ad",
        tissue="brain",
        organism="Mus musculus",
        assay="tabula_muris_facs_raw_counts",
    ),
    DatasetSpec(
        output_name="Tabula_Muris_kidney_filtered_raw_counts",
        kind="tabula_muris",
        source="data/scclubench_sources/tabula_muris_kidney_facs.h5ad",
        tissue="kidney",
        organism="Mus musculus",
        assay="tabula_muris_facs_raw_counts",
    ),
    DatasetSpec(
        output_name="Tabula_Muris_limb_muscle_filtered_raw_counts",
        kind="tabula_muris",
        source="data/scclubench_sources/tabula_muris_limb_muscle_facs.h5ad",
        tissue="limb_muscle",
        organism="Mus musculus",
        assay="tabula_muris_facs_raw_counts",
    ),
    DatasetSpec(
        output_name="Tabula_Muris_liver_filtered_raw_counts",
        kind="tabula_muris",
        source="data/scclubench_sources/tabula_muris_liver_facs.h5ad",
        tissue="liver",
        organism="Mus musculus",
        assay="tabula_muris_facs_raw_counts",
    ),
    DatasetSpec(
        output_name="Tabula_Muris_lung_filtered_raw_counts",
        kind="tabula_muris",
        source="data/scclubench_sources/tabula_muris_lung_facs.h5ad",
        tissue="lung",
        organism="Mus musculus",
        assay="tabula_muris_facs_raw_counts",
    ),
    DatasetSpec(
        output_name="Tabula_Sapiens_lung_10percent_filtered_raw_counts",
        kind="tabula_sapiens",
        source="data/scclubench_sources/ts_lung.zip",
        tissue="lung",
        organism="Homo sapiens",
        assay="tabula_sapiens_raw_counts",
        sample_fraction=0.10,
        sample_seed=42,
    ),
    DatasetSpec(
        output_name="Tabula_Sapiens_trachea_filtered_raw_counts",
        kind="tabula_sapiens",
        source="/tmp/ts_trachea.zip",
        tissue="trachea",
        organism="Homo sapiens",
        assay="tabula_sapiens_raw_counts",
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
        "--datasets",
        default="all",
        help="Comma-separated output dataset names to build, or `all`.",
    )
    p.add_argument(
        "--compression",
        default="gzip",
        choices=["gzip", "lzf", "none"],
        help="Compression used when writing `.h5ad` files.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild files even if they already exist.",
    )
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


def _string_like_columns(frame: pd.DataFrame) -> Iterable[str]:
    for col in frame.columns:
        series = frame[col]
        if is_string_dtype(series) or is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            yield col


def _categoricalize_string_columns(frame: pd.DataFrame) -> None:
    for col in _string_like_columns(frame):
        frame[col] = pd.Categorical(frame[col].astype(str))


def _extract_h5ad_from_zip(zip_path: Path) -> Path:
    extract_dir = SOURCES_DIR / f"{zip_path.stem}_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [name for name in zf.namelist() if name.endswith(".h5ad")]
        if not members:
            raise FileNotFoundError(f"No `.h5ad` payload found in {zip_path}")
        member = members[0]
        out_path = extract_dir / member
        if not out_path.exists():
            LOGGER.info("Extracting %s -> %s", zip_path, out_path)
            zf.extract(member, extract_dir)
    return out_path


def _set_common_obs_columns(
    obs: pd.DataFrame,
    *,
    cell_type_col: str,
    dataset_name: str,
    organism: str,
    tissue: str,
    assay: str,
    source_tag: str,
) -> pd.DataFrame:
    obs = obs.copy()
    obs.index = obs.index.astype(str)
    obs.index.name = "cell_id"
    obs["cell_type"] = obs[cell_type_col].astype(str)
    obs["Group"] = obs["cell_type"]
    obs["label"] = obs["cell_type"]
    obs["labels"] = obs["cell_type"]
    if "batch" in obs.columns:
        obs["original_batch"] = obs["batch"].astype(str)
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
        "script": "scripts/prepare_scclubench_extra_h5ads.py",
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "nnz": int(adata.X.nnz),
        "label_column": "label",
        "batch_column": "batch",
    }
    if extra_uns:
        adata.uns.update(extra_uns)
    return adata


def _prepare_mouse_pancreas(spec: DatasetSpec) -> ad.AnnData:
    source_path = _resolve_path(spec.source)
    adata_src = ad.read_h5ad(source_path)
    obs = _set_common_obs_columns(
        adata_src.obs,
        cell_type_col="assigned_cluster",
        dataset_name=spec.output_name,
        organism=spec.organism,
        tissue=spec.tissue,
        assay=spec.assay,
        source_tag=source_path.stem,
    )
    obs["source_batch"] = source_path.stem
    obs["batch"] = pd.Categorical(obs["batch"].astype(str))
    obs["source_batch"] = pd.Categorical(obs["source_batch"].astype(str))
    extra_uns = {
        "benchmark_family": "scCluBench",
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


def _prepare_mauro_pancreas(spec: DatasetSpec) -> ad.AnnData:
    source_path = _resolve_path(spec.source)
    adata_src = ad.read_h5ad(source_path)
    mask = adata_src.obs["study"].astype(str) == "muraro"
    adata_src = adata_src[mask].copy()
    obs = _set_common_obs_columns(
        adata_src.obs,
        cell_type_col="cell_type",
        dataset_name=spec.output_name,
        organism=spec.organism,
        tissue=spec.tissue,
        assay=spec.assay,
        source_tag="muraro_from_pancreas_raw_counts",
    )
    extra_uns = {
        "benchmark_family": "scCluBench",
        "source_study": "Muraro2016",
        "selection": {"study": "muraro"},
    }
    return _finalize_adata(
        X=adata_src.X,
        obs=obs,
        var=adata_src.var.copy(),
        spec=spec,
        source_path=source_path,
        extra_uns=extra_uns,
    )


def _prepare_tabula_muris(spec: DatasetSpec) -> ad.AnnData:
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
        cell_type_col="cell_ontology_class",
        dataset_name=spec.output_name,
        organism=spec.organism,
        tissue=spec.tissue,
        assay=spec.assay,
        source_tag=source_path.stem,
    )
    extra_uns = {
        "benchmark_family": "scCluBench",
        "source_collection": "Tabula Muris Senis official annotations",
    }
    return _finalize_adata(
        X=X,
        obs=obs,
        var=var,
        spec=spec,
        source_path=source_path,
        extra_uns=extra_uns,
    )


def _prepare_tabula_sapiens(spec: DatasetSpec) -> ad.AnnData:
    source_path = _resolve_path(spec.source)
    h5ad_path = _extract_h5ad_from_zip(source_path) if source_path.suffix == ".zip" else source_path
    adata_src = ad.read_h5ad(h5ad_path)
    sample_meta: dict[str, int | float] = {}
    if spec.sample_fraction is not None and 0 < spec.sample_fraction < 1:
        n_keep = max(1, int(round(adata_src.n_obs * spec.sample_fraction)))
        rng = np.random.default_rng(spec.sample_seed)
        keep_idx = np.sort(rng.choice(adata_src.n_obs, size=n_keep, replace=False))
        adata_src = adata_src[keep_idx].copy()
        sample_meta = {
            "fraction": float(spec.sample_fraction),
            "seed": int(spec.sample_seed),
            "selected_cells": int(adata_src.n_obs),
        }
    if "raw_counts" in adata_src.layers:
        X = adata_src.layers["raw_counts"]
    elif adata_src.raw is not None:
        X = adata_src.raw.X
    else:
        X = adata_src.X
    var = adata_src.var.copy()
    obs = _set_common_obs_columns(
        adata_src.obs,
        cell_type_col="cell_ontology_class",
        dataset_name=spec.output_name,
        organism=spec.organism,
        tissue=spec.tissue,
        assay=spec.assay,
        source_tag=h5ad_path.stem,
    )
    extra_uns = {
        "benchmark_family": "scCluBench",
        "source_collection": "Tabula Sapiens public organ h5ad",
    }
    if sample_meta:
        extra_uns["sampling"] = sample_meta
    return _finalize_adata(
        X=X,
        obs=obs,
        var=var,
        spec=spec,
        source_path=h5ad_path,
        extra_uns=extra_uns,
    )


def _prepare_dataset(spec: DatasetSpec) -> ad.AnnData:
    if spec.kind == "mouse_pancreas":
        return _prepare_mouse_pancreas(spec)
    if spec.kind == "mauro_pancreas":
        return _prepare_mauro_pancreas(spec)
    if spec.kind == "tabula_muris":
        return _prepare_tabula_muris(spec)
    if spec.kind == "tabula_sapiens":
        return _prepare_tabula_sapiens(spec)
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

    if args.datasets.strip().lower() == "all":
        selected_specs = list(DATASETS)
    else:
        wanted = {name.strip() for name in args.datasets.split(",") if name.strip()}
        selected_specs = [spec for spec in DATASETS if spec.output_name in wanted]
        missing = sorted(wanted - {spec.output_name for spec in selected_specs})
        if missing:
            raise ValueError(f"Unknown dataset names: {missing}")

    summary: list[dict[str, object]] = []
    for spec in selected_specs:
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

    summary_path = output_dir / "scclubench_extra_preparation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    LOGGER.info("Summary written to %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
