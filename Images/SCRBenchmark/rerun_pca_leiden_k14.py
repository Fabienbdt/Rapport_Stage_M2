#!/usr/bin/env python3
"""Rerun PCA+Leiden on Baron with a resolution sweep constrained to k=14 clusters."""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier


SCRBENCHMARK_SRC = Path("/data2/fbidet/SCRBenchmark/src/scrbenchmark")
if str(SCRBENCHMARK_SRC) not in sys.path:
    sys.path.insert(0, str(SCRBENCHMARK_SRC))

from utils.pca_utils import get_pca_with_auto_components  # noqa: E402


SPLIT_ROOT = Path(
    "/data2/fbidet/SCRBenchmark/results/"
    "baron_split_70_10_20_existing_algorithms_5seeds_20260526_103715/"
    "gpu2_scname_classic"
)
FULL_ROOT = Path(
    "/data2/fbidet/SCRBenchmark/results/"
    "baron_full_existing_algorithms_5seeds_20260522_162936/"
    "gpu2_scname_classic"
)
OUTPUT_ROOT = Path(
    "/data2/fbidet/SCRBenchmark/results/"
    "baron_pca_leiden_k14_resolution_sweep_20260527"
)

RESOLUTIONS = [round(x, 2) for x in np.arange(0.1, 1.0 + 1e-12, 0.01)]
SEEDS = [42, 43, 44, 45, 46]
TARGET_K = 14
N_NEIGHBORS = 19


def dense_x(adata: ad.AnnData) -> np.ndarray:
    x = adata.X
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x)


def fit_auto_pca(x: np.ndarray, seed: int) -> tuple[PCA, np.ndarray, int]:
    _, n_components, _ = get_pca_with_auto_components(
        x,
        n_components=0,
        random_state=seed,
        max_components_for_elbow=100,
    )
    pca = PCA(n_components=n_components, random_state=seed)
    embedding = pca.fit_transform(x)
    return pca, embedding, int(n_components)


def search_k14_resolution(
    embedding: np.ndarray,
    seed: int,
    n_neighbors: int = N_NEIGHBORS,
) -> dict[str, float | int]:
    adata_tmp = ad.AnnData(np.zeros((embedding.shape[0], 1), dtype=np.float32))
    adata_tmp.obsm["X_pca"] = embedding.astype(np.float32, copy=False)
    sc.pp.neighbors(
        adata_tmp,
        n_neighbors=n_neighbors,
        use_rep="X_pca",
        method="gauss",
        random_state=seed,
    )

    candidates: list[dict[str, float | int]] = []
    observed: list[dict[str, float | int]] = []
    for resolution in RESOLUTIONS:
        key = f"leiden_{str(resolution).replace('.', '_')}"
        sc.tl.leiden(
            adata_tmp,
            resolution=float(resolution),
            n_iterations=2,
            flavor="igraph",
            directed=False,
            random_state=seed,
            key_added=key,
        )
        labels = adata_tmp.obs[key].astype(int).to_numpy()
        n_clusters = int(np.unique(labels).size)
        score = float("nan")
        if n_clusters > 1:
            score = float(silhouette_score(embedding, labels, metric="euclidean"))
        observed.append(
            {
                "resolution": float(resolution),
                "n_clusters": n_clusters,
                "silhouette": score,
            }
        )
        if n_clusters == TARGET_K:
            candidates.append(
                {
                    "resolution": float(resolution),
                    "n_clusters": n_clusters,
                    "silhouette": score,
                }
            )

    if not candidates:
        observed_text = ", ".join(
            f"{row['resolution']:.1f}:k={row['n_clusters']}"
            for row in observed
        )
        raise RuntimeError(
            f"No resolution in {RESOLUTIONS} produced k={TARGET_K}. Observed {observed_text}"
        )

    best = max(candidates, key=lambda row: float(row["silhouette"]))
    summary = {
        "selected_resolution": float(best["resolution"]),
        "selected_silhouette": float(best["silhouette"]),
        "n_clusters": int(best["n_clusters"]),
    }
    for row in observed:
        summary[f"k_at_resolution_{row['resolution']:.2f}"] = int(row["n_clusters"])
        summary[f"silhouette_at_resolution_{row['resolution']:.2f}"] = float(row["silhouette"])
    return summary


def leiden_fixed_resolution(
    embedding: np.ndarray,
    seed: int,
    resolution: float,
    n_neighbors: int = N_NEIGHBORS,
) -> tuple[np.ndarray, dict[str, float | int]]:
    adata_tmp = ad.AnnData(np.zeros((embedding.shape[0], 1), dtype=np.float32))
    adata_tmp.obsm["X_pca"] = embedding.astype(np.float32, copy=False)
    sc.pp.neighbors(
        adata_tmp,
        n_neighbors=n_neighbors,
        use_rep="X_pca",
        method="gauss",
        random_state=seed,
    )
    sc.tl.leiden(
        adata_tmp,
        resolution=float(resolution),
        n_iterations=2,
        flavor="igraph",
        directed=False,
        random_state=seed,
        key_added="leiden_fixed",
    )
    labels = adata_tmp.obs["leiden_fixed"].astype(int).to_numpy()
    n_clusters = int(np.unique(labels).size)
    score = float("nan")
    if n_clusters > 1:
        score = float(silhouette_score(embedding, labels, metric="euclidean"))
    return labels, {
        "selected_resolution": float(resolution),
        "fixed_resolution_silhouette": score,
        "n_clusters": n_clusters,
    }


def labels_frame(predicted: np.ndarray, adata: ad.AnnData) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "predicted_label": predicted.astype(str),
            "true_label": adata.obs["cell_type"].astype(str).to_numpy(),
            "batch": adata.obs["batch"].astype(str).to_numpy(),
        }
    )


def run_inductive() -> list[dict[str, float | int | str]]:
    train = ad.read_h5ad(SPLIT_ROOT / "data" / "benchmark" / "train.h5ad")
    test = ad.read_h5ad(SPLIT_ROOT / "data" / "benchmark" / "test.h5ad")
    x_train = dense_x(train)
    x_test = dense_x(test)

    labels_dir = OUTPUT_ROOT / "inductive" / "results" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []

    _, search_embedding, search_n_components = fit_auto_pca(x_train, SEEDS[0])
    search_summary = search_k14_resolution(search_embedding, SEEDS[0])
    selected_resolution = float(search_summary["selected_resolution"])

    print(
        f"inductive search once: res={selected_resolution:.2f}, "
        f"silhouette={search_summary['selected_silhouette']:.4f}, "
        f"k={search_summary['n_clusters']}, n_components={search_n_components}",
        flush=True,
    )

    for run_id, seed in enumerate(SEEDS):
        pca, train_embedding, n_components = fit_auto_pca(x_train, seed)
        train_labels, fixed_summary = leiden_fixed_resolution(
            train_embedding,
            seed,
            selected_resolution,
        )
        test_embedding = pca.transform(x_test)

        knn = KNeighborsClassifier(n_neighbors=min(15, len(train_labels)))
        knn.fit(train_embedding, train_labels)
        test_labels = knn.predict(test_embedding)

        labels_frame(train_labels, train).to_csv(
            labels_dir / f"benchmark_pca_leiden_run{run_id}_train.csv",
            index=False,
        )
        labels_frame(test_labels, test).to_csv(
            labels_dir / f"benchmark_pca_leiden_run{run_id}_test.csv",
            index=False,
        )
        rows.append(
            {
                "mode": "inductive",
                "run_id": run_id,
                "seed": seed,
                "n_components": n_components,
                "source_seed": SEEDS[0],
                "search_n_components": search_n_components,
                "search_selected_silhouette": search_summary["selected_silhouette"],
                "single_resolution_search": 1,
                **fixed_summary,
            }
        )
        print(
            f"inductive run{run_id}: fixed res={fixed_summary['selected_resolution']:.2f}, "
            f"silhouette={fixed_summary['fixed_resolution_silhouette']:.4f}, "
            f"k={fixed_summary['n_clusters']}",
            flush=True,
        )
    return rows


def run_transductive() -> list[dict[str, float | int | str]]:
    full = ad.read_h5ad(FULL_ROOT / "data" / "processed.h5ad")
    x_full = dense_x(full)

    labels_dir = OUTPUT_ROOT / "transductive" / "results" / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []

    _, search_embedding, search_n_components = fit_auto_pca(x_full, SEEDS[0])
    search_summary = search_k14_resolution(search_embedding, SEEDS[0])
    selected_resolution = float(search_summary["selected_resolution"])

    print(
        f"transductive search once: res={selected_resolution:.2f}, "
        f"silhouette={search_summary['selected_silhouette']:.4f}, "
        f"k={search_summary['n_clusters']}, n_components={search_n_components}",
        flush=True,
    )

    for run_id, seed in enumerate(SEEDS):
        _, embedding, n_components = fit_auto_pca(x_full, seed)
        labels, fixed_summary = leiden_fixed_resolution(
            embedding,
            seed,
            selected_resolution,
        )

        labels_frame(labels, full).to_csv(
            labels_dir / f"labels_pca_leiden_run{run_id}.csv",
            index=False,
        )
        rows.append(
            {
                "mode": "transductive",
                "run_id": run_id,
                "seed": seed,
                "n_components": n_components,
                "source_seed": SEEDS[0],
                "search_n_components": search_n_components,
                "search_selected_silhouette": search_summary["selected_silhouette"],
                "single_resolution_search": 1,
                **fixed_summary,
            }
        )
        print(
            f"transductive run{run_id}: fixed res={fixed_summary['selected_resolution']:.2f}, "
            f"silhouette={fixed_summary['fixed_resolution_silhouette']:.4f}, "
            f"k={fixed_summary['n_clusters']}",
            flush=True,
        )
    return rows


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = run_inductive() + run_transductive()
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / "pca_leiden_k14_resolution_sweep_summary.csv", index=False)
    print(f"Saved labels and summary under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
