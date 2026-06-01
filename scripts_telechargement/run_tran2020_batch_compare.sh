#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${1:?usage: run_tran2020_batch_compare.sh OUTPUT_ROOT}"

SCRAW_GPU_1="${SCRAW_GPU_1:-1}"
SCRAW_GPU_2="${SCRAW_GPU_2:-2}"
SEED_LIST_VALUE="${SEED_LIST:-42}"
SCIB_N_JOBS_VALUE="${SCIB_N_JOBS:-1}"
RUN_DESC_VALUE="${RUN_DESC:-1}"
RUN_SCANORAMA_VALUE="${RUN_SCANORAMA:-1}"
RUN_HARMONY_VALUE="${RUN_HARMONY:-1}"
RUN_COMBAT_VALUE="${RUN_COMBAT:-1}"
INCLUDE_PANCREAS_FOUR_BATCHES_VALUE="${INCLUDE_PANCREAS_FOUR_BATCHES:-0}"

PBMC_ANN="${ROOT_DIR}/data/pbmc_3prime_5prime_bbknn_annotated.h5ad"
PBMC_3P="${ROOT_DIR}/data/pbmc8k_filtered_gene_bc_matrices.tar.gz"
PBMC_5P="${ROOT_DIR}/data/vdj_v1_hs_pbmc_5gex_filtered_gene_bc_matrices.tar.gz"
PBMC_H5AD="${ROOT_DIR}/data/pbmc_3prime_5prime_paper_raw_counts.h5ad"

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/logs"

download_if_missing() {
  local target="$1"
  local url="$2"
  if [[ -f "${target}" ]]; then
    echo "[reuse] ${target}"
    return
  fi
  echo "[download] ${url} -> ${target}"
  curl -L -o "${target}" "${url}"
}

prepare_pbmc_if_needed() {
  if [[ -f "${PBMC_H5AD}" ]]; then
    echo "[reuse] ${PBMC_H5AD}"
    return
  fi

  download_if_missing \
    "${PBMC_ANN}" \
    "ftp://ngs.sanger.ac.uk/production/teichmann/BBKNN/PBMC.merged.h5ad"
  download_if_missing \
    "${PBMC_3P}" \
    "http://cf.10xgenomics.com/samples/cell-exp/2.1.0/pbmc8k/pbmc8k_filtered_gene_bc_matrices.tar.gz"
  download_if_missing \
    "${PBMC_5P}" \
    "http://cf.10xgenomics.com/samples/cell-vdj/2.2.0/vdj_v1_hs_pbmc_5gex/vdj_v1_hs_pbmc_5gex_filtered_gene_bc_matrices.tar.gz"

  echo "[prepare] PBMC 3' vs 5' paper dataset"
  "${ROOT_DIR}/my_venv_312/bin/python" \
    "${ROOT_DIR}/scripts/prepare_pbmc_3prime_5prime_paper_h5ad.py" \
    --counts-3p "${PBMC_3P}" \
    --counts-5p "${PBMC_5P}" \
    --annotations "${PBMC_ANN}" \
    --output "${PBMC_H5AD}"
}

run_scraw_bg() {
  local dataset_slug="$1"
  local data_path="$2"
  local gpu_id="$3"
  local batch_key="$4"
  local out_dir="${OUTPUT_ROOT}/${dataset_slug}/scraw"
  local log_file="${OUTPUT_ROOT}/logs/${dataset_slug}_scraw.log"
  local seed_value=""
  local analysis_csv=""
  local missing_outputs=0

  mkdir -p "${OUTPUT_ROOT}/${dataset_slug}"
  for seed_value in ${SEED_LIST_VALUE//,/ }; do
    analysis_csv="${out_dir}/runs/seed_${seed_value}/results/analysis_results.csv"
    if [[ ! -f "${analysis_csv}" ]]; then
      missing_outputs=1
      break
    fi
  done

  if [[ "${missing_outputs}" != "1" ]]; then
    echo "[skip] ${dataset_slug} scraw" >&2
    LAST_BG_PID=""
    return
  fi

  echo "[launch-bg] scRAW ${dataset_slug} gpu=${gpu_id}" >&2
  env \
    NUMBA_DISABLE_JIT=1 \
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    SEED_LIST="${SEED_LIST_VALUE}" \
    SKIP_EXISTING=1 \
    COMPUTE_SCIB_METRICS=on \
    SCIB_N_JOBS="${SCIB_N_JOBS_VALUE}" \
    CAPTURE_SNAPSHOTS=off \
    METRICS_ONLY=1 \
    BATCH_KEY="${batch_key}" \
    bash "${ROOT_DIR}/scripts/run_default_preset_multiseed.sh" "${data_path}" "${out_dir}" \
    > "${log_file}" 2>&1 &
  LAST_BG_PID="$!"
}

run_baseline_if_needed() {
  local dataset_slug="$1"
  local data_path="$2"
  local method="$3"
  local label_key="$4"
  local batch_key="$5"
  local out_dir="${OUTPUT_ROOT}/${dataset_slug}/${method}"
  local log_file="${OUTPUT_ROOT}/logs/${dataset_slug}_${method}.log"

  if [[ -f "${out_dir}/results/analysis_results.csv" ]]; then
    echo "[skip] ${dataset_slug} ${method}"
    return
  fi

  echo "[run] ${dataset_slug} ${method}"
  env \
    CUDA_VISIBLE_DEVICES= \
    "${ROOT_DIR}/my_venv_312/bin/python" \
    "${ROOT_DIR}/scripts/run_batch_baseline_benchmark.py" \
    --data "${data_path}" \
    --output "${out_dir}" \
    --method "${method}" \
    --label-key "${label_key}" \
    --batch-key "${batch_key}" \
    --compute-scib \
    --scib-n-jobs "${SCIB_N_JOBS_VALUE}" \
    > "${log_file}" 2>&1
}

run_desc_if_needed() {
  local dataset_slug="$1"
  local data_path="$2"
  local label_key="$3"
  local batch_key="$4"
  local out_dir="${OUTPUT_ROOT}/${dataset_slug}/desc"
  local log_file="${OUTPUT_ROOT}/logs/${dataset_slug}_desc.log"

  if [[ "${RUN_DESC_VALUE}" != "1" ]]; then
    return
  fi
  if [[ -f "${out_dir}/results/analysis_results.csv" ]]; then
    echo "[skip] ${dataset_slug} desc"
    return
  fi

  echo "[run] ${dataset_slug} desc"
  "${ROOT_DIR}/envs/desc_py311/bin/python" \
    "${ROOT_DIR}/scripts/run_desc_benchmark.py" \
    --data "${data_path}" \
    --output "${out_dir}" \
    --label-key "${label_key}" \
    --batch-key "${batch_key}" \
    > "${log_file}" 2>&1
}

prepare_pbmc_if_needed

declare -a DATASETS=(
  "pancreas_raw_counts|${ROOT_DIR}/data/pancreas_raw_counts.h5ad|cell_type|batch"
  "pbmc_3prime_5prime_paper_raw_counts|${PBMC_H5AD}|cell_type|batch"
)

if [[ "${INCLUDE_PANCREAS_FOUR_BATCHES_VALUE}" == "1" ]]; then
  DATASETS+=(
    "pancreas_four_batches|${ROOT_DIR}/data/pancreas_raw_counts_four_batches_celseq_celseq2_fluidigmc1_smartseq2.h5ad|cell_type|batch"
  )
fi

SCRAW_PIDS=()
LAST_BG_PID=""
GPU_INDEX=0
for entry in "${DATASETS[@]}"; do
  IFS="|" read -r dataset_slug data_path label_key batch_key <<< "${entry}"
  gpu_id="${SCRAW_GPU_1}"
  if [[ "${GPU_INDEX}" -eq 1 ]]; then
    gpu_id="${SCRAW_GPU_2}"
  fi
  GPU_INDEX=$(( (GPU_INDEX + 1) % 2 ))
  run_scraw_bg "${dataset_slug}" "${data_path}" "${gpu_id}" "${batch_key}"
  if [[ -n "${LAST_BG_PID}" ]]; then
    SCRAW_PIDS+=("${LAST_BG_PID}")
  fi
done

for entry in "${DATASETS[@]}"; do
  IFS="|" read -r dataset_slug data_path label_key batch_key <<< "${entry}"
  if [[ "${RUN_HARMONY_VALUE}" == "1" ]]; then
    run_baseline_if_needed "${dataset_slug}" "${data_path}" "harmony" "${label_key}" "${batch_key}"
  fi
  if [[ "${RUN_COMBAT_VALUE}" == "1" ]]; then
    run_baseline_if_needed "${dataset_slug}" "${data_path}" "combat" "${label_key}" "${batch_key}"
  fi
  if [[ "${RUN_SCANORAMA_VALUE}" == "1" ]]; then
    run_baseline_if_needed "${dataset_slug}" "${data_path}" "scanorama" "${label_key}" "${batch_key}"
  fi
  run_desc_if_needed "${dataset_slug}" "${data_path}" "${label_key}" "${batch_key}"
done

for pid in "${SCRAW_PIDS[@]}"; do
  if [[ -n "${pid}" ]]; then
    wait "${pid}"
  fi
done

"${ROOT_DIR}/my_venv_312/bin/python" \
  "${ROOT_DIR}/scripts/aggregate_tran2020_compare.py" \
  --root "${OUTPUT_ROOT}"

echo "[done] Tran2020 comparison completed: ${OUTPUT_ROOT}"
