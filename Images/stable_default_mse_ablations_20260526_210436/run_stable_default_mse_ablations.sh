#!/usr/bin/env bash
set -euo pipefail

# Stable-default MSE ablations matching the variants in scRAW_Scientific_Paper.tex.
# This launcher deliberately starts from the `stable_default` preset and only
# changes the ablated switches, unlike the historical Trial206 scripts that
# hardcoded the older Baron-specific hyperparameters.

RUN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRAW_ROOT="${SCRAW_ROOT:-/data2/fbidet/scRAW_EXPERIMENTAL}"
export PYTHONPATH="${SCRAW_ROOT}/src:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-/data2/fbidet/scrbenchmark_venv/bin/python}"
PRESET="${PRESET:-stable_default}"
DATA="${DATA:-${SCRAW_ROOT}/data/baron_human_pancreas.h5ad}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES

COMPUTE_SCIB_METRICS="${COMPUTE_SCIB_METRICS:-on}"
SCIB_N_JOBS="${SCIB_N_JOBS:-1}"
OUTPUT_PROFILE="${OUTPUT_PROFILE:-search_minimal}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CAPTURE_SNAPSHOTS="${CAPTURE_SNAPSHOTS:-off}"
SAVE_PROCESSED_DATA="${SAVE_PROCESSED_DATA:-off}"
LEIDEN_TARGET_CLUSTERS="${LEIDEN_TARGET_CLUSTERS:-14}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/runs" "${RUN_ROOT}/summaries"

BASE_CMD=(
  "${PYTHON_BIN}" "-m" "scraw_dedicated.cli"
  "--preset" "${PRESET}"
  "--data" "${DATA}"
  "--seed" "${SEED}"
  "--device" "${DEVICE}"
  "--output-profile" "${OUTPUT_PROFILE}"
  "--metrics-only"
  "--capture-snapshots" "${CAPTURE_SNAPSHOTS}"
  "--save-processed-data" "${SAVE_PROCESSED_DATA}"
  "--compute-scib-metrics" "${COMPUTE_SCIB_METRICS}"
  "--scib-n-jobs" "${SCIB_N_JOBS}"
  "--auto-hparams" "off"
  "--batch-key" "batch"
  "--leiden-target-clusters" "${LEIDEN_TARGET_CLUSTERS}"
  "--param" "batch_correction_key=batch"
  "--param" "clustering_method=hdbscan"
  "--param" "reconstruction_distribution=mse"
  "--param" "capture_embedding_snapshots=false"
)

is_complete() {
  local name="$1"
  local run_dir="${RUN_ROOT}/runs/${name}"
  [[ -f "${run_dir}/results/analysis_results.csv" && -f "${run_dir}/results/clustering_final/final_clustering_comparison.csv" ]]
}

run_variant() {
  local name="$1"
  shift
  if [[ "${SKIP_EXISTING}" == "1" ]] && is_complete "${name}"; then
    echo "Skipping ${name} (already complete)."
    return 0
  fi

  echo "Running ${name}..."
  "${BASE_CMD[@]}" \
    --output "${RUN_ROOT}/runs/${name}" \
    "$@" \
    > "${RUN_ROOT}/logs/${name}.log" 2>&1
}

# Paper row: MSE. Disable the phase-2 weighted regime by making warmup cover all epochs.
run_variant "01_plain_mse" \
  --param "warmup_epochs=120" \
  --param "rare_triplet_weight=0.0" \
  --param "adversarial_batch_weight=0.0" \
  --param "use_batch_conditioning=false" \
  --param "masking_rate=0.0"

# Paper row: Weighted MSE.
run_variant "02_weighted_mse" \
  --param "rare_triplet_weight=0.0" \
  --param "adversarial_batch_weight=0.0" \
  --param "use_batch_conditioning=false"

# Paper row: Weighted MSE + triplet.
run_variant "03_weighted_mse_triplet" \
  --param "adversarial_batch_weight=0.0" \
  --param "use_batch_conditioning=false"

# Paper row: Weighted MSE + DANN.
run_variant "04_weighted_mse_dann" \
  --param "rare_triplet_weight=0.0"

# Paper row: Weighted MSE + DANN + triplet (full stable_default).
run_variant "05_weighted_mse_dann_triplet_full"

# Paper row: full variant, single dynamic-weight update.
run_variant "06_weighted_mse_dann_triplet_single_update" \
  --param "dynamic_weight_update_interval=0"

# Paper row: full variant, density-only weighting (remove cluster-size component).
run_variant "07_weighted_mse_dann_triplet_density_only" \
  --param "cluster_weight_power=0.0"

# Paper row: full variant, cluster-size-only weighting (remove density component).
run_variant "08_weighted_mse_dann_triplet_cluster_only" \
  --param "density_weight_power=0.0"

"${PYTHON_BIN}" "${RUN_ROOT}/summarize_stable_default_mse_ablations.py" "${RUN_ROOT}" \
  > "${RUN_ROOT}/logs/summary.log" 2>&1 || true

echo "Completed stable_default MSE ablation launcher."
echo "RUN_ROOT=${RUN_ROOT}"
