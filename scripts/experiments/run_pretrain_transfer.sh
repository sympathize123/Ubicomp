#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

DEFAULT_ARGS=(
  --fine-tune-ratios 0.0
  --model-types dl_mldg dl_siamese
  --seed 42
  --split-strategy stratified_group_kfold
  --scenarios "GLOBEM+D-2->D-3"
)

# 0.2 0.4 0.6
# recevied results dl_erm dl_clustering dl_reorder dl_masf dl_dann dl_irm dl_csd dl_mldg
# error dl_siamese

if [[ "$#" -eq 0 ]]; then
  args=("${DEFAULT_ARGS[@]}")
else
  args=("$@")
fi

models=()
base_args=()
freeze_backbone=false
timestamp="$(date +"%Y%m%d_%H%M%S")"
split_strategy="stratified_shuffle"

i=0
while [[ $i -lt ${#args[@]} ]]; do
  arg="${args[$i]}"
  if [[ "$arg" == "--freeze-backbone" ]]; then
    freeze_backbone=true
    i=$((i + 1))
    continue
  fi
  if [[ "$arg" == "--model-types" ]]; then
    i=$((i + 1))
    while [[ $i -lt ${#args[@]} && "${args[$i]}" != --* ]]; do
      models+=("${args[$i]}")
      i=$((i + 1))
    done
    continue
  fi
  if [[ "$arg" == --model-types=* ]]; then
    models_str="${arg#*=}"
    IFS=',' read -r -a parsed_models <<< "$models_str"
    if [[ ${#parsed_models[@]} -gt 0 ]]; then
      models+=("${parsed_models[@]}")
    fi
    i=$((i + 1))
    continue
  fi
  if [[ "$arg" == "--split-strategy" ]]; then
    split_strategy="${args[$((i + 1))]}"
    base_args+=("$arg" "$split_strategy")
    i=$((i + 2))
    continue
  fi
  if [[ "$arg" == --split-strategy=* ]]; then
    split_strategy="${arg#*=}"
    base_args+=("$arg")
    i=$((i + 1))
    continue
  fi
  if [[ "$arg" == "--output-csv" ]]; then
    i=$((i + 2))
    continue
  fi
  if [[ "$arg" == --output-csv=* ]]; then
    i=$((i + 1))
    continue
  fi
  base_args+=("$arg")
  i=$((i + 1))
done

if [[ ${#models[@]} -eq 0 ]]; then
  models=(tree transformer cdtrans)
fi

for model in "${models[@]}"; do
  output_dir="${ROOT_DIR}/results/${model}"
  mkdir -p "$output_dir"
  suffix="unfrozen"
  if [[ "$freeze_backbone" == "true" ]]; then
    suffix="frozen"
  fi
  output_csv="${output_dir}/domain_adaptation_results_${suffix}_${split_strategy}_${timestamp}.csv"
  python3 "${ROOT_DIR}/scripts/experiments/pretrain_transfer.py" \
    "${base_args[@]}" \
    $([[ "$freeze_backbone" == "true" ]] && echo "--freeze-backbone") \
    --model-types "$model" \
    --output-csv "$output_csv"
done
