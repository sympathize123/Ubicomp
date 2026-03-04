#!/bin/bash
set -euo pipefail

# Usage:
#   bash run_cross_dataset.sh [MODEL] [BACKBONE]
# Example:
#   bash run_cross_dataset.sh XGB MLP

MODEL="${1:-XGB}"
BACKBONE="${2:-MLP}"

labels=("arousal" "disturbance" "valence")

for label in "${labels[@]}"; do
  echo "========================================="
  echo "[ANALYZE] label=${label}"
  echo "========================================="
  python3 execute_cross_dataset.py \
    --label "${label}" \
    --mode analyze \
    --feature_report "results/cross_dataset_feature_report_${label}.csv"

done

for label in "${labels[@]}"; do
  echo "========================================="
  echo "[RUN] label=${label}, model=${MODEL}, backbone=${BACKBONE}"
  echo "========================================="
  python3 execute_cross_dataset.py \
    --label "${label}" \
    --mode run \
    --run_setting all \
    --model "${MODEL}" \
    --backbone "${BACKBONE}" \
    --output "results/cross_dataset_${label}_${MODEL}.csv"

done

echo "Done. Outputs are under results/."
