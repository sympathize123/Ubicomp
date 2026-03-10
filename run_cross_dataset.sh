#!/bin/bash
set -euo pipefail

# Cross-dataset benchmark driver.
# Runs every valid cross-dataset direction for every label shared by D-1, D-2, D-3.
#
# Usage:
#   bash run_cross_dataset.sh [MODEL|ALL] [BACKBONE|ALL] [RUN_SETTING] [HPO_TRIALS]
# Examples:
#   bash run_cross_dataset.sh
#   bash run_cross_dataset.sh XGB MLP all 5
#   bash run_cross_dataset.sh ALL ALL all 5
#   bash run_cross_dataset.sh DANN MLP two_to_one 5
#
# Notes:
#   - Cross-dataset evaluation only supports labels common to all datasets.
#   - In the current pipeline those labels are: arousal, disturbance, valence.

MODEL="${1:-ALL}"
BACKBONE="${2:-MLP}"
RUN_SETTING="${3:-all}"
HPO_TRIALS="${4:-5}"

COMMON_LABELS=("arousal" "disturbance" "valence")

ALL_MODELS=(
  "XGB" "LGB" "MLP" "ResNet"
  "DANN" "CDAN" "DAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST" "CGDM"
  "TabNet" "SAINT" "TabTransformer" "FTTransformer" "DCN" "AutoInt"
  "IRM" "VREx" "GroupDRO" "MixStyle" "ERM_DG" "MLDG" "MASF" "Fish" "CSD" "SagNet"
)

ALL_BACKBONES=("MLP")
BACKBONE_AWARE_MODELS=(
  "DANN" "CDAN" "DAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST" "CGDM"
  "IRM" "VREx" "GroupDRO" "MixStyle" "ERM_DG" "MLDG" "MASF" "Fish" "CSD" "SagNet"
)

is_backbone_aware() {
  local model="$1"
  for m in "${BACKBONE_AWARE_MODELS[@]}"; do
    if [[ "$m" == "$model" ]]; then
      return 0
    fi
  done
  return 1
}

if [[ "$MODEL" == "ALL" ]]; then
  MODELS_TO_RUN=("${ALL_MODELS[@]}")
else
  MODELS_TO_RUN=("$MODEL")
fi

mkdir -p results

for label in "${COMMON_LABELS[@]}"; do
  echo "========================================="
  echo "[ANALYZE] label=${label}"
  echo "========================================="
  python3 execute_cross_dataset.py \
    --label "${label}" \
    --mode analyze \
    --feature_report "results/cross_dataset_feature_report_${label}.csv"
done

TOTAL_JOBS=0
for model in "${MODELS_TO_RUN[@]}"; do
  if [[ "$BACKBONE" == "ALL" ]]; then
    if is_backbone_aware "$model"; then
      backbones=("${ALL_BACKBONES[@]}")
    else
      backbones=("MLP")
    fi
  else
    backbones=("$BACKBONE")
  fi
  TOTAL_JOBS=$((TOTAL_JOBS + ${#backbones[@]} * ${#COMMON_LABELS[@]}))
done

JOB_IDX=0
for model in "${MODELS_TO_RUN[@]}"; do
  if [[ "$BACKBONE" == "ALL" ]]; then
    if is_backbone_aware "$model"; then
      backbones=("${ALL_BACKBONES[@]}")
    else
      backbones=("MLP")
    fi
  else
    backbones=("$BACKBONE")
  fi

  for bb in "${backbones[@]}"; do
    for label in "${COMMON_LABELS[@]}"; do
      JOB_IDX=$((JOB_IDX + 1))
      echo "========================================="
      echo "[RUN ${JOB_IDX}/${TOTAL_JOBS}] label=${label}, model=${model}, backbone=${bb}, setting=${RUN_SETTING}"
      echo "========================================="
      python3 execute_cross_dataset.py \
        --label "${label}" \
        --mode run \
        --run_setting "${RUN_SETTING}" \
        --model "${model}" \
        --backbone "${bb}" \
        --hpo_trials "${HPO_TRIALS}" \
        --output "results/cross_dataset_${label}_${model}_${bb}_${RUN_SETTING}.csv"
    done
  done
done

echo "Done. Outputs are under results/."
