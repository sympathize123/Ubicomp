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

COMMON_LABELS=("stress_binary") #"arousal" "disturbance" "valence"
ALL_MODELS=(
  "XGB" "LGB" "MLP" "ResNet"
  #"DANN" "CDAN" "DAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST" 
  #"CGDM"
  #"TabNet" 
  #"SAINT" 
  #"TabTransformer" "FTTransformer" "DCN"
  #"IRM" "VREx" "GroupDRO" "MixStyle" "ERM_DG" "MLDG" "Fish" "CSD" "SagNet" "MASF" 
)
ALL_BACKBONES=("MLP" "ResNet" "Transformer")
BACKBONE_AWARE_MODELS=(
  "DANN" "CDAN" "DAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST" "CGDM"
  "IRM" "VREx" "GroupDRO" "MixStyle" "ERM_DG" "MLDG" "MASF" "Fish" "CSD" "SagNet"
)

is_backbone_aware() {
  local model="$1"
  for m in "${BACKBONE_AWARE_MODELS[@]}"; do
    [[ "$m" == "$model" ]] && return 0
  done
  return 1
}

is_finished() {
  local label="$1" model="$2" bb="$3"
  local output_file="results/cross_dataset_${label}_${model}_${bb}_${RUN_SETTING}.csv"
  [[ -f "$output_file" ]] && return 0
  return 1
}

if [[ "$MODEL" == "ALL" ]]; then
  MODELS_TO_RUN=("${ALL_MODELS[@]}")
else
  MODELS_TO_RUN=("$MODEL")
fi

mkdir -p results

for label in "${COMMON_LABELS[@]}"; do
  feature_report="results/cross_dataset_feature_report_${label}.csv"
  if [[ -f "$feature_report" ]]; then
    echo "[SKIP ANALYZE] label=${label} — feature report already exists"
    continue
  fi
  echo "========================================="
  echo "[ANALYZE] label=${label}"
  echo "========================================="
  python3 execute_cross_dataset.py \
    --label "${label}" \
    --mode analyze \
    --feature_report "$feature_report"
done

TOTAL_JOBS=0
SKIPPED_JOBS=0
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
      TOTAL_JOBS=$((TOTAL_JOBS + 1))
      if is_finished "$label" "$model" "$bb"; then
        SKIPPED_JOBS=$((SKIPPED_JOBS + 1))
      fi
    done
  done
done

echo "========================================="
echo "[INFO] Total jobs: ${TOTAL_JOBS} | Already finished (will skip): ${SKIPPED_JOBS} | To run: $((TOTAL_JOBS - SKIPPED_JOBS))"
echo "========================================="

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

      if is_finished "$label" "$model" "$bb"; then
        echo "[SKIP ${JOB_IDX}/${TOTAL_JOBS}] label=${label}, model=${model}, backbone=${bb} — output already exists"
        continue
      fi

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