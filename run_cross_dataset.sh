#!/bin/bash
set -euo pipefail

# Cross-dataset benchmark driver.
#
# Usage:
#   bash run_cross_dataset.sh [RUN_SETTING] [HPO_TRIALS]
# Examples:
#   bash run_cross_dataset.sh
#   bash run_cross_dataset.sh all 5
#   bash run_cross_dataset.sh two_to_one 5

RUN_SETTING="${1:-all}"
HPO_TRIALS="${2:-5}"

COMMON_LABELS=("arousal" "disturbance" "valence" "stress_binary")

baselines=("XGB" "LGB" "MLP" "ResNet")

tabular_dl=("TabNet" "SAINT" "TabTransformer" "FTTransformer" "DCN" "AutoInt")

dg_models=("IRM" "VREx" "GroupDRO" "MixStyle" "MLDG" "MASF" "Fish" "CSD" "SagNet")

da_models=("CDAN" "DAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST" "CGDM") #DANN

backbones=("MLP")

output_dir="results"
mkdir -p "$output_dir"

is_finished() {
    local label="$1" model="$2" backbone="$3"
    [[ -f "${output_dir}/cross_dataset_${label}_${model}_${backbone}_${RUN_SETTING}.csv" ]]
}

run_experiment() {
    local label="$1" model="$2" backbone="$3"
    shift 3
    local extra_args=("$@")

    local out="${output_dir}/cross_dataset_${label}_${model}_${backbone}_${RUN_SETTING}.csv"

    if is_finished "$label" "$model" "$backbone"; then
        echo "[SKIP] label=${label}, model=${model}, backbone=${backbone} — output already exists"
        return
    fi

    echo "========================================="
    echo "[RUN] label=${label}, model=${model}, backbone=${backbone}, setting=${RUN_SETTING}"
    echo "========================================="
    python3 execute_cross_dataset.py \
        --label "${label}" \
        --mode run \
        --run_setting "${RUN_SETTING}" \
        --model "${model}" \
        --backbone "${backbone}" \
        --hpo_trials "${HPO_TRIALS}" \
        --hpo_mode single_split \
        --output "${out}" \
        "${extra_args[@]}"
}

# Feature analysis (once per label)
for label in "${COMMON_LABELS[@]}"; do
    feature_report="${output_dir}/cross_dataset_feature_report_${label}.csv"
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

# echo ""
# echo "========================================="
# echo "CATEGORY 1: Standard Baselines"
# echo "========================================="
# for label in "${COMMON_LABELS[@]}"; do
#     for model in "${baselines[@]}"; do
#         run_experiment "$label" "$model" "MLP"
#     done
# done

# echo ""
# echo "========================================="
# echo "CATEGORY 2: Tabular DL"
# echo "========================================="
# for label in "${COMMON_LABELS[@]}"; do
#     for model in "${tabular_dl[@]}"; do
#         extra=()
#         case "$model" in
#             SAINT|TabTransformer|FTTransformer) extra=("--efficient_attention") ;;
#         esac
#         run_experiment "$label" "$model" "MLP" "${extra[@]}"
#     done
# done

# echo ""
# echo "========================================="
# echo "CATEGORY 3: Domain Generalization"
# echo "========================================="
# for label in "${COMMON_LABELS[@]}"; do
#     for model in "${dg_models[@]}"; do
#         for backbone in "${backbones[@]}"; do
#             run_experiment "$label" "$model" "$backbone"
#         done
#     done
# done

echo ""
echo "========================================="
echo "CATEGORY 4: Domain Adaptation (UDA)"
echo "========================================="
for label in "${COMMON_LABELS[@]}"; do
    for model in "${da_models[@]}"; do
        for backbone in "${backbones[@]}"; do
            run_experiment "$label" "$model" "$backbone" "--uda"
        done
    done
done

echo ""
echo "Done. Results saved to ${output_dir}/"