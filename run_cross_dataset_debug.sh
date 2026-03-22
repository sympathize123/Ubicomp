#!/bin/bash
# Quick cross-dataset debug run (fast smoke checks across models).
# Usage:
#   bash run_cross_dataset_debug.sh [label|ALL] [run_setting] [backbone]
# Defaults:
#   label=arousal, run_setting=two_to_one, backbone=MLP

set -u

LABEL_ARG="${1:-arousal}"
RUN_SETTING="${2:-two_to_one}"
BACKBONE="${3:-MLP}"

HPO_TRIALS="${HPO_TRIALS:-1}"
LIMIT_EXPERIMENTS="${LIMIT_EXPERIMENTS:-1}"
EPOCHS_OVERRIDE="${EPOCHS_OVERRIDE:-3}"

OUTPUT_DIR="${OUTPUT_DIR:-results/cross_dataset_debug}"
FEATURE_DIR="${FEATURE_DIR:-$OUTPUT_DIR/feature_reports}"

COMMON_LABELS=("arousal" "disturbance" "valence")

if [ "$LABEL_ARG" = "ALL" ]; then
    LABELS=("${COMMON_LABELS[@]}")
else
    LABELS=("$LABEL_ARG")
fi

baselines=("XGB" "LGB" "MLP" "ResNet")
tabular_dl=("TabNet" "SAINT" "TabTransformer" "FTTransformer" "DCN" "AutoInt")
dg_models=("IRM" "VREx" "GroupDRO" "MixStyle" "ERM_DG" "MLDG" "MASF" "Fish" "CSD" "SagNet")
da_models=("DANN" "CDAN" "DAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST" "CGDM")

mkdir -p "$OUTPUT_DIR" "$FEATURE_DIR"

echo "=== CROSS-DATASET DEBUG RUN ==="
echo "Labels: ${LABELS[*]}"
echo "Run setting: $RUN_SETTING"
echo "Backbone: $BACKBONE"
echo "Debug flags: hpo_trials=$HPO_TRIALS limit_experiments=$LIMIT_EXPERIMENTS epochs_override=$EPOCHS_OVERRIDE"
echo "Output dir: $OUTPUT_DIR"

analyze_label() {
    local label="$1"
    local report="$FEATURE_DIR/cross_dataset_feature_report_${label}.csv"
    echo ""
    echo "[ANALYZE] label=$label"
    python3 execute_cross_dataset.py \
        --label "$label" \
        --mode analyze \
        --feature_report "$report"
}

run_model() {
    local label="$1"
    local model="$2"
    shift 2

    local output_csv="$OUTPUT_DIR/cross_dataset_debug_${label}_${model}_${BACKBONE}_${RUN_SETTING}.csv"
    if [ -s "$output_csv" ]; then
        echo "--- $label | $model : SKIP (already exists: $output_csv) ---"
        return 0
    fi

    echo "--- $label | $model : RUN ---"
    python3 execute_cross_dataset.py \
        --label "$label" \
        --mode run \
        --run_setting "$RUN_SETTING" \
        --limit_experiments "$LIMIT_EXPERIMENTS" \
        --model "$model" \
        --backbone "$BACKBONE" \
        --hpo_trials "$HPO_TRIALS" \
        --epochs_override "$EPOCHS_OVERRIDE" \
        --output "$output_csv" \
        "$@" \
        && echo "[$label/$model] OK" \
        || echo "[$label/$model] FAILED"
}

for label in "${LABELS[@]}"; do
    analyze_label "$label"

    for model in "${baselines[@]}"; do
        run_model "$label" "$model"
    done

    for model in "${tabular_dl[@]}"; do
        eff=""
        if [ "$model" = "SAINT" ] || [ "$model" = "TabTransformer" ] || [ "$model" = "FTTransformer" ]; then
            eff="--efficient_attention"
        fi
        run_model "$label" "$model" $eff
    done

    for model in "${dg_models[@]}"; do
        run_model "$label" "$model"
    done

    for model in "${da_models[@]}"; do
        run_model "$label" "$model"
    done
done

echo "=== CROSS-DATASET DEBUG DONE ==="
