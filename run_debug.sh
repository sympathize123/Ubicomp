#!/bin/bash
# Quick debug run: 1 HPO trial, 1 fold, 3 epochs per model
# Usage: bash run_debug.sh [dataset] [label]
# Defaults: dataset=D-3, label=stress_binary

DATASET="${1:-D-1}"
LABEL="${2:-arousal}"
DEBUG_FLAGS="--hpo_trials 1 --hpo_mode fold1 --max_folds 1 --epochs_override 3"

echo "=== DEBUG RUN: dataset=$DATASET label=$LABEL ==="

baselines=("XGB" "LGB" "MLP" "ResNet")
tabular_dl=("TabNet" "SAINT" "TabTransformer" "FTTransformer" "DCN" "AutoInt")
dg_models=("IRM" "VREx" "GroupDRO" "MixStyle" "MLDG" "MASF" "Fish" "CSD" "SagNet")
da_models=("DANN" "CDAN" "DAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST" "CGDM")

run_model() {
    local model=$1
    shift
    echo "--- $model ---"
    python3 execute_benchmark.py --dataset "$DATASET" --label "$LABEL" --model "$model" $DEBUG_FLAGS "$@" \
        && echo "[$model] OK" \
        || echo "[$model] FAILED"
}

for model in "${baselines[@]}"; do
    run_model "$model"
done

for model in "${tabular_dl[@]}"; do
    eff=""
    if [ "$model" == "SAINT" ] || [ "$model" == "TabTransformer" ] || [ "$model" == "FTTransformer" ]; then
        eff="--efficient_attention"
    fi
    run_model "$model" $eff
done

for model in "${dg_models[@]}"; do
    run_model "$model" --backbone MLP
done

for model in "${da_models[@]}"; do
    run_model "$model" --backbone MLP --uda
done

echo "=== DEBUG DONE ==="
