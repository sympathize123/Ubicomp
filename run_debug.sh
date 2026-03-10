#!/bin/bash
# Quick debug run: 1 HPO trial, 1 fold, 3 epochs per model
# Usage: bash run_debug.sh [dataset] [label]
# Defaults: dataset=D-1, label=arousal

set -u

DATASET="${1:-D-1}"
LABEL="${2:-arousal}"
DEBUG_FLAGS="--hpo_trials 1 --hpo_mode fold1 --max_folds 1 --epochs_override 3"
PROGRESS_CSV="${PROGRESS_CSV:-results/benchmark_results_da_hpo_progress.csv}"
REQUIRED_FOLDS="${REQUIRED_FOLDS:-1}"

echo "=== DEBUG RUN (resume): dataset=$DATASET label=$LABEL ==="
echo "Progress CSV: $PROGRESS_CSV (required folds per model: $REQUIRED_FOLDS)"

baselines=("XGB" "LGB" "MLP" "ResNet")
tabular_dl=("TabNet" "SAINT" "TabTransformer" "FTTransformer" "DCN" "AutoInt")
dg_models=("IRM" "VREx" "GroupDRO" "MixStyle" "MLDG" "MASF" "Fish" "CSD" "SagNet")
da_models=("DANN" "CDAN" "DAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST" "CGDM")

is_done() {
    local model="$1"
    local backbone="$2"

    if [ ! -f "$PROGRESS_CSV" ]; then
        return 1
    fi

    python3 - "$PROGRESS_CSV" "$DATASET" "$LABEL" "$model" "$backbone" "$REQUIRED_FOLDS" <<'PY'
import csv
import sys

path, dataset, label, model, backbone, required = sys.argv[1:]
required = int(required)
count = 0

with open(path, newline='') as f:
    for row in csv.DictReader(f):
        if row.get('Phase') != 'final':
            continue
        if (row.get('Dataset') == dataset and row.get('Label') == label and
            row.get('Model') == model and row.get('Backbone') == backbone):
            count += 1

sys.exit(0 if count >= required else 1)
PY
}

run_model() {
    local model=$1
    local backbone_key=$2
    shift 2

    if is_done "$model" "$backbone_key"; then
        echo "--- $model (backbone=$backbone_key) : SKIP (already done) ---"
        return 0
    fi

    echo "--- $model (backbone=$backbone_key) : RUN ---"
    python3 execute_benchmark.py --dataset "$DATASET" --label "$LABEL" --model "$model" $DEBUG_FLAGS "$@" \
        && echo "[$model] OK" \
        || echo "[$model] FAILED"
}

for model in "${baselines[@]}"; do
    run_model "$model" "MLP"
done

for model in "${tabular_dl[@]}"; do
    eff=""
    if [ "$model" == "SAINT" ] || [ "$model" == "TabTransformer" ] || [ "$model" == "FTTransformer" ]; then
        eff="--efficient_attention"
    fi
    run_model "$model" "MLP" $eff
done

for model in "${dg_models[@]}"; do
    run_model "$model" "MLP" --backbone MLP
done

for model in "${da_models[@]}"; do
    run_model "$model" "MLP" --backbone MLP --uda
done

echo "=== DEBUG DONE ==="
