#!/bin/bash
# Resume benchmark from progress CSV.
# Last crash point (from tmux log): D-1 / disturbance / AutoInt.
# AutoInt, TabPFN are excluded (OOM/instability).
# This script skips combos that already have >= 5 completed folds in:
# results/benchmark_results_da_hpo_progress.csv

set -euo pipefail

PROGRESS_CSV="results/benchmark_results_da_hpo_progress.csv"
REQUIRED_FOLDS=5

# DA models (require --uda)
da_models=("DANN" "CDAN" "DAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST" "CGDM")
excluded_models=("AutoInt" "TabPFN")

is_da() {
    local m="$1"
    for da in "${da_models[@]}"; do
        [ "$m" == "$da" ] && return 0
    done
    return 1
}

needs_eff() {
    local m="$1"
    [ "$m" == "SAINT" ] || [ "$m" == "TabTransformer" ] || [ "$m" == "FTTransformer" ]
}

is_excluded() {
    local m="$1"
    for x in "${excluded_models[@]}"; do
        [ "$m" == "$x" ] && return 0
    done
    return 1
}

is_done() {
    local dataset="$1" label="$2" model="$3" backbone="$4"
    if [ ! -f "$PROGRESS_CSV" ]; then
        return 1
    fi
    local folds
    folds=$(python3 - "$PROGRESS_CSV" "$dataset" "$label" "$model" "$backbone" <<'PY'
import csv, sys
path, dataset, label, model, backbone = sys.argv[1:]
count = 0
with open(path, newline='') as f:
    for row in csv.DictReader(f):
        if row.get('Phase') != 'final':
            continue
        if (row.get('Dataset') == dataset and row.get('Label') == label and
            row.get('Model') == model and row.get('Backbone') == backbone):
            count += 1
print(count)
PY
)
    [ "${folds:-0}" -ge "$REQUIRED_FOLDS" ]
}

run_model() {
    local dataset="$1" label="$2" model="$3" backbone="${4:-}"
    local backbone_key="${backbone:-MLP}"
    if is_excluded "$model"; then
        echo "SKIP (excluded): Dataset=$dataset, Label=$label, Model=$model, Backbone=$backbone_key"
        return 0
    fi
    if is_done "$dataset" "$label" "$model" "$backbone_key"; then
        echo "SKIP (already ${REQUIRED_FOLDS} folds): Dataset=$dataset, Label=$label, Model=$model, Backbone=$backbone_key"
        return 0
    fi
    echo "================================================"
    echo "Running: Dataset=$dataset, Label=$label, Model=$model, Backbone=$backbone"
    echo "================================================"
    local extra_args=""
    if needs_eff "$model"; then extra_args="--efficient_attention"; fi
    if is_da "$model"; then extra_args="$extra_args --uda"; fi
    if [ -n "$backbone" ]; then extra_args="$extra_args --backbone $backbone"; fi
    python3 execute_benchmark.py --dataset "$dataset" --label "$label" --model "$model" \
        --hpo_trials 5 --hpo_mode nested $extra_args
}

# ============================================================
# 1) D-1 / disturbance — resume from crash point
# ============================================================
# AutoInt excluded

# D-1 / disturbance — all DG/DA
for model in "IRM" "VREx" "GroupDRO" "MixStyle" "MLDG" "MASF" "Fish" "CSD" "SagNet" \
             "DANN" "CDAN" "DAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST" "CGDM"; do
    run_model "D-1" "disturbance" "$model" "MLP"
done

# ============================================================
# 2) D-1 / valence — ALL models
# ============================================================
for model in "XGB" "LGB" "MLP" "ResNet"; do
    run_model "D-1" "valence" "$model"
done
for model in "TabNet" "SAINT" "TabTransformer" "FTTransformer" "DCN"; do
    run_model "D-1" "valence" "$model"
done
for model in "IRM" "VREx" "GroupDRO" "MixStyle" "MLDG" "MASF" "Fish" "CSD" "SagNet" \
             "DANN" "CDAN" "DAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST" "CGDM"; do
    run_model "D-1" "valence" "$model" "MLP"
done

# ============================================================
# 3) D-2 — ALL labels × ALL models
# ============================================================
for label in "arousal" "disturbance" "valence"; do
    for model in "XGB" "LGB" "MLP" "ResNet"; do
        run_model "D-2" "$label" "$model"
    done
    for model in "TabNet" "SAINT" "TabTransformer" "FTTransformer" "DCN"; do
        run_model "D-2" "$label" "$model"
    done
    for model in "IRM" "VREx" "GroupDRO" "MixStyle" "MLDG" "MASF" "Fish" "CSD" "SagNet" \
                 "DANN" "CDAN" "DAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST" "CGDM"; do
        run_model "D-2" "$label" "$model" "MLP"
    done
done

# ============================================================
# 4) D-3 — ALL labels × ALL models
# ============================================================
for label in "angry" "arousal" "disturbance" "happy" "valence"; do
    for model in "XGB" "LGB" "MLP" "ResNet"; do
        run_model "D-3" "$label" "$model"
    done
    for model in "TabNet" "SAINT" "TabTransformer" "FTTransformer" "DCN"; do
        run_model "D-3" "$label" "$model"
    done
    for model in "IRM" "VREx" "GroupDRO" "MixStyle" "MLDG" "MASF" "Fish" "CSD" "SagNet" \
                 "DANN" "CDAN" "DAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST" "CGDM"; do
        run_model "D-3" "$label" "$model" "MLP"
    done
done

echo ""
echo "=========================================="
echo "  ALL REMAINING BENCHMARKS COMPLETED"
echo "=========================================="
