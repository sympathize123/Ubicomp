#!/bin/bash
# Resume temporal baseline benchmark from progress CSV.
# Temporal setup:
# - Per-user chronological split (60/20/20)
# - Keep only first 30 days per user
# - Drop users that cannot satisfy temporal split/class diversity constraints
# - Concatenate all users' train/val/test slices globally

set -euo pipefail

SUMMARY_CSV="${SUMMARY_CSV:-results/benchmark_results_temporal_hpo.csv}"
PROGRESS_CSV="${PROGRESS_CSV:-results/benchmark_results_temporal_hpo_progress.csv}"
REQUIRED_FOLDS="${REQUIRED_FOLDS:-1}"
HPO_TRIALS="${HPO_TRIALS:-5}"
HPO_MODE="${HPO_MODE:-nested}"

# Requested temporal benchmark models.
# NOTE: NODE is currently not implemented in execute_benchmark.py.
models=("XGB" "LGB" "MLP" "ResNet" "TabNet" "SAINT" "TabTransformer" "FTTransformer" "NODE")

needs_eff() {
    local m="$1"
    [ "$m" == "SAINT" ] || [ "$m" == "TabTransformer" ] || [ "$m" == "FTTransformer" ]
}

is_supported() {
    local m="$1"
    [ "$m" != "NODE" ]
}

is_done() {
    local dataset="$1" label="$2" model="$3"
    local folds_progress=0
    local folds_summary=0

    if [ -f "$PROGRESS_CSV" ]; then
        folds_progress=$(python3 - "$PROGRESS_CSV" "$dataset" "$label" "$model" <<'PY'
import csv, sys
path, dataset, label, model = sys.argv[1:]
folds = set()
with open(path, newline='') as f:
    for row in csv.DictReader(f):
        if row.get('Phase') != 'final':
            continue
        if not (row.get('Dataset') == dataset and row.get('Label') == label and row.get('Model') == model):
            continue
        fold = (row.get('Fold') or '').strip()
        if fold:
            folds.add(fold)
print(len(folds))
PY
)
    fi

    if [ -f "$SUMMARY_CSV" ]; then
        folds_summary=$(python3 - "$SUMMARY_CSV" "$dataset" "$label" "$model" <<'PY'
import csv, sys
path, dataset, label, model = sys.argv[1:]
best = 0
with open(path, newline='') as f:
    for row in csv.DictReader(f):
        if not (row.get('Dataset') == dataset and row.get('Label') == label and row.get('Model') == model):
            continue
        try:
            n_folds = int(float(row.get('N_Folds', '0') or 0))
        except Exception:
            n_folds = 0
        if n_folds > best:
            best = n_folds
print(best)
PY
)
    fi

    [ "${folds_progress:-0}" -ge "$REQUIRED_FOLDS" ] || [ "${folds_summary:-0}" -ge "$REQUIRED_FOLDS" ]
}

run_model() {
    local dataset="$1" label="$2" model="$3"

    if ! is_supported "$model"; then
        echo "SKIP (unsupported in current codebase): Dataset=$dataset, Label=$label, Model=$model"
        return 0
    fi

    if is_done "$dataset" "$label" "$model"; then
        echo "SKIP (already ${REQUIRED_FOLDS} fold): Dataset=$dataset, Label=$label, Model=$model"
        return 0
    fi

    echo "================================================"
    echo "Running Temporal Baseline: Dataset=$dataset, Label=$label, Model=$model"
    echo "================================================"

    local extra_args=""
    if needs_eff "$model"; then
        extra_args="--efficient_attention"
    fi

    python3 execute_benchmark.py \
        --dataset "$dataset" \
        --label "$label" \
        --model "$model" \
        --split_strategy temporal \
        --temporal_train_ratio 0.6 \
        --temporal_val_ratio 0.2 \
        --temporal_drop_days 30 \
        --hpo_trials "$HPO_TRIALS" \
        --hpo_mode "$HPO_MODE" \
        --output "$SUMMARY_CSV" \
        $extra_args
}

run_dataset() {
    local dataset="$1"
    shift
    local labels=("$@")

    for label in "${labels[@]}"; do
        for model in "${models[@]}"; do
            run_model "$dataset" "$label" "$model"
        done
    done
}

# Common labels for D-1/D-2/D-3
# "stress" is mapped to stress_binary in current data files.
common_labels=("valence" "arousal" "stress_binary" "disturbance")

# D-3-specific labels
d3_specific_labels=("happy" "relaxed" "cheerful" "content" "sad" "anxious" "depressed" "angry")

echo "Starting temporal baseline resume run..."
echo "Summary CSV:  $SUMMARY_CSV"
echo "Progress CSV: $PROGRESS_CSV"
echo "Required folds per combo: $REQUIRED_FOLDS"

run_dataset "D-1" "${common_labels[@]}"
run_dataset "D-2" "${common_labels[@]}"
run_dataset "D-3" "${common_labels[@]}" "${d3_specific_labels[@]}"

echo ""
echo "=========================================="
echo "  TEMPORAL BENCHMARK RUN COMPLETED"
echo "=========================================="
