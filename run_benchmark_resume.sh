#!/bin/bash
# Resume within-dataset benchmark runs across the full supported model matrix.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SUMMARY_CSV="${SUMMARY_CSV:-results/benchmark_results_da_hpo.csv}"
PROGRESS_CSV="${PROGRESS_CSV:-results/benchmark_results_da_hpo_progress.csv}"
REQUIRED_FOLDS="${REQUIRED_FOLDS:-5}"
HPO_TRIALS="${HPO_TRIALS:-30}"
HPO_MODE="${HPO_MODE:-fold1}"
SEEDS_STR="${SEEDS:-42}"
BACKBONES_STR="${BACKBONES:-MLP}"
EXCLUDED_MODELS_STR="${EXCLUDED_MODELS:-AutoInt}"

read -r -a SEED_LIST <<< "$SEEDS_STR"
read -r -a BACKBONE_LIST <<< "$BACKBONES_STR"
read -r -a EXCLUDED_MODELS <<< "$EXCLUDED_MODELS_STR"

COMMON_LABELS=("arousal" "disturbance" "stress_binary" "valence")
D3_EXTRA_LABELS=("angry" "anxious" "cheerful" "content" "depressed" "happy" "relaxed" "sad")

BASELINES=("XGB" "LGB" "MLP" "ResNet")
TABULAR_DL=("TabNet" "SAINT" "TabTransformer" "FTTransformer" "DCN")
DG_MODELS=("ERM_DG" "IRM" "VREx" "GroupDRO" "MixStyle" "MLDG" "MASF" "Fish" "CSD" "SagNet")
DA_MODELS=("DANN" "CDAN" "DAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST" "CGDM")

in_list() {
    local needle="$1"
    shift || true
    local item
    for item in "$@"; do
        [ "$needle" = "$item" ] && return 0
    done
    return 1
}

is_da() {
    in_list "$1" "${DA_MODELS[@]}"
}

requires_backbone() {
    in_list "$1" "${DG_MODELS[@]}" || in_list "$1" "${DA_MODELS[@]}"
}

needs_eff() {
    local model="$1"
    [ "$model" = "SAINT" ] || [ "$model" = "TabTransformer" ] || [ "$model" = "FTTransformer" ]
}

is_excluded() {
    local model="$1"
    [ "${#EXCLUDED_MODELS[@]}" -gt 0 ] && in_list "$model" "${EXCLUDED_MODELS[@]}"
}

expected_runs() {
    echo $(( REQUIRED_FOLDS * ${#SEED_LIST[@]} ))
}

is_done() {
    local dataset="$1"
    local label="$2"
    local model="$3"
    local backbone="$4"
    local required_runs
    local progress_runs=0
    local summary_runs=0
    local summary_seed_count=0

    required_runs="$(expected_runs)"

    if [ -f "$PROGRESS_CSV" ]; then
        progress_runs=$("$PYTHON_BIN" - "$PROGRESS_CSV" "$dataset" "$label" "$model" "$backbone" <<'PY'
import csv
import sys

path, dataset, label, model, backbone = sys.argv[1:]
runs = set()
with open(path, newline='') as f:
    for row in csv.DictReader(f):
        if row.get("Phase") != "final":
            continue
        if row.get("Dataset") != dataset or row.get("Label") != label or row.get("Model") != model:
            continue
        if (row.get("Backbone") or "MLP") != backbone:
            continue
        fold = (row.get("Fold") or "").strip()
        seed = (row.get("Seed") or "").strip()
        if fold and seed:
            runs.add((fold, seed))
print(len(runs))
PY
)
    fi

    if [ -f "$SUMMARY_CSV" ]; then
        read -r summary_runs summary_seed_count < <("$PYTHON_BIN" - "$SUMMARY_CSV" "$dataset" "$label" "$model" "$backbone" <<'PY'
import csv
import sys

path, dataset, label, model, backbone = sys.argv[1:]
best_runs = 0
best_seed_count = 0
with open(path, newline='') as f:
    for row in csv.DictReader(f):
        if row.get("Dataset") != dataset or row.get("Label") != label or row.get("Model") != model:
            continue
        if (row.get("Backbone") or "MLP") != backbone:
            continue
        try:
            n_runs = int(float(row.get("N_Runs", row.get("N_Folds", "0")) or 0))
        except Exception:
            n_runs = 0
        try:
            seed_count = int(float(row.get("Seed_Count", "0") or 0))
        except Exception:
            seed_count = 0
        if (n_runs, seed_count) > (best_runs, best_seed_count):
            best_runs = n_runs
            best_seed_count = seed_count
print(best_runs, best_seed_count)
PY
)
    fi

    [ "${progress_runs:-0}" -ge "$required_runs" ] || {
        [ "${summary_runs:-0}" -ge "$required_runs" ] && [ "${summary_seed_count:-0}" -ge "${#SEED_LIST[@]}" ]
    }
}

run_model() {
    local dataset="$1"
    local label="$2"
    local model="$3"
    local backbone="${4:-MLP}"
    local cmd=(
        "$PYTHON_BIN" execute_benchmark.py
        --dataset "$dataset"
        --label "$label"
        --model "$model"
        --hpo_trials "$HPO_TRIALS"
        --hpo_mode "$HPO_MODE"
        --output "$SUMMARY_CSV"
        --seeds "${SEED_LIST[@]}"
    )

    if is_excluded "$model"; then
        echo "SKIP (excluded): dataset=$dataset label=$label model=$model backbone=$backbone"
        return 0
    fi

    if is_done "$dataset" "$label" "$model" "$backbone"; then
        echo "SKIP (complete): dataset=$dataset label=$label model=$model backbone=$backbone"
        return 0
    fi

    if needs_eff "$model"; then
        cmd+=(--efficient_attention)
    fi
    if is_da "$model"; then
        cmd+=(--uda)
    fi
    if requires_backbone "$model"; then
        cmd+=(--backbone "$backbone")
    fi

    echo "================================================"
    echo "Running: dataset=$dataset label=$label model=$model backbone=$backbone seeds=${SEED_LIST[*]}"
    echo "================================================"
    "${cmd[@]}"
}

run_label_block() {
    local dataset="$1"
    shift
    local labels=("$@")
    local label
    local model
    local backbone

    for label in "${labels[@]}"; do
        for model in "${BASELINES[@]}"; do
            run_model "$dataset" "$label" "$model"
        done
        for model in "${TABULAR_DL[@]}"; do
            run_model "$dataset" "$label" "$model"
        done
        for model in "${DG_MODELS[@]}"; do
            for backbone in "${BACKBONE_LIST[@]}"; do
                run_model "$dataset" "$label" "$model" "$backbone"
            done
        done
        for model in "${DA_MODELS[@]}"; do
            for backbone in "${BACKBONE_LIST[@]}"; do
                run_model "$dataset" "$label" "$model" "$backbone"
            done
        done
    done
}

echo "Starting within-dataset benchmark resume run..."
echo "Summary CSV:  $SUMMARY_CSV"
echo "Progress CSV: $PROGRESS_CSV"
echo "Seeds:        ${SEED_LIST[*]}"
echo "Backbones:    ${BACKBONE_LIST[*]}"
echo "HPO:          trials=$HPO_TRIALS mode=$HPO_MODE"
echo "Required runs per combo: $(expected_runs)"

run_label_block "D-1" "${COMMON_LABELS[@]}"
run_label_block "D-2" "${COMMON_LABELS[@]}"
run_label_block "D-3" "${COMMON_LABELS[@]}" "${D3_EXTRA_LABELS[@]}"

echo ""
echo "=========================================="
echo "  WITHIN-DATASET BENCHMARK RUN COMPLETED"
echo "=========================================="
