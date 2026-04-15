#!/bin/bash
# Resume-aware cross-dataset benchmark driver across the full supported model matrix.
#
# Usage:
#   bash run_cross_dataset.sh [RUN_SETTING] [HPO_TRIALS]

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
RUN_SETTING="${1:-${RUN_SETTING:-all}}"
HPO_TRIALS="${2:-${HPO_TRIALS:-30}}"
SEEDS_STR="${SEEDS:-42}"
BACKBONES_STR="${BACKBONES:-MLP}"
EXCLUDED_MODELS_STR="${EXCLUDED_MODELS:-AutoInt}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
FEATURE_DIR="${FEATURE_DIR:-$OUTPUT_DIR}"

read -r -a SEED_LIST <<< "$SEEDS_STR"
read -r -a BACKBONE_LIST <<< "$BACKBONES_STR"
read -r -a EXCLUDED_MODELS <<< "$EXCLUDED_MODELS_STR"

COMMON_LABELS=("arousal" "disturbance" "valence" "stress_binary")

BASELINES=("XGB" "LGB" "MLP" "ResNet")
TABULAR_DL=("TabNet" "SAINT" "TabTransformer" "FTTransformer" "DCN")
DG_MODELS=("ERM_DG" "IRM" "VREx" "GroupDRO" "MixStyle" "MLDG" "MASF" "Fish" "CSD" "SagNet")
DA_MODELS=("DANN" "CDAN" "DAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST" "CGDM")

mkdir -p "$OUTPUT_DIR" "$FEATURE_DIR"

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

needs_eff() {
    local model="$1"
    [ "$model" = "SAINT" ] || [ "$model" = "TabTransformer" ] || [ "$model" = "FTTransformer" ]
}

is_excluded() {
    local model="$1"
    [ "${#EXCLUDED_MODELS[@]}" -gt 0 ] && in_list "$model" "${EXCLUDED_MODELS[@]}"
}

expected_settings() {
    case "$RUN_SETTING" in
        two_to_one) echo 3 ;;
        one_to_one) echo 6 ;;
        all) echo 9 ;;
        *)
            echo "Unsupported RUN_SETTING: $RUN_SETTING" >&2
            exit 1
            ;;
    esac
}

is_complete_output() {
    local output_csv="$1"
    local expected_row_count
    local expected_setting_count

    [ -s "$output_csv" ] || return 1

    expected_setting_count="$(expected_settings)"
    expected_row_count=$(( expected_setting_count * ${#SEED_LIST[@]} ))

    "$PYTHON_BIN" - "$output_csv" "$expected_row_count" "$expected_setting_count" "${#SEED_LIST[@]}" <<'PY'
import csv
import sys

path, expected_rows, expected_settings, expected_seed_count = sys.argv[1:]
expected_rows = int(expected_rows)
expected_settings = int(expected_settings)
expected_seed_count = int(expected_seed_count)

rows = 0
settings = set()
seeds = set()
with open(path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows += 1
        settings.add((row.get("Train_Datasets"), row.get("Test_Dataset")))
        seed = (row.get("Seed") or "").strip()
        if seed:
            seeds.add(seed)

sys.exit(0 if rows >= expected_rows and len(settings) >= expected_settings and len(seeds) >= expected_seed_count else 1)
PY
}

analyze_label() {
    local label="$1"
    local feature_report="${FEATURE_DIR}/cross_dataset_feature_report_${label}.csv"
    if [ -s "$feature_report" ]; then
        echo "[SKIP ANALYZE] label=$label"
        return 0
    fi
    echo "========================================="
    echo "[ANALYZE] label=$label"
    echo "========================================="
    "$PYTHON_BIN" execute_cross_dataset.py \
        --label "$label" \
        --mode analyze \
        --feature_report "$feature_report"
}

run_model() {
    local label="$1"
    local model="$2"
    local backbone="${3:-MLP}"
    local output_csv="${OUTPUT_DIR}/cross_dataset_${label}_${model}_${backbone}_${RUN_SETTING}.csv"
    local cmd=(
        "$PYTHON_BIN" execute_cross_dataset.py
        --label "$label"
        --mode run
        --run_setting "$RUN_SETTING"
        --model "$model"
        --backbone "$backbone"
        --hpo_trials "$HPO_TRIALS"
        --hpo_mode single_split
        --output "$output_csv"
        --seeds "${SEED_LIST[@]}"
    )

    if is_excluded "$model"; then
        echo "[SKIP EXCLUDED] label=$label model=$model backbone=$backbone"
        return 0
    fi

    if is_complete_output "$output_csv"; then
        echo "[SKIP COMPLETE] label=$label model=$model backbone=$backbone"
        return 0
    fi

    if needs_eff "$model"; then
        cmd+=(--efficient_attention)
    fi
    if is_da "$model"; then
        cmd+=(--uda)
    fi

    echo "========================================="
    echo "[RUN] label=$label model=$model backbone=$backbone setting=$RUN_SETTING seeds=${SEED_LIST[*]}"
    echo "========================================="
    "${cmd[@]}"
}

run_label_block() {
    local label="$1"
    local model
    local backbone

    analyze_label "$label"

    for model in "${BASELINES[@]}"; do
        run_model "$label" "$model" "MLP"
    done
    for model in "${TABULAR_DL[@]}"; do
        run_model "$label" "$model" "MLP"
    done
    for model in "${DG_MODELS[@]}"; do
        for backbone in "${BACKBONE_LIST[@]}"; do
            run_model "$label" "$model" "$backbone"
        done
    done
    for model in "${DA_MODELS[@]}"; do
        for backbone in "${BACKBONE_LIST[@]}"; do
            run_model "$label" "$model" "$backbone"
        done
    done
}

echo "Starting cross-dataset benchmark driver..."
echo "Run setting: $RUN_SETTING"
echo "HPO trials:  $HPO_TRIALS"
echo "Seeds:       ${SEED_LIST[*]}"
echo "Backbones:   ${BACKBONE_LIST[*]}"
echo "Output dir:  $OUTPUT_DIR"
echo "Expected source-target settings per model: $(expected_settings)"

for label in "${COMMON_LABELS[@]}"; do
    run_label_block "$label"
done

echo ""
echo "Done. Results saved under ${OUTPUT_DIR}/"
