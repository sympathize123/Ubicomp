#!/bin/bash

# Datasets
DATASETS=("D-1" "D-2" "D-3")

# Target Labels (Feel free to modify this list as needed)
# "stress_binary" is the default original label, removed per request
LABELS=("valence" "arousal" "disturbance")

# Models (Excluding Skipped: SAINT, Removed: FastFormer, Perceiver)
MODELS=(
    "XGB" "LGB" "MLP" "ResNet"
    "TabNet" "TabTransformer" "NODE" "DCN"
    "DANN" "CDAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST"
    "IRM" "VREx" "GroupDRO" "MixStyle" "ERM_DG" "MLDG" "MASF" "Fish" "CSD" "SagNet"
)

# Output directory
mkdir -p results_final

echo "Starting Full Benchmark Suite (hpo_trials=5)"

for label in "${LABELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        
        # Avoid running D-1 and D-2 with D-3 specific labels if they aren't available
        # (e.g. happy/angry only exist in D-3). You can customize this logic.
        if [[ "$label" == "happy" || "$label" == "angry" ]]; then
            if [[ "$dataset" != "D-3" ]]; then
                continue
            fi
        fi
        
        for model in "${MODELS[@]}"; do
            echo "[$(date)] Running Model: $model on Dataset: $dataset for Label: $label"
            
            # Clean command with 5 HPO trials
            CMD="python3 execute_benchmark.py --label $label --dataset $dataset --model $model --hpo_trials 5"
            
            # Log file incorporates the label
            LOG_FILE="results_final/${dataset}_${label}_${model}.log"
            
            # Execute
            $CMD 2>&1 | tee "$LOG_FILE"
        done
    done
done

echo "All experiments completed."
