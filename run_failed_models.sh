#!/bin/bash

DATASETS=("D-1" "D-2" "D-3")
MODELS=("NODE" "TabTransformer")

echo "Re-running failed models with HPO fix..."

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "[$(date)] Running Model: $model on Dataset: $dataset"
        CMD="python3 execute_benchmark.py --dataset $dataset --model $model --hpo_trials 5"
        LOG_FILE="results_final/${dataset}_${model}.log"
        $CMD 2>&1 | tee "$LOG_FILE"
    done
done

echo "Failed models re-run completed."
