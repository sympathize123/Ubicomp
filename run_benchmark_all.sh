#!/bin/bash
# Script to run full benchmark for all datasets and models

# Datasets
datasets=("D-1" "D-2" "D-3")

# Models
models=("XGB" "LGB" "MLP" "ResNet" "DANN")

print("Starting Full Benchmark Run...")

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        echo "------------------------------------------------"
        echo "Running Benchmark: Dataset=$dataset, Model=$model"
        echo "------------------------------------------------"
        python3 execute_benchmark.py --dataset "$dataset" --model "$model"
    done
done

print("Benchmark Completed. Results saved to results/benchmark_results.csv")
