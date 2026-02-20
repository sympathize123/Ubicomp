#!/bin/bash
set -euo pipefail

cd /home/iclab/tomiris/Ubicomp

mkdir -p results
out_results=results/dg_benchmark_results.csv
out_time=results/dg_benchmark_timing.csv

# CSV header for timing
printf "Dataset,Model,Backbone,Epochs,ElapsedSeconds\n" > "$out_time"

datasets=(D-1 D-2 D-3)
backbones=(MLP ResNet Transformer)
models=(ERM_DG IRM VREx GroupDRO MixStyle MLDG MASF)

echo "Starting DG benchmark runs..."

for dataset in "${datasets[@]}"; do
  for backbone in "${backbones[@]}"; do
    for model in "${models[@]}"; do
      echo "Running dataset=$dataset model=$model backbone=$backbone"
      /usr/bin/time -f "${dataset},${model},${backbone},50,%e" \
        -o "$out_time" -a \
        python3 execute_benchmark.py \
          --dataset "$dataset" \
          --model "$model" \
          --backbone "$backbone" \
          --epochs 50 \
          --output "$out_results"
    done
  done
 done

echo "DG benchmark completed."
