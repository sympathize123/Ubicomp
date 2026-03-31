#!/usr/bin/env bash
set -euo pipefail
cd /home/iclab/minseo/Ubicomp
models=(SAINT TabTransformer FTTransformer)
datasets=(D-1 D-2 D-3)
common_labels=(arousal disturbance stress_binary valence)
d3_labels=(happy relaxed cheerful content sad anxious depressed angry)

run_one(){
  local dataset="$1" label="$2" model="$3"
  local eff=""
  if [ "$model" = "SAINT" ] || [ "$model" = "TabTransformer" ] || [ "$model" = "FTTransformer" ]; then
    eff="--efficient_attention"
  fi
  echo "=== RUN dataset=$dataset label=$label model=$model ==="
  python3 execute_benchmark.py \
    --dataset "$dataset" \
    --label "$label" \
    --model "$model" \
    --hpo_trials 5 \
    --hpo_mode nested \
    $eff \
    --output results/benchmark_results_da_hpo.csv
}

for d in "${datasets[@]}"; do
  for l in "${common_labels[@]}"; do
    for m in "${models[@]}"; do
      run_one "$d" "$l" "$m"
    done
  done
  if [ "$d" = "D-3" ]; then
    for l in "${d3_labels[@]}"; do
      for m in "${models[@]}"; do
        run_one "$d" "$l" "$m"
      done
    done
  fi
done
