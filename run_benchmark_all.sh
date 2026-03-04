#!/bin/bash
# Script to run full benchmark for all datasets and models

# Models Categories
# 1. Standard Baselines (Fixed Architecture)
baselines=("XGB" "LGB" "MLP" "ResNet")

# 2. Tabular DL (Fixed Architecture)
tabular_dl=("TabNet" "SAINT" "TabTransformer" "FTTransformer" "DCN" "AutoInt")

# 3. Domain Generalization & Adaptation (Backbone Agnostic)
dg_da_models=(
    # DG
    "IRM" "VREx" "GroupDRO" "MixStyle" "MLDG" "MASF" "Fish" "CSD" "SagNet"
    # DA
    "DANN" "CDAN" "DAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST" "CGDM"
)

# DA models (require --uda)
da_models=(
    "DANN" "CDAN" "DAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST" "CGDM"
)

# Backbones for DG/DA
backbones=("MLP")

echo "Starting Full Benchmark Run..."

# Per-dataset label lists (only labels with existing data files)
declare -A dataset_labels
dataset_labels["D-1"]="arousal disturbance valence"
dataset_labels["D-2"]="arousal disturbance valence"
dataset_labels["D-3"]="angry arousal disturbance happy valence"

for dataset in "D-1" "D-2" "D-3"; do
  read -ra labels <<< "${dataset_labels[$dataset]}"
  for label in "${labels[@]}"; do

    # 1. Run Baselines
    for model in "${baselines[@]}"; do
        echo "------------------------------------------------"
        echo "Running Baseline: Dataset=$dataset, Label=$label, Model=$model"
        echo "------------------------------------------------"
        python3 execute_benchmark.py --dataset "$dataset" --label "$label" --model "$model" --hpo_trials 5 --hpo_mode nested
    done

    # 2. Run Tabular DL
    for model in "${tabular_dl[@]}"; do
        echo "------------------------------------------------"
        echo "Running Tabular DL: Dataset=$dataset, Label=$label, Model=$model"
        echo "------------------------------------------------"
        use_eff=""
        if [ "$model" == "SAINT" ] || [ "$model" == "TabTransformer" ] || [ "$model" == "FTTransformer" ]; then
            use_eff="--efficient_attention"
        fi
        python3 execute_benchmark.py --dataset "$dataset" --label "$label" --model "$model" --hpo_trials 5 --hpo_mode nested $use_eff
    done

    # 3. Run DG/DA with Backbones
    for model in "${dg_da_models[@]}"; do
        for backbone in "${backbones[@]}"; do
            echo "------------------------------------------------"
            echo "Running DG/DA: Dataset=$dataset, Label=$label, Model=$model, Backbone=$backbone"
            echo "------------------------------------------------"
            use_uda=""
            for da_model in "${da_models[@]}"; do
                if [ "$model" == "$da_model" ]; then
                    use_uda="--uda"
                    break
                fi
            done
            python3 execute_benchmark.py --dataset "$dataset" --label "$label" --model "$model" --backbone "$backbone" --hpo_trials 5 --hpo_mode nested $use_uda
        done
    done

  done
done

echo "Benchmark Completed. Results saved to results/benchmark_results_da_hpo.csv"
echo "Progress results saved to results/benchmark_results_da_hpo_progress.csv"
