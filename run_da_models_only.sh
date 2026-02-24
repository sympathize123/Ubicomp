# Script to run ONLY Domain Adaptation models for timing experiments
# Datasets
datasets=("D-1" "D-2" "D-3")
# DA Models Only (9 models)
da_models=(
    "DANN" 
    "CDAN" 
    "DAN"
    "DeepCORAL" 
    "MCC" 
    "ADDA" 
    "MCD" 
    "JAN" 
    "SHOT" 
    "CBST"
    "CGDM"
)
# Backbones for DA models
backbones=("MLP" "ResNet" "Transformer")

# HPO Configuration - Using 20 trials as per DomainBed standard
HPO_TRIALS=20

echo "Starting DA Models Benchmark Run..."
echo "Models: ${da_models[@]}"
echo "Datasets: ${datasets[@]}"
echo "Backbones: ${backbones[@]}"
echo "HPO Trials: $HPO_TRIALS per model (DomainBed standard)"
echo "Total experiments: $((${#da_models[@]} * ${#datasets[@]} * ${#backbones[@]}))"
echo ""
for dataset in "${datasets[@]}"; do
    for model in "${da_models[@]}"; do
        for backbone in "${backbones[@]}"; do
            echo "------------------------------------------------"
            echo "Running: Dataset=$dataset, Model=$model, Backbone=$backbone"
            echo "HPO Trials: $HPO_TRIALS"
            echo "------------------------------------------------"
            python3 execute_benchmark.py --dataset "$dataset" --model "$model" --backbone "$backbone" --epochs 50  --hpo_trials $HPO_TRIALS --uda
            echo ""
        done
    done
done
echo "DA Models Benchmark Completed!"
echo "Performance results saved to: results/benchmark_results_da_hpo.csv"
echo "Timing results saved to: results/timing_results_da_hpo.csv"

echo "Summary:"
echo "- HPO Trials: $HPO_TRIALS per model (following DomainBed protocol)"
echo "- Total experiments: $((${#da_models[@]} * ${#datasets[@]} * ${#backbones[@]}))"
echo "- Each model tuned optimal hyperparameters via Optuna"
