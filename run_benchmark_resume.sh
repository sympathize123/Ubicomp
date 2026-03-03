#!/bin/bash
# Resume benchmark from where it stopped.
# Completed: D-1/arousal (ALL models), D-1/disturbance (XGB,LGB,MLP,ResNet,TabNet,TabPFN,SAINT,TabTransformer)
# Remaining: D-1/disturbance (FTTransformer,DCN,AutoInt + all DG/DA), D-1/valence (ALL), D-2 (ALL), D-3 (ALL)

set -e

# DA models (require --uda)
da_models=("DANN" "CDAN" "DAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST" "CGDM")

is_da() {
    local m="$1"
    for da in "${da_models[@]}"; do
        [ "$m" == "$da" ] && return 0
    done
    return 1
}

needs_eff() {
    local m="$1"
    [ "$m" == "SAINT" ] || [ "$m" == "TabTransformer" ] || [ "$m" == "FTTransformer" ]
}

run_model() {
    local dataset="$1" label="$2" model="$3" backbone="${4:-}"
    echo "================================================"
    echo "Running: Dataset=$dataset, Label=$label, Model=$model, Backbone=$backbone"
    echo "================================================"
    local extra_args=""
    if needs_eff "$model"; then extra_args="--efficient_attention"; fi
    if is_da "$model"; then extra_args="$extra_args --uda"; fi
    if [ -n "$backbone" ]; then extra_args="$extra_args --backbone $backbone"; fi
    python3 execute_benchmark.py --dataset "$dataset" --label "$label" --model "$model" \
        --hpo_trials 5 --hpo_mode nested $extra_args
}

# ============================================================
# 1) D-1 / disturbance — remaining Tabular DL
# ============================================================
for model in "FTTransformer" "DCN" "AutoInt"; do
    run_model "D-1" "disturbance" "$model"
done

# D-1 / disturbance — all DG/DA
for model in "IRM" "VREx" "GroupDRO" "MixStyle" "MLDG" "MASF" "Fish" "CSD" "SagNet" \
             "DANN" "CDAN" "DAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST" "CGDM"; do
    run_model "D-1" "disturbance" "$model" "MLP"
done

# ============================================================
# 2) D-1 / valence — ALL models
# ============================================================
for model in "XGB" "LGB" "MLP" "ResNet"; do
    run_model "D-1" "valence" "$model"
done
for model in "TabNet" "TabPFN" "SAINT" "TabTransformer" "FTTransformer" "DCN" "AutoInt"; do
    run_model "D-1" "valence" "$model"
done
for model in "IRM" "VREx" "GroupDRO" "MixStyle" "MLDG" "MASF" "Fish" "CSD" "SagNet" \
             "DANN" "CDAN" "DAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST" "CGDM"; do
    run_model "D-1" "valence" "$model" "MLP"
done

# ============================================================
# 3) D-2 — ALL labels × ALL models
# ============================================================
for label in "arousal" "disturbance" "valence"; do
    for model in "XGB" "LGB" "MLP" "ResNet"; do
        run_model "D-2" "$label" "$model"
    done
    for model in "TabNet" "TabPFN" "SAINT" "TabTransformer" "FTTransformer" "DCN" "AutoInt"; do
        run_model "D-2" "$label" "$model"
    done
    for model in "IRM" "VREx" "GroupDRO" "MixStyle" "MLDG" "MASF" "Fish" "CSD" "SagNet" \
                 "DANN" "CDAN" "DAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST" "CGDM"; do
        run_model "D-2" "$label" "$model" "MLP"
    done
done

# ============================================================
# 4) D-3 — ALL labels × ALL models
# ============================================================
for label in "angry" "arousal" "disturbance" "happy" "valence"; do
    for model in "XGB" "LGB" "MLP" "ResNet"; do
        run_model "D-3" "$label" "$model"
    done
    for model in "TabNet" "TabPFN" "SAINT" "TabTransformer" "FTTransformer" "DCN" "AutoInt"; do
        run_model "D-3" "$label" "$model"
    done
    for model in "IRM" "VREx" "GroupDRO" "MixStyle" "MLDG" "MASF" "Fish" "CSD" "SagNet" \
                 "DANN" "CDAN" "DAN" "DeepCORAL" "MCC" "ADDA" "MCD" "JAN" "SHOT" "CBST" "CGDM"; do
        run_model "D-3" "$label" "$model" "MLP"
    done
done

echo ""
echo "=========================================="
echo "  ALL REMAINING BENCHMARKS COMPLETED"
echo "=========================================="
