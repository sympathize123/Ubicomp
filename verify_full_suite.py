import subprocess
import sys
import os

# List of all models to verify
MODELS = [
    # Baselines
    'XGB', 'LGB', 'MLP', 'ResNet', 
    # DL / Transformers
    'TabNet', 'TabTransformer', 'NODE', 'DCN',
    # DA
    'DANN', 'CDAN', 'DeepCORAL', 'MCC', 'ADDA', 'MCD', 'JAN', 'SHOT', 'CBST', 
    # DG
    'IRM', 'VREx', 'GroupDRO', 'MixStyle', 'ERM_DG', 'MLDG', 'MASF', 'Fish', 'CSD', 'SagNet'
]

# Additional configurations
UDA_MODELS = ['DANN'] # Add others if UDA enabled for them

def run_verification():
    failed_models = []
    passed_models = []
    
    print("Starting Full Suite Verification...")
    print(f"Total Models to Verify: {len(MODELS) + len(UDA_MODELS)}")
    
    # 1. Standard Verification (DG/Supervised Mode)
    for model in MODELS:
        print(f"\n[Verifying] {model} (Standard Mode)...")
        try:
            cmd = [
                "python3", "execute_benchmark.py",
                "--dataset", "D-3",
                "--model", model,
                "--hpo_trials", "0",
                "--epochs", "1",
                "--patience", "1",
                "--backbone", "MLP" # Default
            ]
            
            # Special case args
            # Special case args
            if model == 'TabTransformer':
                # User expects "Linear Attention" for TabTransformer -> Enable efficient_attention
                cmd.append("--efficient_attention")
            
            if model == 'NODE':
                # Dataset size 12929 % 24 = 17. Batch size 24 is safe and faster.
                cmd.append("--batch_size")
                cmd.append("24")
                
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"  -> PASSED")
                passed_models.append(model)
            else:
                print(f"  -> FAILED")
                print(f"  Error Output:\n{result.stderr[-500:]}")
                failed_models.append(model)
                
        except Exception as e:
            print(f"  -> EXCEPTION: {e}")
            failed_models.append(model)

    # 2. UDA Verification
    for model in UDA_MODELS:
        print(f"\n[Verifying] {model} (UDA Mode)...")
        try:
            cmd = [
                "python3", "execute_benchmark.py",
                "--dataset", "D-3",
                "--model", model,
                "--hpo_trials", "0",
                "--epochs", "1",
                "--patience", "1",
                "--backbone", "MLP",
                "--uda"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"  -> PASSED")
                passed_models.append(f"{model}_UDA")
            else:
                print(f"  -> FAILED")
                print(f"  Error Output:\n{result.stderr[-500:]}")
                failed_models.append(f"{model}_UDA")
                
        except Exception as e:
            print(f"  -> EXCEPTION: {e}")
            failed_models.append(f"{model}_UDA")

    print("\n------------------------------------------------")
    print("Verification Summary")
    print("------------------------------------------------")
    print(f"Passed: {len(passed_models)}")
    print(f"Failed: {len(failed_models)}")
    
    if failed_models:
        print("\nFailed Models:")
        for m in failed_models:
            print(f" - {m}")
        sys.exit(1)
    else:
        print("\nALL MODELS VERIFIED SUCCESSFULLY!")
        sys.exit(0)

if __name__ == "__main__":
    run_verification()
