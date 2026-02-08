import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import argparse
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import sys
from src.data_loader import StressDataset
from src.models import XGBoostWrapper, LightGBMWrapper, MLP, ResNet, TabNetWrapper, TabPFNWrapper, WidedeepWrapper, PytorchTabularWrapper, DeepCTRWrapper, train_torch_model, evaluate_model
from src.da_models import DANN, train_dann
from src.domainbed_algos import ERM as DG_ERM, IRM, VREx, GroupDRO, MixStyle, train_dg_model
from sklearn.preprocessing import LabelEncoder

# Constants
DATASETS = {
    'D-1': '/home/iclab/minseo/CHI/data/Archived/stress_binary_personal-full.pkl',
    'D-2': '/home/iclab/minseo/CHI/data/Archived/stress_binary_personal-full_D#2.pkl',
    'D-3': '/home/iclab/minseo/CHI/data/Archived/stress_binary_personal-full_D#3.pkl'
}

class DANNInferenceWrapper(nn.Module):
    def __init__(self, dann_model):
        super().__init__()
        self.model = dann_model
        
    def forward(self, x):
        class_out, _ = self.model(x, alpha=0)
        return class_out

def get_args():
    parser = argparse.ArgumentParser(description="Run Within-Dataset Benchmark")
    parser.add_argument('--dataset', type=str, required=True, choices=['D-1', 'D-2', 'D-3'], help='Dataset to benchmark')
    parser.add_argument('--model', type=str, required=True, choices=['XGB', 'LGB', 'MLP', 'ResNet', 'DANN', 'TabNet', 'TabPFN', 'SAINT', 'TabTransformer', 'NODE', 'DCN', 'IRM', 'VREx', 'GroupDRO', 'MixStyle', 'ERM_DG'], help='Model to use')
    parser.add_argument('--backbone', type=str, default='MLP', choices=['MLP', 'ResNet', 'Transformer'], help='Backbone for DG/DA models')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs for DL models')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for DL models')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--output', type=str, default='results/benchmark_results.csv', help='Output results file')
    return parser.parse_args()

def main():
    args = get_args()
    
    # 1. Load Data
    print(f"Loading {args.dataset}...")
    dataset_path = DATASETS[args.dataset]
    ds = StressDataset(args.dataset, dataset_path)
    
    train_idx, val_idx, test_idx = ds.get_temporal_splits()
    
    X_train, y_train = ds.X[train_idx], ds.y[train_idx]
    X_val, y_val = ds.X[val_idx], ds.y[val_idx]
    X_test, y_test = ds.X[test_idx], ds.y[test_idx]
    
    # Domain Labels (Users) for DANN and DG Models
    DG_MODELS = ['DANN', 'IRM', 'VREx', 'GroupDRO', 'MixStyle', 'ERM_DG']
    
    if args.model in DG_MODELS:
        print(f"Preparing Domain Labels for {args.model}...")
        le = LabelEncoder()
        # Fit on all users to ensure consistency
        all_users_encoded = le.fit_transform(ds.users)
        d_train = all_users_encoded[train_idx]
        d_val = all_users_encoded[val_idx]
        
        num_domains = len(le.classes_)
        print(f"Number of Domains (Users): {num_domains}")
    else:
        # Default for non-DG/DA models
        d_train, d_val, num_domains = None, None, 0

    print(f"Data Splits: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    
    seeds = [42, 0, 1]
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        # Set seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 2. Initialize Model
        # Use simple standard hyperparams for benchmark
        input_dim = X_train.shape[1]
        model = None
        
        if args.model == 'XGB':
            model = XGBoostWrapper(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=seed)
        elif args.model == 'LGB':
            model = LightGBMWrapper(n_estimators=100, num_leaves=31, learning_rate=0.1, random_state=seed)
        elif args.model == 'TabNet':
            model = TabNetWrapper(optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),
                                  scheduler_params={"step_size":10, "gamma":0.9},
                                  scheduler_fn=torch.optim.lr_scheduler.StepLR, mask_type='entmax',
                                  seed=seed)
        elif args.model == 'TabPFN':
            # TabPFN usually doesn't need seed for initialization as it's a pre-trained prior?
            # But we can try passing it if supported or used in sampling.
            model = TabPFNWrapper(seed=seed)
        elif args.model == 'SAINT':
            model = WidedeepWrapper(model_type='SAINT')
        elif args.model == 'TabTransformer':
            model = WidedeepWrapper(model_type='TabTransformer')
        elif args.model == 'NODE':
            model = PytorchTabularWrapper(model_type='NODE')
        elif args.model == 'DCN':
            model = DeepCTRWrapper(model_type='DCN')
        elif args.model == 'MLP':
            net = MLP(input_dim=input_dim, hidden_dim=256)
            model = train_torch_model(net, X_train, y_train, X_val, y_val, 
                                      epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        elif args.model == 'ResNet':
            net = ResNet(input_dim=input_dim, hidden_dim=256, num_blocks=2, dropout=0.3) 
            model = train_torch_model(net, X_train, y_train, X_val, y_val, 
                                      epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        elif args.model == 'DANN':
            net = DANN(input_dim=input_dim, num_domains=num_domains, hidden_dim=256, dropout=0.3)
            model = train_dann(net, X_train, y_train, d_train, X_val, y_val, d_val,
                               epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
            # Wrap for evaluation
            model = DANNInferenceWrapper(model)
        
        # DG Models
        elif args.model == 'IRM':
            model = IRM(input_dim=input_dim, num_classes=2, hparams={'lr': args.lr, 'backbone': args.backbone})
        elif args.model == 'VREx':
            model = VREx(input_dim=input_dim, num_classes=2, hparams={'lr': args.lr, 'backbone': args.backbone})
        elif args.model == 'GroupDRO':
            model = GroupDRO(input_dim=input_dim, num_classes=2, hparams={'lr': args.lr, 'num_domains': num_domains, 'backbone': args.backbone})
        elif args.model == 'MixStyle':
            model = MixStyle(input_dim=input_dim, num_classes=2, hparams={'lr': args.lr, 'backbone': args.backbone})
        elif args.model == 'ERM_DG':
            # Baseline ERM within the DG framework (same backbone)
            model = DG_ERM(input_dim=input_dim, num_classes=2, hparams={'lr': args.lr, 'backbone': args.backbone})
            
        # 3. Train (if not already trained inside helper)
        if args.model in ['XGB', 'LGB', 'TabNet', 'TabPFN', 'SAINT', 'TabTransformer', 'NODE', 'DCN']:
            model.fit(X_train, y_train, X_val, y_val)
        elif args.model in ['IRM', 'VREx', 'GroupDRO', 'MixStyle', 'ERM_DG']:
            # Use specialized DG training loop
            model = train_dg_model(model, X_train, y_train, d_train, X_val, y_val, d_val,
                                   epochs=args.epochs, batch_size=32, domains_per_batch=8)
            
        # 4. Evaluate
        print("Evaluating...")
        
        # Helper for evaluate_model handles torch module or wrapper
        metrics = evaluate_model(model, X_test, y_test)
        
        print(f"Results for {args.dataset} - {args.model} - Seed {seed}: {metrics}")
        
        # 5. Save Results
        results = {
            'Dataset': args.dataset,
            'Model': args.model,
            'Seed': seed,
            'Accuracy': metrics['Accuracy'],
            'F1': metrics['F1'],
            'AUROC': metrics['AUROC']
        }
        
        # Check if file exists to write header
        header = not os.path.exists(args.output)
        df_res = pd.DataFrame([results])
        df_res.to_csv(args.output, mode='a', header=header, index=False)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
