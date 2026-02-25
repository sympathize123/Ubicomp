import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import argparse
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import sys
import time  # ADDED: For timing measurements
from src.data_loader import StressDataset
from src.models import XGBoostWrapper, LightGBMWrapper, MLP, ResNet, TabNetWrapper, TabPFNWrapper, WidedeepWrapper, PytorchTabularWrapper, DeepCTRWrapper, train_torch_model, evaluate_model
from src.da_models import DANN, CDAN, DAN, DeepCORAL, MCC, ADDA, MCD, JAN, SHOT, CBST, CGDM, train_adversarial_da, train_mcd, train_dann, train_cdan, train_adda, train_jan, train_shot, train_cbst, train_deepcoral, train_mcc, train_dan, train_cgdm
from src.domainbed_algos import ERM as DG_ERM, IRM, VREx, GroupDRO, MixStyle, MLDG, MASF, Fish, CSD, SagNet, train_dg_model
from src.hparams_registry import get_hparams
from sklearn.preprocessing import LabelEncoder

# We will define the base directory and templates dynamically based on args.label
BASE_DATA_DIR = '/home/iclab/minseo/Ubicomp/data'

# ADDED: Timing results file
TIMING_OUTPUT = 'results/timing_results_da_hpo.csv'

class DANNInferenceWrapper(nn.Module):
    def __init__(self, dann_model):
        super().__init__()
        self.model = dann_model
        
    def forward(self, x):
        return self.model.predict(x)

class CDANInferenceWrapper(nn.Module):
    def __init__(self, cdan_model):
        super().__init__()
        self.model = cdan_model
        
    def forward(self, x):
        return self.model.predict(x)

class MCDInferenceWrapper(nn.Module):
    def __init__(self, mcd_model):
        super().__init__()
        self.model = mcd_model
        
    def forward(self, x):
        o1, o2 = self.model(x)
        return (o1 + o2) / 2.0 # Ensemble average

class CGDMInferenceWrapper(nn.Module):
    def __init__(self, cgdm_model):
        super().__init__()
        self.model = cgdm_model
        
    def forward(self, x):
        return self.model.predict(x)

def get_args():
    parser = argparse.ArgumentParser(description="Run Within-Dataset Benchmark")
    parser.add_argument('--dataset', type=str, required=True, choices=['D-1', 'D-2', 'D-3'], help='Dataset to benchmark')
    parser.add_argument('--label', type=str, default='stress_binary', help='Target label to predict (e.g., stress_binary, arousal, valence, disturbance, happy, angry)')
    parser.add_argument('--model', type=str, required=True, choices=['XGB', 'LGB', 'MLP', 'ResNet', 'DANN', 'CDAN', 'DAN', 'DeepCORAL', 'MCC', 'ADDA', 'MCD', 'JAN', 'SHOT', 'CBST', 'CGDM', 'TabNet', 'TabPFN', 'SAINT', 'TabTransformer', 'FastFormer', 'Perceiver', 'NODE', 'DCN', 'IRM', 'VREx', 'GroupDRO', 'MixStyle', 'ERM_DG', 'MLDG', 'MASF', 'Fish', 'CSD', 'SagNet'], help='Model to use')
    parser.add_argument('--backbone', type=str, default='MLP', choices=['MLP', 'ResNet', 'Transformer'], help='Backbone for DG/DA models')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs for DL models')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for DL models')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--output', type=str, default='results/benchmark_results_da_hpo.csv', help='Output results file')
    parser.add_argument('--hpo_trials', type=int, default=0, help='Number of HPO trials (0 for default params)')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--efficient_attention', action='store_true', help='Use efficient attention (FastFormer backend) for Transformer models')
    parser.add_argument('--uda', action='store_true', help='Enable Unsupervised Domain Adaptation (Source -> Target)')
    
    return parser.parse_args()

def train_model(args, X_train, y_train, d_train, X_val, y_val, d_val, 
                input_dim, num_classes, num_domains, hparams, seed=42, patience=20, X_target=None):

    print(f"  [DEBUG] train_model params: Backbone={args.backbone}, Model={args.model}, LR={hparams.get('lr')}")
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = None
    
    # Update args with hparams
    backbone = hparams.get('backbone', args.backbone)
    lr = hparams.get('lr', args.lr)
    batch_size = hparams.get('batch_size', args.batch_size)
    epochs = args.epochs
    dropout = hparams.get('dropout', 0.3)
    hidden_dim = hparams.get('hidden_dim', 256)
    num_layers = hparams.get('num_layers', 3)
    num_blocks = hparams.get('num_blocks', 2)
    nhead = hparams.get('nhead', 4)

    if args.model == 'XGB':
        model = XGBoostWrapper(n_estimators=hparams.get('n_estimators', 100), max_depth=hparams.get('max_depth', 6), 
                               learning_rate=hparams.get('learning_rate', 0.1), subsample=hparams.get('subsample', 1.0), 
                               colsample_bytree=hparams.get('colsample_bytree', 1.0), random_state=seed, patience=patience)
    elif args.model == 'LGB':
        model = LightGBMWrapper(n_estimators=hparams.get('n_estimators', 100), num_leaves=hparams.get('num_leaves', 31), 
                                learning_rate=hparams.get('learning_rate', 0.1), min_child_samples=hparams.get('min_child_samples', 20),
                                random_state=seed, patience=patience)
    elif args.model == 'TabNet':
        opt_params = dict(lr=lr)
        if 'weight_decay' in hparams: opt_params['weight_decay'] = hparams['weight_decay']
        
        model = TabNetWrapper(optimizer_fn=torch.optim.Adam, optimizer_params=opt_params,
                                scheduler_params={"step_size":10, "gamma":0.9},
                                scheduler_fn=torch.optim.lr_scheduler.StepLR, mask_type='entmax',
                                seed=seed, 
                                n_d=hparams.get('n_d', 8), n_a=hparams.get('n_a', 8), n_steps=hparams.get('n_steps', 3),
                                gamma=hparams.get('gamma', 1.3), lambda_sparse=hparams.get('lambda_sparse', 1e-3),
                                batch_size=batch_size, epochs=epochs, patience=patience)
    elif args.model == 'TabPFN':
        model = TabPFNWrapper(seed=seed, subsample_seed=hparams.get('subsample_seed', None))
    elif args.model == 'SAINT':
        model = WidedeepWrapper(model_type='SAINT', input_dim=hparams.get('input_dim', 32), n_heads=hparams.get('n_heads', 4), 
                                n_blocks=hparams.get('n_blocks', 2), dropout=hparams.get('dropout', 0.1), mlp_dropout=hparams.get('dropout', 0.1),
                                epochs=epochs, patience=patience, batch_size=batch_size, 
                                efficient_attention=args.efficient_attention, **hparams)
    elif args.model == 'TabTransformer':
        # Default to efficient_attention=True (Linear Attention) to prevent OOM
        use_efficient = True if not args.efficient_attention else args.efficient_attention
        
        # Pop keys that are explicitly passed to avoid "multiple values" error during HPO
        _input_dim = hparams.pop('input_dim', 32)
        _n_heads = hparams.pop('n_heads', 4)
        _n_blocks = hparams.pop('n_blocks', 2)
        _dropout = hparams.pop('dropout', 0.1)
        
        model = WidedeepWrapper(model_type='TabTransformer', input_dim=_input_dim, n_heads=_n_heads, 
                                n_blocks=_n_blocks, dropout=_dropout,
                                epochs=epochs, patience=patience, batch_size=batch_size, 
                                efficient_attention=use_efficient, **hparams)
    elif args.model == 'FastFormer':
        # FastFormer for efficient attention
        model = WidedeepWrapper(model_type='FastFormer', input_dim=hparams.get('input_dim', 32), n_heads=hparams.get('n_heads', 4), 
                                epochs=epochs, patience=patience, batch_size=batch_size, **hparams)
    elif args.model == 'Perceiver':
        # Perceiver for ultra-efficient attention on high-dim
        model = WidedeepWrapper(model_type='Perceiver', input_dim=hparams.get('input_dim', 32), 
                                n_latents=hparams.get('n_latents', 32), latent_dim=hparams.get('latent_dim', 64),
                                epochs=epochs, patience=patience, batch_size=batch_size, **hparams)
    elif args.model == 'NODE':
        # Default to batch_size=24 to prevent BatchNorm error
        if batch_size == 64 and 'batch_size' not in hparams:
             batch_size = 24
             
        # Pop keys that are explicitly passed to avoid "multiple values" error during HPO
        _num_layers = hparams.pop('num_layers', 2)
        _num_trees = hparams.pop('num_trees', 512)
        _depth = hparams.pop('depth', 6)
        
        model = PytorchTabularWrapper(model_type='NODE', num_layers=_num_layers, num_trees=_num_trees, 
                                      depth=_depth, batch_size=batch_size, epochs=epochs, patience=patience, **hparams)
    elif args.model == 'DCN':
        _dnn_hidden_units = hparams.pop('dnn_hidden_units', (256, 128))
        _dropout = hparams.pop('dropout', 0.1)
        model = DeepCTRWrapper(model_type='DCN', dnn_hidden_units=_dnn_hidden_units, 
                               dnn_dropout=_dropout, batch_size=batch_size, epochs=epochs, patience=patience, **hparams)
    elif args.model == 'MLP':
        net = MLP(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        model = train_torch_model(net, X_train, y_train, X_val, y_val, 
                                    epochs=epochs, batch_size=batch_size, lr=lr, patience=patience)
    elif args.model == 'ResNet':
        net = ResNet(input_dim=input_dim, hidden_dim=hidden_dim, num_blocks=num_blocks, dropout=dropout) 
        model = train_torch_model(net, X_train, y_train, X_val, y_val, 
                                    epochs=epochs, batch_size=batch_size, lr=lr, patience=patience)
    elif args.model == 'DANN':
        net = DANN(input_dim=input_dim, num_classes=2, num_domains=num_domains, 
            hparams={**hparams, 'lr': lr, 'backbone': backbone, 'dropout': dropout,
                     'hidden_dim': hidden_dim, 'num_layers': num_layers, 'num_blocks': num_blocks, 'nhead': nhead})
        model = train_dann(net, X_train, y_train, d_train, X_val, y_val, d_val,
                        epochs=epochs, batch_size=batch_size, patience=patience, X_target=X_target)
        model = DANNInferenceWrapper(model)
    
    elif args.model == 'DAN':
        net = DAN(input_dim=input_dim, num_classes=2,
            hparams={**hparams, 'lr': lr, 'backbone': backbone, 'dropout': dropout,
                     'hidden_dim': hidden_dim, 'num_layers': num_layers, 'num_blocks': num_blocks, 'nhead': nhead})
        model = train_dan(net, X_train, y_train, d_train, X_val, y_val, d_val,
                        epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, X_target=X_target)

    
    # DG Models
    elif args.model == 'IRM':
        model = IRM(input_dim=input_dim, num_classes=2, hparams={'lr': lr, 'backbone': backbone, **hparams})
    elif args.model == 'VREx':
        model = VREx(input_dim=input_dim, num_classes=2, hparams={'lr': lr, 'backbone': backbone, **hparams})
    elif args.model == 'GroupDRO':
        model = GroupDRO(input_dim=input_dim, num_classes=2, hparams={'lr': lr, 'num_domains': num_domains, 'backbone': backbone, **hparams})
    elif args.model == 'MixStyle':
        model = MixStyle(input_dim=input_dim, num_classes=2, hparams={'lr': lr, 'backbone': backbone, **hparams})
    elif args.model == 'ERM_DG':
        model = DG_ERM(input_dim=input_dim, num_classes=2, hparams={'lr': lr, 'backbone': backbone, **hparams})
    elif args.model == 'MLDG':
        model = MLDG(input_dim=input_dim, num_classes=2, hparams={'lr': lr, 'backbone': backbone, **hparams})
    elif args.model == 'MASF':
        model = MASF(input_dim=input_dim, num_classes=2, hparams={'lr': lr, 'backbone': backbone, **hparams})
    elif args.model == 'Fish':
        model = Fish(input_dim=input_dim, num_classes=2, hparams={'lr': lr, 'backbone': backbone, **hparams})
    elif args.model == 'CSD':
        model = CSD(input_dim=input_dim, num_classes=2, hparams={'lr': lr, 'backbone': backbone, **hparams})
    elif args.model == 'SagNet':
        model = SagNet(input_dim=input_dim, num_classes=2, hparams={'lr': lr, 'backbone': backbone, **hparams})
    
    # DA Models
    elif args.model == 'ADDA':
        net = ADDA(input_dim=input_dim, num_classes=2,
                   hparams={**hparams, 'lr': lr, 'backbone': backbone, 'dropout': dropout,
                            'hidden_dim': hidden_dim, 'num_layers': num_layers, 'num_blocks': num_blocks, 'nhead': nhead})
        model = train_adda(net, X_train, y_train, d_train, X_val, y_val, d_val, epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, X_target=X_target)
        
    elif args.model == 'MCD':
        net = MCD(input_dim=input_dim, num_classes=2,
                  hparams={**hparams, 'lr': lr, 'backbone': backbone, 'dropout': dropout,
                           'hidden_dim': hidden_dim, 'num_layers': num_layers, 'num_blocks': num_blocks, 'nhead': nhead})
        model = train_mcd(net, X_train, y_train, d_train, X_val, y_val, d_val, epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, X_target=X_target)
        model = MCDInferenceWrapper(model)
        
    elif args.model == 'JAN':
        net = JAN(input_dim=input_dim, num_classes=2,
                  hparams={**hparams, 'lr': lr, 'backbone': backbone, 'dropout': dropout,
                           'hidden_dim': hidden_dim, 'num_layers': num_layers, 'num_blocks': num_blocks, 'nhead': nhead})
        model = train_jan(net, X_train, y_train, d_train, X_val, y_val, d_val, epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, X_target=X_target)

    elif args.model == 'SHOT':
        net = SHOT(input_dim=input_dim, num_classes=2,
                   hparams={**hparams, 'lr': lr, 'backbone': backbone, 'dropout': dropout,
                            'hidden_dim': hidden_dim, 'num_layers': num_layers, 'num_blocks': num_blocks, 'nhead': nhead})
        model = train_shot(net, X_train, y_train, d_train, X_val, y_val, d_val, epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, X_target=X_target)

    elif args.model == 'CBST':
        net = CBST(input_dim=input_dim, num_classes=2,
                   hparams={**hparams, 'lr': lr, 'backbone': backbone, 'dropout': dropout,
                            'hidden_dim': hidden_dim, 'num_layers': num_layers, 'num_blocks': num_blocks, 'nhead': nhead})
        model = train_cbst(net, X_train, y_train, d_train, X_val, y_val, d_val, epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, X_target=X_target)

    elif args.model == 'CDAN':
        net = CDAN(input_dim=input_dim, num_classes=2, num_domains=num_domains, 
            hparams={**hparams, 'lr': lr, 'backbone': backbone, 'dropout': dropout,
                     'hidden_dim': hidden_dim, 'num_layers': num_layers, 'num_blocks': num_blocks, 'nhead': nhead})
        model = train_cdan(net, X_train, y_train, d_train, X_val, y_val, d_val,
                        epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, X_target=X_target)
        model = CDANInferenceWrapper(model)
    
    elif args.model == 'DeepCORAL':
        net = DeepCORAL(input_dim=input_dim, num_classes=2, 
            hparams={**hparams, 'lr': lr, 'backbone': backbone, 'dropout': dropout,
                     'hidden_dim': hidden_dim, 'num_layers': num_layers, 'num_blocks': num_blocks, 'nhead': nhead})
        model = train_deepcoral(net, X_train, y_train, d_train, X_val, y_val, d_val,
                        epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, X_target=X_target)
    
    elif args.model == 'MCC':
        net = MCC(input_dim=input_dim, num_classes=2, 
            hparams={**hparams, 'lr': lr, 'backbone': backbone, 'dropout': dropout,
                     'hidden_dim': hidden_dim, 'num_layers': num_layers, 'num_blocks': num_blocks, 'nhead': nhead})
        model = train_mcc(net, X_train, y_train, d_train, X_val, y_val, d_val,
                        epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, X_target=X_target)
    elif args.model == 'CGDM':
        if X_target is None:
            raise ValueError("CGDM requires X_target (use --uda).")
        net = CGDM(input_dim=input_dim, num_classes=num_classes)
        model = train_cgdm(net, X_train, y_train, X_target,
                           X_val=X_val, y_val=y_val,
                           epochs=epochs, batch_size=batch_size, lr=lr,
                           weight_decay=hparams.get('weight_decay', 5e-4))
        model = CGDMInferenceWrapper(model)

    # Train (if not already trained inside helper)
    if args.model in ['XGB', 'LGB', 'TabNet', 'TabPFN', 'SAINT', 'TabTransformer', 'FastFormer', 'Perceiver', 'NODE', 'DCN']:
        model.fit(X_train, y_train, X_val, y_val)
    elif args.model in ['IRM', 'VREx', 'GroupDRO', 'MixStyle', 'ERM_DG', 'MLDG', 'MASF', 'Fish', 'CSD', 'SagNet']:
        model = train_dg_model(model, X_train, y_train, d_train, X_val, y_val, d_val,
                               epochs=epochs, batch_size=batch_size, domains_per_batch=8, patience=patience)
        
    return model

def main():
    args = get_args()
    
    # 1. Load Data
    print(f"Loading {args.dataset} (Label: {args.label})...")
    
    # Construct paths dynamically
    if args.dataset == 'D-1':
        dataset_path = os.path.join(BASE_DATA_DIR, f"{args.label}_personal-full_D#2.pkl")
    elif args.dataset == 'D-2':
        dataset_path = os.path.join(BASE_DATA_DIR, f"{args.label}_personal-full_D#3.pkl")
    elif args.dataset == 'D-3':
        dataset_path = os.path.join(BASE_DATA_DIR, f"{args.label}_personal-full.pkl")
    else:
        raise ValueError("Unknown dataset")
        
    if not os.path.exists(dataset_path):
        # Fallback to absolute archived path if it's the original stress_binary and local copy doesn't exist
        fallback_path = f"/home/iclab/minseo/CHI/data/Archived/{os.path.basename(dataset_path)}"
        if os.path.exists(fallback_path):
            dataset_path = fallback_path

    ds = StressDataset(args.dataset, dataset_path)
    
    train_idx, val_idx, test_idx = ds.get_temporal_splits()
    
    # Normalize features using only training data statistics to prevent data leakage
    ds.normalize_features(train_idx, val_idx, test_idx)
    
    X_train, y_train = ds.X[train_idx], ds.y[train_idx]
    X_val, y_val = ds.X[val_idx], ds.y[val_idx]
    X_test, y_test = ds.X[test_idx], ds.y[test_idx]
    
    # Domain Labels
    DG_MODELS = ['DANN', 'CDAN', 'DAN', 'DeepCORAL', 'MCC', 'CGDM', 'IRM', 'VREx', 'GroupDRO', 'MixStyle', 'ERM_DG', 'MLDG', 'MASF', 'ADDA', 'MCD', 'JAN', 'SHOT', 'CBST', 'Fish', 'CSD', 'SagNet']
    
    d_train, d_val, num_domains = None, None, 0
    X_target = None # For UDA

    if args.model in DG_MODELS:
        if args.uda:
            print(f"Preparing UDA Mode for {args.model} (Source=Train, Target=Test)...")
            # UDA Mode: Domain 0 = Source (Train), Domain 1 = Target (Test)
            num_domains = 2
            
            # d_train is all 0s (Source)
            d_train = np.zeros(len(X_train), dtype=int)
            d_val = np.zeros(len(X_val), dtype=int) # Val is also Source
            
            # X_target is the Unlabeled Test Set (Domain 1)
            X_target = X_test
            print(f"UDA Enabled: X_target shape {X_target.shape}")
            
        else:
            print(f"Preparing DG Mode for {args.model} (Multi-Source Users)...")
            le = LabelEncoder()
            all_users_encoded = le.fit_transform(ds.users)
            d_train = all_users_encoded[train_idx]
            d_val = all_users_encoded[val_idx]
            
            num_domains = len(le.classes_)
            print(f"Number of Domains (Users): {num_domains}")

    print(f"Data Splits: Train {X_train.shape}, Val {X_val.shape}")
    input_dim = X_train.shape[1]
    num_classes = 2
    # 4. Hyperparameter Optimization
    # 4. Hyperparameter Optimization
    if args.hpo_trials > 0:
        # Only print backbone if it's a DG/DA model that uses it
        dg_da_models = ['DANN', 'CDAN', 'DAN', 'DeepCORAL', 'MCC', 'ADDA', 'MCD', 'JAN', 'SHOT', 'CBST', 
                        'IRM', 'VREx', 'GroupDRO', 'MixStyle', 'MLDG', 'MASF', 'Fish', 'CSD', 'SagNet', 'ERM_DG']
        
        if args.model in dg_da_models:
             print(f"Starting HPO with {args.hpo_trials} trials for {args.model} (Backbone: {args.backbone})...")
        else:
             print(f"Starting HPO with {args.hpo_trials} trials for {args.model}...")
        import optuna
        from src.hparams_registry import get_hparams
        
        def objective(trial):
            # Get search space
            hparams = get_hparams(args.model, args.dataset, backbone=args.backbone) # Pass backbone!
            
            # Suggest params
            trial_params = {}
            for k, v in hparams.items():
                if callable(v):
                    trial_params[k] = v(trial)
                else:
                    trial_params[k] = v
            
            # Train with these params
            # We need to capture validation metric. 
            # To avoid overhead, we might do a shorter run or partial data? 
            # For now, full run on this split.
            
            try:
                model = train_model(args, X_train, y_train, d_train, X_val, y_val, d_val, 
                                    input_dim, num_classes, num_domains, trial_params, seed=42, patience=args.patience, X_target=X_target)
                
                # Evaluate on Val
                val_metrics = evaluate_model(model, X_val, y_val)
                return val_metrics['AUROC'] # Maximize AUROC
            except Exception as e:
                print(f"HPO Trial failed: {e}")
                return 0.0

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=args.hpo_trials)
        
        print("Best HPO params:", study.best_params)
        best_hparams = study.best_params
    else:
        # Default params
        # We still need to call get_hparams but it returns callables. 
        # We need a 'default' set or just manually set empty so train_model uses defaults.
        # Actually, hparams_registry returns callables for random search.
        # If not HPO, we pass empty dict and let train_model use its internal defaults 
        # OR we could pick 'middle' values. 
        # Current logic: train_model has defaults if hparams is empty.
        best_hparams = {}
    
    seeds = [42]
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        
        # ADDED: Start timing
        start_time = time.time()
        
        model = train_model(args, X_train, y_train, d_train, X_val, y_val, d_val, 
                            input_dim, num_classes, num_domains, hparams=best_hparams, seed=seed, patience=args.patience, X_target=X_target)
        
        # ADDED: End timing
        end_time = time.time()
        total_time = end_time - start_time
        time_per_epoch = total_time / args.epochs
        
        print(f"Training completed in {total_time:.2f} seconds ({time_per_epoch:.2f} seconds/epoch)")
        
        # 4. Evaluate
        print("Evaluating on Train, Validation and Test sets...")
        
        train_metrics = evaluate_model(model, X_train, y_train)
        val_metrics = evaluate_model(model, X_val, y_val)
        test_metrics = evaluate_model(model, X_test, y_test)
        
        print(f"Results for {args.dataset} - {args.model} - Seed {seed}:")
        print(f"  Train AUROC: {train_metrics['AUROC']:.4f}, Acc: {train_metrics['Accuracy']:.4f}")
        print(f"  Val   AUROC: {val_metrics['AUROC']:.4f}, Acc: {val_metrics['Accuracy']:.4f}")
        print(f"  Test  AUROC: {test_metrics['AUROC']:.4f}, Acc: {test_metrics['Accuracy']:.4f}")
        
        # 5. Save Results
        results = {
            'Dataset': args.dataset,
            'Label': args.label,
            'Model': args.model,
            'Backbone': args.backbone, # Added Backbone
            'Seed': seed,
            'Train_Accuracy': train_metrics['Accuracy'],
            'Train_AUROC': train_metrics['AUROC'],
            'Val_Accuracy': val_metrics['Accuracy'],
            'Val_AUROC': val_metrics['AUROC'],
            'Test_Accuracy': test_metrics['Accuracy'],
            'Test_F1': test_metrics['F1'],
            'Test_AUROC': test_metrics['AUROC']
        }
        
        if os.path.dirname(args.output):
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            
        header = not os.path.exists(args.output)
        df_res = pd.DataFrame([results])
        df_res.to_csv(args.output, mode='a', header=header, index=False)
        print(f"Results saved to {args.output}")
        
        # ADDED: Save timing results
        timing_results = {
            'Dataset': args.dataset,
            'Model': args.model,
            'Backbone': args.backbone if args.model in DG_MODELS else 'N/A',
            'Seed': seed,
            'Total_Time_Seconds': round(total_time, 2),
            'Time_Per_Epoch': round(time_per_epoch, 2),
            'Epochs': args.epochs
        }
        
        if os.path.dirname(TIMING_OUTPUT):
            os.makedirs(os.path.dirname(TIMING_OUTPUT), exist_ok=True)
            
        timing_header = not os.path.exists(TIMING_OUTPUT)
        df_timing = pd.DataFrame([timing_results])
        df_timing.to_csv(TIMING_OUTPUT, mode='a', header=timing_header, index=False)
        print(f"Timing results saved to {TIMING_OUTPUT}")

if __name__ == "__main__":
    main()
