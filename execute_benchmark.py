import os
import argparse
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from src.data_loader import BenchmarkDataset
from src.models import XGBoostWrapper, LightGBMWrapper, MLP, ResNet, TabNetWrapper, WidedeepWrapper, DeepCTRWrapper, train_torch_model, evaluate_model
from src.da_models import DANN, CDAN, DAN, DeepCORAL, MCC, ADDA, MCD, JAN, SHOT, CBST, CGDM, MCDInferenceWrapper, train_mcd, train_dann, train_cdan, train_adda, train_jan, train_shot, train_cbst, train_deepcoral, train_mcc, train_dan, train_cgdm
from src.domainbed_algos import ERM as DG_ERM, IRM, VREx, GroupDRO, MixStyle, MLDG, MASF, Fish, CSD, SagNet, train_dg_model
from src.hparams_registry import get_hparams
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

BASE_DATA_DIR = str((Path(__file__).resolve().parent / 'data').resolve())

PROGRESS_COLUMNS = [
    'Dataset', 'Label', 'Model', 'Backbone', 'Seed', 'Fold', 'Phase', 'Trial',
    'Train_Accuracy', 'Train_AUROC', 'Val_Accuracy', 'Val_AUROC',
    'Test_Accuracy', 'Test_F1', 'Test_AUROC', 'Hparams_JSON'
]

def append_row(path, row, columns):
    path = Path(path)
    if path.parent != Path('.'):
        path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    df = pd.DataFrame([row], columns=columns)
    df.to_csv(path, mode='a', header=header, index=False)


def make_groupwise_val_split(train_idx, labels, groups, seed=42, max_splits=5):
    train_idx = np.asarray(train_idx, dtype=int)
    train_labels = labels[train_idx]
    train_groups = groups[train_idx]
    unique_groups = np.unique(train_groups)
    if unique_groups.size < 2:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        dummy = np.zeros_like(train_labels)
        train_rel, val_rel = next(sss.split(dummy, train_labels))
        return train_idx[train_rel], train_idx[val_rel]
    n_splits = int(min(max_splits, unique_groups.size))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros_like(train_labels)
    try:
        train_rel, val_rel = next(splitter.split(dummy, train_labels, train_groups))
    except ValueError:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_rel, val_rel = next(sss.split(dummy, train_labels))
    return train_idx[train_rel], train_idx[val_rel]



def get_args():
    parser = argparse.ArgumentParser(description="Run Within-Dataset Benchmark")
    parser.add_argument('--dataset', type=str, required=True, choices=['D-1', 'D-2', 'D-3'])
    parser.add_argument('--label', type=str, default='stress_binary')
    parser.add_argument('--model', type=str, required=True, choices=['XGB', 'LGB', 'MLP', 'ResNet', 'DANN', 'CDAN', 'DAN', 'DeepCORAL', 'MCC', 'ADDA', 'MCD', 'JAN', 'SHOT', 'CBST', 'CGDM', 'TabNet', 'SAINT', 'TabTransformer', 'FTTransformer', 'DCN', 'AutoInt', 'IRM', 'VREx', 'GroupDRO', 'MixStyle', 'ERM_DG', 'MLDG', 'MASF', 'Fish', 'CSD', 'SagNet'])
    parser.add_argument('--backbone', type=str, default='MLP', choices=['MLP', 'ResNet', 'Transformer'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output', type=str, default='results/benchmark_results_da_hpo.csv')
    parser.add_argument('--hpo_trials', type=int, default=0)
    parser.add_argument('--hpo_mode', type=str, default='fold1', choices=['fold1', 'cv', 'nested'])
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--efficient_attention', action='store_true')
    parser.add_argument('--uda', action='store_true')
    parser.add_argument('--max_folds', type=int, default=None, help='Limit number of folds (e.g. 1 for quick debug)')
    parser.add_argument('--epochs_override', type=int, default=None, help='Override epochs for quick debug runs')
    return parser.parse_args()


def train_model(args, X_train, y_train, d_train, X_val, y_val, d_val,
                input_dim, num_classes, num_domains, hparams, seed=42, patience=20, X_target=None):

    print(f"  [DEBUG] train_model params: Backbone={args.backbone}, Model={args.model}, LR={hparams.get('lr')}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = None

    backbone = hparams.get('backbone', args.backbone)
    lr = hparams.get('lr', args.lr)
    batch_size = hparams.get('batch_size', args.batch_size)
    epochs = args.epochs_override if args.epochs_override else args.epochs
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
                              scheduler_params={"step_size": 10, "gamma": 0.9},
                              scheduler_fn=torch.optim.lr_scheduler.StepLR, mask_type='entmax',
                              seed=seed,
                              n_d=hparams.get('n_d', 8), n_a=hparams.get('n_a', 8), n_steps=hparams.get('n_steps', 3),
                              gamma=hparams.get('gamma', 1.3), lambda_sparse=hparams.get('lambda_sparse', 1e-3),
                              batch_size=batch_size, epochs=epochs, patience=patience)
    elif args.model == 'SAINT':
        _input_dim = hparams.pop('input_dim', 32)
        _n_heads = hparams.pop('n_heads', 4)
        _n_blocks = hparams.pop('n_blocks', 2)
        _dropout = hparams.pop('dropout', 0.1)
        hparams.pop('lr', None)  # lr passed separately via WideTrainer
        model = WidedeepWrapper(model_type='SAINT', input_dim=_input_dim, n_heads=_n_heads,
                                n_blocks=_n_blocks, dropout=_dropout, mlp_dropout=_dropout,
                                epochs=epochs, patience=patience, batch_size=batch_size,
                                efficient_attention=args.efficient_attention, **hparams)
    elif args.model == 'TabTransformer':
        use_efficient = True if not args.efficient_attention else args.efficient_attention
        _input_dim = hparams.pop('input_dim', 32)
        _n_heads = hparams.pop('n_heads', 4)
        _n_blocks = hparams.pop('n_blocks', 2)
        _dropout = hparams.pop('dropout', 0.1)
        model = WidedeepWrapper(model_type='TabTransformer', input_dim=_input_dim, n_heads=_n_heads,
                                n_blocks=_n_blocks, dropout=_dropout,
                                epochs=epochs, patience=patience, batch_size=batch_size,
                                efficient_attention=use_efficient, **hparams)
    elif args.model == 'FTTransformer':
        _input_dim = hparams.pop('input_dim', 32)
        _n_heads = hparams.pop('n_heads', 4)
        _n_blocks = hparams.pop('n_blocks', 2)
        _dropout = hparams.pop('dropout', 0.1)
        model = WidedeepWrapper(model_type='FTTransformer', input_dim=_input_dim, n_heads=_n_heads,
                                n_blocks=_n_blocks, dropout=_dropout,
                                epochs=epochs, patience=patience, batch_size=batch_size,
                                efficient_attention=args.efficient_attention, **hparams)
    elif args.model == 'DCN':
        _dnn_hidden_units = hparams.pop('dnn_hidden_units', (256, 128))
        _dropout = hparams.pop('dropout', 0.1)
        model = DeepCTRWrapper(model_type='DCN', dnn_hidden_units=_dnn_hidden_units,
                               dnn_dropout=_dropout, batch_size=batch_size, epochs=epochs, patience=patience, **hparams)
    elif args.model == 'AutoInt':
        _dropout = hparams.pop('dropout', 0.1)
        # deepctr_torch AutoInt does not support att_embedding_dim in this environment.
        hparams.pop('att_embedding_dim', None)
        model = DeepCTRWrapper(model_type='AutoInt', dnn_dropout=_dropout,
                               batch_size=batch_size, epochs=epochs, patience=patience, **hparams)
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
    elif args.model == 'DAN':
        net = DAN(input_dim=input_dim, num_classes=2,
                  hparams={**hparams, 'lr': lr, 'backbone': backbone, 'dropout': dropout,
                           'hidden_dim': hidden_dim, 'num_layers': num_layers, 'num_blocks': num_blocks, 'nhead': nhead})
        model = train_dan(net, X_train, y_train, d_train, X_val, y_val, d_val,
                          epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, X_target=X_target)
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
                           patience=patience,
                           weight_decay=hparams.get('weight_decay', 5e-4))

    if args.model in ['XGB', 'LGB', 'TabNet', 'SAINT', 'TabTransformer', 'FTTransformer', 'DCN', 'AutoInt']:
        model.fit(X_train, y_train, X_val, y_val)
    elif args.model in ['IRM', 'VREx', 'GroupDRO', 'MixStyle', 'MLDG', 'MASF', 'Fish', 'CSD', 'SagNet']:
        model = train_dg_model(model, X_train, y_train, d_train, X_val, y_val, d_val,
                               epochs=epochs, batch_size=batch_size, domains_per_batch=8, patience=patience)

    return model


def main():
    args = get_args()
    output_path = Path(args.output)
    progress_output_path = output_path.with_name(output_path.stem + "_progress.csv")

    print(f"Loading {args.dataset} (Label: {args.label})...")

    if args.dataset == 'D-1':
        dataset_path = os.path.join(BASE_DATA_DIR, f"{args.label}_personal-full_D#2.pkl")
    elif args.dataset == 'D-2':
        dataset_path = os.path.join(BASE_DATA_DIR, f"{args.label}_personal-full_D#3.pkl")
    elif args.dataset == 'D-3':
        dataset_path = os.path.join(BASE_DATA_DIR, f"{args.label}_personal-full.pkl")
    else:
        raise ValueError("Unknown dataset")

    ds = BenchmarkDataset(args.dataset, dataset_path)

    group_folds = 5
    split_seed = 42
    labels = ds.y
    groups = ds.users
    X_raw = ds.X.copy()
    splitter = StratifiedGroupKFold(n_splits=group_folds, shuffle=True, random_state=split_seed)

    fold_splits = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(np.zeros_like(labels), labels, groups)):
        train_idx, val_idx = make_groupwise_val_split(train_idx, labels, groups, seed=split_seed + fold_id)
        fold_splits.append((fold_id, train_idx, val_idx, test_idx))

    if args.max_folds is not None:
        fold_splits = fold_splits[:args.max_folds]

    le = LabelEncoder()
    user_domain_ids = le.fit_transform(ds.users)
    num_domains_all = len(le.classes_)

    fold_data = []
    for fold_id, train_idx, val_idx, test_idx in fold_splits:
        ds.X = X_raw.copy()
        ds.normalize_features(train_idx, val_idx, test_idx)
        fold_data.append({
            "fold_id": fold_id,
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
            "X_train": ds.X[train_idx],
            "y_train": ds.y[train_idx],
            "X_val": ds.X[val_idx],
            "y_val": ds.y[val_idx],
            "X_test": ds.X[test_idx],
            "y_test": ds.y[test_idx],
        })

    DG_MODELS = ['DANN', 'CDAN', 'DAN', 'DeepCORAL', 'MCC', 'CGDM', 'IRM', 'VREx', 'GroupDRO', 'MixStyle', 'ERM_DG', 'MLDG', 'MASF', 'ADDA', 'MCD', 'JAN', 'SHOT', 'CBST', 'Fish', 'CSD', 'SagNet']
    DA_MODELS = ['DANN', 'CDAN', 'DAN', 'DeepCORAL', 'MCC', 'ADDA', 'MCD', 'JAN', 'SHOT', 'CBST', 'CGDM']

    def _prepare_domain_info(entry):
        if args.model not in DG_MODELS:
            return None, None, 0, None
        if args.uda:
            d_train = np.zeros(len(entry["y_train"]), dtype=int)
            d_val = np.zeros(len(entry["y_val"]), dtype=int)
            return d_train, d_val, 2, entry["X_test"]
        d_train = user_domain_ids[entry["train_idx"]]
        d_val = user_domain_ids[entry["val_idx"]]
        return d_train, d_val, num_domains_all, None

    best_hparams = {}

    def _run_hpo(folds_for_hpo, *, label):
        print(f"Starting {label} with {args.hpo_trials} trials...")
        import optuna
        from src.hparams_registry import get_hparams

        def objective(trial):
            hparams = get_hparams(args.model, args.dataset, backbone=args.backbone)
            trial_params = {}
            for k, v in hparams.items():
                if callable(v):
                    trial_params[k] = v(trial)
                else:
                    trial_params[k] = v

            scores = []
            for entry in folds_for_hpo:
                d_train, d_val, num_domains, X_target = _prepare_domain_info(entry)
                X_val_eval = entry["X_val"]
                y_val_eval = entry["y_val"]
                if args.uda and args.model in DA_MODELS:
                    X_val_eval = entry["X_test"]
                    y_val_eval = entry["y_test"]
                try:
                    model = train_model(
                        args,
                        entry["X_train"], entry["y_train"], d_train,
                        X_val_eval, y_val_eval, d_val,
                        entry["X_train"].shape[1], 2, num_domains,
                        trial_params, seed=42, patience=args.patience, X_target=X_target,
                    )
                    val_metrics = evaluate_model(model, X_val_eval, y_val_eval)
                    scores.append(val_metrics['AUROC'])
                except Exception as e:
                    print(f"HPO Trial failed on fold {entry['fold_id'] + 1}: {e}")
                    return 0.0
            return float(np.mean(scores)) if scores else 0.0

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=args.hpo_trials)
        print("Best HPO params:", study.best_params)
        return study.best_params

    if args.hpo_trials > 0 and args.hpo_mode in ("fold1", "cv"):
        folds_for_hpo = fold_data if args.hpo_mode == "cv" else [fold_data[0]]
        best_hparams = _run_hpo(folds_for_hpo, label="CV-HPO" if args.hpo_mode == "cv" else "Fold-1 HPO")

    seeds = [42]

    # Collect per-fold results for final aggregation
    METRIC_KEYS = ['Train_Accuracy', 'Train_AUROC', 'Val_Accuracy', 'Val_AUROC',
                   'Test_Accuracy', 'Test_F1', 'Test_AUROC']
    all_fold_results = []

    for entry in fold_data:
        fold_id = entry["fold_id"]
        print(f"\n=== Fold {fold_id + 1}/{group_folds} ===")

        if args.hpo_trials > 0 and args.hpo_mode == "nested":
            best_hparams = _run_hpo([entry], label=f"Nested HPO (Fold {fold_id + 1})")

        d_train, d_val, num_domains, X_target = _prepare_domain_info(entry)
        if args.model in DG_MODELS:
            if args.uda:
                print(f"Preparing UDA Mode for {args.model} (Source=Train, Target=Test)...")
                print(f"UDA Enabled: X_target shape {entry['X_test'].shape}")
            else:
                print(f"Preparing DG Mode for {args.model} (Multi-Source Users)...")
                print(f"Number of Domains (Users): {num_domains}")

        print(f"Data Splits: Train {entry['X_train'].shape}, Val {entry['X_val'].shape}, Test {entry['X_test'].shape}")

        for seed in seeds:
            print(f"\n--- Fold {fold_id + 1} | Seed {seed} ---")
            X_val_train = entry["X_val"]
            y_val_train = entry["y_val"]
            if args.uda and args.model in DA_MODELS:
                X_val_train = entry["X_test"]
                y_val_train = entry["y_test"]

            model = train_model(
                args,
                entry["X_train"], entry["y_train"], d_train,
                X_val_train, y_val_train, d_val,
                entry["X_train"].shape[1], 2, num_domains,
                hparams=best_hparams, seed=seed, patience=args.patience, X_target=X_target,
            )

            print("Evaluating on Train, Validation and Test sets...")

            train_metrics = evaluate_model(model, entry["X_train"], entry["y_train"])
            val_metrics   = evaluate_model(model, entry["X_val"],   entry["y_val"])
            test_metrics  = evaluate_model(model, entry["X_test"],  entry["y_test"])

            print(f"Results for {args.dataset} - {args.model} - Fold {fold_id + 1} - Seed {seed}:")
            print(f"  Train AUROC: {train_metrics['AUROC']:.4f}, Acc: {train_metrics['Accuracy']:.4f}")
            print(f"  Val   AUROC: {val_metrics['AUROC']:.4f}, Acc: {val_metrics['Accuracy']:.4f}")
            print(f"  Test  AUROC: {test_metrics['AUROC']:.4f}, Acc: {test_metrics['Accuracy']:.4f}")

            fold_result = {
                'Train_Accuracy': train_metrics['Accuracy'], 'Train_AUROC': train_metrics['AUROC'],
                'Val_Accuracy': val_metrics['Accuracy'], 'Val_AUROC': val_metrics['AUROC'],
                'Test_Accuracy': test_metrics['Accuracy'], 'Test_F1': test_metrics['F1'],
                'Test_AUROC': test_metrics['AUROC']
            }
            all_fold_results.append(fold_result)

            # Progress CSV: per-fold immediate save (unchanged)
            final_progress_row = {
                'Dataset': args.dataset, 'Label': args.label, 'Model': args.model,
                'Backbone': args.backbone, 'Seed': seed, 'Fold': fold_id + 1,
                'Phase': 'final', 'Trial': '',
                'Train_Accuracy': train_metrics.get('Accuracy'), 'Train_AUROC': train_metrics.get('AUROC'),
                'Val_Accuracy': val_metrics.get('Accuracy'), 'Val_AUROC': val_metrics.get('AUROC'),
                'Test_Accuracy': test_metrics.get('Accuracy'), 'Test_F1': test_metrics.get('F1'),
                'Test_AUROC': test_metrics.get('AUROC'),
                'Hparams_JSON': json.dumps(best_hparams, default=str)
            }
            append_row(progress_output_path, final_progress_row, PROGRESS_COLUMNS)
            print(f"Progress results saved to {progress_output_path}")

    # --- Final CSV: aggregate mean ± std across all folds ---
    if all_fold_results:
        summary = {'Dataset': args.dataset, 'Label': args.label, 'Model': args.model,
                   'Backbone': args.backbone, 'N_Folds': len(all_fold_results)}
        for key in METRIC_KEYS:
            vals = [r[key] for r in all_fold_results]
            summary[f'{key}_Mean'] = float(np.mean(vals))
            summary[f'{key}_Std'] = float(np.std(vals))

        if os.path.dirname(args.output):
            os.makedirs(os.path.dirname(args.output), exist_ok=True)

        header = not os.path.exists(args.output)
        df_summary = pd.DataFrame([summary])
        df_summary.to_csv(args.output, mode='a', header=header, index=False)
        print(f"\n=== Summary (Mean ± Std over {len(all_fold_results)} folds) ===")
        for key in METRIC_KEYS:
            print(f"  {key}: {summary[f'{key}_Mean']:.4f} ± {summary[f'{key}_Std']:.4f}")
        print(f"Summary saved to {args.output}")



if __name__ == "__main__":
    main()
