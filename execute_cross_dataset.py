#!/usr/bin/env python3
"""
Cross-dataset benchmark runner.

Supports:
1) Feature-space analysis with canonicalized feature name intersection.
2) Leave-one-dataset-out evaluation:
   - train on two datasets -> test on one dataset
   - train on one dataset -> test on one dataset
"""

import argparse
import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

import copy
from execute_benchmark import train_model
from src.hparams_registry import get_hparams
from src.models import evaluate_model
from benchmark_logger import BenchmarkLogger, evaluate_extended

BASE_DATA_DIR = str((Path(__file__).resolve().parent / 'data').resolve())
COMMON_LABELS = ['arousal', 'disturbance', 'valence', 'stress_binary']

DA_MODELS = ['DANN', 'CDAN', 'DAN', 'DeepCORAL', 'MCC', 'ADDA', 'MCD', 'JAN', 'SHOT', 'CBST', 'CGDM']

RESULT_COLUMNS = [
    'Setting', 'Label', 'Model', 'Backbone', 'Seed', 'Val_Ratio', 'HPO_Trials',
    'Train_Datasets', 'Test_Dataset',
    'Common_Features', 'Train_Samples', 'Val_Samples', 'Test_Samples',
    'Train_Users', 'Val_Users', 'Test_Users',
    'Train_PosRatio', 'Val_PosRatio', 'Test_PosRatio',
    'Train_Accuracy', 'Train_AUROC', 'Train_F1', 'Train_Precision', 'Train_Recall',
    'Val_Accuracy', 'Val_AUROC', 'Val_F1', 'Val_Precision', 'Val_Recall',
    'Test_Accuracy', 'Test_F1', 'Test_AUROC', 'Test_Precision', 'Test_Recall',
    'Best_Epoch', 'Early_Stopped', 'HPO_Best_AUROC',
    'Total_Wall_S', 'HPO_Wall_S', 'Train_Wall_S',
    'Device_Name', 'Peak_GPU_MB',
    'Hparams_JSON', 'Experiment_ID',
]


def canonicalize_feature_name(name: str) -> str:
    """Normalize feature aliases across datasets."""
    x = str(name).strip()

    # Known categorical alias mismatch across datasets
    x = x.replace('UNKNOWN', 'UNDEFINED')
    x = x.replace('unknown', 'undefined')

    # Logger-format differences
    x = x.replace('##', '#')
    x = x.replace('#_', '#')
    x = x.replace('_Today', 'Today')

    # Common identity fields with case differences
    pif_alias = {
        'PIF#AGE': 'PIF#age',
        'PIF#GENDER': 'PIF#gender',
        'PIF#ANDROID': 'PIF#android',
        'PIF#IOS': 'PIF#ios',
        'PIF#OPENNESS': 'PIF#openness',
        'PIF#CONSCIENTIOUSNESS': 'PIF#conscientiousness',
        'PIF#EXTRAVERSION': 'PIF#extraversion',
        'PIF#AGREEABLENESS': 'PIF#agreeableness',
        'PIF#NEUROTICISM': 'PIF#neuroticism',
        'PIF#PARTICIPATIONSTARTTIMESTAMP': 'PIF#participationStartTimestamp',
    }
    return pif_alias.get(x, x)


def _dataset_path(dataset: str, label: str) -> str:
    if label == "stress_binary":
        if dataset == 'D-1':
            dataset_path = os.path.join(BASE_DATA_DIR, "stress_binary_personal-full_D-1.pkl")
        elif dataset == 'D-2':
            dataset_path = os.path.join(BASE_DATA_DIR, "stress_binary_personal-full_D-2.pkl")
        elif dataset == 'D-3':
            dataset_path = os.path.join(BASE_DATA_DIR, "stress_binary_personal-full_D-3.pkl")
        else:
            raise ValueError("Unknown dataset")
    elif dataset == 'D-1':
        dataset_path = os.path.join(BASE_DATA_DIR, f"{label}_personal-full_D#2.pkl")
    elif dataset == 'D-2':
        dataset_path = os.path.join(BASE_DATA_DIR, f"{label}_personal-full_D#3.pkl")
    elif dataset == 'D-3':
        dataset_path = os.path.join(BASE_DATA_DIR, f"{label}_personal-full.pkl")
    else:
        raise ValueError("Unknown dataset")
    return dataset_path


def _load_dataset_raw(dataset: str, label: str) -> Dict[str, np.ndarray]:
    path = _dataset_path(dataset, label)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing dataset file: {path}')

    with open(path, 'rb') as f:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message=r'numpy\.core\.numeric is deprecated',
                category=DeprecationWarning,
            )
            data = pickle.load(f)

    if not (isinstance(data, (tuple, list)) and len(data) >= 5):
        raise ValueError(f'Unexpected pickle format: {path}')

    X = data[0]
    y = np.asarray(data[1], dtype=np.int64)
    users = np.asarray(data[2])

    if isinstance(X, pd.DataFrame):
        feature_names = [str(c) for c in X.columns]
        X = X.values
    else:
        X = np.asarray(X)
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    keep_mask = np.ones(len(feature_names), dtype=bool)
    for i, name in enumerate(feature_names):
        name_lower = name.lower()
        if 'timestamp' in name_lower or 'participant' in name_lower or 'label' in name_lower:
            keep_mask[i] = False

    X = np.asarray(X[:, keep_mask], dtype=np.float32)
    feature_names = [feature_names[i] for i in np.where(keep_mask)[0]]

    feat_to_idx = {}
    duplicate_alias_count = 0
    for i, raw in enumerate(feature_names):
        canonical = canonicalize_feature_name(raw)
        if canonical in feat_to_idx:
            duplicate_alias_count += 1
            continue
        feat_to_idx[canonical] = i

    return {
        'X': X,
        'y': y,
        'users': users,
        'feature_names_raw': feature_names,
        'feature_to_idx': feat_to_idx,
        'duplicate_alias_count': duplicate_alias_count,
    }


def _compute_feature_report(label: str, bundles: Dict[str, Dict]) -> Tuple[pd.DataFrame, List[str]]:
    raw_sets = {k: set(v['feature_names_raw']) for k, v in bundles.items()}
    can_sets = {k: set(v['feature_to_idx'].keys()) for k, v in bundles.items()}

    raw_union = set().union(*raw_sets.values())
    raw_inter = set.intersection(*raw_sets.values())
    can_union = set().union(*can_sets.values())
    can_inter = set.intersection(*can_sets.values())

    rows = []
    rows.append({
        'Label': label,
        'Metric': 'raw',
        'Triple_Intersection': len(raw_inter),
        'Union': len(raw_union),
        'Intersection_Over_Union': len(raw_inter) / max(1, len(raw_union)),
        'D1_Coverage': len(raw_inter) / max(1, len(raw_sets['D-1'])),
        'D2_Coverage': len(raw_inter) / max(1, len(raw_sets['D-2'])),
        'D3_Coverage': len(raw_inter) / max(1, len(raw_sets['D-3'])),
    })
    rows.append({
        'Label': label,
        'Metric': 'canonicalized',
        'Triple_Intersection': len(can_inter),
        'Union': len(can_union),
        'Intersection_Over_Union': len(can_inter) / max(1, len(can_union)),
        'D1_Coverage': len(can_inter) / max(1, len(can_sets['D-1'])),
        'D2_Coverage': len(can_inter) / max(1, len(can_sets['D-2'])),
        'D3_Coverage': len(can_inter) / max(1, len(can_sets['D-3'])),
    })

    for a, b in [('D-1', 'D-2'), ('D-1', 'D-3'), ('D-2', 'D-3')]:
        inter = len(can_sets[a] & can_sets[b])
        union = len(can_sets[a] | can_sets[b])
        rows.append({
            'Label': label,
            'Metric': f'pair_{a}_{b}_canonicalized',
            'Triple_Intersection': inter,
            'Union': union,
            'Intersection_Over_Union': inter / max(1, union),
            'D1_Coverage': np.nan,
            'D2_Coverage': np.nan,
            'D3_Coverage': np.nan,
        })

    return pd.DataFrame(rows), sorted(can_inter)


def _select_common_features(bundles: Dict[str, Dict], common_features: List[str]) -> Dict[str, Dict]:
    out = {}
    for dataset, bundle in bundles.items():
        idx = [bundle['feature_to_idx'][f] for f in common_features]
        out[dataset] = {
            'X': bundle['X'][:, idx].astype(np.float32, copy=False),
            'y': bundle['y'],
            'users': bundle['users'],
        }
    return out


def _clip_by_train(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip outliers using the 99.9th percentile of absolute train values."""
    clip = float(np.percentile(np.abs(X_train.reshape(-1)), 99.9))
    clip = max(10.0, clip)
    return (
        np.clip(X_train, -clip, clip).astype(np.float32),
        np.clip(X_val,   -clip, clip).astype(np.float32),
        np.clip(X_test,  -clip, clip).astype(np.float32),
        clip,
    )


def _normalize_per_user(X: np.ndarray, users: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """Per-user z-score normalization: compute mean/std from each user's train samples,
    apply to all of that user's samples."""
    X_out = X.copy()
    for user in np.unique(users):
        u_mask = users == user
        u_train_mask = u_mask & train_mask
        if u_train_mask.any():
            mean = X[u_train_mask].mean(axis=0)
            std = X[u_train_mask].std(axis=0)
        else:
            mean = X[u_mask].mean(axis=0)
            std = X[u_mask].std(axis=0)
        std[std < 1e-6] = 1.0
        X_out[u_mask] = (X[u_mask] - mean) / std
    return X_out.astype(np.float32)


def _make_stratified_split(y: np.ndarray, seed: int, val_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    if len(np.unique(y)) < 2:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(indices)
        split_at = max(1, int(round(len(indices) * (1.0 - val_ratio))))
        split_at = min(split_at, len(indices) - 1)
        return shuffled[:split_at], shuffled[split_at:]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y)), y))
    return train_idx, val_idx


def _build_cross_dataset_splits(aligned: Dict[str, Dict], train_datasets: List[str], test_dataset: str, seed: int, val_ratio: float):
    X_train_parts, y_train_parts, u_train_parts, g_train_parts = [], [], [], []

    for ds in train_datasets:
        X_train_parts.append(aligned[ds]['X'])
        y_train_parts.append(aligned[ds]['y'])
        # Prefix user IDs with dataset name to avoid collisions across datasets
        users = aligned[ds]['users']
        u_train_parts.append(np.array([f'{ds}:{u}' for u in users], dtype=object))
        groups = np.array([f'{ds}:{u}' for u in users], dtype=object)
        g_train_parts.append(groups)

    X_src = np.concatenate(X_train_parts, axis=0)
    y_src = np.concatenate(y_train_parts, axis=0)
    u_src = np.concatenate(u_train_parts, axis=0)
    g_src = np.concatenate(g_train_parts, axis=0)

    # Per-user normalization on source data (train mask covers all source samples for initial norm,
    # then val split is carved out; users entirely in val use their own mean/std)
    tr_idx, va_idx = _make_stratified_split(y_src, seed=seed, val_ratio=val_ratio)
    train_mask_src = np.zeros(len(y_src), dtype=bool)
    train_mask_src[tr_idx] = True

    X_src = _normalize_per_user(X_src, u_src, train_mask_src)

    # Per-user normalization on test dataset (no train samples — uses own stats)
    X_te_raw = aligned[test_dataset]['X']
    y_te = aligned[test_dataset]['y']
    u_te = np.array([f'{test_dataset}:{u}' for u in aligned[test_dataset]['users']], dtype=object)
    train_mask_te = np.zeros(len(u_te), dtype=bool)  # all False: test users normalize by themselves
    X_te = _normalize_per_user(X_te_raw, u_te, train_mask_te)

    X_tr = X_src[tr_idx]
    y_tr = y_src[tr_idx]
    g_tr = g_src[tr_idx]

    X_va = X_src[va_idx]
    y_va = y_src[va_idx]
    g_va = g_src[va_idx]

    X_tr, X_va, X_te, clip_val = _clip_by_train(X_tr, X_va, X_te)

    le = LabelEncoder()
    le.fit(g_src)
    d_tr = le.transform(g_tr)
    d_va = le.transform(g_va)

    u_src_tr = u_src[tr_idx]
    u_src_va = u_src[va_idx]
    num_domains = len(le.classes_)
    return X_tr, y_tr, d_tr, u_src_tr, X_va, y_va, d_va, u_src_va, X_te, y_te, u_te, num_domains, clip_val


def _sample_hparams(args, train_dataset_key: str, trial):
    search_space = get_hparams(args.model, train_dataset_key, backbone=args.backbone)
    sampled = {}
    for key, value in search_space.items():
        sampled[key] = value(trial) if callable(value) else value
    return sampled


def _run_hpo(args, train_dataset_key: str, X_tr, y_tr, d_tr, X_va, y_va, d_va, X_target, num_domains):
    if args.hpo_trials <= 0:
        return {}

    print(f'  Running HPO on source train/val split only: trials={args.hpo_trials}')

    def objective(trial):
        trial_hparams = _sample_hparams(args, train_dataset_key, trial)
        try:
            model = train_model(
                args=args,
                X_train=X_tr,
                y_train=y_tr,
                d_train=d_tr,
                X_val=X_va,
                y_val=y_va,
                d_val=d_va,
                input_dim=X_tr.shape[1],
                num_classes=2,
                num_domains=num_domains,
                hparams=trial_hparams,
                seed=args.seed,
                patience=args.patience,
                X_target=X_target,
            )
            val_metrics = evaluate_model(model, X_va, y_va)
            return float(val_metrics['AUROC'])
        except Exception as exc:
            print(f'  HPO trial failed: {exc}')
            return 0.0

    import optuna as _optuna
    _optuna.logging.set_verbosity(_optuna.logging.WARNING)
    study = _optuna.create_study(direction='maximize', sampler=_optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.hpo_trials)
    print(f'  Best HPO params: {study.best_params}')
    return study.best_params, study


def _run_experiment(args, aligned: Dict[str, Dict], common_features: List[str],
                    label: str, train_datasets: List[str], test_dataset: str,
                    records_dir: str):
    exp_args = copy.copy(args)
    exp_args.uda = bool(args.uda or (args.model in DA_MODELS and not args.disable_auto_uda))

    X_tr, y_tr, d_tr, u_tr, X_va, y_va, d_va, u_va, X_te, y_te, u_te, num_domains, clip_val = \
        _build_cross_dataset_splits(aligned, train_datasets, test_dataset, args.seed, args.val_ratio)

    setting = 'train2_test1' if len(train_datasets) == 2 else 'train1_test1'
    logger = BenchmarkLogger(output_dir=records_dir, benchmark_type="cross_dataset")
    logger.set_setting(
        label=label, model=args.model, backbone=args.backbone, seed=args.seed,
        val_ratio=args.val_ratio, hpo_trials=args.hpo_trials,
        max_epochs=args.epochs, patience=args.patience,
        train_datasets=train_datasets, test_dataset=test_dataset,
        setting_type=setting, n_common_features=len(common_features),
        common_feature_list=common_features,
    )
    logger.set_preprocessing(clip_value=clip_val)
    logger.set_split_stats(y_tr, u_tr, y_va, u_va, y_te, u_te)

    X_target = X_te if exp_args.uda and args.model in DA_MODELS else None
    with logger.time_hpo():
        best_hparams, study = _run_hpo(
            args=exp_args, train_dataset_key=train_datasets[0],
            X_tr=X_tr, y_tr=y_tr, d_tr=d_tr,
            X_va=X_va, y_va=y_va, d_va=d_va,
            X_target=X_target, num_domains=num_domains,
        )
    if study is not None:
        logger.record_hpo(study, best_hparams)
    else:
        logger.record_hpo_no_study(best_hparams)

    with logger.time_train():
        model = train_model(
            args=exp_args, X_train=X_tr, y_train=y_tr, d_train=d_tr,
            X_val=X_va, y_val=y_va, d_val=d_va,
            input_dim=X_tr.shape[1], num_classes=2, num_domains=num_domains,
            hparams=best_hparams, seed=args.seed, patience=args.patience,
            X_target=X_target,
        )
    logger.record_training()

    with logger.time_eval():
        train_m = evaluate_extended(model, X_tr, y_tr)
        val_m   = evaluate_extended(model, X_va, y_va)
        test_m  = evaluate_extended(model, X_te, y_te)

    logger.record_metrics(train_m, val_m, test_m)
    record = logger.finalize()
    rt = record["runtime"]
    tr = record["training"]
    ss = record["split_stats"]

    return {
        'Setting':         setting,
        'Label':           label,
        'Model':           args.model,
        'Backbone':        args.backbone,
        'Seed':            args.seed,
        'Val_Ratio':       args.val_ratio,
        'HPO_Trials':      args.hpo_trials,
        'Train_Datasets':  '+'.join(train_datasets),
        'Test_Dataset':    test_dataset,
        'Common_Features': len(common_features),
        'Train_Samples':   ss['train']['n_samples'],
        'Val_Samples':     ss['val']['n_samples'],
        'Test_Samples':    ss['test']['n_samples'],
        'Train_Users':     ss['train']['n_users'],
        'Val_Users':       ss['val']['n_users'],
        'Test_Users':      ss['test']['n_users'],
        'Train_PosRatio':  ss['train']['positive_ratio'],
        'Val_PosRatio':    ss['val']['positive_ratio'],
        'Test_PosRatio':   ss['test']['positive_ratio'],
        'Train_Accuracy':  train_m['accuracy'],
        'Train_AUROC':     train_m['auroc'],
        'Train_F1':        train_m['f1'],
        'Train_Precision': train_m['precision'],
        'Train_Recall':    train_m['recall'],
        'Val_Accuracy':    val_m['accuracy'],
        'Val_AUROC':       val_m['auroc'],
        'Val_F1':          val_m['f1'],
        'Val_Precision':   val_m['precision'],
        'Val_Recall':      val_m['recall'],
        'Test_Accuracy':   test_m['accuracy'],
        'Test_F1':         test_m['f1'],
        'Test_AUROC':      test_m['auroc'],
        'Test_Precision':  test_m['precision'],
        'Test_Recall':     test_m['recall'],
        'Best_Epoch':      tr.get('best_epoch'),
        'Early_Stopped':   tr.get('early_stopped'),
        'HPO_Best_AUROC':  record['hpo'].get('best_value'),
        'Total_Wall_S':    rt.get('total_wall_s'),
        'HPO_Wall_S':      rt.get('hpo_wall_s'),
        'Train_Wall_S':    rt.get('train_wall_s'),
        'Device_Name':     rt.get('device_name'),
        'Peak_GPU_MB':     rt.get('peak_gpu_memory_mb'),
        'Hparams_JSON':    json.dumps(best_hparams, default=str),
        'Experiment_ID':   record['experiment_id'],
    }


def _experiment_plan():
    two_to_one = [
        (['D-1', 'D-3'], 'D-2'),
        (['D-1', 'D-2'], 'D-3'),
        (['D-2', 'D-3'], 'D-1'),
    ]
    one_to_one = [
        (['D-1'], 'D-2'),
        (['D-3'], 'D-2'),
        (['D-1'], 'D-3'),
        (['D-2'], 'D-3'),
        (['D-2'], 'D-1'),
        (['D-3'], 'D-1'),
    ]
    return two_to_one, one_to_one


def get_args():
    parser = argparse.ArgumentParser(description='Cross-dataset benchmark with common feature intersection')
    parser.add_argument('--label', type=str, default='arousal', choices=COMMON_LABELS)
    parser.add_argument('--model', type=str, default='XGB',
                        choices=['XGB', 'LGB', 'MLP', 'ResNet', 'DANN', 'CDAN', 'DAN', 'DeepCORAL', 'MCC',
                                 'ADDA', 'MCD', 'JAN', 'SHOT', 'CBST', 'CGDM', 'TabNet', 'SAINT',
                                 'TabTransformer', 'FTTransformer', 'DCN', 'AutoInt', 'IRM', 'VREx', 'GroupDRO',
                                 'MixStyle', 'ERM_DG', 'MLDG', 'MASF', 'Fish', 'CSD', 'SagNet'])
    parser.add_argument('--backbone', type=str, default='MLP', choices=['MLP', 'ResNet', 'Transformer'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--efficient_attention', action='store_true')
    parser.add_argument('--uda', action='store_true')
    parser.add_argument('--disable_auto_uda', action='store_true',
                        help='If set, do not auto-enable UDA for DA models')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Source-only validation ratio for a single stratified split')

    parser.add_argument('--mode', type=str, default='both', choices=['analyze', 'run', 'both'])
    parser.add_argument('--run_setting', type=str, default='all', choices=['all', 'two_to_one', 'one_to_one'])
    parser.add_argument('--limit_experiments', type=int, default=None,
                        help='Optional cap on number of planned experiments (for quick smoke tests)')
    parser.add_argument('--output', type=str, default='results/cross_dataset_results.csv')
    parser.add_argument('--feature_report', type=str, default='results/cross_dataset_feature_report.csv')

    parser.add_argument('--hpo_trials', type=int, default=5,
                        help='Optuna trials on the source train/val split only')
    parser.add_argument('--hpo_mode', type=str, default='single_split',
                        help='Compatibility arg; cross-dataset uses a single source split')
    parser.add_argument('--max_folds', type=int, default=None)
    parser.add_argument('--epochs_override', type=int, default=None)

    return parser.parse_args()


def main():
    args = get_args()

    print(f'Loading cross-dataset bundles for label={args.label} ...')
    bundles = {d: _load_dataset_raw(d, args.label) for d in ['D-1', 'D-2', 'D-3']}

    for d in ['D-1', 'D-2', 'D-3']:
        print(
            f"{d}: samples={bundles[d]['X'].shape[0]}, features(raw)={len(bundles[d]['feature_names_raw'])}, "
            f"alias-collisions={bundles[d]['duplicate_alias_count']}"
        )

    feature_report_df, common_features = _compute_feature_report(args.label, bundles)

    if args.mode in ('analyze', 'both'):
        feature_report_path = Path(args.feature_report)
        feature_report_path.parent.mkdir(parents=True, exist_ok=True)
        feature_report_df.to_csv(feature_report_path, index=False)

        common_feature_json = feature_report_path.with_suffix('.common_features.json')
        with open(common_feature_json, 'w') as f:
            json.dump({'label': args.label, 'common_features': common_features}, f, indent=2)

        print('\n=== Feature Intersection Report ===')
        print(feature_report_df.to_string(index=False))
        print(f'Feature report saved to: {feature_report_path}')
        print(f'Common feature list saved to: {common_feature_json}')

    if args.mode in ('run', 'both'):
        aligned = _select_common_features(bundles, common_features)

        two_to_one, one_to_one = _experiment_plan()
        plans = []
        if args.run_setting in ('all', 'two_to_one'):
            plans.extend(two_to_one)
        if args.run_setting in ('all', 'one_to_one'):
            plans.extend(one_to_one)
        if args.limit_experiments is not None:
            plans = plans[:max(0, args.limit_experiments)]

        out_path = Path(args.output)
        records_dir = str(out_path.parent / 'records')
        rows = []
        print(
            f'\nRunning {len(plans)} cross-dataset experiments with {len(common_features)} common features '
            f'(val_ratio={args.val_ratio}, hpo_trials={args.hpo_trials})...'
        )
        for i, (train_ds, test_ds) in enumerate(plans, start=1):
            print(f'[{i}/{len(plans)}] Train={"+".join(train_ds)} -> Test={test_ds}')
            row = _run_experiment(args, aligned, common_features, args.label, train_ds, test_ds, records_dir)
            rows.append(row)
            print(
                f"  Test AUROC={row['Test_AUROC']:.4f}, Test F1={row['Test_F1']:.4f}, "
                f"Test ACC={row['Test_Accuracy']:.4f}"
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(out_path, index=False)
        print(f'\nCross-dataset results saved to: {out_path}')


if __name__ == '__main__':
    main()