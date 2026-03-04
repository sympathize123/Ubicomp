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
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from execute_benchmark import make_groupwise_val_split, train_model
from src.models import evaluate_model

BASE_DATA_DIR = '/home/iclab/minseo/Ubicomp/data'
COMMON_LABELS = ['arousal', 'disturbance', 'valence']
DATASET_PATH_TMPL = {
    'D-1': '{label}_personal-full_D#2.pkl',
    'D-2': '{label}_personal-full_D#3.pkl',
    'D-3': '{label}_personal-full.pkl',
}
DA_MODELS = ['DANN', 'CDAN', 'DAN', 'DeepCORAL', 'MCC', 'ADDA', 'MCD', 'JAN', 'SHOT', 'CBST', 'CGDM']

RESULT_COLUMNS = [
    'Setting', 'Label', 'Model', 'Backbone', 'Train_Datasets', 'Test_Dataset',
    'Common_Features', 'Train_Samples', 'Val_Samples', 'Test_Samples',
    'Train_Accuracy', 'Train_AUROC', 'Val_Accuracy', 'Val_AUROC',
    'Test_Accuracy', 'Test_F1', 'Test_AUROC'
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
    return os.path.join(BASE_DATA_DIR, DATASET_PATH_TMPL[dataset].format(label=label))


def _load_dataset_raw(dataset: str, label: str) -> Dict[str, np.ndarray]:
    path = _dataset_path(dataset, label)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing dataset file: {path}')

    with open(path, 'rb') as f:
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


def _standardize_by_train(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std < 1e-6] = 1.0

    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std
    X_test_n = (X_test - mean) / std

    # Robust clipping by train stats only
    clip = np.percentile(np.abs(X_train_n.reshape(-1)), 99.9)
    clip = max(10.0, float(clip))

    X_train_n = np.clip(X_train_n, -clip, clip).astype(np.float32)
    X_val_n = np.clip(X_val_n, -clip, clip).astype(np.float32)
    X_test_n = np.clip(X_test_n, -clip, clip).astype(np.float32)
    return X_train_n, X_val_n, X_test_n


def _build_cross_dataset_splits(aligned: Dict[str, Dict], train_datasets: List[str], test_dataset: str, seed: int):
    X_train_parts, y_train_parts, g_train_parts = [], [], []

    for ds in train_datasets:
        X_train_parts.append(aligned[ds]['X'])
        y_train_parts.append(aligned[ds]['y'])
        users = aligned[ds]['users']
        groups = np.array([f'{ds}:{u}' for u in users], dtype=object)
        g_train_parts.append(groups)

    X_src = np.concatenate(X_train_parts, axis=0)
    y_src = np.concatenate(y_train_parts, axis=0)
    g_src = np.concatenate(g_train_parts, axis=0)

    source_idx = np.arange(len(y_src))
    tr_idx, va_idx = make_groupwise_val_split(source_idx, y_src, g_src, seed=seed)

    X_tr = X_src[tr_idx]
    y_tr = y_src[tr_idx]
    g_tr = g_src[tr_idx]

    X_va = X_src[va_idx]
    y_va = y_src[va_idx]
    g_va = g_src[va_idx]

    X_te = aligned[test_dataset]['X']
    y_te = aligned[test_dataset]['y']

    X_tr, X_va, X_te = _standardize_by_train(X_tr, X_va, X_te)

    le = LabelEncoder()
    le.fit(g_src)
    d_tr = le.transform(g_tr)
    d_va = le.transform(g_va)

    num_domains = len(le.classes_)
    return X_tr, y_tr, d_tr, X_va, y_va, d_va, X_te, y_te, num_domains


def _run_experiment(args, aligned: Dict[str, Dict], common_features: List[str], label: str, train_datasets: List[str], test_dataset: str):
    X_tr, y_tr, d_tr, X_va, y_va, d_va, X_te, y_te, num_domains = _build_cross_dataset_splits(
        aligned=aligned,
        train_datasets=train_datasets,
        test_dataset=test_dataset,
        seed=args.seed,
    )

    # Enable UDA automatically for DA models unless disabled
    use_uda = args.uda or (args.model in DA_MODELS and not args.disable_auto_uda)
    args.uda = bool(use_uda)

    X_target = X_te if args.uda and args.model in DA_MODELS else None

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
        hparams={},
        seed=args.seed,
        patience=args.patience,
        X_target=X_target,
    )

    train_metrics = evaluate_model(model, X_tr, y_tr)
    val_metrics = evaluate_model(model, X_va, y_va)
    test_metrics = evaluate_model(model, X_te, y_te)

    setting = 'train2_test1' if len(train_datasets) == 2 else 'train1_test1'
    return {
        'Setting': setting,
        'Label': label,
        'Model': args.model,
        'Backbone': args.backbone,
        'Train_Datasets': '+'.join(train_datasets),
        'Test_Dataset': test_dataset,
        'Common_Features': len(common_features),
        'Train_Samples': int(len(y_tr)),
        'Val_Samples': int(len(y_va)),
        'Test_Samples': int(len(y_te)),
        'Train_Accuracy': train_metrics['Accuracy'],
        'Train_AUROC': train_metrics['AUROC'],
        'Val_Accuracy': val_metrics['Accuracy'],
        'Val_AUROC': val_metrics['AUROC'],
        'Test_Accuracy': test_metrics['Accuracy'],
        'Test_F1': test_metrics['F1'],
        'Test_AUROC': test_metrics['AUROC'],
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
                                 'ADDA', 'MCD', 'JAN', 'SHOT', 'CBST', 'CGDM', 'TabNet', 'TabPFN', 'SAINT',
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

    parser.add_argument('--mode', type=str, default='both', choices=['analyze', 'run', 'both'])
    parser.add_argument('--run_setting', type=str, default='all', choices=['all', 'two_to_one', 'one_to_one'])
    parser.add_argument('--limit_experiments', type=int, default=None,
                        help='Optional cap on number of planned experiments (for quick smoke tests)')
    parser.add_argument('--output', type=str, default='results/cross_dataset_results.csv')
    parser.add_argument('--feature_report', type=str, default='results/cross_dataset_feature_report.csv')

    # Kept for compatibility with execute_benchmark.train_model signature
    parser.add_argument('--hpo_trials', type=int, default=0)
    parser.add_argument('--hpo_mode', type=str, default='fold1')
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

        rows = []
        print(f'\nRunning {len(plans)} cross-dataset experiments with {len(common_features)} common features...')
        for i, (train_ds, test_ds) in enumerate(plans, start=1):
            print(f'[{i}/{len(plans)}] Train={"+".join(train_ds)} -> Test={test_ds}')
            row = _run_experiment(args, aligned, common_features, args.label, train_ds, test_ds)
            rows.append(row)
            print(
                f"  Test AUROC={row['Test_AUROC']:.4f}, Test F1={row['Test_F1']:.4f}, "
                f"Test ACC={row['Test_Accuracy']:.4f}"
            )

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(out_path, index=False)
        print(f'\nCross-dataset results saved to: {out_path}')


if __name__ == '__main__':
    main()
