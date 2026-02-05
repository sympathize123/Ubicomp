#!/usr/bin/env python3
"""LOSO evalu ation on the D-4 dataset usingthe domain_adaptation LightGBM pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from domain_adaptation.data_utils import (  # noqa: E402
    DatasetBundle,
    augment_temporal_and_group_features,
    load_dataset,
    loso_splits,
)
from domain_adaptation.models.tree import LightGBMConfig, LightGBMPipeline  # noqa: E402
from domain_adaptation.pipeline import (  # noqa: E402
    _fit_scaler,
    _prepare_target_datasets,
    _split_array_dataset,
    _split_target_indices,
    _to_array_dataset,
)


def _make_config(num_threads: int) -> LightGBMConfig:
    return LightGBMConfig(
        pretrain_rounds=256,
        finetune_rounds=128,
        adapt_rounds=64,
        learning_rate=0.05,
        num_leaves=64,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_child_samples=20,
        num_threads=num_threads,
        early_stopping_rounds=25,
    )


def _load_d4_bundle(path: Path) -> DatasetBundle:
    bundle = load_dataset(path, name="D-4")
    # Preserve the feature engineering used throughout the domain_adaptation package.
    bundle = augment_temporal_and_group_features([bundle])[0]
    return bundle


def run_loso_lightgbm(
    *,
    dataset_path: Path,
    output_csv: Path,
    fine_tune_ratios: Sequence[float],
    fine_tune_val_ratio: float,
    pretrain_val_ratio: float,
    feature_clip_value: float,
    num_threads: int,
    seed: int = 42,
) -> pd.DataFrame:
    bundle = _load_d4_bundle(dataset_path)
    users = np.unique(bundle.groups)
    base_config = _make_config(num_threads)
    baseline_config = LightGBMConfig(
        pretrain_rounds=0,
        finetune_rounds=base_config.finetune_rounds,
        adapt_rounds=0,
        learning_rate=base_config.learning_rate,
        num_leaves=base_config.num_leaves,
        feature_fraction=base_config.feature_fraction,
        bagging_fraction=base_config.bagging_fraction,
        bagging_freq=base_config.bagging_freq,
        min_child_samples=base_config.min_child_samples,
        num_threads=base_config.num_threads,
        early_stopping_rounds=base_config.early_stopping_rounds,
    )

    results: list[dict[str, object]] = []
    loso_pairs = list(loso_splits(bundle.groups))
    for fold_id, (train_idx, test_idx) in enumerate(loso_pairs):
        test_user = str(bundle.groups[test_idx][0])
        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)

        train_frame = bundle.features.iloc[train_idx].reset_index(drop=True)
        train_labels = bundle.labels[train_idx].astype(int)
        test_frame = bundle.features.iloc[test_idx].reset_index(drop=True)
        test_labels = bundle.labels[test_idx].astype(int)

        if train_frame.empty or test_frame.empty:
            continue

        scaler = _fit_scaler([train_frame], [test_frame])
        pretrain_full = _to_array_dataset(train_frame, train_labels, scaler, clip_value=feature_clip_value)
        pretrain_train, pretrain_val = _split_array_dataset(
            pretrain_full,
            pretrain_val_ratio,
            seed=seed + fold_id,
        )

        for ft_ratio in fine_tune_ratios:
            train_split_idx, val_split_idx, eval_idx, effective_ratio = _split_target_indices(
                test_labels,
                ft_ratio,
                fine_tune_val_ratio,
                seed=seed,
            )

            train_ds, val_ds, eval_ds, target_counts = _prepare_target_datasets(
                test_frame,
                test_labels,
                scaler,
                train_split_idx,
                val_split_idx,
                eval_idx,
                clip_value=feature_clip_value,
                domain_id=None,
            )

            pipeline = LightGBMPipeline(base_config)
            run_result = pipeline.run(
                pretrain=pretrain_train,
                pretrain_val=pretrain_val,
                train=train_ds,
                val=val_ds,
                adapt=None,
                evaluation=eval_ds,
            )
            results.append(
                {
                    "user": test_user,
                    "fold_id": fold_id,
                    "mode": "pretrain_finetune",
                    "ft_ratio": float(ft_ratio),
                    "ft_ratio_effective": effective_ratio,
                    "train_samples": float(len(train_ds.y)),
                    "val_samples": float(len(val_ds.y)),
                    "eval_samples": float(len(eval_ds.y)),
                    "pretrain_samples": float(len(pretrain_train.y)),
                    "pretrain_val_samples": float(len(pretrain_val.y)) if pretrain_val is not None else 0.0,
                    "train_auroc": run_result.train_auroc,
                    "val_auroc": run_result.val_auroc,
                    "test_auroc": run_result.test_auroc,
                    "train_accuracy": run_result.train_accuracy,
                    "val_accuracy": run_result.val_accuracy,
                    "test_accuracy": run_result.test_accuracy,
                    "train_auprc": run_result.train_auprc,
                    "val_auprc": run_result.val_auprc,
                    "test_auprc": run_result.test_auprc,
                    "pretrain_val_auroc": run_result.pretrain_val_auroc,
                    "pretrain_val_accuracy": run_result.pretrain_val_accuracy,
                    "pretrain_val_auprc": run_result.pretrain_val_auprc,
                    "pretrain_seconds": run_result.stage_durations.get("pretrain_seconds"),
                    "finetune_seconds": run_result.stage_durations.get("finetune_seconds"),
                    "adapt_seconds": run_result.stage_durations.get("adapt_seconds"),
                    "best_iteration": run_result.best_iteration,
                }
            )

            if ft_ratio > 0 and train_ds.X.size > 0:
                baseline_pipeline = LightGBMPipeline(baseline_config)
                baseline_result = baseline_pipeline.run(
                    pretrain=None,
                    pretrain_val=None,
                    train=train_ds,
                    val=val_ds,
                    adapt=None,
                    evaluation=eval_ds,
                )
                results.append(
                    {
                        "user": test_user,
                        "fold_id": fold_id,
                        "mode": "target_only",
                        "ft_ratio": float(ft_ratio),
                        "ft_ratio_effective": effective_ratio,
                        "train_samples": float(len(train_ds.y)),
                        "val_samples": float(len(val_ds.y)),
                        "eval_samples": float(len(eval_ds.y)),
                        "pretrain_samples": 0.0,
                        "pretrain_val_samples": 0.0,
                        "train_auroc": baseline_result.train_auroc,
                        "val_auroc": baseline_result.val_auroc,
                        "test_auroc": baseline_result.test_auroc,
                        "train_accuracy": baseline_result.train_accuracy,
                        "val_accuracy": baseline_result.val_accuracy,
                        "test_accuracy": baseline_result.test_accuracy,
                        "train_auprc": baseline_result.train_auprc,
                        "val_auprc": baseline_result.val_auprc,
                        "test_auprc": baseline_result.test_auprc,
                        "pretrain_val_auroc": baseline_result.pretrain_val_auroc,
                        "pretrain_val_accuracy": baseline_result.pretrain_val_accuracy,
                        "pretrain_val_auprc": baseline_result.pretrain_val_auprc,
                        "pretrain_seconds": baseline_result.stage_durations.get("pretrain_seconds"),
                        "finetune_seconds": baseline_result.stage_durations.get("finetune_seconds"),
                        "adapt_seconds": baseline_result.stage_durations.get("adapt_seconds"),
                        "best_iteration": baseline_result.best_iteration,
                    }
                )

    df = pd.DataFrame(results)
    if not df.empty:
        float_cols = df.select_dtypes(include=[np.floating]).columns
        df[float_cols] = df[float_cols].round(4)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LOSO LightGBM evaluation on the D-4 dataset.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/var/nfs_share/Overfitting/D-4/Intermediate/stress_binary_personal-current.pkl"),
        help="Path to the D-4 pickle file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/d4_loso_lightgbm.csv"),
        help="Where to write the per-user metrics CSV.",
    )
    parser.add_argument(
        "--fine-tune-ratios",
        type=float,
        nargs="+",
        default=[0.0, 0.2],
        help="Portion of target-user samples used for fine-tuning (0 means zero-shot).",
    )
    parser.add_argument(
        "--fine-tune-val-ratio",
        type=float,
        default=0.2,
        help="Share of fine-tuning samples kept for validation.",
    )
    parser.add_argument(
        "--pretrain-val-ratio",
        type=float,
        default=0.1,
        help="Validation share for the pretrain (other-user) split.",
    )
    parser.add_argument(
        "--feature-clip-value",
        type=float,
        default=5.0,
        help="Clip standardized features into [-value, value] before training.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=8,
        help="LightGBM thread count.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for splits.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_loso_lightgbm(
        dataset_path=args.dataset,
        output_csv=args.output_csv,
        fine_tune_ratios=args.fine_tune_ratios,
        fine_tune_val_ratio=args.fine_tune_val_ratio,
        pretrain_val_ratio=args.pretrain_val_ratio,
        feature_clip_value=args.feature_clip_value,
        num_threads=args.num_threads,
        seed=args.seed,
    )
    if df.empty:
        print("No results were generated.")
        return
    print(f"Wrote {len(df)} rows to {args.output_csv}")
    summary = (
        df.groupby(["mode", "ft_ratio"])["test_auroc"]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
    )
    print("\nPer-mode AUROC summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
