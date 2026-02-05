#!/usr/bin/env python3
"""LOSO zero-shot evaluation on D-4 using multiple domain_adaptation models."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence

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
from domain_adaptation.models.common import ArrayDataset  # noqa: E402
from domain_adaptation.models.dl_erm import DLERMConfig, DLERMPipeline  # noqa: E402
from domain_adaptation.models.dl_dann import DLDANNConfig, DLDANNPipeline  # noqa: E402
from domain_adaptation.models.dl_irm import DLIRMConfig, DLIRMPipeline  # noqa: E402
from domain_adaptation.models.dl_csd import DLCSConfig, DLCSDPipeline  # noqa: E402
from domain_adaptation.models.dl_mldg import DLMldgConfig, DLMldgPipeline  # noqa: E402
from domain_adaptation.models.tree import LightGBMConfig, LightGBMPipeline  # noqa: E402
from domain_adaptation.models.transformer import TransformerConfig, TransformerPipeline  # noqa: E402
from domain_adaptation.pipeline import (  # noqa: E402
    _fit_scaler,
    _split_array_dataset,
    _to_array_dataset,
)


def _load_bundle(dataset_path: Path) -> DatasetBundle:
    bundle = load_dataset(dataset_path, name="D-4")
    return augment_temporal_and_group_features([bundle])[0]


def _empty_dataset(feature_names: List[str], scaler) -> ArrayDataset:
    return _to_array_dataset(
        pd.DataFrame(columns=feature_names),
        np.empty((0,), dtype=np.float32),
        scaler,
    )


def _make_lightgbm_config(num_threads: int) -> LightGBMConfig:
    return LightGBMConfig(
        pretrain_rounds=256,
        finetune_rounds=0,
        adapt_rounds=0,
        learning_rate=0.05,
        num_leaves=64,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_child_samples=20,
        num_threads=num_threads,
        early_stopping_rounds=25,
    )


def _make_dl_configs(input_dim: int) -> Dict[str, object]:
    base_dims = (256, 256)
    dl_erm = DLERMConfig(
        input_dim=input_dim,
        hidden_dims=base_dims,
        dropout=0.3,
        pretrain_epochs=50,
        finetune_epochs=0,
        adapt_epochs=0,
        batch_size=128,
        pretrain_lr=1e-3,
        finetune_lr=5e-4,
        adapt_lr=2e-4,
        weight_decay=1e-4,
        early_stopping_patience=10,
    )
    dl_dann = DLDANNConfig(
        input_dim=input_dim,
        hidden_dims=base_dims,
        domain_hidden_dims=(128,),
        domain_dropout=0.1,
        pretrain_epochs=50,
        finetune_epochs=0,
        adapt_epochs=0,
        batch_size=128,
    )
    dl_irm = DLIRMConfig(
        input_dim=input_dim,
        hidden_dims=base_dims,
        pretrain_epochs=50,
        finetune_epochs=0,
        adapt_epochs=0,
        batch_size=128,
    )
    dl_csd = DLCSConfig(
        input_dim=input_dim,
        hidden_dims=base_dims,
        domain_count=2,
        pretrain_epochs=50,
        finetune_epochs=0,
        adapt_epochs=0,
        batch_size=128,
    )
    dl_mldg = DLMldgConfig(
        input_dim=input_dim,
        hidden_dims=base_dims,
        pretrain_epochs=50,
        finetune_epochs=0,
        adapt_epochs=0,
        batch_size=128,
    )
    return {
        "dl_erm": dl_erm,
        "dl_dann": dl_dann,
        "dl_irm": dl_irm,
        "dl_csd": dl_csd,
        "dl_mldg": dl_mldg,
    }


def _make_transformer_config(input_dim: int, feature_names: List[str]) -> TransformerConfig:
    return TransformerConfig(
        input_dim=input_dim,
        feature_names=tuple(feature_names),
        d_model=128,
        n_heads=4,
        num_layers=2,
        dropout=0.1,
        pretrain_epochs=30,
        finetune_epochs=0,
        adapt_epochs=0,
        batch_size=128,
        pretrain_lr=1e-3,
        finetune_lr=5e-4,
        adapt_lr=2e-4,
        weight_decay=1e-4,
        grad_clip=1.0,
        use_cosine_scheduler=False,
        use_plateau_scheduler=False,
    )


def run_zero_shot(
    *,
    dataset_path: Path,
    output_csv: Path,
    models: Sequence[str],
    pretrain_val_ratio: float,
    feature_clip_value: float,
    include_target_in_scaler: bool,
    num_threads: int,
    seed: int,
) -> pd.DataFrame:
    bundle = _load_bundle(dataset_path)
    feature_names = bundle.feature_names
    dl_configs = _make_dl_configs(len(feature_names))
    transformer_config = _make_transformer_config(len(feature_names), feature_names)
    lgbm_config = _make_lightgbm_config(num_threads)

    loso_pairs = list(loso_splits(bundle.groups))
    results: List[Dict[str, object]] = []

    for fold_id, (train_idx, test_idx) in enumerate(loso_pairs):
        test_user = str(bundle.groups[test_idx][0])
        train_frame = bundle.features.iloc[train_idx].reset_index(drop=True)
        train_labels = bundle.labels[train_idx].astype(int)
        test_frame = bundle.features.iloc[test_idx].reset_index(drop=True)
        test_labels = bundle.labels[test_idx].astype(int)

        if train_frame.empty or test_frame.empty:
            continue

        extra_frames = [test_frame] if include_target_in_scaler else []
        scaler = _fit_scaler([train_frame], extra_frames)
        pretrain_full = _to_array_dataset(train_frame, train_labels, scaler, clip_value=feature_clip_value)
        pretrain_train, pretrain_val = _split_array_dataset(
            pretrain_full,
            pretrain_val_ratio,
            seed=seed + fold_id,
        )
        eval_ds = _to_array_dataset(test_frame, test_labels, scaler, clip_value=feature_clip_value)
        empty_train = _empty_dataset(feature_names, scaler)

        common_meta = {
            "user": test_user,
            "fold_id": fold_id,
            "train_samples": float(len(pretrain_train.y)),
            "val_samples": float(len(pretrain_val.y)) if pretrain_val is not None else 0.0,
            "eval_samples": float(len(eval_ds.y)),
        }

        for model_name in models:
            if model_name == "lightgbm":
                pipeline = LightGBMPipeline(lgbm_config)
                result = pipeline.run(
                    pretrain=pretrain_train,
                    pretrain_val=pretrain_val,
                    train=empty_train,
                    val=empty_train,
                    adapt=None,
                    evaluation=eval_ds,
                )
                results.append(
                    {
                        **common_meta,
                        "model": "lightgbm",
                        "mode": "zero_shot",
                        "train_auroc": result.train_auroc,
                        "val_auroc": result.val_auroc,
                        "test_auroc": result.test_auroc,
                        "train_accuracy": result.train_accuracy,
                        "val_accuracy": result.val_accuracy,
                        "test_accuracy": result.test_accuracy,
                        "train_auprc": result.train_auprc,
                        "val_auprc": result.val_auprc,
                        "test_auprc": result.test_auprc,
                        "pretrain_val_auroc": result.pretrain_val_auroc,
                        "pretrain_val_accuracy": result.pretrain_val_accuracy,
                        "pretrain_val_auprc": result.pretrain_val_auprc,
                        "pretrain_seconds": result.stage_durations.get("pretrain_seconds"),
                        "finetune_seconds": result.stage_durations.get("finetune_seconds"),
                        "adapt_seconds": result.stage_durations.get("adapt_seconds"),
                        "best_iteration": result.best_iteration,
                    }
                )
                continue

            if model_name == "transformer":
                transformer_cfg = transformer_config
                pipeline = TransformerPipeline(transformer_cfg)
                result = pipeline.run(
                    seed=seed,
                    pretrain=pretrain_train,
                    pretrain_val=pretrain_val,
                    train=empty_train,
                    val=empty_train,
                    adapt=None,
                    evaluation=eval_ds,
                )
                results.append(
                    {
                        **common_meta,
                        "model": "transformer",
                        "mode": "zero_shot",
                        "train_auroc": result.train_auroc,
                        "val_auroc": result.val_auroc,
                        "test_auroc": result.test_auroc,
                        "train_accuracy": result.train_accuracy,
                        "val_accuracy": result.val_accuracy,
                        "test_accuracy": result.test_accuracy,
                        "train_auprc": result.train_auprc,
                        "val_auprc": result.val_auprc,
                        "test_auprc": result.test_auprc,
                        "pretrain_val_auroc": result.pretrain_val_auroc,
                        "pretrain_val_accuracy": result.pretrain_val_accuracy,
                        "pretrain_val_auprc": result.pretrain_val_auprc,
                        "pretrain_seconds": result.stage_durations.get("pretrain_seconds"),
                        "finetune_seconds": result.stage_durations.get("finetune_seconds"),
                        "adapt_seconds": result.stage_durations.get("adapt_seconds"),
                        "best_iteration": None,
                    }
                )
                continue

            if model_name in dl_configs:
                cfg = dl_configs[model_name]
                if model_name == "dl_erm":
                    pipeline = DLERMPipeline(cfg)  # type: ignore[arg-type]
                elif model_name == "dl_dann":
                    pipeline = DLDANNPipeline(cfg)  # type: ignore[arg-type]
                elif model_name == "dl_irm":
                    pipeline = DLIRMPipeline(cfg)  # type: ignore[arg-type]
                elif model_name == "dl_csd":
                    pipeline = DLCSDPipeline(cfg)  # type: ignore[arg-type]
                elif model_name == "dl_mldg":
                    pipeline = DLMldgPipeline(cfg)  # type: ignore[arg-type]
                else:
                    continue

                result = pipeline.run(
                    seed=seed,
                    pretrain=pretrain_train,
                    pretrain_val=pretrain_val,
                    train=empty_train,
                    val=empty_train,
                    adapt=None,
                    evaluation=eval_ds,
                )
                results.append(
                    {
                        **common_meta,
                        "model": model_name,
                        "mode": "zero_shot",
                        "train_auroc": result.train_auroc,
                        "val_auroc": result.val_auroc,
                        "test_auroc": result.test_auroc,
                        "train_accuracy": result.train_accuracy,
                        "val_accuracy": result.val_accuracy,
                        "test_accuracy": result.test_accuracy,
                        "train_auprc": result.train_auprc,
                        "val_auprc": result.val_auprc,
                        "test_auprc": result.test_auprc,
                        "pretrain_val_auroc": result.pretrain_val_auroc,
                        "pretrain_val_accuracy": result.pretrain_val_accuracy,
                        "pretrain_val_auprc": result.pretrain_val_auprc,
                        "pretrain_seconds": result.stage_durations.get("pretrain_seconds"),
                        "finetune_seconds": result.stage_durations.get("finetune_seconds"),
                        "adapt_seconds": result.stage_durations.get("adapt_seconds"),
                        "best_iteration": None,
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
    parser = argparse.ArgumentParser(description="Run zero-shot LOSO on D-4 with multiple models (no fine-tuning).")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/var/nfs_share/Overfitting/D-4/Intermediate/stress_binary_personal-current.pkl"),
        help="Path to the D-4 pickle file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/d4_loso_zero_shot.csv"),
        help="Where to write the per-user metrics CSV.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["dl_erm"],
        choices=["lightgbm", "transformer", "dl_erm", "dl_dann", "dl_irm", "dl_csd", "dl_mldg"],
        help="Model pipelines to run.",
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
        "--include-target-in-scaler",
        action="store_true",
        help="Include the held-out user's features when fitting the scaler.",
    )
    parser.add_argument(
        "--exclude-target-in-scaler",
        action="store_false",
        dest="include_target_in_scaler",
        help="Fit the scaler using only source users.",
    )
    parser.set_defaults(include_target_in_scaler=True)
    parser.add_argument("--num-threads", type=int, default=8, help="LightGBM thread count.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_zero_shot(
        dataset_path=args.dataset,
        output_csv=args.output_csv,
        models=args.models,
        pretrain_val_ratio=args.pretrain_val_ratio,
        feature_clip_value=args.feature_clip_value,
        include_target_in_scaler=args.include_target_in_scaler,
        num_threads=args.num_threads,
        seed=args.seed,
    )
    if df.empty:
        print("No results were generated.")
        return
    print(f"Wrote {len(df)} rows to {args.output_csv}")
    summary = (
        df.groupby(["model"])["test_auroc"]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
    )
    print("\nPer-model AUROC summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
