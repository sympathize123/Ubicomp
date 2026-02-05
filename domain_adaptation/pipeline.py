"""High-level orchestration for pretrain → fine-tune experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover
    LogisticRegression = None

from .cache_utils import CacheManager
from .data_utils import (
    DEFAULT_DATASET_PATHS,
    DatasetStore,
    align_feature_intersection,
    augment_temporal_and_group_features,
    load_datasets,
    loso_splits,
    stratified_group_kfold_splits,
)
from .models.cdtrans import CDTransConfig, CDTransPipeline
from .models.common import ArrayDataset
from .models.dl_erm import DLERMConfig, DLERMPipeline, DLERMRunResult
from .models.dl_dann import DLDANNConfig, DLDANNPipeline, DLDANNRunResult
from .models.dl_irm import DLIRMConfig, DLIRMPipeline, DLIRMRunResult
from .models.dl_csd import DLCSConfig, DLCSDPipeline, DLCSRunResult
from .models.dl_mldg import DLMldgConfig, DLMldgPipeline, DLMldgRunResult
from .models.dl_clustering import DLClusteringConfig, DLClusteringPipeline, DLClusteringRunResult
from .models.dl_siamese import DLSiameseConfig, DLSiamesePipeline, DLSiameseRunResult
from .models.dl_reorder import DLReorderConfig, DLReorderPipeline, DLReorderRunResult
from .models.dl_masf import DLMASFConfig, DLMASFPipeline, DLMASFRunResult
from .models.tabpfn import TabPFNConfig, TabPFNPipeline, TabPFNRunResult
from .models.tree import LightGBMConfig, LightGBMPipeline
from .models.transformer import TransformerConfig, TransformerPipeline

try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None


@dataclass
class ExperimentScenario:
    target: str
    sources: List[str]

    @property
    def name(self) -> str:
        return f"{'+'.join(self.sources)}→{self.target}"


@dataclass
class TargetSplits:
    train_features: pd.DataFrame
    val_features: pd.DataFrame
    test_features: pd.DataFrame
    train_labels: np.ndarray
    val_labels: np.ndarray
    test_labels: np.ndarray
@dataclass
class ScenarioSplit:
    split: TargetSplits
    metadata: Dict[str, object]


def generate_all_scenarios(dataset_names: Sequence[str]) -> List[ExperimentScenario]:
    unique_ordered = list(dict.fromkeys(dataset_names))
    scenarios: List[ExperimentScenario] = []
    for target in unique_ordered:
        sources = [name for name in unique_ordered if name != target]
        for count in range(1, len(sources) + 1):
            for combo in combinations(sources, count):
                scenarios.append(ExperimentScenario(target=target, sources=list(combo)))
    scenarios.sort(key=lambda scenario: (scenario.target, len(scenario.sources), tuple(scenario.sources)))
    return scenarios


def _make_stratified_shuffle_split(bundle: DatasetBundle, *, seed: int, val_size: float, test_size: float) -> TargetSplits:
    X = bundle.features.values
    y = bundle.labels.astype(int)

    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(sss_test.split(X, y))

    train_features = bundle.features.iloc[train_idx]
    train_labels = y[train_idx]
    test_features = bundle.features.iloc[test_idx]
    test_labels = y[test_idx]

    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_sub_idx, val_idx = next(sss_val.split(train_features.values, train_labels))

    final_train_features = train_features.iloc[train_sub_idx].reset_index(drop=True)
    final_train_labels = train_labels[train_sub_idx]
    val_features = train_features.iloc[val_idx].reset_index(drop=True)
    val_labels = train_labels[val_idx]
    test_features = test_features.reset_index(drop=True)

    return TargetSplits(
        train_features=final_train_features,
        val_features=val_features,
        test_features=test_features,
        train_labels=final_train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
    )

def _fit_scaler(
    source_frames: Sequence[pd.DataFrame],
    extra_frames: Sequence[pd.DataFrame],
) -> StandardScaler:
    scaler = StandardScaler()
    matrices: List[np.ndarray] = []
    for frame in source_frames:
        if not frame.empty:
            matrices.append(frame.values)
    for frame in extra_frames:
        if not frame.empty:
            matrices.append(frame.values)
    if not matrices:
        raise ValueError("Unable to fit scaler without any data")
    combined = np.vstack(matrices)
    scaler.fit(combined)
    return scaler


def _to_array_dataset(
    frame: pd.DataFrame,
    labels: np.ndarray,
    scaler: StandardScaler,
    *,
    clip_value: Optional[float] = None,
    domain_values: Optional[np.ndarray] = None,
    domain_id: Optional[int] = None,
) -> ArrayDataset:
    feature_dim = scaler.mean_.shape[0]
    if frame.empty:
        empty_X = np.empty((0, feature_dim), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.float32)
        if domain_values is not None:
            domain_arr = np.empty((0,), dtype=np.int64)
        elif domain_id is not None:
            domain_arr = np.empty((0,), dtype=np.int64)
        else:
            domain_arr = None
        return ArrayDataset(empty_X, empty_y, domain_arr)

    matrix = scaler.transform(frame.values).astype(np.float32)
    if clip_value is not None and clip_value > 0:
        matrix = np.clip(matrix, -clip_value, clip_value)
    domain_arr: Optional[np.ndarray]
    if domain_values is not None:
        domain_arr = np.asarray(domain_values, dtype=np.int64)
    elif domain_id is not None:
        domain_arr = np.full(labels.shape, int(domain_id), dtype=np.int64)
    else:
        domain_arr = None
    return ArrayDataset(matrix, labels.astype(np.float32), domain_arr)


def _combine_sources(datasets: Sequence[ArrayDataset]) -> ArrayDataset:
    if not datasets:
        return ArrayDataset(np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32))
    X = np.vstack([ds.X for ds in datasets]) if len(datasets) > 1 else datasets[0].X
    y = np.concatenate([ds.y for ds in datasets]) if len(datasets) > 1 else datasets[0].y
    if all(ds.domains is not None for ds in datasets):
        domains = np.concatenate([ds.domains for ds in datasets]) if len(datasets) > 1 else datasets[0].domains
    else:
        domains = None
    return ArrayDataset(X, y, domains)


def _split_array_dataset(
    dataset: ArrayDataset,
    val_ratio: float,
    seed: int,
) -> Tuple[ArrayDataset, Optional[ArrayDataset]]:
    if dataset.X.size == 0 or val_ratio <= 0.0:
        return dataset, None
    ratio = max(0.0, min(float(val_ratio), 0.5 if val_ratio < 1.0 else 1.0))
    stratify = dataset.y if len(np.unique(dataset.y)) > 1 else None
    indices = np.arange(dataset.X.shape[0])
    try:
        idx_train, idx_val = train_test_split(
            indices,
            test_size=ratio,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        return dataset, None
    X_train = dataset.X[idx_train]
    y_train = dataset.y[idx_train]
    X_val = dataset.X[idx_val]
    y_val = dataset.y[idx_val]
    if dataset.domains is not None:
        d_train = dataset.domains[idx_train]
        d_val = dataset.domains[idx_val]
    else:
        d_train = d_val = None
    return ArrayDataset(X_train, y_train, d_train), ArrayDataset(X_val, y_val, d_val)


def _resolve_target_ratios(
    train_ratio: Optional[float],
    val_ratio: Optional[float],
    test_ratio: Optional[float],
    fallback_val_size: float,
    fallback_test_size: float,
) -> Tuple[float, float, float]:
    test = fallback_test_size if test_ratio is None else float(test_ratio)
    if not 0 <= test < 1:
        raise ValueError("target test ratio must be in [0, 1)")
    remaining = 1.0 - test
    if remaining <= 0:
        raise ValueError("target test ratio leaves no room for train/val splits")

    if train_ratio is not None and val_ratio is not None:
        train = float(train_ratio)
        val = float(val_ratio)
    elif train_ratio is not None:
        train = float(train_ratio)
        val = remaining - train
    elif val_ratio is not None:
        val = float(val_ratio)
        train = remaining - val
    else:
        default_val = remaining * fallback_val_size
        val = default_val
        train = remaining - val

    for name, value in (("train", train), ("val", val), ("test", test)):
        if value < 0:
            raise ValueError(f"target {name} ratio became negative ({value})")
    total = train + val + test
    if total <= 0:
        raise ValueError("target split ratios sum to zero")
    if not np.isclose(total, 1.0):
        train /= total
        val /= total
        test /= total
    return float(train), float(val), float(test)


def _resolve_feature_strategy(strategy: str, fine_tune_ratios: Sequence[float]) -> str:
    normalized = strategy.lower()
    valid = {"auto", "all_union", "source_union", "intersection"}
    if normalized not in valid:
        raise ValueError(f"feature_strategy must be one of {sorted(valid)}")
    if normalized != "auto":
        return normalized
    ratios = list(fine_tune_ratios)
    if not ratios:
        return "all_union"
    if all(r <= 0.0 for r in ratios):
        return "source_union"
    return "all_union"


def _tuning_ratio(fine_tune_ratios: Sequence[float]) -> Optional[float]:
    for ratio in fine_tune_ratios:
        if ratio > 0:
            return float(ratio)
    return None


def _suggest_dl_hyperparams(trial: "optuna.Trial") -> Dict[str, object]:
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 1, 4)
    pretrain_epochs = trial.suggest_categorical("pretrain_epochs", [50, 100, 200, 300])
    finetune_epochs = trial.suggest_categorical("finetune_epochs", [10, 25, 50, 100])
    adapt_epochs = trial.suggest_categorical("adapt_epochs", [5, 10, 20, 40])
    return {
        "hidden_dims": tuple([int(hidden_dim)] * int(num_layers)),
        "pretrain_epochs": int(pretrain_epochs),
        "finetune_epochs": int(finetune_epochs),
        "adapt_epochs": int(adapt_epochs),
    }


def _run_dl_tuning_trial(
    model_name: str,
    config: object,
    *,
    pretrain_train: ArrayDataset,
    pretrain_val: Optional[ArrayDataset],
    train: ArrayDataset,
    val: ArrayDataset,
    evaluation: ArrayDataset,
    seed: int,
) -> float:
    if val.X.size == 0:
        return float("-inf")
    if model_name == "dl_erm":
        pipeline = DLERMPipeline(config)
    elif model_name == "dl_clustering":
        pipeline = DLClusteringPipeline(config)
    elif model_name == "dl_siamese":
        pipeline = DLSiamesePipeline(config)
    elif model_name == "dl_reorder":
        pipeline = DLReorderPipeline(config)
    elif model_name == "dl_masf":
        pipeline = DLMASFPipeline(config)
    elif model_name == "dl_dann":
        pipeline = DLDANNPipeline(config)
    elif model_name == "dl_irm":
        pipeline = DLIRMPipeline(config)
    elif model_name == "dl_csd":
        pipeline = DLCSDPipeline(config)
    elif model_name == "dl_mldg":
        pipeline = DLMldgPipeline(config)
        source_dataset = pretrain_train
        if source_dataset.X.size == 0 or source_dataset.domains is None or source_dataset.domains.size == 0:
            return float("-inf")
        val_source = pretrain_val if pretrain_val is not None else source_dataset
        result = pipeline.run(
            seed=seed,
            pretrain=source_dataset,
            pretrain_val=val_source,
            train=source_dataset,
            val=val_source,
            adapt=None,
            evaluation=evaluation,
        )
        return float(result.val_auroc)
    else:
        raise ValueError(f"Unsupported DL model '{model_name}' for tuning")

    result = pipeline.run(
        seed=seed,
        pretrain=pretrain_train,
        pretrain_val=pretrain_val,
        train=train,
        val=val,
        adapt=None,
        evaluation=evaluation,
    )
    return float(result.val_auroc)


def _tune_dl_model(
    model_name: str,
    base_config: object,
    *,
    pretrain_train: ArrayDataset,
    pretrain_val: Optional[ArrayDataset],
    train: ArrayDataset,
    val: ArrayDataset,
    evaluation: ArrayDataset,
    seed: int,
    trials: int,
) -> object:
    if optuna is None:
        raise ImportError("optuna is required to tune DL models")
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: "optuna.Trial") -> float:
        params = _suggest_dl_hyperparams(trial)
        tuned_config = replace(base_config, **params)
        score = _run_dl_tuning_trial(
            model_name,
            tuned_config,
            pretrain_train=pretrain_train,
            pretrain_val=pretrain_val,
            train=train,
            val=val,
            evaluation=evaluation,
            seed=seed + trial.number,
        )
        if not np.isfinite(score):
            return float("-inf")
        return float(score)

    study.optimize(objective, n_trials=max(1, int(trials)))
    if not study.best_trials:
        return base_config
    best_params = study.best_trial.params
    tuned_params = {
        "hidden_dims": tuple([int(best_params["hidden_dim"])] * int(best_params["num_layers"])),
        "pretrain_epochs": int(best_params["pretrain_epochs"]),
        "finetune_epochs": int(best_params["finetune_epochs"]),
        "adapt_epochs": int(best_params["adapt_epochs"]),
    }
    return replace(base_config, **tuned_params)


def _apply_tuned_config(base_config: object, tuned_config: object) -> object:
    if tuned_config is None:
        return base_config
    if is_dataclass(tuned_config):
        tuned_dict = asdict(tuned_config)
    else:
        tuned_dict = dict(tuned_config)
    allowed = set(getattr(base_config, "__dataclass_fields__", {}).keys())
    excluded = {"input_dim", "domain_count", "feature_names"}
    filtered = {k: v for k, v in tuned_dict.items() if k in allowed and k not in excluded}
    if "hidden_dims" in filtered and isinstance(filtered["hidden_dims"], list):
        filtered["hidden_dims"] = tuple(filtered["hidden_dims"])
    if not filtered:
        return base_config
    return replace(base_config, **filtered)


def _filter_features_by_source_stats(
    bundles: Sequence[DatasetBundle],
    feature_names: Sequence[str],
    *,
    min_fraction: float,
    min_std: float,
    zero_tolerance: float,
) -> Tuple[List[DatasetBundle], List[str]]:
    if not bundles:
        return [], []
    min_fraction = max(0.0, float(min_fraction))
    min_std = max(0.0, float(min_std))
    if min_fraction <= 0.0 and min_std <= 0.0:
        return list(bundles), list(feature_names)
    if not feature_names:
        return list(bundles), list(feature_names)

    sources = list(bundles[:-1])
    if not sources:
        return list(bundles), list(feature_names)

    feature_count = len(feature_names)
    total_rows = 0
    activity_counts = np.zeros(feature_count, dtype=np.float64) if min_fraction > 0 else None
    sum_vals = np.zeros(feature_count, dtype=np.float64) if min_std > 0 else None
    sum_sq = np.zeros(feature_count, dtype=np.float64) if min_std > 0 else None

    for bundle in sources:
        values = np.asarray(bundle.features.values, dtype=np.float32)
        if values.size == 0:
            continue
        rows = values.shape[0]
        total_rows += rows
        if min_fraction > 0 and activity_counts is not None:
            active = np.count_nonzero(np.abs(values) > float(zero_tolerance), axis=0)
            activity_counts += active
        if min_std > 0 and sum_vals is not None and sum_sq is not None:
            sum_vals += values.sum(axis=0, dtype=np.float64)
            sum_sq += np.sum(values * values, axis=0, dtype=np.float64)

    if total_rows == 0:
        return list(bundles), list(feature_names)

    keep_mask = np.ones(feature_count, dtype=bool)
    if min_fraction > 0 and activity_counts is not None:
        freq = activity_counts / float(total_rows)
        keep_mask &= freq >= min_fraction
    if min_std > 0 and sum_vals is not None and sum_sq is not None:
        mean = sum_vals / float(total_rows)
        var = np.maximum(sum_sq / float(total_rows) - np.square(mean), 0.0)
        std = np.sqrt(var)
        keep_mask &= std >= min_std

    if keep_mask.all():
        return list(bundles), list(feature_names)

    keep_indices = np.where(keep_mask)[0]
    if keep_indices.size == 0:
        return list(bundles), list(feature_names)
    selected = [feature_names[idx] for idx in keep_indices]
    filtered = [bundle.select_features(selected) for bundle in bundles]
    return filtered, selected


def _filter_features_by_correlation(
    bundles: Sequence[DatasetBundle],
    feature_names: Sequence[str],
    *,
    max_corr: float,
    sample_rows: int,
    random_state: int,
) -> Tuple[List[DatasetBundle], List[str]]:
    if not bundles or max_corr <= 0 or max_corr >= 1:
        return list(bundles), list(feature_names)
    sources = bundles[:-1]
    if not sources:
        return list(bundles), list(feature_names)
    rng = np.random.default_rng(random_state)
    sampled_frames: List[pd.DataFrame] = []
    for bundle in sources:
        frame = bundle.features
        if frame.empty:
            continue
        if sample_rows > 0 and len(frame) > sample_rows:
            idx = rng.choice(len(frame), sample_rows, replace=False)
            sampled_frames.append(frame.iloc[idx])
        else:
            sampled_frames.append(frame)
    if not sampled_frames:
        return list(bundles), list(feature_names)
    sample_df = pd.concat(sampled_frames, axis=0, ignore_index=True)
    sample_df = sample_df.reindex(columns=list(feature_names), fill_value=0.0)
    if sample_df.shape[0] < 2:
        return list(bundles), list(feature_names)
    sample_matrix = sample_df.values.astype(np.float32, copy=False)
    keep_mask = np.ones(len(feature_names), dtype=bool)
    prefix_to_indices: Dict[str, List[int]] = {}
    for idx, name in enumerate(feature_names):
        prefix = name.split("#", 1)[0]
        prefix_to_indices.setdefault(prefix, []).append(idx)
    for indices in prefix_to_indices.values():
        active_indices = [i for i in indices if keep_mask[i]]
        if len(active_indices) <= 1:
            continue
        subset = sample_matrix[:, active_indices]
        if subset.shape[0] < 2:
            continue
        corr_matrix = np.corrcoef(subset, rowvar=False)
        if np.isnan(corr_matrix).all():
            continue
        for i_idx, global_i in enumerate(active_indices):
            if not keep_mask[global_i]:
                continue
            for j_idx in range(i_idx + 1, len(active_indices)):
                global_j = active_indices[j_idx]
                if not keep_mask[global_j]:
                    continue
                corr = corr_matrix[i_idx, j_idx]
                if np.isnan(corr):
                    continue
                if abs(float(corr)) >= max_corr:
                    keep_mask[global_j] = False
    if keep_mask.all():
        return list(bundles), list(feature_names)
    selected = [name for keep, name in zip(keep_mask, feature_names) if keep]
    filtered = [bundle.select_features(selected) for bundle in bundles]
    return filtered, selected


def _filter_features_by_class_shift(
    bundles: Sequence[DatasetBundle],
    feature_names: Sequence[str],
    *,
    threshold: float,
    zero_tolerance: float,
    max_samples: int,
    random_state: int,
) -> Tuple[List[DatasetBundle], List[str]]:
    if not bundles or threshold <= 0:
        return list(bundles), list(feature_names)
    sources = bundles[:-1]
    if not sources:
        return list(bundles), list(feature_names)
    target_bundle = bundles[-1]
    src_frames = [bundle.features for bundle in sources]
    src_labels = [bundle.labels.astype(int) for bundle in sources]
    src_df = pd.concat(src_frames, axis=0, ignore_index=True)
    src_y = np.concatenate(src_labels)
    tgt_df = target_bundle.features.reset_index(drop=True)
    tgt_y = target_bundle.labels.astype(int)

    def _sample(frame: pd.DataFrame, labels: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        if max_samples > 0 and len(frame) > max_samples:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(frame), max_samples, replace=False)
            return frame.iloc[idx], labels[idx]
        return frame, labels

    src_df, src_y = _sample(src_df, src_y)
    tgt_df, tgt_y = _sample(tgt_df, tgt_y)
    if not (src_y == 1).any() or not (tgt_y == 1).any():
        return list(bundles), list(feature_names)
    src_df = src_df.reindex(columns=list(feature_names), fill_value=0.0)
    tgt_df = tgt_df.reindex(columns=list(feature_names), fill_value=0.0)
    src_matrix = src_df.values.astype(np.float32, copy=False)
    tgt_matrix = tgt_df.values.astype(np.float32, copy=False)
    src_pos_mask = src_y == 1
    tgt_pos_mask = tgt_y == 1
    src_pos_matrix = np.abs(src_matrix[src_pos_mask]) > zero_tolerance
    tgt_pos_matrix = np.abs(tgt_matrix[tgt_pos_mask]) > zero_tolerance
    src_pos_activation = src_pos_matrix.mean(axis=0)
    tgt_pos_activation = tgt_pos_matrix.mean(axis=0)
    diff = np.abs(src_pos_activation - tgt_pos_activation)
    keep_mask = diff <= float(threshold)
    if keep_mask.all():
        return list(bundles), list(feature_names)
    selected = [name for keep, name in zip(keep_mask, feature_names) if keep]
    filtered = [bundle.select_features(selected) for bundle in bundles]
    return filtered, selected


def _select_features_via_l1(
    bundles: Sequence[DatasetBundle],
    feature_names: Sequence[str],
    *,
    keep_ratio: float,
    max_samples: int,
    random_state: int,
) -> Tuple[List[DatasetBundle], List[str]]:
    if not bundles or keep_ratio <= 0 or LogisticRegression is None:
        return list(bundles), list(feature_names)
    sources = bundles[:-1]
    if not sources:
        return list(bundles), list(feature_names)
    frames = [bundle.features for bundle in sources]
    labels = [bundle.labels.astype(int) for bundle in sources]
    frame = pd.concat(frames, axis=0, ignore_index=True)
    y = np.concatenate(labels)
    if frame.empty or len(np.unique(y)) < 2:
        return list(bundles), list(feature_names)
    rng = np.random.default_rng(random_state)
    if max_samples > 0 and len(frame) > max_samples:
        idx = rng.choice(len(frame), max_samples, replace=False)
        frame = frame.iloc[idx]
        y = y[idx]
    frame = frame.reindex(columns=list(feature_names), fill_value=0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(frame.values.astype(np.float32, copy=False))
    clf = LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=200,
        class_weight="balanced",
        n_jobs=None,
    )
    try:
        clf.fit(X_scaled, y)
    except Exception:
        return list(bundles), list(feature_names)
    coefs = np.abs(clf.coef_).mean(axis=0)
    keep_count = max(1, int(len(feature_names) * keep_ratio))
    top_indices = np.argsort(coefs)[-keep_count:]
    keep_mask = np.zeros(len(feature_names), dtype=bool)
    keep_mask[top_indices] = True
    selected = [name for keep, name in zip(keep_mask, feature_names) if keep]
    filtered = [bundle.select_features(selected) for bundle in bundles]
    return filtered, selected


def _filter_features_by_target_shift(
    bundles: Sequence[DatasetBundle],
    feature_names: Sequence[str],
    *,
    shift_std_threshold: float,
    zero_tolerance: float,
) -> Tuple[List[DatasetBundle], List[str]]:
    if not bundles:
        return [], []
    if shift_std_threshold is None or shift_std_threshold <= 0:
        return list(bundles), list(feature_names)
    if len(bundles) < 2:
        return list(bundles), list(feature_names)
    feature_count = len(feature_names)
    if feature_count == 0:
        return list(bundles), list(feature_names)

    sources = bundles[:-1]
    target = bundles[-1]

    def _accumulate(entries: Sequence[DatasetBundle]) -> Tuple[int, np.ndarray, np.ndarray]:
        total = 0
        sum_vals = np.zeros(feature_count, dtype=np.float64)
        sum_sq = np.zeros(feature_count, dtype=np.float64)
        for bundle in entries:
            values = np.asarray(bundle.features.values, dtype=np.float64)
            if values.size == 0:
                continue
            total += values.shape[0]
            sum_vals += values.sum(axis=0)
            sum_sq += np.square(values).sum(axis=0)
        return total, sum_vals, sum_sq

    src_total, src_sum, src_sq = _accumulate(sources)
    tgt_total, tgt_sum, tgt_sq = _accumulate([target])
    if src_total == 0 or tgt_total == 0:
        return list(bundles), list(feature_names)

    mean_src = src_sum / float(src_total)
    var_src = np.maximum(src_sq / float(src_total) - np.square(mean_src), 0.0)
    std_src = np.maximum(np.sqrt(var_src), float(zero_tolerance))

    mean_tgt = tgt_sum / float(tgt_total)
    shift = np.abs(mean_tgt - mean_src)
    shift_std = shift / std_src

    keep_mask = shift_std <= float(shift_std_threshold)
    if keep_mask.all():
        return list(bundles), list(feature_names)
    keep_indices = np.where(keep_mask)[0]
    if keep_indices.size == 0:
        return list(bundles), list(feature_names)
    selected = [feature_names[idx] for idx in keep_indices]
    filtered = [bundle.select_features(selected) for bundle in bundles]
    return filtered, selected


def _split_target_indices(
    labels: np.ndarray,
    ft_ratio: float,
    ft_val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    total = len(labels)
    if total == 0:
        return (
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            0.0,
        )
    if total <= 1 or ft_ratio <= 0:
        eval_idx = np.arange(total, dtype=int)
        return (
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            eval_idx,
            0.0,
        )

    ratio = float(ft_ratio)
    if ratio >= 1.0:
        ratio = 0.999
    adapt_count = max(1, int(round(total * ratio)))
    if adapt_count >= total:
        adapt_count = total - 1
    if adapt_count <= 0:
        eval_idx = np.arange(total, dtype=int)
        return (
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            eval_idx,
            0.0,
        )

    adapt_ratio = adapt_count / total
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=(total - adapt_count) / total,
        random_state=seed,
    )
    dummy = np.zeros(total)
    adapt_idx, eval_idx = next(sss.split(dummy, labels))
    if adapt_idx.size == 0 or eval_idx.size == 0:
        eval_idx = np.arange(total, dtype=int)
        return (
            np.zeros((0,), dtype=int),
            np.zeros((0,), dtype=int),
            eval_idx,
            0.0,
        )

    val_ratio = max(0.0, min(float(ft_val_ratio), 0.5))
    if val_ratio > 0 and adapt_idx.size > 1 and len(np.unique(labels[adapt_idx])) > 1:
        sss_val = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_ratio,
            random_state=seed,
        )
        dummy_adapt = np.zeros(adapt_idx.size)
        train_rel, val_rel = next(sss_val.split(dummy_adapt, labels[adapt_idx]))
        train_idx = adapt_idx[train_rel]
        val_idx = adapt_idx[val_rel]
    else:
        train_idx = adapt_idx
        val_idx = np.zeros((0,), dtype=int)

    actual_ratio = (train_idx.size + val_idx.size) / total
    return train_idx, val_idx, eval_idx, float(actual_ratio)


def _prepare_target_datasets(
    target_features: pd.DataFrame,
    target_labels: np.ndarray,
    scaler: StandardScaler,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    eval_idx: np.ndarray,
    *,
    clip_value: Optional[float] = None,
    domain_id: Optional[int] = None,
) -> Tuple[ArrayDataset, ArrayDataset, ArrayDataset, Dict[str, int]]:
    train_frame = target_features.iloc[train_idx].reset_index(drop=True) if train_idx.size else pd.DataFrame(columns=target_features.columns)
    val_frame = target_features.iloc[val_idx].reset_index(drop=True) if val_idx.size else pd.DataFrame(columns=target_features.columns)
    eval_frame = target_features.iloc[eval_idx].reset_index(drop=True) if eval_idx.size else pd.DataFrame(columns=target_features.columns)

    train_dataset = _to_array_dataset(
        train_frame,
        target_labels[train_idx] if train_idx.size else np.empty((0,), dtype=target_labels.dtype),
        scaler,
        clip_value=clip_value,
        domain_id=domain_id,
    )
    val_dataset = _to_array_dataset(
        val_frame,
        target_labels[val_idx] if val_idx.size else np.empty((0,), dtype=target_labels.dtype),
        scaler,
        clip_value=clip_value,
        domain_id=domain_id,
    )
    eval_dataset = _to_array_dataset(
        eval_frame,
        target_labels[eval_idx] if eval_idx.size else np.empty((0,), dtype=target_labels.dtype),
        scaler,
        clip_value=clip_value,
        domain_id=domain_id,
    )

    counts = {
        "adapt_train": int(train_idx.size),
        "adapt_val": int(val_idx.size),
        "eval": int(eval_idx.size),
    }
    return train_dataset, val_dataset, eval_dataset, counts


def _indices_to_target_split(
    bundle: DatasetBundle,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> TargetSplits:
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    train_features = bundle.features.iloc[train_idx].reset_index(drop=True)
    val_features = bundle.features.iloc[val_idx].reset_index(drop=True)
    test_features = bundle.features.iloc[test_idx].reset_index(drop=True)

    labels = bundle.labels.astype(int)

    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]

    return TargetSplits(
        train_features=train_features,
        val_features=val_features,
        test_features=test_features,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
    )


def _make_groupwise_validation_split(
    train_idx: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    train_idx = np.asarray(train_idx, dtype=int)
    train_labels = labels[train_idx]
    train_groups = groups[train_idx]
    unique_groups = np.unique(train_groups)
    if unique_groups.size < 2:
        raise ValueError("Validation split requires at least two distinct groups")

    n_splits = int(min(5, unique_groups.size))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy_features = np.zeros_like(train_labels)
    inner_train_idx, val_idx = next(splitter.split(dummy_features, train_labels, train_groups))
    return train_idx[inner_train_idx], train_idx[val_idx]


def _generate_scenario_splits(
    bundle: DatasetBundle,
    *,
    strategy: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    group_folds: int,
) -> List[ScenarioSplit]:
    labels = bundle.labels.astype(int)
    groups = np.asarray(bundle.groups)

    if strategy == "stratified_shuffle":
        split = _make_stratified_shuffle_split(
            bundle,
            seed=seed,
            val_size=val_ratio / (train_ratio + val_ratio) if (train_ratio + val_ratio) > 0 else 0.0,
            test_size=test_ratio,
        )
        return [ScenarioSplit(split=split, metadata={"split_strategy": strategy, "fold_id": 0, "test_groups": None})]

    if strategy == "loso":
        loso_pairs = loso_splits(groups)
        scenario_splits: List[ScenarioSplit] = []
        for fold_id, (train_idx, test_idx) in enumerate(loso_pairs):
            train_idx = np.asarray(train_idx, dtype=int)
            test_idx = np.asarray(test_idx, dtype=int)
            if train_idx.size == 0 or test_idx.size == 0:
                continue
            train_final, val_idx = _make_groupwise_validation_split(train_idx, labels, groups, seed + fold_id)
            split = _indices_to_target_split(bundle, train_final, val_idx, test_idx)
            test_groups = "+".join(str(g) for g in np.unique(groups[test_idx]))
            metadata = {
                "split_strategy": strategy,
                "fold_id": fold_id,
                "test_groups": test_groups,
            }
            scenario_splits.append(ScenarioSplit(split=split, metadata=metadata))
        if not scenario_splits:
            raise ValueError("LOSO split requested but no valid folds were generated")
        return scenario_splits

    if strategy == "stratified_group_kfold":
        if group_folds < 2:
            raise ValueError("group_folds must be at least 2 for stratified_group_kfold strategy")
        fold_pairs = stratified_group_kfold_splits(
            labels,
            groups,
            n_splits=group_folds,
            shuffle=True,
            random_state=seed,
        )
        scenario_splits: List[ScenarioSplit] = []
        for fold_id, (train_idx, test_idx) in enumerate(fold_pairs):
            train_idx = np.asarray(train_idx, dtype=int)
            test_idx = np.asarray(test_idx, dtype=int)
            train_final, val_idx = _make_groupwise_validation_split(train_idx, labels, groups, seed + fold_id)
            split = _indices_to_target_split(bundle, train_final, val_idx, test_idx)
            test_groups = "+".join(str(g) for g in np.unique(groups[test_idx]))
            metadata = {
                "split_strategy": strategy,
                "fold_id": fold_id,
                "test_groups": test_groups,
            }
            scenario_splits.append(ScenarioSplit(split=split, metadata=metadata))
        if not scenario_splits:
            raise ValueError("StratifiedGroupKFold split failed to generate folds")
        return scenario_splits

    raise ValueError(f"Unsupported split strategy '{strategy}'")


def run_experiment_scenario(
    scenario: ExperimentScenario,
    *,
    cache: CacheManager,
    overrides: Optional[Dict[str, str]] = None,
    fine_tune_ratios: Sequence[float] = (0.0, 0.2, 0.4, 0.6),
    fine_tune_val_ratio: float = 0.2,
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    model_types: Sequence[str] = ("tree", "transformer"),
    val_size: float = 0.2,
    test_size: float = 0.2,
    pretrain_val_ratio: float = 0.1,
    target_train_ratio: Optional[float] = None,
    target_val_ratio: Optional[float] = None,
    target_test_ratio: Optional[float] = None,
    split_strategy: str = "stratified_shuffle",
    group_folds: int = 5,
    lightgbm_config: Optional[LightGBMConfig] = None,
    transformer_config: Optional[TransformerConfig] = None,
    cdtrans_config: Optional[CDTransConfig] = None,
    feature_strategy: str = "all_union",
    min_source_feature_frac: float = 0.0,
    min_source_feature_std: float = 0.0,
    feature_zero_tolerance: float = 1e-6,
    include_target_in_scaler: bool = True,
    feature_clip_value: Optional[float] = 5.0,
    transformer_overrides: Optional[Dict[str, object]] = None,
    dl_erm_config: Optional[DLERMConfig] = None,
    dl_erm_overrides: Optional[Dict[str, object]] = None,
    dl_dann_config: Optional[DLDANNConfig] = None,
    dl_dann_overrides: Optional[Dict[str, object]] = None,
    dl_irm_config: Optional[DLIRMConfig] = None,
    dl_irm_overrides: Optional[Dict[str, object]] = None,
    dl_csd_config: Optional[DLCSConfig] = None,
    dl_csd_overrides: Optional[Dict[str, object]] = None,
    dl_mldg_config: Optional[DLMldgConfig] = None,
    dl_mldg_overrides: Optional[Dict[str, object]] = None,
    dl_clustering_config: Optional[DLClusteringConfig] = None,
    dl_clustering_overrides: Optional[Dict[str, object]] = None,
    dl_siamese_config: Optional[DLSiameseConfig] = None,
    dl_siamese_overrides: Optional[Dict[str, object]] = None,
    dl_reorder_config: Optional[DLReorderConfig] = None,
    dl_reorder_overrides: Optional[Dict[str, object]] = None,
    dl_masf_config: Optional[DLMASFConfig] = None,
    dl_masf_overrides: Optional[Dict[str, object]] = None,
    # tabpfn_config: Optional[TabPFNConfig] = None,
    # tabpfn_overrides: Optional[Dict[str, object]] = None,
    max_target_shift_std: float = 0.0,
    max_feature_correlation: float = 0.0,
    correlation_sample_rows: int = 2000,
    max_class_conditional_shift: float = 0.0,
    class_shift_max_samples: int = 5000,
    l1_feature_keep_ratio: float = 0.0,
    l1_max_samples: int = 5000,
    optuna_tune: bool = False,
    optuna_trials: int = 50,
    optuna_seed: int = 42,
    tuned_configs: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    dataset_names = scenario.sources + [scenario.target]
    split_seed = seeds[0] if seeds else 42
    bundles = load_datasets(dataset_names, cache, overrides=overrides)
    bundles = augment_temporal_and_group_features(bundles)
    aligned_bundles, feature_list = align_feature_intersection(bundles, strategy=feature_strategy)
    aligned_bundles, feature_list = _filter_features_by_source_stats(
        aligned_bundles,
        feature_list,
        min_fraction=min_source_feature_frac,
        min_std=min_source_feature_std,
        zero_tolerance=feature_zero_tolerance,
    )
    aligned_bundles, feature_list = _filter_features_by_correlation(
        aligned_bundles,
        feature_list,
        max_corr=max_feature_correlation,
        sample_rows=correlation_sample_rows,
        random_state=split_seed,
    )
    aligned_bundles, feature_list = _select_features_via_l1(
        aligned_bundles,
        feature_list,
        keep_ratio=l1_feature_keep_ratio,
        max_samples=l1_max_samples,
        random_state=split_seed,
    )
    aligned_bundles, feature_list = _filter_features_by_target_shift(
        aligned_bundles,
        feature_list,
        shift_std_threshold=max_target_shift_std,
        zero_tolerance=feature_zero_tolerance,
    )
    aligned_bundles, feature_list = _filter_features_by_class_shift(
        aligned_bundles,
        feature_list,
        threshold=max_class_conditional_shift,
        zero_tolerance=feature_zero_tolerance,
        max_samples=class_shift_max_samples,
        random_state=split_seed,
    )

    source_bundles = aligned_bundles[:-1]
    target_bundle = aligned_bundles[-1]
    train_ratio, val_ratio, test_ratio = _resolve_target_ratios(
        target_train_ratio,
        target_val_ratio,
        target_test_ratio,
        val_size,
        test_size,
    )
    scenario_splits = _generate_scenario_splits(
        target_bundle,
        strategy=split_strategy,
        seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        group_folds=group_folds,
    )

    results: List[Dict[str, object]] = []

    if lightgbm_config is None:
        lightgbm_config = LightGBMConfig()
    feature_tuple = tuple(feature_list)
    domain_count = len(source_bundles) + 1
    if transformer_config is None:
        transformer_config = TransformerConfig(input_dim=len(feature_list), feature_names=feature_tuple)
    else:
        transformer_config = replace(transformer_config, input_dim=len(feature_list), feature_names=feature_tuple)
    if transformer_overrides:
        try:
            transformer_config = replace(transformer_config, **transformer_overrides)
        except TypeError as exc:
            raise ValueError(f"Invalid transformer override: {exc}") from exc
    if dl_erm_config is None:
        dl_erm_config = DLERMConfig(input_dim=len(feature_list))
    else:
        dl_erm_config = replace(dl_erm_config, input_dim=len(feature_list))
    if dl_erm_overrides:
        try:
            dl_erm_config = replace(dl_erm_config, **dl_erm_overrides)
        except TypeError as exc:
            raise ValueError(f"Invalid dl_erm override: {exc}") from exc
    if dl_dann_config is None:
        dl_dann_config = DLDANNConfig(input_dim=len(feature_list))
    else:
        dl_dann_config = replace(dl_dann_config, input_dim=len(feature_list))
    if dl_dann_overrides:
        try:
            dl_dann_config = replace(dl_dann_config, **dl_dann_overrides)
        except TypeError as exc:
            raise ValueError(f"Invalid dl_dann override: {exc}") from exc
    if dl_irm_config is None:
        dl_irm_config = DLIRMConfig(input_dim=len(feature_list))
    else:
        dl_irm_config = replace(dl_irm_config, input_dim=len(feature_list))
    if dl_irm_overrides:
        try:
            dl_irm_config = replace(dl_irm_config, **dl_irm_overrides)
        except TypeError as exc:
            raise ValueError(f"Invalid dl_irm override: {exc}") from exc
    if dl_csd_config is None:
        dl_csd_config = DLCSConfig(input_dim=len(feature_list), domain_count=domain_count)
    else:
        dl_csd_config = replace(dl_csd_config, input_dim=len(feature_list), domain_count=domain_count)
    if dl_csd_overrides:
        try:
            dl_csd_config = replace(dl_csd_config, **dl_csd_overrides)
        except TypeError as exc:
            raise ValueError(f"Invalid dl_csd override: {exc}") from exc
    if dl_mldg_config is None:
        dl_mldg_config = DLMldgConfig(input_dim=len(feature_list), hidden_dims=tuple(dl_erm_config.hidden_dims))
    else:
        dl_mldg_config = replace(dl_mldg_config, input_dim=len(feature_list))
    if dl_mldg_overrides:
        try:
            dl_mldg_config = replace(dl_mldg_config, **dl_mldg_overrides)
        except TypeError as exc:
            raise ValueError(f"Invalid dl_mldg override: {exc}") from exc
    if dl_clustering_config is None:
        dl_clustering_config = DLClusteringConfig(input_dim=len(feature_list))
    else:
        dl_clustering_config = replace(dl_clustering_config, input_dim=len(feature_list))
    if dl_clustering_overrides:
        try:
            dl_clustering_config = replace(dl_clustering_config, **dl_clustering_overrides)
        except TypeError as exc:
            raise ValueError(f"Invalid dl_clustering override: {exc}") from exc
    if dl_siamese_config is None:
        dl_siamese_config = DLSiameseConfig(input_dim=len(feature_list))
    else:
        dl_siamese_config = replace(dl_siamese_config, input_dim=len(feature_list))
    if dl_siamese_overrides:
        try:
            dl_siamese_config = replace(dl_siamese_config, **dl_siamese_overrides)
        except TypeError as exc:
            raise ValueError(f"Invalid dl_siamese override: {exc}") from exc
    if dl_reorder_config is None:
        dl_reorder_config = DLReorderConfig(input_dim=len(feature_list))
    else:
        dl_reorder_config = replace(dl_reorder_config, input_dim=len(feature_list))
    if dl_reorder_overrides:
        try:
            dl_reorder_config = replace(dl_reorder_config, **dl_reorder_overrides)
        except TypeError as exc:
            raise ValueError(f"Invalid dl_reorder override: {exc}") from exc
    if dl_masf_config is None:
        dl_masf_config = DLMASFConfig(input_dim=len(feature_list))
    else:
        dl_masf_config = replace(dl_masf_config, input_dim=len(feature_list))
    if dl_masf_overrides:
        try:
            dl_masf_config = replace(dl_masf_config, **dl_masf_overrides)
        except TypeError as exc:
            raise ValueError(f"Invalid dl_masf override: {exc}") from exc
    cdtrans_base_config = (
        CDTransConfig(input_dim=len(feature_list), feature_names=feature_tuple)
        if cdtrans_config is None
        else replace(cdtrans_config, input_dim=len(feature_list), feature_names=feature_tuple)
    )
    # if tabpfn_config is None:
    #     tabpfn_config = TabPFNConfig()
    # if tabpfn_overrides:
    #     try:
    #         tabpfn_config = replace(tabpfn_config, **tabpfn_overrides)
    #     except TypeError as exc:
    #         raise ValueError(f"Invalid TabPFN override: {exc}") from exc

    tuned_configs = tuned_configs if tuned_configs is not None else {}
    tuning_ratio = _tuning_ratio(fine_tune_ratios)
    if optuna_tune and tuning_ratio is not None:
        split_entry = scenario_splits[0]
        split = split_entry.split
        split_meta = dict(split_entry.metadata)
        split_meta.setdefault("split_seed", split_seed)
        source_frames = [bundle.features.reset_index(drop=True) for bundle in source_bundles]
        if split_meta.get("split_strategy") == "stratified_shuffle":
            target_frame_full = target_bundle.features.reset_index(drop=True)
            target_labels_full = target_bundle.labels.astype(int)
        else:
            target_frame_full = split.test_features.reset_index(drop=True)
            target_labels_full = split.test_labels.astype(int)
        extra_frames = [target_frame_full] if include_target_in_scaler else []
        scaler = _fit_scaler(source_frames, extra_frames)
        source_datasets = []
        for domain_idx, (frame, bundle) in enumerate(zip(source_frames, source_bundles)):
            source_datasets.append(
                _to_array_dataset(
                    frame,
                    bundle.labels.astype(np.float32),
                    scaler,
                    clip_value=feature_clip_value,
                    domain_id=domain_idx,
                )
            )
        combined_pretrain = _combine_sources(source_datasets)
        pretrain_train_dataset, pretrain_val_dataset = _split_array_dataset(
            combined_pretrain,
            pretrain_val_ratio,
            seed=split_seed,
        )
        train_idx, val_idx, eval_idx, _ = _split_target_indices(
            target_labels_full,
            tuning_ratio,
            fine_tune_val_ratio,
            optuna_seed,
        )
        train_dataset, val_dataset, eval_dataset, _ = _prepare_target_datasets(
            target_frame_full,
            target_labels_full,
            scaler,
            train_idx,
            val_idx,
            eval_idx,
            clip_value=feature_clip_value,
            domain_id=len(source_datasets),
        )
        if "dl_erm" in model_types and "dl_erm" not in tuned_configs:
            tuned_configs["dl_erm"] = _tune_dl_model(
                "dl_erm",
                dl_erm_config,
                pretrain_train=pretrain_train_dataset,
                pretrain_val=pretrain_val_dataset,
                train=train_dataset,
                val=val_dataset,
                evaluation=eval_dataset,
                seed=optuna_seed,
                trials=optuna_trials,
            )
        if "dl_clustering" in model_types and "dl_clustering" not in tuned_configs:
            tuned_configs["dl_clustering"] = _tune_dl_model(
                "dl_clustering",
                dl_clustering_config,
                pretrain_train=pretrain_train_dataset,
                pretrain_val=pretrain_val_dataset,
                train=train_dataset,
                val=val_dataset,
                evaluation=eval_dataset,
                seed=optuna_seed,
                trials=optuna_trials,
            )
        if "dl_siamese" in model_types and "dl_siamese" not in tuned_configs:
            tuned_configs["dl_siamese"] = _tune_dl_model(
                "dl_siamese",
                dl_siamese_config,
                pretrain_train=pretrain_train_dataset,
                pretrain_val=pretrain_val_dataset,
                train=train_dataset,
                val=val_dataset,
                evaluation=eval_dataset,
                seed=optuna_seed,
                trials=optuna_trials,
            )
        if "dl_reorder" in model_types and "dl_reorder" not in tuned_configs:
            tuned_configs["dl_reorder"] = _tune_dl_model(
                "dl_reorder",
                dl_reorder_config,
                pretrain_train=pretrain_train_dataset,
                pretrain_val=pretrain_val_dataset,
                train=train_dataset,
                val=val_dataset,
                evaluation=eval_dataset,
                seed=optuna_seed,
                trials=optuna_trials,
            )
        if "dl_masf" in model_types and "dl_masf" not in tuned_configs:
            tuned_configs["dl_masf"] = _tune_dl_model(
                "dl_masf",
                dl_masf_config,
                pretrain_train=pretrain_train_dataset,
                pretrain_val=pretrain_val_dataset,
                train=train_dataset,
                val=val_dataset,
                evaluation=eval_dataset,
                seed=optuna_seed,
                trials=optuna_trials,
            )
        if "dl_dann" in model_types and "dl_dann" not in tuned_configs:
            tuned_configs["dl_dann"] = _tune_dl_model(
                "dl_dann",
                dl_dann_config,
                pretrain_train=pretrain_train_dataset,
                pretrain_val=pretrain_val_dataset,
                train=train_dataset,
                val=val_dataset,
                evaluation=eval_dataset,
                seed=optuna_seed,
                trials=optuna_trials,
            )
        if "dl_irm" in model_types and "dl_irm" not in tuned_configs:
            tuned_configs["dl_irm"] = _tune_dl_model(
                "dl_irm",
                dl_irm_config,
                pretrain_train=pretrain_train_dataset,
                pretrain_val=pretrain_val_dataset,
                train=train_dataset,
                val=val_dataset,
                evaluation=eval_dataset,
                seed=optuna_seed,
                trials=optuna_trials,
            )
        if "dl_csd" in model_types and "dl_csd" not in tuned_configs:
            tuned_configs["dl_csd"] = _tune_dl_model(
                "dl_csd",
                dl_csd_config,
                pretrain_train=pretrain_train_dataset,
                pretrain_val=pretrain_val_dataset,
                train=train_dataset,
                val=val_dataset,
                evaluation=eval_dataset,
                seed=optuna_seed,
                trials=optuna_trials,
            )
        if "dl_mldg" in model_types and "dl_mldg" not in tuned_configs:
            tuned_configs["dl_mldg"] = _tune_dl_model(
                "dl_mldg",
                dl_mldg_config,
                pretrain_train=combined_pretrain,
                pretrain_val=pretrain_val_dataset,
                train=combined_pretrain,
                val=pretrain_val_dataset if pretrain_val_dataset is not None else combined_pretrain,
                evaluation=eval_dataset,
                seed=optuna_seed,
                trials=optuna_trials,
            )
    if "dl_erm" in tuned_configs:
        dl_erm_config = _apply_tuned_config(dl_erm_config, tuned_configs["dl_erm"])
    if "dl_clustering" in tuned_configs:
        dl_clustering_config = _apply_tuned_config(dl_clustering_config, tuned_configs["dl_clustering"])
    if "dl_siamese" in tuned_configs:
        dl_siamese_config = _apply_tuned_config(dl_siamese_config, tuned_configs["dl_siamese"])
    if "dl_reorder" in tuned_configs:
        dl_reorder_config = _apply_tuned_config(dl_reorder_config, tuned_configs["dl_reorder"])
    if "dl_masf" in tuned_configs:
        dl_masf_config = _apply_tuned_config(dl_masf_config, tuned_configs["dl_masf"])
    if "dl_dann" in tuned_configs:
        dl_dann_config = _apply_tuned_config(dl_dann_config, tuned_configs["dl_dann"])
    if "dl_irm" in tuned_configs:
        dl_irm_config = _apply_tuned_config(dl_irm_config, tuned_configs["dl_irm"])
    if "dl_csd" in tuned_configs:
        dl_csd_config = _apply_tuned_config(dl_csd_config, tuned_configs["dl_csd"])
    if "dl_mldg" in tuned_configs:
        dl_mldg_config = _apply_tuned_config(dl_mldg_config, tuned_configs["dl_mldg"])

    for split_entry in scenario_splits:
        split = split_entry.split
        split_meta = dict(split_entry.metadata)
        split_meta.setdefault("split_seed", split_seed)

        source_frames = [bundle.features.reset_index(drop=True) for bundle in source_bundles]
        if split_meta.get("split_strategy") == "stratified_shuffle":
            target_frame_full = target_bundle.features.reset_index(drop=True)
            target_labels_full = target_bundle.labels.astype(int)
        else:
            target_frame_full = split.test_features.reset_index(drop=True)
            target_labels_full = split.test_labels.astype(int)

        extra_frames = [target_frame_full] if include_target_in_scaler else []
        scaler = _fit_scaler(source_frames, extra_frames)

        source_datasets = []
        for domain_idx, (frame, bundle) in enumerate(zip(source_frames, source_bundles)):
            source_datasets.append(
                _to_array_dataset(
                    frame,
                    bundle.labels.astype(np.float32),
                    scaler,
                    clip_value=feature_clip_value,
                    domain_id=domain_idx,
                )
            )
        combined_pretrain = _combine_sources(source_datasets)
        target_domain_id = len(source_datasets)
        domain_count = target_domain_id + 1
        pretrain_train_dataset, pretrain_val_dataset = _split_array_dataset(
            combined_pretrain,
            pretrain_val_ratio,
            seed=split_seed,
        )

        total_target_samples = int(target_labels_full.size)

        # tabpfn_pipeline = TabPFNPipeline(tabpfn_config) if "tabpfn" in model_types else None

        for ratio in fine_tune_ratios:
            for seed in seeds:
                train_idx, val_idx, eval_idx, effective_ratio = _split_target_indices(
                    target_labels_full,
                    ratio,
                    fine_tune_val_ratio,
                    seed,
                )

                train_dataset, val_dataset, eval_dataset, target_counts = _prepare_target_datasets(
                    target_frame_full,
                    target_labels_full,
                    scaler,
                    train_idx,
                    val_idx,
                    eval_idx,
                    clip_value=feature_clip_value,
                    domain_id=target_domain_id,
                )

                common_payload = {
                    "scenario": scenario.name,
                    "target": scenario.target,
                    "sources": "+".join(scenario.sources),
                    "feature_count": len(feature_list),
                    "ft_ratio": ratio,
                    "seed": seed,
                    "split_strategy": split_meta.get("split_strategy"),
                    "fold_id": int(split_meta.get("fold_id", 0)),
                    "test_groups": split_meta.get("test_groups"),
                    "split_seed": split_meta.get("split_seed"),
                    "train_samples": float(len(train_dataset.y)),
                    "val_samples": float(len(val_dataset.y)),
                    "pretrain_samples": float(len(combined_pretrain.y)),
                    "pretrain_train_samples": float(len(pretrain_train_dataset.y)) if pretrain_train_dataset.X.size else 0.0,
                    "pretrain_val_samples": float(len(pretrain_val_dataset.y)) if pretrain_val_dataset is not None else 0.0,
                    "adapt_samples": float(target_counts["adapt_train"] + target_counts["adapt_val"]),
                    "adapt_train_samples": float(target_counts["adapt_train"]),
                    "adapt_val_samples": float(target_counts["adapt_val"]),
                    "eval_samples": float(target_counts["eval"]),
                    "target_total_samples": float(total_target_samples),
                    "ft_ratio_effective": effective_ratio,
                }

                if "tree" in model_types:
                    lgbm_pipeline = LightGBMPipeline(lightgbm_config)
                    result = lgbm_pipeline.run(
                        pretrain=pretrain_train_dataset,
                        pretrain_val=pretrain_val_dataset,
                        train=train_dataset,
                        val=val_dataset,
                        adapt=None,
                        evaluation=eval_dataset,
                    )
                    results.append({
                        **common_payload,
                        "model": "lightgbm",
                        "mode": "pretrain_finetune",
                        "train_auroc": result.train_auroc,
                        "val_auroc": result.val_auroc,
                        "test_auroc": result.test_auroc,
                        "train_accuracy": result.train_accuracy,
                        "val_accuracy": result.val_accuracy,
                        "test_accuracy": result.test_accuracy,
                        "train_auprc": result.train_auprc,
                        "val_auprc": result.val_auprc,
                        "test_auprc": result.test_auprc,
                        "pretrain_seconds": result.stage_durations.get("pretrain_seconds"),
                        "finetune_seconds": result.stage_durations.get("finetune_seconds"),
                        "adapt_seconds": result.stage_durations.get("adapt_seconds"),
                        "pretrain_epochs": None,
                        "finetune_epochs": None,
                        "adapt_epochs": None,
                        "pretrain_val_auroc": result.pretrain_val_auroc,
                        "pretrain_val_accuracy": result.pretrain_val_accuracy,
                        "pretrain_val_auprc": result.pretrain_val_auprc,
                        "best_iteration": result.best_iteration,
                    })

                    if train_dataset.X.size > 0:
                        baseline_config = replace(lightgbm_config, pretrain_rounds=0)
                        baseline_pipeline = LightGBMPipeline(baseline_config)
                        baseline_result = baseline_pipeline.run(
                            pretrain=None,
                            pretrain_val=None,
                            train=train_dataset,
                            val=val_dataset,
                            adapt=None,
                            evaluation=eval_dataset,
                        )
                        results.append({
                            **common_payload,
                            "model": "lightgbm",
                            "mode": "target_only",
                            "train_auroc": baseline_result.train_auroc,
                            "val_auroc": baseline_result.val_auroc,
                            "test_auroc": baseline_result.test_auroc,
                            "train_accuracy": baseline_result.train_accuracy,
                            "val_accuracy": baseline_result.val_accuracy,
                            "test_accuracy": baseline_result.test_accuracy,
                            "train_auprc": baseline_result.train_auprc,
                            "val_auprc": baseline_result.val_auprc,
                            "test_auprc": baseline_result.test_auprc,
                            "pretrain_seconds": baseline_result.stage_durations.get("pretrain_seconds"),
                            "finetune_seconds": baseline_result.stage_durations.get("finetune_seconds"),
                            "adapt_seconds": baseline_result.stage_durations.get("adapt_seconds"),
                            "pretrain_epochs": None,
                            "finetune_epochs": None,
                            "adapt_epochs": None,
                            "pretrain_val_auroc": baseline_result.pretrain_val_auroc,
                            "pretrain_val_accuracy": baseline_result.pretrain_val_accuracy,
                            "pretrain_val_auprc": baseline_result.pretrain_val_auprc,
                            "best_iteration": baseline_result.best_iteration,
                        })

                if "transformer" in model_types:
                    transformer_cfg = replace(transformer_config, input_dim=len(feature_list), feature_names=feature_tuple)
                    transformer_pipeline = TransformerPipeline(transformer_cfg)
                    transformer_result = transformer_pipeline.run(
                        seed=seed,
                        pretrain=pretrain_train_dataset,
                        pretrain_val=pretrain_val_dataset,
                        train=train_dataset,
                        val=val_dataset,
                        adapt=None,
                        evaluation=eval_dataset,
                    )
                    results.append({
                        **common_payload,
                        "model": "transformer",
                        "mode": "pretrain_finetune",
                        "train_auroc": transformer_result.train_auroc,
                        "val_auroc": transformer_result.val_auroc,
                        "test_auroc": transformer_result.test_auroc,
                        "train_accuracy": transformer_result.train_accuracy,
                        "val_accuracy": transformer_result.val_accuracy,
                        "test_accuracy": transformer_result.test_accuracy,
                        "train_auprc": transformer_result.train_auprc,
                        "val_auprc": transformer_result.val_auprc,
                        "test_auprc": transformer_result.test_auprc,
                        "pretrain_seconds": transformer_result.stage_durations.get("pretrain_seconds"),
                        "finetune_seconds": transformer_result.stage_durations.get("finetune_seconds"),
                        "adapt_seconds": transformer_result.stage_durations.get("adapt_seconds"),
                        "pretrain_epochs": transformer_result.stage_epochs.get("pretrain_epochs"),
                        "finetune_epochs": transformer_result.stage_epochs.get("finetune_epochs"),
                        "adapt_epochs": transformer_result.stage_epochs.get("adapt_epochs"),
                        "pretrain_val_auroc": transformer_result.pretrain_val_auroc,
                        "pretrain_val_accuracy": transformer_result.pretrain_val_accuracy,
                        "pretrain_val_auprc": transformer_result.pretrain_val_auprc,
                        "best_iteration": None,
                    })

                    if train_dataset.X.size > 0:
                        baseline_cfg = replace(transformer_config, input_dim=len(feature_list), feature_names=feature_tuple)
                        baseline_pipeline = TransformerPipeline(baseline_cfg)
                        baseline_result = baseline_pipeline.run(
                            seed=seed,
                            pretrain=None,
                            pretrain_val=None,
                            train=train_dataset,
                            val=val_dataset,
                            adapt=None,
                            evaluation=eval_dataset,
                        )
                        results.append({
                            **common_payload,
                            "model": "transformer",
                            "mode": "target_only",
                            "train_auroc": baseline_result.train_auroc,
                            "val_auroc": baseline_result.val_auroc,
                            "test_auroc": baseline_result.test_auroc,
                            "train_accuracy": baseline_result.train_accuracy,
                            "val_accuracy": baseline_result.val_accuracy,
                            "test_accuracy": baseline_result.test_accuracy,
                            "train_auprc": baseline_result.train_auprc,
                            "val_auprc": baseline_result.val_auprc,
                            "test_auprc": baseline_result.test_auprc,
                            "pretrain_seconds": baseline_result.stage_durations.get("pretrain_seconds"),
                            "finetune_seconds": baseline_result.stage_durations.get("finetune_seconds"),
                            "adapt_seconds": baseline_result.stage_durations.get("adapt_seconds"),
                            "pretrain_epochs": baseline_result.stage_epochs.get("pretrain_epochs"),
                            "finetune_epochs": baseline_result.stage_epochs.get("finetune_epochs"),
                            "adapt_epochs": baseline_result.stage_epochs.get("adapt_epochs"),
                            "pretrain_val_auroc": baseline_result.pretrain_val_auroc,
                            "pretrain_val_accuracy": baseline_result.pretrain_val_accuracy,
                            "pretrain_val_auprc": baseline_result.pretrain_val_auprc,
                            "best_iteration": None,
                        })

                if "dl_erm" in model_types:
                    dl_erm_cfg = replace(dl_erm_config, input_dim=len(feature_list))
                    dl_erm_pipeline = DLERMPipeline(dl_erm_cfg)
                    dl_erm_result = dl_erm_pipeline.run(
                        seed=seed,
                        pretrain=pretrain_train_dataset,
                        pretrain_val=pretrain_val_dataset,
                        train=train_dataset,
                        val=val_dataset,
                        adapt=None,
                        evaluation=eval_dataset,
                    )
                    results.append({
                        **common_payload,
                        "model": "dl_erm",
                        "mode": "pretrain_finetune",
                        "train_auroc": dl_erm_result.train_auroc,
                        "val_auroc": dl_erm_result.val_auroc,
                        "test_auroc": dl_erm_result.test_auroc,
                        "train_accuracy": dl_erm_result.train_accuracy,
                        "val_accuracy": dl_erm_result.val_accuracy,
                        "test_accuracy": dl_erm_result.test_accuracy,
                        "train_auprc": dl_erm_result.train_auprc,
                        "val_auprc": dl_erm_result.val_auprc,
                        "test_auprc": dl_erm_result.test_auprc,
                        "pretrain_seconds": dl_erm_result.stage_durations.get("pretrain_seconds"),
                        "finetune_seconds": dl_erm_result.stage_durations.get("finetune_seconds"),
                        "adapt_seconds": dl_erm_result.stage_durations.get("adapt_seconds"),
                        "pretrain_epochs": dl_erm_result.stage_epochs.get("pretrain_epochs"),
                        "finetune_epochs": dl_erm_result.stage_epochs.get("finetune_epochs"),
                        "adapt_epochs": dl_erm_result.stage_epochs.get("adapt_epochs"),
                        "pretrain_val_auroc": dl_erm_result.pretrain_val_auroc,
                        "pretrain_val_accuracy": dl_erm_result.pretrain_val_accuracy,
                        "pretrain_val_auprc": dl_erm_result.pretrain_val_auprc,
                        "best_iteration": None,
                    })

                    if train_dataset.X.size > 0:
                        dl_erm_target_cfg = replace(dl_erm_cfg, pretrain_epochs=0)
                        dl_erm_target_pipeline = DLERMPipeline(dl_erm_target_cfg)
                        dl_erm_target_result = dl_erm_target_pipeline.run(
                            seed=seed,
                            pretrain=None,
                            pretrain_val=None,
                            train=train_dataset,
                            val=val_dataset,
                            adapt=None,
                            evaluation=eval_dataset,
                        )
                        results.append({
                            **common_payload,
                            "model": "dl_erm",
                            "mode": "target_only",
                            "train_auroc": dl_erm_target_result.train_auroc,
                            "val_auroc": dl_erm_target_result.val_auroc,
                            "test_auroc": dl_erm_target_result.test_auroc,
                            "train_accuracy": dl_erm_target_result.train_accuracy,
                            "val_accuracy": dl_erm_target_result.val_accuracy,
                            "test_accuracy": dl_erm_target_result.test_accuracy,
                            "train_auprc": dl_erm_target_result.train_auprc,
                            "val_auprc": dl_erm_target_result.val_auprc,
                            "test_auprc": dl_erm_target_result.test_auprc,
                            "pretrain_seconds": dl_erm_target_result.stage_durations.get("pretrain_seconds"),
                            "finetune_seconds": dl_erm_target_result.stage_durations.get("finetune_seconds"),
                            "adapt_seconds": dl_erm_target_result.stage_durations.get("adapt_seconds"),
                            "pretrain_epochs": dl_erm_target_result.stage_epochs.get("pretrain_epochs"),
                            "finetune_epochs": dl_erm_target_result.stage_epochs.get("finetune_epochs"),
                            "adapt_epochs": dl_erm_target_result.stage_epochs.get("adapt_epochs"),
                            "pretrain_val_auroc": dl_erm_target_result.pretrain_val_auroc,
                            "pretrain_val_accuracy": dl_erm_target_result.pretrain_val_accuracy,
                            "pretrain_val_auprc": dl_erm_target_result.pretrain_val_auprc,
                            "best_iteration": None,
                        })

                if "dl_clustering" in model_types:
                    dl_clustering_cfg = replace(dl_clustering_config, input_dim=len(feature_list))
                    dl_clustering_pipeline = DLClusteringPipeline(dl_clustering_cfg)
                    dl_clustering_result = dl_clustering_pipeline.run(
                        seed=seed,
                        pretrain=pretrain_train_dataset,
                        pretrain_val=pretrain_val_dataset,
                        train=train_dataset,
                        val=val_dataset,
                        adapt=None,
                        evaluation=eval_dataset,
                    )
                    results.append({
                        **common_payload,
                        "model": "dl_clustering",
                        "mode": "pretrain_finetune",
                        "train_auroc": dl_clustering_result.train_auroc,
                        "val_auroc": dl_clustering_result.val_auroc,
                        "test_auroc": dl_clustering_result.test_auroc,
                        "train_accuracy": dl_clustering_result.train_accuracy,
                        "val_accuracy": dl_clustering_result.val_accuracy,
                        "test_accuracy": dl_clustering_result.test_accuracy,
                        "train_auprc": dl_clustering_result.train_auprc,
                        "val_auprc": dl_clustering_result.val_auprc,
                        "test_auprc": dl_clustering_result.test_auprc,
                        "pretrain_seconds": dl_clustering_result.stage_durations.get("pretrain_seconds"),
                        "finetune_seconds": dl_clustering_result.stage_durations.get("finetune_seconds"),
                        "adapt_seconds": dl_clustering_result.stage_durations.get("adapt_seconds"),
                        "pretrain_epochs": dl_clustering_result.stage_epochs.get("pretrain_epochs"),
                        "finetune_epochs": dl_clustering_result.stage_epochs.get("finetune_epochs"),
                        "adapt_epochs": dl_clustering_result.stage_epochs.get("adapt_epochs"),
                        "pretrain_val_auroc": dl_clustering_result.pretrain_val_auroc,
                        "pretrain_val_accuracy": dl_clustering_result.pretrain_val_accuracy,
                        "pretrain_val_auprc": dl_clustering_result.pretrain_val_auprc,
                        "best_iteration": None,
                    })

                    if train_dataset.X.size > 0:
                        dl_clustering_target_cfg = replace(dl_clustering_cfg, pretrain_epochs=0)
                        dl_clustering_target_pipeline = DLClusteringPipeline(dl_clustering_target_cfg)
                        dl_clustering_target_result = dl_clustering_target_pipeline.run(
                            seed=seed,
                            pretrain=None,
                            pretrain_val=None,
                            train=train_dataset,
                            val=val_dataset,
                            adapt=None,
                            evaluation=eval_dataset,
                        )
                        results.append({
                            **common_payload,
                            "model": "dl_clustering",
                            "mode": "target_only",
                            "train_auroc": dl_clustering_target_result.train_auroc,
                            "val_auroc": dl_clustering_target_result.val_auroc,
                            "test_auroc": dl_clustering_target_result.test_auroc,
                            "train_accuracy": dl_clustering_target_result.train_accuracy,
                            "val_accuracy": dl_clustering_target_result.val_accuracy,
                            "test_accuracy": dl_clustering_target_result.test_accuracy,
                            "train_auprc": dl_clustering_target_result.train_auprc,
                            "val_auprc": dl_clustering_target_result.val_auprc,
                            "test_auprc": dl_clustering_target_result.test_auprc,
                            "pretrain_seconds": dl_clustering_target_result.stage_durations.get("pretrain_seconds"),
                            "finetune_seconds": dl_clustering_target_result.stage_durations.get("finetune_seconds"),
                            "adapt_seconds": dl_clustering_target_result.stage_durations.get("adapt_seconds"),
                            "pretrain_epochs": dl_clustering_target_result.stage_epochs.get("pretrain_epochs"),
                            "finetune_epochs": dl_clustering_target_result.stage_epochs.get("finetune_epochs"),
                            "adapt_epochs": dl_clustering_target_result.stage_epochs.get("adapt_epochs"),
                            "pretrain_val_auroc": dl_clustering_target_result.pretrain_val_auroc,
                            "pretrain_val_accuracy": dl_clustering_target_result.pretrain_val_accuracy,
                            "pretrain_val_auprc": dl_clustering_target_result.pretrain_val_auprc,
                            "best_iteration": None,
                        })

                if "dl_siamese" in model_types:
                    dl_siamese_cfg = replace(dl_siamese_config, input_dim=len(feature_list))
                    dl_siamese_pipeline = DLSiamesePipeline(dl_siamese_cfg)
                    dl_siamese_result = dl_siamese_pipeline.run(
                        seed=seed,
                        pretrain=pretrain_train_dataset,
                        pretrain_val=pretrain_val_dataset,
                        train=train_dataset,
                        val=val_dataset,
                        adapt=None,
                        evaluation=eval_dataset,
                    )
                    results.append({
                        **common_payload,
                        "model": "dl_siamese",
                        "mode": "pretrain_finetune",
                        "train_auroc": dl_siamese_result.train_auroc,
                        "val_auroc": dl_siamese_result.val_auroc,
                        "test_auroc": dl_siamese_result.test_auroc,
                        "train_accuracy": dl_siamese_result.train_accuracy,
                        "val_accuracy": dl_siamese_result.val_accuracy,
                        "test_accuracy": dl_siamese_result.test_accuracy,
                        "train_auprc": dl_siamese_result.train_auprc,
                        "val_auprc": dl_siamese_result.val_auprc,
                        "test_auprc": dl_siamese_result.test_auprc,
                        "pretrain_seconds": dl_siamese_result.stage_durations.get("pretrain_seconds"),
                        "finetune_seconds": dl_siamese_result.stage_durations.get("finetune_seconds"),
                        "adapt_seconds": dl_siamese_result.stage_durations.get("adapt_seconds"),
                        "pretrain_epochs": dl_siamese_result.stage_epochs.get("pretrain_epochs"),
                        "finetune_epochs": dl_siamese_result.stage_epochs.get("finetune_epochs"),
                        "adapt_epochs": dl_siamese_result.stage_epochs.get("adapt_epochs"),
                        "pretrain_val_auroc": dl_siamese_result.pretrain_val_auroc,
                        "pretrain_val_accuracy": dl_siamese_result.pretrain_val_accuracy,
                        "pretrain_val_auprc": dl_siamese_result.pretrain_val_auprc,
                        "best_iteration": None,
                    })

                    if train_dataset.X.size > 0:
                        dl_siamese_target_cfg = replace(dl_siamese_cfg, pretrain_epochs=0)
                        dl_siamese_target_pipeline = DLSiamesePipeline(dl_siamese_target_cfg)
                        dl_siamese_target_result = dl_siamese_target_pipeline.run(
                            seed=seed,
                            pretrain=None,
                            pretrain_val=None,
                            train=train_dataset,
                            val=val_dataset,
                            adapt=None,
                            evaluation=eval_dataset,
                        )
                        results.append({
                            **common_payload,
                            "model": "dl_siamese",
                            "mode": "target_only",
                            "train_auroc": dl_siamese_target_result.train_auroc,
                            "val_auroc": dl_siamese_target_result.val_auroc,
                            "test_auroc": dl_siamese_target_result.test_auroc,
                            "train_accuracy": dl_siamese_target_result.train_accuracy,
                            "val_accuracy": dl_siamese_target_result.val_accuracy,
                            "test_accuracy": dl_siamese_target_result.test_accuracy,
                            "train_auprc": dl_siamese_target_result.train_auprc,
                            "val_auprc": dl_siamese_target_result.val_auprc,
                            "test_auprc": dl_siamese_target_result.test_auprc,
                            "pretrain_seconds": dl_siamese_target_result.stage_durations.get("pretrain_seconds"),
                            "finetune_seconds": dl_siamese_target_result.stage_durations.get("finetune_seconds"),
                            "adapt_seconds": dl_siamese_target_result.stage_durations.get("adapt_seconds"),
                            "pretrain_epochs": dl_siamese_target_result.stage_epochs.get("pretrain_epochs"),
                            "finetune_epochs": dl_siamese_target_result.stage_epochs.get("finetune_epochs"),
                            "adapt_epochs": dl_siamese_target_result.stage_epochs.get("adapt_epochs"),
                            "pretrain_val_auroc": dl_siamese_target_result.pretrain_val_auroc,
                            "pretrain_val_accuracy": dl_siamese_target_result.pretrain_val_accuracy,
                            "pretrain_val_auprc": dl_siamese_target_result.pretrain_val_auprc,
                            "best_iteration": None,
                        })

                if "dl_reorder" in model_types:
                    dl_reorder_cfg = replace(dl_reorder_config, input_dim=len(feature_list))
                    dl_reorder_pipeline = DLReorderPipeline(dl_reorder_cfg)
                    dl_reorder_result = dl_reorder_pipeline.run(
                        seed=seed,
                        pretrain=pretrain_train_dataset,
                        pretrain_val=pretrain_val_dataset,
                        train=train_dataset,
                        val=val_dataset,
                        adapt=None,
                        evaluation=eval_dataset,
                    )
                    results.append({
                        **common_payload,
                        "model": "dl_reorder",
                        "mode": "pretrain_finetune",
                        "train_auroc": dl_reorder_result.train_auroc,
                        "val_auroc": dl_reorder_result.val_auroc,
                        "test_auroc": dl_reorder_result.test_auroc,
                        "train_accuracy": dl_reorder_result.train_accuracy,
                        "val_accuracy": dl_reorder_result.val_accuracy,
                        "test_accuracy": dl_reorder_result.test_accuracy,
                        "train_auprc": dl_reorder_result.train_auprc,
                        "val_auprc": dl_reorder_result.val_auprc,
                        "test_auprc": dl_reorder_result.test_auprc,
                        "pretrain_seconds": dl_reorder_result.stage_durations.get("pretrain_seconds"),
                        "finetune_seconds": dl_reorder_result.stage_durations.get("finetune_seconds"),
                        "adapt_seconds": dl_reorder_result.stage_durations.get("adapt_seconds"),
                        "pretrain_epochs": dl_reorder_result.stage_epochs.get("pretrain_epochs"),
                        "finetune_epochs": dl_reorder_result.stage_epochs.get("finetune_epochs"),
                        "adapt_epochs": dl_reorder_result.stage_epochs.get("adapt_epochs"),
                        "pretrain_val_auroc": dl_reorder_result.pretrain_val_auroc,
                        "pretrain_val_accuracy": dl_reorder_result.pretrain_val_accuracy,
                        "pretrain_val_auprc": dl_reorder_result.pretrain_val_auprc,
                        "best_iteration": None,
                    })

                    if train_dataset.X.size > 0:
                        dl_reorder_target_cfg = replace(dl_reorder_cfg, pretrain_epochs=0)
                        dl_reorder_target_pipeline = DLReorderPipeline(dl_reorder_target_cfg)
                        dl_reorder_target_result = dl_reorder_target_pipeline.run(
                            seed=seed,
                            pretrain=None,
                            pretrain_val=None,
                            train=train_dataset,
                            val=val_dataset,
                            adapt=None,
                            evaluation=eval_dataset,
                        )
                        results.append({
                            **common_payload,
                            "model": "dl_reorder",
                            "mode": "target_only",
                            "train_auroc": dl_reorder_target_result.train_auroc,
                            "val_auroc": dl_reorder_target_result.val_auroc,
                            "test_auroc": dl_reorder_target_result.test_auroc,
                            "train_accuracy": dl_reorder_target_result.train_accuracy,
                            "val_accuracy": dl_reorder_target_result.val_accuracy,
                            "test_accuracy": dl_reorder_target_result.test_accuracy,
                            "train_auprc": dl_reorder_target_result.train_auprc,
                            "val_auprc": dl_reorder_target_result.val_auprc,
                            "test_auprc": dl_reorder_target_result.test_auprc,
                            "pretrain_seconds": dl_reorder_target_result.stage_durations.get("pretrain_seconds"),
                            "finetune_seconds": dl_reorder_target_result.stage_durations.get("finetune_seconds"),
                            "adapt_seconds": dl_reorder_target_result.stage_durations.get("adapt_seconds"),
                            "pretrain_epochs": dl_reorder_target_result.stage_epochs.get("pretrain_epochs"),
                            "finetune_epochs": dl_reorder_target_result.stage_epochs.get("finetune_epochs"),
                            "adapt_epochs": dl_reorder_target_result.stage_epochs.get("adapt_epochs"),
                            "pretrain_val_auroc": dl_reorder_target_result.pretrain_val_auroc,
                            "pretrain_val_accuracy": dl_reorder_target_result.pretrain_val_accuracy,
                            "pretrain_val_auprc": dl_reorder_target_result.pretrain_val_auprc,
                            "best_iteration": None,
                        })

                if "dl_masf" in model_types:
                    dl_masf_cfg = replace(dl_masf_config, input_dim=len(feature_list))
                    dl_masf_pipeline = DLMASFPipeline(dl_masf_cfg)
                    dl_masf_result = dl_masf_pipeline.run(
                        seed=seed,
                        pretrain=pretrain_train_dataset,
                        pretrain_val=pretrain_val_dataset,
                        train=train_dataset,
                        val=val_dataset,
                        adapt=None,
                        evaluation=eval_dataset,
                    )
                    results.append({
                        **common_payload,
                        "model": "dl_masf",
                        "mode": "pretrain_finetune",
                        "train_auroc": dl_masf_result.train_auroc,
                        "val_auroc": dl_masf_result.val_auroc,
                        "test_auroc": dl_masf_result.test_auroc,
                        "train_accuracy": dl_masf_result.train_accuracy,
                        "val_accuracy": dl_masf_result.val_accuracy,
                        "test_accuracy": dl_masf_result.test_accuracy,
                        "train_auprc": dl_masf_result.train_auprc,
                        "val_auprc": dl_masf_result.val_auprc,
                        "test_auprc": dl_masf_result.test_auprc,
                        "pretrain_seconds": dl_masf_result.stage_durations.get("pretrain_seconds"),
                        "finetune_seconds": dl_masf_result.stage_durations.get("finetune_seconds"),
                        "adapt_seconds": dl_masf_result.stage_durations.get("adapt_seconds"),
                        "pretrain_epochs": dl_masf_result.stage_epochs.get("pretrain_epochs"),
                        "finetune_epochs": dl_masf_result.stage_epochs.get("finetune_epochs"),
                        "adapt_epochs": dl_masf_result.stage_epochs.get("adapt_epochs"),
                        "pretrain_val_auroc": dl_masf_result.pretrain_val_auroc,
                        "pretrain_val_accuracy": dl_masf_result.pretrain_val_accuracy,
                        "pretrain_val_auprc": dl_masf_result.pretrain_val_auprc,
                        "best_iteration": None,
                    })

                    if train_dataset.X.size > 0:
                        dl_masf_target_cfg = replace(dl_masf_cfg, pretrain_epochs=0)
                        dl_masf_target_pipeline = DLMASFPipeline(dl_masf_target_cfg)
                        dl_masf_target_result = dl_masf_target_pipeline.run(
                            seed=seed,
                            pretrain=None,
                            pretrain_val=None,
                            train=train_dataset,
                            val=val_dataset,
                            adapt=None,
                            evaluation=eval_dataset,
                        )
                        results.append({
                            **common_payload,
                            "model": "dl_masf",
                            "mode": "target_only",
                            "train_auroc": dl_masf_target_result.train_auroc,
                            "val_auroc": dl_masf_target_result.val_auroc,
                            "test_auroc": dl_masf_target_result.test_auroc,
                            "train_accuracy": dl_masf_target_result.train_accuracy,
                            "val_accuracy": dl_masf_target_result.val_accuracy,
                            "test_accuracy": dl_masf_target_result.test_accuracy,
                            "train_auprc": dl_masf_target_result.train_auprc,
                            "val_auprc": dl_masf_target_result.val_auprc,
                            "test_auprc": dl_masf_target_result.test_auprc,
                            "pretrain_seconds": dl_masf_target_result.stage_durations.get("pretrain_seconds"),
                            "finetune_seconds": dl_masf_target_result.stage_durations.get("finetune_seconds"),
                            "adapt_seconds": dl_masf_target_result.stage_durations.get("adapt_seconds"),
                            "pretrain_epochs": dl_masf_target_result.stage_epochs.get("pretrain_epochs"),
                            "finetune_epochs": dl_masf_target_result.stage_epochs.get("finetune_epochs"),
                            "adapt_epochs": dl_masf_target_result.stage_epochs.get("adapt_epochs"),
                            "pretrain_val_auroc": dl_masf_target_result.pretrain_val_auroc,
                            "pretrain_val_accuracy": dl_masf_target_result.pretrain_val_accuracy,
                            "pretrain_val_auprc": dl_masf_target_result.pretrain_val_auprc,
                            "best_iteration": None,
                        })

                if "dl_dann" in model_types:
                    dl_dann_cfg = replace(dl_dann_config, input_dim=len(feature_list))
                    dl_dann_pipeline = DLDANNPipeline(dl_dann_cfg)
                    dl_dann_result = dl_dann_pipeline.run(
                        seed=seed,
                        pretrain=pretrain_train_dataset,
                        pretrain_val=pretrain_val_dataset,
                        train=train_dataset,
                        val=val_dataset,
                        adapt=None,
                        evaluation=eval_dataset,
                    )
                    results.append({
                        **common_payload,
                        "model": "dl_dann",
                        "mode": "pretrain_finetune",
                        "train_auroc": dl_dann_result.train_auroc,
                        "val_auroc": dl_dann_result.val_auroc,
                        "test_auroc": dl_dann_result.test_auroc,
                        "train_accuracy": dl_dann_result.train_accuracy,
                        "val_accuracy": dl_dann_result.val_accuracy,
                        "test_accuracy": dl_dann_result.test_accuracy,
                        "train_auprc": dl_dann_result.train_auprc,
                        "val_auprc": dl_dann_result.val_auprc,
                        "test_auprc": dl_dann_result.test_auprc,
                        "pretrain_seconds": dl_dann_result.stage_durations.get("pretrain_seconds"),
                        "finetune_seconds": dl_dann_result.stage_durations.get("finetune_seconds"),
                        "adapt_seconds": dl_dann_result.stage_durations.get("adapt_seconds"),
                        "pretrain_epochs": dl_dann_result.stage_epochs.get("pretrain_epochs"),
                        "finetune_epochs": dl_dann_result.stage_epochs.get("finetune_epochs"),
                        "adapt_epochs": dl_dann_result.stage_epochs.get("adapt_epochs"),
                        "pretrain_val_auroc": dl_dann_result.pretrain_val_auroc,
                        "pretrain_val_accuracy": dl_dann_result.pretrain_val_accuracy,
                        "pretrain_val_auprc": dl_dann_result.pretrain_val_auprc,
                        "best_iteration": None,
                    })

                    if train_dataset.X.size > 0:
                        dl_dann_target_cfg = replace(dl_dann_cfg, pretrain_epochs=0)
                        dl_dann_target_pipeline = DLDANNPipeline(dl_dann_target_cfg)
                        dl_dann_target_result = dl_dann_target_pipeline.run(
                            seed=seed,
                            pretrain=None,
                            pretrain_val=None,
                            train=train_dataset,
                            val=val_dataset,
                            adapt=None,
                            evaluation=eval_dataset,
                        )
                        results.append({
                            **common_payload,
                            "model": "dl_dann",
                            "mode": "target_only",
                            "train_auroc": dl_dann_target_result.train_auroc,
                            "val_auroc": dl_dann_target_result.val_auroc,
                            "test_auroc": dl_dann_target_result.test_auroc,
                            "train_accuracy": dl_dann_target_result.train_accuracy,
                            "val_accuracy": dl_dann_target_result.val_accuracy,
                            "test_accuracy": dl_dann_target_result.test_accuracy,
                            "train_auprc": dl_dann_target_result.train_auprc,
                            "val_auprc": dl_dann_target_result.val_auprc,
                            "test_auprc": dl_dann_target_result.test_auprc,
                            "pretrain_seconds": dl_dann_target_result.stage_durations.get("pretrain_seconds"),
                            "finetune_seconds": dl_dann_target_result.stage_durations.get("finetune_seconds"),
                            "adapt_seconds": dl_dann_target_result.stage_durations.get("adapt_seconds"),
                            "pretrain_epochs": dl_dann_target_result.stage_epochs.get("pretrain_epochs"),
                            "finetune_epochs": dl_dann_target_result.stage_epochs.get("finetune_epochs"),
                            "adapt_epochs": dl_dann_target_result.stage_epochs.get("adapt_epochs"),
                            "pretrain_val_auroc": dl_dann_target_result.pretrain_val_auroc,
                            "pretrain_val_accuracy": dl_dann_target_result.pretrain_val_accuracy,
                            "pretrain_val_auprc": dl_dann_target_result.pretrain_val_auprc,
                            "best_iteration": None,
                        })

                if "dl_irm" in model_types:
                    dl_irm_cfg = replace(dl_irm_config, input_dim=len(feature_list))
                    dl_irm_pipeline = DLIRMPipeline(dl_irm_cfg)
                    dl_irm_result = dl_irm_pipeline.run(
                        seed=seed,
                        pretrain=pretrain_train_dataset,
                        pretrain_val=pretrain_val_dataset,
                        train=train_dataset,
                        val=val_dataset,
                        adapt=None,
                        evaluation=eval_dataset,
                    )
                    results.append({
                        **common_payload,
                        "model": "dl_irm",
                        "mode": "pretrain_finetune",
                        "train_auroc": dl_irm_result.train_auroc,
                        "val_auroc": dl_irm_result.val_auroc,
                        "test_auroc": dl_irm_result.test_auroc,
                        "train_accuracy": dl_irm_result.train_accuracy,
                        "val_accuracy": dl_irm_result.val_accuracy,
                        "test_accuracy": dl_irm_result.test_accuracy,
                        "train_auprc": dl_irm_result.train_auprc,
                        "val_auprc": dl_irm_result.val_auprc,
                        "test_auprc": dl_irm_result.test_auprc,
                        "pretrain_seconds": dl_irm_result.stage_durations.get("pretrain_seconds"),
                        "finetune_seconds": dl_irm_result.stage_durations.get("finetune_seconds"),
                        "adapt_seconds": dl_irm_result.stage_durations.get("adapt_seconds"),
                        "pretrain_epochs": dl_irm_result.stage_epochs.get("pretrain_epochs"),
                        "finetune_epochs": dl_irm_result.stage_epochs.get("finetune_epochs"),
                        "adapt_epochs": dl_irm_result.stage_epochs.get("adapt_epochs"),
                        "pretrain_val_auroc": dl_irm_result.pretrain_val_auroc,
                        "pretrain_val_accuracy": dl_irm_result.pretrain_val_accuracy,
                        "pretrain_val_auprc": dl_irm_result.pretrain_val_auprc,
                        "best_iteration": None,
                    })

                    if train_dataset.X.size > 0:
                        dl_irm_target_cfg = replace(dl_irm_cfg, pretrain_epochs=0)
                        dl_irm_target_pipeline = DLIRMPipeline(dl_irm_target_cfg)
                        dl_irm_target_result = dl_irm_target_pipeline.run(
                            seed=seed,
                            pretrain=None,
                            pretrain_val=None,
                            train=train_dataset,
                            val=val_dataset,
                            adapt=None,
                            evaluation=eval_dataset,
                        )
                        results.append({
                            **common_payload,
                            "model": "dl_irm",
                            "mode": "target_only",
                            "train_auroc": dl_irm_target_result.train_auroc,
                            "val_auroc": dl_irm_target_result.val_auroc,
                            "test_auroc": dl_irm_target_result.test_auroc,
                            "train_accuracy": dl_irm_target_result.train_accuracy,
                            "val_accuracy": dl_irm_target_result.val_accuracy,
                            "test_accuracy": dl_irm_target_result.test_accuracy,
                            "train_auprc": dl_irm_target_result.train_auprc,
                            "val_auprc": dl_irm_target_result.val_auprc,
                            "test_auprc": dl_irm_target_result.test_auprc,
                            "pretrain_seconds": dl_irm_target_result.stage_durations.get("pretrain_seconds"),
                            "finetune_seconds": dl_irm_target_result.stage_durations.get("finetune_seconds"),
                            "adapt_seconds": dl_irm_target_result.stage_durations.get("adapt_seconds"),
                            "pretrain_epochs": dl_irm_target_result.stage_epochs.get("pretrain_epochs"),
                            "finetune_epochs": dl_irm_target_result.stage_epochs.get("finetune_epochs"),
                            "adapt_epochs": dl_irm_target_result.stage_epochs.get("adapt_epochs"),
                            "pretrain_val_auroc": dl_irm_target_result.pretrain_val_auroc,
                            "pretrain_val_accuracy": dl_irm_target_result.pretrain_val_accuracy,
                            "pretrain_val_auprc": dl_irm_target_result.pretrain_val_auprc,
                            "best_iteration": None,
                        })

                if "dl_csd" in model_types:
                    dl_csd_cfg = replace(dl_csd_config, input_dim=len(feature_list), domain_count=domain_count)
                    dl_csd_pipeline = DLCSDPipeline(dl_csd_cfg)
                    dl_csd_result = dl_csd_pipeline.run(
                        seed=seed,
                        pretrain=pretrain_train_dataset,
                        pretrain_val=pretrain_val_dataset,
                        train=train_dataset,
                        val=val_dataset,
                        adapt=None,
                        evaluation=eval_dataset,
                    )
                    results.append({
                        **common_payload,
                        "model": "dl_csd",
                        "mode": "pretrain_finetune",
                        "train_auroc": dl_csd_result.train_auroc,
                        "val_auroc": dl_csd_result.val_auroc,
                        "test_auroc": dl_csd_result.test_auroc,
                        "train_accuracy": dl_csd_result.train_accuracy,
                        "val_accuracy": dl_csd_result.val_accuracy,
                        "test_accuracy": dl_csd_result.test_accuracy,
                        "train_auprc": dl_csd_result.train_auprc,
                        "val_auprc": dl_csd_result.val_auprc,
                        "test_auprc": dl_csd_result.test_auprc,
                        "pretrain_seconds": dl_csd_result.stage_durations.get("pretrain_seconds"),
                        "finetune_seconds": dl_csd_result.stage_durations.get("finetune_seconds"),
                        "adapt_seconds": dl_csd_result.stage_durations.get("adapt_seconds"),
                        "pretrain_epochs": dl_csd_result.stage_epochs.get("pretrain_epochs"),
                        "finetune_epochs": dl_csd_result.stage_epochs.get("finetune_epochs"),
                        "adapt_epochs": dl_csd_result.stage_epochs.get("adapt_epochs"),
                        "pretrain_val_auroc": dl_csd_result.pretrain_val_auroc,
                        "pretrain_val_accuracy": dl_csd_result.pretrain_val_accuracy,
                        "pretrain_val_auprc": dl_csd_result.pretrain_val_auprc,
                        "best_iteration": None,
                    })

                    if train_dataset.X.size > 0:
                        dl_csd_target_cfg = replace(dl_csd_cfg, pretrain_epochs=0)
                        dl_csd_target_pipeline = DLCSDPipeline(dl_csd_target_cfg)
                        dl_csd_target_result = dl_csd_target_pipeline.run(
                            seed=seed,
                            pretrain=None,
                            pretrain_val=None,
                            train=train_dataset,
                            val=val_dataset,
                            adapt=None,
                            evaluation=eval_dataset,
                        )
                        results.append({
                            **common_payload,
                            "model": "dl_csd",
                            "mode": "target_only",
                            "train_auroc": dl_csd_target_result.train_auroc,
                            "val_auroc": dl_csd_target_result.val_auroc,
                            "test_auroc": dl_csd_target_result.test_auroc,
                            "train_accuracy": dl_csd_target_result.train_accuracy,
                            "val_accuracy": dl_csd_target_result.val_accuracy,
                            "test_accuracy": dl_csd_target_result.test_accuracy,
                            "train_auprc": dl_csd_target_result.train_auprc,
                            "val_auprc": dl_csd_target_result.val_auprc,
                            "test_auprc": dl_csd_target_result.test_auprc,
                            "pretrain_seconds": dl_csd_target_result.stage_durations.get("pretrain_seconds"),
                            "finetune_seconds": dl_csd_target_result.stage_durations.get("finetune_seconds"),
                            "adapt_seconds": dl_csd_target_result.stage_durations.get("adapt_seconds"),
                            "pretrain_epochs": dl_csd_target_result.stage_epochs.get("pretrain_epochs"),
                            "finetune_epochs": dl_csd_target_result.stage_epochs.get("finetune_epochs"),
                            "adapt_epochs": dl_csd_target_result.stage_epochs.get("adapt_epochs"),
                        "pretrain_val_auroc": dl_csd_target_result.pretrain_val_auroc,
                        "pretrain_val_accuracy": dl_csd_target_result.pretrain_val_accuracy,
                        "pretrain_val_auprc": dl_csd_target_result.pretrain_val_auprc,
                        "best_iteration": None,
                        })

                if "dl_mldg" in model_types:
                    source_dataset = combined_pretrain
                    if source_dataset.X.size == 0 or source_dataset.domains is None or source_dataset.domains.size == 0:
                        continue
                    val_source = pretrain_val_dataset if pretrain_val_dataset is not None else source_dataset
                    dl_mldg_cfg = replace(dl_mldg_config, input_dim=len(feature_list))
                    dl_mldg_pipeline = DLMldgPipeline(dl_mldg_cfg)
                    dl_mldg_result = dl_mldg_pipeline.run(
                        seed=seed,
                        pretrain=source_dataset,
                        pretrain_val=val_source,
                        train=source_dataset,
                        val=val_source,
                        adapt=None,
                        evaluation=eval_dataset,
                    )
                    results.append({
                        **common_payload,
                        "model": "dl_mldg",
                        "mode": "pretrain_only",
                        "train_auroc": dl_mldg_result.train_auroc,
                        "val_auroc": dl_mldg_result.val_auroc,
                        "test_auroc": dl_mldg_result.test_auroc,
                        "train_accuracy": dl_mldg_result.train_accuracy,
                        "val_accuracy": dl_mldg_result.val_accuracy,
                        "test_accuracy": dl_mldg_result.test_accuracy,
                        "train_auprc": dl_mldg_result.train_auprc,
                        "val_auprc": dl_mldg_result.val_auprc,
                        "test_auprc": dl_mldg_result.test_auprc,
                        "pretrain_seconds": dl_mldg_result.stage_durations.get("pretrain_seconds"),
                        "finetune_seconds": dl_mldg_result.stage_durations.get("finetune_seconds"),
                        "adapt_seconds": dl_mldg_result.stage_durations.get("adapt_seconds"),
                        "pretrain_epochs": dl_mldg_result.stage_epochs.get("pretrain_epochs"),
                        "finetune_epochs": dl_mldg_result.stage_epochs.get("finetune_epochs"),
                        "adapt_epochs": dl_mldg_result.stage_epochs.get("adapt_epochs"),
                        "pretrain_val_auroc": dl_mldg_result.pretrain_val_auroc,
                        "pretrain_val_accuracy": dl_mldg_result.pretrain_val_accuracy,
                        "pretrain_val_auprc": dl_mldg_result.pretrain_val_auprc,
                        "best_iteration": None,
                    })

                # if "tabpfn" in model_types and tabpfn_pipeline is not None:
                #     def _append_tabpfn(result_obj: "TabPFNRunResult", mode: str) -> None:
                #         results.append({
                #             **common_payload,
                #             "model": "tabpfn",
                #             "mode": mode,
                #             "train_auroc": result_obj.train_auroc,
                #             "val_auroc": result_obj.val_auroc,
                #             "test_auroc": result_obj.test_auroc,
                #             "train_accuracy": result_obj.train_accuracy,
                #             "val_accuracy": result_obj.val_accuracy,
                #             "test_accuracy": result_obj.test_accuracy,
                #             "train_auprc": result_obj.train_auprc,
                #             "val_auprc": result_obj.val_auprc,
                #             "test_auprc": result_obj.test_auprc,
                #             "pretrain_seconds": None,
                #             "finetune_seconds": result_obj.stage_durations.get("train_seconds"),
                #             "adapt_seconds": None,
                #             "pretrain_epochs": None,
                #             "finetune_epochs": None,
                #             "adapt_epochs": None,
                #             "pretrain_val_auroc": float("nan"),
                #             "pretrain_val_accuracy": float("nan"),
                #             "pretrain_val_auprc": float("nan"),
                #             "best_iteration": None,
                #         })

                #     if pretrain_train_dataset.X.size > 0:
                #         try:
                #             tabpfn_result = tabpfn_pipeline.run(
                #                 seed=seed,
                #                 train=train_dataset,
                #                 val=val_dataset,
                #                 evaluation=eval_dataset,
                #                 extra_train=pretrain_train_dataset,
                #             )
                #         except ValueError:
                #             tabpfn_result = None
                #         if tabpfn_result is not None:
                #             _append_tabpfn(tabpfn_result, "pretrain_finetune")

                #     if train_dataset.X.size > 0:
                #         try:
                #             target_only_result = tabpfn_pipeline.run(
                #                 seed=seed,
                #                 train=train_dataset,
                #                 val=val_dataset,
                #                 evaluation=eval_dataset,
                #                 extra_train=None,
                #             )
                #         except ValueError:
                #             target_only_result = None
                #         if target_only_result is not None:
                #             _append_tabpfn(target_only_result, "target_only")

                if "cdtrans" in model_types:
                    cdtrans_adapt_dataset = eval_dataset
                    cdtrans_val_dataset = eval_dataset
                    cdtrans_payload = {
                        **common_payload,
                        "adapt_samples": float(len(cdtrans_adapt_dataset.y)),
                        "adapt_train_samples": float(len(cdtrans_adapt_dataset.y)),
                        "adapt_val_samples": float(len(cdtrans_val_dataset.y)),
                    }
                    print(
                        "[Pipeline][CDTrans]",
                        f"scenario={scenario.name}",
                        f"seed={seed}",
                        f"sources_shape={pretrain_train_dataset.X.shape}",
                        f"adapt_shape={cdtrans_adapt_dataset.X.shape}",
                        f"d_model={cdtrans_base_config.d_model}",
                        flush=True,
                    )
                    cdtrans_pipeline = CDTransPipeline(cdtrans_base_config)
                    cdtrans_result = cdtrans_pipeline.run(
                        seed=seed,
                        source_train=pretrain_train_dataset,
                        source_val=pretrain_val_dataset,
                        target_train=cdtrans_adapt_dataset,
                        target_val=cdtrans_val_dataset,
                        target_eval=eval_dataset,
                    )
                    pretrain_val = cdtrans_result.pretrain_val_metrics
                    pretrain_test = cdtrans_result.pretrain_test_metrics
                    results.append({
                        **cdtrans_payload,
                        "model": "cdtrans",
                        "mode": "pretrain_only",
                        "train_auroc": float("nan"),
                        "val_auroc": pretrain_val.get("auroc"),
                        "test_auroc": pretrain_test.get("auroc"),
                        "train_accuracy": float("nan"),
                        "val_accuracy": pretrain_val.get("accuracy"),
                        "test_accuracy": pretrain_test.get("accuracy"),
                        "train_auprc": float("nan"),
                        "val_auprc": pretrain_val.get("auprc"),
                        "test_auprc": pretrain_test.get("auprc"),
                        "pretrain_seconds": cdtrans_result.stage_durations.get("pretrain_seconds"),
                        "finetune_seconds": None,
                        "adapt_seconds": cdtrans_result.stage_durations.get("adapt_seconds"),
                        "pretrain_epochs": cdtrans_result.stage_epochs.get("pretrain_epochs"),
                        "finetune_epochs": None,
                        "adapt_epochs": None,
                        "pretrain_val_auroc": pretrain_val.get("auroc"),
                        "pretrain_val_accuracy": pretrain_val.get("accuracy"),
                        "pretrain_val_auprc": pretrain_val.get("auprc"),
                        "best_iteration": None,
                        })
                    if cdtrans_result.adapt_train_metrics is not None and cdtrans_result.adapt_val_metrics is not None:
                        adapt_train = cdtrans_result.adapt_train_metrics
                        adapt_val = cdtrans_result.adapt_val_metrics
                        adapt_test = cdtrans_result.adapt_test_metrics
                        results.append({
                            **cdtrans_payload,
                            "model": "cdtrans",
                            "mode": "adapt",
                            "train_auroc": adapt_train.get("auroc"),
                            "val_auroc": adapt_val.get("auroc"),
                            "test_auroc": adapt_test.get("auroc"),
                            "train_accuracy": adapt_train.get("accuracy"),
                            "val_accuracy": adapt_val.get("accuracy"),
                            "test_accuracy": adapt_test.get("accuracy"),
                            "train_auprc": adapt_train.get("auprc"),
                            "val_auprc": adapt_val.get("auprc"),
                            "test_auprc": adapt_test.get("auprc"),
                            "pretrain_seconds": cdtrans_result.stage_durations.get("pretrain_seconds"),
                            "finetune_seconds": None,
                            "adapt_seconds": cdtrans_result.stage_durations.get("adapt_seconds"),
                            "pretrain_epochs": cdtrans_result.stage_epochs.get("pretrain_epochs"),
                            "finetune_epochs": None,
                            "adapt_epochs": cdtrans_result.stage_epochs.get("adapt_epochs"),
                            "pretrain_val_auroc": pretrain_val.get("auroc"),
                            "pretrain_val_accuracy": pretrain_val.get("accuracy"),
                            "pretrain_val_auprc": pretrain_val.get("auprc"),
                            "best_iteration": None,
                        })
    return results


def scenarios_default(dataset_names: Optional[Sequence[str]] = None) -> List[ExperimentScenario]:
    if dataset_names is None:
        dataset_names = sorted(DEFAULT_DATASET_PATHS.keys())
    return generate_all_scenarios(dataset_names)


def run_all_scenarios(
    *,
    cache_dir: Path,
    overrides: Optional[Dict[str, str]] = None,
    fine_tune_ratios: Sequence[float] = (0.0, 0.2, 0.4, 0.6),
    fine_tune_val_ratio: float = 0.2,
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
    model_types: Sequence[str] = ("tree", "transformer"),
    val_size: float = 0.2,
    test_size: float = 0.2,
    pretrain_val_ratio: float = 0.1,
    target_train_ratio: Optional[float] = None,
    target_val_ratio: Optional[float] = None,
    target_test_ratio: Optional[float] = None,
    split_strategy: str = "stratified_shuffle",
    group_folds: int = 5,
    cdtrans_config: Optional[CDTransConfig] = None,
    feature_strategy: str = "auto",
    min_source_feature_frac: float = 0.005,
    min_source_feature_std: float = 0.0,
    feature_zero_tolerance: float = 1e-6,
    include_target_in_scaler: bool = True,
    feature_clip_value: Optional[float] = 5.0,
    transformer_overrides: Optional[Dict[str, object]] = None,
    dl_erm_config: Optional[DLERMConfig] = None,
    dl_erm_overrides: Optional[Dict[str, object]] = None,
    dl_dann_config: Optional[DLDANNConfig] = None,
    dl_dann_overrides: Optional[Dict[str, object]] = None,
    dl_irm_config: Optional[DLIRMConfig] = None,
    dl_irm_overrides: Optional[Dict[str, object]] = None,
    dl_csd_config: Optional[DLCSConfig] = None,
    dl_csd_overrides: Optional[Dict[str, object]] = None,
    dl_mldg_config: Optional[DLMldgConfig] = None,
    dl_mldg_overrides: Optional[Dict[str, object]] = None,
    dl_clustering_config: Optional[DLClusteringConfig] = None,
    dl_clustering_overrides: Optional[Dict[str, object]] = None,
    dl_siamese_config: Optional[DLSiameseConfig] = None,
    dl_siamese_overrides: Optional[Dict[str, object]] = None,
    dl_reorder_config: Optional[DLReorderConfig] = None,
    dl_reorder_overrides: Optional[Dict[str, object]] = None,
    dl_masf_config: Optional[DLMASFConfig] = None,
    dl_masf_overrides: Optional[Dict[str, object]] = None,
    # tabpfn_config: Optional[TabPFNConfig] = None,
    # tabpfn_overrides: Optional[Dict[str, object]] = None,
    max_target_shift_std: float = 15.0,
    max_feature_correlation: float = 0.0,
    correlation_sample_rows: int = 2000,
    max_class_conditional_shift: float = 0.0,
    class_shift_max_samples: int = 5000,
    l1_feature_keep_ratio: float = 0.0,
    l1_max_samples: int = 5000,
    optuna_tune: bool = False,
    optuna_trials: int = 50,
    optuna_seed: int = 42,
    tuned_configs: Optional[Dict[str, object]] = None,
    tuned_configs_out: Optional[Dict[str, object]] = None,
    scenario_specs: Optional[Sequence[ExperimentScenario]] = None,
    output_csv: Optional[Path] = None,
) -> pd.DataFrame:
    cache = CacheManager(cache_dir)
    all_results: List[Dict[str, object]] = []

    store = DatasetStore.from_defaults(cache=cache, overrides=overrides)
    dataset_names = sorted(store.configs.keys())
    if scenario_specs:
        scenario_list = list(scenario_specs)
        valid_names = set(dataset_names)
        for scenario in scenario_list:
            missing = [name for name in scenario.sources + [scenario.target] if name not in valid_names]
            if missing:
                raise ValueError(f"Scenario {scenario.name} references unknown datasets: {missing}")
    else:
        scenario_list = scenarios_default(dataset_names)
        if not scenario_list:
            raise ValueError("No dataset combinations available for scenario generation")

    resolved_feature_strategy = _resolve_feature_strategy(feature_strategy, fine_tune_ratios)
    tuned_configs = tuned_configs if tuned_configs is not None else {}

    for scenario in scenario_list:
        scenario_results = run_experiment_scenario(
            scenario,
            cache=cache,
            overrides=overrides,
            fine_tune_ratios=fine_tune_ratios,
            fine_tune_val_ratio=fine_tune_val_ratio,
            seeds=seeds,
            model_types=model_types,
            val_size=val_size,
            test_size=test_size,
            pretrain_val_ratio=pretrain_val_ratio,
            target_train_ratio=target_train_ratio,
            target_val_ratio=target_val_ratio,
            target_test_ratio=target_test_ratio,
            split_strategy=split_strategy,
            group_folds=group_folds,
            cdtrans_config=cdtrans_config,
            feature_strategy=resolved_feature_strategy,
            min_source_feature_frac=min_source_feature_frac,
            min_source_feature_std=min_source_feature_std,
            feature_zero_tolerance=feature_zero_tolerance,
            include_target_in_scaler=include_target_in_scaler,
            feature_clip_value=feature_clip_value,
            transformer_overrides=transformer_overrides,
            dl_erm_config=dl_erm_config,
            dl_erm_overrides=dl_erm_overrides,
            dl_dann_config=dl_dann_config,
            dl_dann_overrides=dl_dann_overrides,
            dl_irm_config=dl_irm_config,
            dl_irm_overrides=dl_irm_overrides,
            dl_csd_config=dl_csd_config,
            dl_csd_overrides=dl_csd_overrides,
            dl_mldg_config=dl_mldg_config,
            dl_mldg_overrides=dl_mldg_overrides,
            dl_clustering_config=dl_clustering_config,
            dl_clustering_overrides=dl_clustering_overrides,
            dl_siamese_config=dl_siamese_config,
            dl_siamese_overrides=dl_siamese_overrides,
            dl_reorder_config=dl_reorder_config,
            dl_reorder_overrides=dl_reorder_overrides,
            dl_masf_config=dl_masf_config,
            dl_masf_overrides=dl_masf_overrides,
            # tabpfn_config=tabpfn_config,
            # tabpfn_overrides=tabpfn_overrides,
            max_target_shift_std=max_target_shift_std,
            max_feature_correlation=max_feature_correlation,
            correlation_sample_rows=correlation_sample_rows,
            max_class_conditional_shift=max_class_conditional_shift,
            class_shift_max_samples=class_shift_max_samples,
            l1_feature_keep_ratio=l1_feature_keep_ratio,
            l1_max_samples=l1_max_samples,
            optuna_tune=optuna_tune,
            optuna_trials=optuna_trials,
            optuna_seed=optuna_seed,
            tuned_configs=tuned_configs,
        )
        all_results.extend(scenario_results)

    df = pd.DataFrame(all_results)
    if not df.empty:
        float_cols = df.select_dtypes(include=[np.floating]).columns
        if len(float_cols) > 0:
            df[float_cols] = df[float_cols].round(3)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    if tuned_configs_out is not None:
        tuned_configs_out.update({
            name: (asdict(cfg) if is_dataclass(cfg) else cfg)
            for name, cfg in tuned_configs.items()
        })
    return df
