from __future__ import annotations

import io
import json
import os
import pickle
import platform
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cpu_model() -> str:
    cpu_name = platform.processor() or platform.uname().processor
    if cpu_name:
        return cpu_name
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                parts = line.split(":", 1)
                if len(parts) == 2 and parts[1].strip():
                    return parts[1].strip()
    return "unknown"


def _hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "device_type": "cpu",
        "device_name": _cpu_model(),
        "gpu_model": None,
        "gpu_models": [],
        "gpu_count": 0,
        "gpu_total_vram_gb": None,
        "gpu_vram_per_device_gb": [],
        "cpu_model": _cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False) if _PSUTIL else None,
        "ram_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2) if _PSUTIL else None,
        "platform": platform.platform(),
        "hostname": platform.node(),
    }
    if _TORCH and torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_models = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        vram_per_device = [
            round(torch.cuda.get_device_properties(i).total_memory / 1024 ** 3, 2)
            for i in range(gpu_count)
        ]
        info.update({
            "device_type": "cuda",
            "device_name": gpu_models[0] if gpu_models else "cuda",
            "gpu_model": gpu_models[0] if gpu_models else None,
            "gpu_models": gpu_models,
            "gpu_count": gpu_count,
            "gpu_total_vram_gb": round(sum(vram_per_device), 2),
            "gpu_vram_per_device_gb": vram_per_device,
        })
    return info


def _peak_gpu_mb() -> Optional[float]:
    if _TORCH and torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2)
    return None


def _peak_cpu_mb() -> Optional[float]:
    if _PSUTIL:
        return round(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, 2)
    return None


def _gpu_hours(seconds: Optional[float], hardware: Dict[str, Any]) -> Optional[float]:
    if seconds is None or hardware.get("device_type") != "cuda":
        return None
    return round(float(seconds) * max(1, int(hardware.get("gpu_count") or 0)) / 3600.0, 6)


def _split_stats(y: np.ndarray, users: Optional[np.ndarray]) -> Dict[str, Any]:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    n = len(y)
    return {
        "n_samples": n,
        "n_users": int(len(np.unique(users))) if users is not None else None,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "positive_ratio": round(n_pos / max(1, n), 6),
        "class_balance": round(min(n_pos, n_neg) / max(1, max(n_pos, n_neg)), 6),
    }


def _sync_cuda():
    if _TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


def _resolve_torch_module(model) -> Optional["nn.Module"]:
    if not _TORCH:
        return None
    queue = [model]
    seen = set()
    while queue:
        current = queue.pop(0)
        if current is None:
            continue
        obj_id = id(current)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        if isinstance(current, nn.Module):
            return current
        for attr in ("model", "trainer", "network", "source_model", "target_encoder"):
            queue.append(getattr(current, attr, None))
        trainer = getattr(current, "trainer", None)
        if trainer is not None:
            queue.append(getattr(trainer, "model", None))
        inner_model = getattr(current, "model", None)
        if inner_model is not None:
            queue.append(getattr(inner_model, "network", None))
    return None


def _framework_name(model) -> str:
    if _TORCH and isinstance(model, nn.Module):
        return "pytorch"
    name = model.__class__.__name__.lower()
    if "xgboost" in name:
        return "xgboost"
    if "lightgbm" in name or "lgb" in name:
        return "lightgbm"
    if "tabnet" in name:
        return "pytorch_tabnet"
    if "widedeep" in name or "saint" in name or "transformer" in name:
        return "pytorch_widedeep"
    if "deepctr" in name or "autoint" in name or "dcn" in name:
        return "deepctr_torch"
    return model.__class__.__module__.split(".")[0]


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs)
    if probs.ndim == 1:
        probs = probs.reshape(-1, 1)
    if probs.shape[1] == 1:
        probs = np.hstack([1.0 - probs, probs])
    return probs


def _predict_probs(model, X: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
    module = _resolve_torch_module(model)
    if _TORCH and isinstance(model, nn.Module):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.eval()
        model.to(device)
        batch_size = int(batch_size or getattr(model, "batch_size", 256) or 256)
        dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=0,
        )
        outputs = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(device)
                logits = model.predict(xb) if hasattr(model, "predict") else model(xb)
                if isinstance(logits, tuple):
                    logits = logits[0]
                outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(outputs, axis=0)
    if hasattr(model, "predict_proba"):
        return _normalize_probs(model.predict_proba(X))
    if module is not None and hasattr(module, "predict_proba"):
        return _normalize_probs(module.predict_proba(X))
    raise TypeError(f"Cannot obtain probabilities from model type {type(model)!r}")


def evaluate_extended(model, X: np.ndarray, y: np.ndarray, batch_size: Optional[int] = None) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    probs = _predict_probs(model, X, batch_size=batch_size)
    preds = np.argmax(probs, axis=1)
    try:
        auroc = float(roc_auc_score(y, probs[:, 1]))
    except Exception:
        auroc = 0.5
    return {
        "accuracy": round(float(accuracy_score(y, preds)), 6),
        "auroc": round(auroc, 6),
        "f1": round(float(f1_score(y, preds, average="macro", zero_division=0)), 6),
        "precision": round(float(precision_score(y, preds, average="macro", zero_division=0)), 6),
        "recall": round(float(recall_score(y, preds, average="macro", zero_division=0)), 6),
    }


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _count_parameters(model) -> Dict[str, Optional[int]]:
    module = _resolve_torch_module(model)
    if module is None:
        return {"parameter_count": None, "trainable_parameter_count": None}
    return {
        "parameter_count": int(sum(p.numel() for p in module.parameters())),
        "trainable_parameter_count": int(sum(p.numel() for p in module.parameters() if p.requires_grad)),
    }


def _estimate_artifact_size_mb(model) -> Optional[float]:
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(getattr(model, "trainer", None), "model", None),
        _resolve_torch_module(model),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            buffer = io.BytesIO()
            if _TORCH and isinstance(candidate, nn.Module):
                torch.save(candidate.state_dict(), buffer)
            else:
                pickle.dump(candidate, buffer, protocol=pickle.HIGHEST_PROTOCOL)
            return round(len(buffer.getbuffer()) / 1024 ** 2, 6)
        except Exception:
            continue
    return None


def _profile_flops_macs(model, input_dim: Optional[int]) -> Dict[str, Any]:
    result = {
        "flops": None,
        "macs": None,
        "profiler": None,
        "available": False,
        "note": None,
    }
    module = _resolve_torch_module(model)
    if module is None or input_dim is None:
        result["note"] = "torch_module_or_input_dim_unavailable"
        return result
    try:
        from thop import profile
    except Exception:
        result["note"] = "thop_not_installed"
        return result
    try:
        device = next(module.parameters()).device if any(True for _ in module.parameters()) else torch.device("cpu")
        dummy = torch.randn(1, int(input_dim), device=device)
        module.eval()
        macs, _ = profile(module, inputs=(dummy,), verbose=False)
        result.update({
            "flops": int(2 * macs) if macs is not None else None,
            "macs": int(macs) if macs is not None else None,
            "profiler": "thop",
            "available": True,
            "note": None,
        })
    except Exception as exc:
        result["note"] = f"profile_failed:{type(exc).__name__}"
    return result


def _benchmark_inference(model, X: np.ndarray, batch_size: Optional[int] = None) -> Dict[str, Any]:
    if X is None or len(X) == 0:
        return {
            "benchmark_split": None,
            "batch_size": batch_size,
            "benchmark_samples": 0,
            "per_batch_latency_ms": None,
            "per_sample_latency_ms": None,
            "throughput_samples_per_s": None,
            "repeats": 0,
            "note": "empty_input",
        }

    batch_size = int(batch_size or getattr(model, "batch_size", 256) or 256)
    batch_size = max(1, min(batch_size, len(X)))
    benchmark_samples = min(len(X), max(batch_size * 8, 256))
    X_ref = np.asarray(X[:benchmark_samples], dtype=np.float32)
    X_batch = np.asarray(X_ref[:batch_size], dtype=np.float32)
    repeats = 5

    try:
        for _ in range(1):
            _sync_cuda()
            _predict_probs(model, X_batch, batch_size=batch_size)
            _sync_cuda()

        batch_times = []
        for _ in range(repeats):
            _sync_cuda()
            t0 = time.perf_counter()
            _predict_probs(model, X_batch, batch_size=batch_size)
            _sync_cuda()
            batch_times.append(time.perf_counter() - t0)

        dataset_times = []
        for _ in range(repeats):
            _sync_cuda()
            t0 = time.perf_counter()
            _predict_probs(model, X_ref, batch_size=batch_size)
            _sync_cuda()
            dataset_times.append(time.perf_counter() - t0)

        mean_batch = float(np.mean(batch_times))
        mean_dataset = float(np.mean(dataset_times))
        return {
            "batch_size": batch_size,
            "benchmark_samples": int(len(X_ref)),
            "per_batch_latency_ms": round(mean_batch * 1000.0, 6),
            "per_sample_latency_ms": round((mean_batch / len(X_batch)) * 1000.0, 6),
            "throughput_samples_per_s": round(len(X_ref) / max(mean_dataset, 1e-12), 6),
            "repeats": repeats,
            "note": None,
        }
    except Exception as exc:
        return {
            "batch_size": batch_size,
            "benchmark_samples": int(len(X_ref)),
            "per_batch_latency_ms": None,
            "per_sample_latency_ms": None,
            "throughput_samples_per_s": None,
            "repeats": repeats,
            "note": f"benchmark_failed:{type(exc).__name__}",
        }


def _flat_summary(r: Dict[str, Any]) -> Dict[str, Any]:
    m = r.get("metrics", {})
    s = r.get("setting", {})
    rt = r.get("runtime", {})
    tr = r.get("training", {})
    hw = r.get("hardware", {})
    model_stats = r.get("model_stats", {})
    infer = r.get("inference_benchmark", {}).get("test", {})
    budget = r.get("compute_budget", {})
    sustainability = r.get("sustainability", {})
    return {
        "experiment_id": r.get("experiment_id"),
        "benchmark_type": r.get("benchmark_type"),
        "timestamp_utc": r.get("timestamp_utc"),
        "model": s.get("model"),
        "label": s.get("label"),
        "dataset": s.get("dataset"),
        "fold_id": s.get("fold_id"),
        "train_datasets": s.get("train_datasets"),
        "test_dataset": s.get("test_dataset"),
        "setting_type": s.get("setting_type"),
        "n_common_features": s.get("n_common_features"),
        "seed": s.get("seed"),
        "seed_count": budget.get("seed_count"),
        "hpo_best_auroc": r.get("hpo", {}).get("best_value"),
        "train_auroc": m.get("train", {}).get("auroc"),
        "val_auroc": m.get("val", {}).get("auroc"),
        "test_auroc": m.get("test", {}).get("auroc"),
        "test_f1": m.get("test", {}).get("f1"),
        "test_accuracy": m.get("test", {}).get("accuracy"),
        "test_precision": m.get("test", {}).get("precision"),
        "test_recall": m.get("test", {}).get("recall"),
        "best_epoch": tr.get("best_epoch"),
        "early_stopped": tr.get("early_stopped"),
        "hpo_wall_s": rt.get("hpo_wall_s"),
        "train_wall_s": rt.get("train_wall_s"),
        "eval_wall_s": rt.get("eval_wall_s"),
        "total_wall_s": rt.get("total_wall_s"),
        "hpo_gpu_hours": rt.get("hpo_gpu_hours"),
        "train_gpu_hours": rt.get("train_gpu_hours"),
        "eval_gpu_hours": rt.get("eval_gpu_hours"),
        "total_gpu_hours": rt.get("total_gpu_hours"),
        "peak_gpu_mb": rt.get("peak_gpu_memory_mb"),
        "peak_cpu_mb": rt.get("peak_cpu_memory_mb"),
        "device_name": hw.get("device_name"),
        "gpu_model": hw.get("gpu_model"),
        "gpu_count": hw.get("gpu_count"),
        "gpu_total_vram_gb": hw.get("gpu_total_vram_gb"),
        "cpu_model": hw.get("cpu_model"),
        "ram_gb": hw.get("ram_gb"),
        "parameter_count": model_stats.get("parameter_count"),
        "trainable_parameter_count": model_stats.get("trainable_parameter_count"),
        "artifact_size_mb": model_stats.get("artifact_size_mb"),
        "inference_batch_latency_ms": infer.get("per_batch_latency_ms"),
        "inference_per_sample_latency_ms": infer.get("per_sample_latency_ms"),
        "inference_throughput_sps": infer.get("throughput_samples_per_s"),
        "flops": model_stats.get("flops"),
        "macs": model_stats.get("macs"),
        "energy_kwh": sustainability.get("energy_kwh"),
        "carbon_kg_co2eq": sustainability.get("carbon_kg_co2eq"),
    }


class BenchmarkLogger:
    """
    Unified structured logger for cross-user and cross-dataset benchmark runs.
    One instance per fold (cross-user) or per train->test pair (cross-dataset).
    """

    def __init__(self, output_dir: str, benchmark_type: str):
        assert benchmark_type in ("cross_user", "cross_dataset")
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._summary_path = self._dir.parent / "summary.jsonl"
        self._type = benchmark_type
        self._t0 = time.perf_counter()
        self._wall: Dict[str, float] = {}
        hardware = _hardware_info()
        self._rec: Dict[str, Any] = {
            "benchmark_type": benchmark_type,
            "experiment_id": None,
            "timestamp_utc": _utc(),
            "setting": {},
            "hardware": hardware,
            "preprocessing": {},
            "split_stats": {},
            "compute_budget": {},
            "comparison_policy": {},
            "hpo": {},
            "training": {},
            "model_stats": {},
            "metrics": {},
            "inference_benchmark": {},
            "runtime": {
                "hpo_wall_s": None,
                "train_wall_s": None,
                "eval_wall_s": None,
                "total_wall_s": None,
                "hpo_gpu_hours": None,
                "train_gpu_hours": None,
                "eval_gpu_hours": None,
                "total_gpu_hours": None,
                "peak_gpu_memory_mb": None,
                "peak_cpu_memory_mb": None,
            },
            "sustainability": {
                "energy_kwh": None,
                "carbon_kg_co2eq": None,
                "tracker": None,
                "note": "energy_and_carbon_tracking_not_enabled",
            },
        }
        if _TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def set_setting(self, **kwargs):
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        if self._type == "cross_user":
            eid = (
                f"{kwargs.get('model')}_{kwargs.get('dataset')}"
                f"_{kwargs.get('label')}_fold{kwargs.get('fold_id')}"
                f"_seed{kwargs.get('seed')}_{ts}"
            )
        else:
            trains = "_".join(kwargs.get("train_datasets", []))
            eid = (
                f"{kwargs.get('model')}_{kwargs.get('label')}"
                f"_{trains}_test{kwargs.get('test_dataset')}"
                f"_seed{kwargs.get('seed')}_{ts}"
            )
        self._rec["experiment_id"] = eid
        self._rec["setting"] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in kwargs.items()
        }

    def set_preprocessing(self, clip_value: float, excluded_columns: Optional[List[str]] = None):
        self._rec["preprocessing"] = {
            "normalization": "per_user_zscore",
            "clip_method": "train_percentile_99.9",
            "clip_value": round(float(clip_value), 4),
            "excluded_columns": excluded_columns or ["timestamp", "participant", "label"],
        }

    def set_split_stats(
        self,
        y_tr: np.ndarray,
        users_tr: Optional[np.ndarray],
        y_va: np.ndarray,
        users_va: Optional[np.ndarray],
        y_te: np.ndarray,
        users_te: Optional[np.ndarray],
    ):
        self._rec["split_stats"] = {
            "train": _split_stats(y_tr, users_tr),
            "val": _split_stats(y_va, users_va),
            "test": _split_stats(y_te, users_te),
        }

    def record_policy(
        self,
        *,
        seeds: List[int],
        planned_hpo_trials: int,
        hpo_mode: str,
        max_epochs: int,
        default_batch_size: int,
        patience: int,
        early_stopping_metric: str = "val_auroc",
        batch_size_policy: str = "selected_from_hparams_or_cli_default",
    ):
        self._rec["compute_budget"] = {
            "planned_hpo_trials": int(planned_hpo_trials),
            "completed_hpo_trials": None,
            "hpo_mode": hpo_mode,
            "max_epochs_per_run": int(max_epochs),
            "planned_hpo_epoch_upper_bound": int(planned_hpo_trials) * int(max_epochs),
            "default_batch_size": int(default_batch_size),
            "selected_batch_size": None,
            "patience": int(patience),
            "seed_count": len(seeds),
            "seeds": [int(s) for s in seeds],
        }
        self._rec["comparison_policy"] = {
            "hardware_shared_across_baselines": True,
            "hpo_budget_shared_across_baselines": True,
            "early_stopping_policy": {
                "enabled": bool(patience and patience > 0),
                "patience": int(patience),
                "metric": early_stopping_metric,
            },
            "max_epoch_policy": {"max_epochs": int(max_epochs)},
            "batch_size_policy": batch_size_policy,
        }

    @contextmanager
    def time_hpo(self):
        t0 = time.perf_counter()
        yield
        self._wall["hpo"] = round(time.perf_counter() - t0, 3)
        self._rec["runtime"]["hpo_wall_s"] = self._wall["hpo"]

    @contextmanager
    def time_train(self):
        t0 = time.perf_counter()
        yield
        self._wall["train"] = round(time.perf_counter() - t0, 3)
        self._rec["runtime"]["train_wall_s"] = self._wall["train"]

    @contextmanager
    def time_eval(self):
        t0 = time.perf_counter()
        yield
        self._wall["eval"] = round(time.perf_counter() - t0, 3)
        self._rec["runtime"]["eval_wall_s"] = self._wall["eval"]

    def record_hpo(self, study, best_params: Dict[str, Any]):
        trials = []
        total_trial_runtime_s = 0.0
        try:
            for t in study.trials:
                duration_s = t.duration.total_seconds() if t.duration else None
                if duration_s is not None:
                    total_trial_runtime_s += duration_s
                trials.append({
                    "trial_idx": t.number,
                    "params": {k: (v.item() if hasattr(v, "item") else v) for k, v in t.params.items()},
                    "resolved_params": dict(t.user_attrs.get("resolved_hparams", {})) or None,
                    "value": float(t.value) if t.value is not None else None,
                    "duration_s": duration_s,
                    "state": str(t.state),
                })
            best_idx = study.best_trial.number
            best_val = float(study.best_value)
        except Exception:
            best_idx, best_val = 0, float("nan")
        self._rec["hpo"] = {
            "method": "optuna_tpe",
            "n_trials": len(trials),
            "metric": "val_auroc",
            "best_trial_idx": best_idx,
            "best_value": best_val,
            "best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in best_params.items()},
            "all_trials": trials,
            "total_trial_runtime_s": round(total_trial_runtime_s, 6),
        }
        self._rec["compute_budget"]["completed_hpo_trials"] = len(trials)
        if self._rec["runtime"]["hpo_wall_s"] is None:
            self._rec["runtime"]["hpo_wall_s"] = round(total_trial_runtime_s, 3)

    def record_hpo_no_study(self, best_params: Dict[str, Any]):
        self._rec["hpo"] = {
            "method": "none",
            "n_trials": 0,
            "metric": "val_auroc",
            "best_trial_idx": None,
            "best_value": None,
            "best_params": best_params,
            "all_trials": [],
            "total_trial_runtime_s": 0.0,
        }
        self._rec["compute_budget"]["completed_hpo_trials"] = 0
        if self._rec["runtime"]["hpo_wall_s"] is None:
            self._rec["runtime"]["hpo_wall_s"] = 0.0

    def record_training(self, model_or_info: Optional[Any] = None):
        info = {}
        if isinstance(model_or_info, dict):
            info = dict(model_or_info)
        elif model_or_info is not None:
            info = dict(getattr(model_or_info, "_training_info", {}))

        self._rec["training"] = {
            "optimizer": info.get("optimizer"),
            "best_epoch": info.get("best_epoch"),
            "early_stopped": info.get("early_stopped", False),
            "early_stop_epoch": info.get("early_stop_epoch"),
            "epochs_ran": info.get("epochs_ran"),
            "max_epochs": info.get("max_epochs"),
            "batch_size": info.get("selected_batch_size", info.get("batch_size")),
            "lr": info.get("selected_lr", info.get("lr")),
            "weight_decay": info.get("weight_decay"),
            "patience": info.get("patience"),
            "model_selection_metric": info.get("model_selection_metric", "val_auroc"),
            "best_metric_value": info.get("best_metric_value"),
            "epoch_history": info.get("epoch_history", []),
            "phases": info.get("phases"),
            "architecture_params": info.get("architecture_params"),
        }
        if self._rec["compute_budget"]:
            self._rec["compute_budget"]["selected_batch_size"] = self._rec["training"].get("batch_size")
            self._rec["compute_budget"]["selected_lr"] = self._rec["training"].get("lr")

    def record_model_stats(self, model, input_dim: Optional[int] = None):
        params = _count_parameters(model)
        flops = _profile_flops_macs(model, input_dim=input_dim)
        self._rec["model_stats"] = {
            "class_name": model.__class__.__name__,
            "framework": _framework_name(model),
            "parameter_count": params["parameter_count"],
            "trainable_parameter_count": params["trainable_parameter_count"],
            "artifact_size_mb": _estimate_artifact_size_mb(model),
            "input_dim": input_dim,
            "flops": flops["flops"],
            "macs": flops["macs"],
            "flop_profiler": flops["profiler"],
            "flop_profile_note": flops["note"],
        }

    def record_metrics(self, train_m: Dict[str, float], val_m: Dict[str, float], test_m: Dict[str, float]):
        self._rec["metrics"] = {"train": train_m, "val": val_m, "test": test_m}

    def record_inference_benchmark(self, model, X: np.ndarray, batch_size: Optional[int] = None, split_name: str = "test"):
        info = _benchmark_inference(model, X, batch_size=batch_size)
        info["benchmark_split"] = split_name
        self._rec["inference_benchmark"][split_name] = info

    def finalize(self) -> Dict[str, Any]:
        self._rec["runtime"]["total_wall_s"] = round(time.perf_counter() - self._t0, 3)
        self._rec["runtime"]["peak_gpu_memory_mb"] = _peak_gpu_mb()
        self._rec["runtime"]["peak_cpu_memory_mb"] = _peak_cpu_mb()
        hardware = self._rec["hardware"]
        self._rec["runtime"]["hpo_gpu_hours"] = _gpu_hours(self._rec["runtime"]["hpo_wall_s"], hardware)
        self._rec["runtime"]["train_gpu_hours"] = _gpu_hours(self._rec["runtime"]["train_wall_s"], hardware)
        self._rec["runtime"]["eval_gpu_hours"] = _gpu_hours(self._rec["runtime"]["eval_wall_s"], hardware)
        self._rec["runtime"]["total_gpu_hours"] = _gpu_hours(self._rec["runtime"]["total_wall_s"], hardware)
        eid = self._rec["experiment_id"] or "unknown"
        with open(self._dir / f"{eid}.json", "w") as f:
            json.dump(self._rec, f, indent=2, default=_json_default)
        with open(self._summary_path, "a") as f:
            f.write(json.dumps(_flat_summary(self._rec), default=_json_default) + "\n")
        return self._rec
