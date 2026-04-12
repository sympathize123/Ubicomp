from __future__ import annotations

import json
import os
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


def _device_info() -> Dict[str, str]:
    if _TORCH and torch.cuda.is_available():
        return {"device_type": "cuda", "device_name": torch.cuda.get_device_name(torch.cuda.current_device())}
    return {"device_type": "cpu", "device_name": platform.processor() or "unknown"}


def _peak_gpu_mb() -> Optional[float]:
    if _TORCH and torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2)
    return None


def _peak_cpu_mb() -> Optional[float]:
    if _PSUTIL:
        return round(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, 2)
    return None


def _split_stats(y: np.ndarray, users: Optional[np.ndarray]) -> Dict:
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


def _get_probs(model, X: np.ndarray) -> np.ndarray:
    if _TORCH:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(model, nn.Module):
            model.eval()
            model.to(device)
            with torch.no_grad():
                t = torch.tensor(X, dtype=torch.float32).to(device)
                out = model.predict(t) if hasattr(model, "predict") else model(t)
                if isinstance(out, tuple):
                    out = out[0]
            return torch.softmax(out, dim=1).cpu().numpy()
    return model.predict_proba(X)


def _metrics(y: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
    preds = np.argmax(probs, axis=1)
    try:
        auroc = float(roc_auc_score(y, probs[:, 1]))
    except Exception:
        auroc = 0.5
    return {
        "accuracy":  round(float(accuracy_score(y, preds)), 6),
        "auroc":     round(auroc, 6),
        "f1":        round(float(f1_score(y, preds, average="macro", zero_division=0)), 6),
        "precision": round(float(precision_score(y, preds, average="macro", zero_division=0)), 6),
        "recall":    round(float(recall_score(y, preds, average="macro", zero_division=0)), 6),
    }


def evaluate_extended(model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    return _metrics(y, _get_probs(model, X))


@contextmanager
def _timer(store: Dict, key: str):
    t0 = time.perf_counter()
    yield
    store[key] = round(time.perf_counter() - t0, 3)


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _flat_summary(r: Dict) -> Dict:
    m = r.get("metrics", {})
    s = r.get("setting", {})
    rt = r.get("runtime", {})
    tr = r.get("training", {})
    return {
        "experiment_id":     r.get("experiment_id"),
        "benchmark_type":    r.get("benchmark_type"),
        "timestamp_utc":     r.get("timestamp_utc"),
        "model":             s.get("model"),
        "label":             s.get("label"),
        "dataset":           s.get("dataset"),
        "fold_id":           s.get("fold_id"),
        "train_datasets":    s.get("train_datasets"),
        "test_dataset":      s.get("test_dataset"),
        "setting_type":      s.get("setting_type"),
        "n_common_features": s.get("n_common_features"),
        "seed":              s.get("seed"),
        "hpo_best_auroc":    r.get("hpo", {}).get("best_value"),
        "train_auroc":       m.get("train", {}).get("auroc"),
        "val_auroc":         m.get("val", {}).get("auroc"),
        "test_auroc":        m.get("test", {}).get("auroc"),
        "test_f1":           m.get("test", {}).get("f1"),
        "test_accuracy":     m.get("test", {}).get("accuracy"),
        "test_precision":    m.get("test", {}).get("precision"),
        "test_recall":       m.get("test", {}).get("recall"),
        "best_epoch":        tr.get("best_epoch"),
        "early_stopped":     tr.get("early_stopped"),
        "hpo_wall_s":        rt.get("hpo_wall_s"),
        "train_wall_s":      rt.get("train_wall_s"),
        "total_wall_s":      rt.get("total_wall_s"),
        "peak_gpu_mb":       rt.get("peak_gpu_memory_mb"),
        "device_name":       rt.get("device_name"),
    }


class BenchmarkLogger:
    """
    Unified structured logger for cross-user and cross-dataset benchmark runs.
    One instance per fold (cross-user) or per train→test pair (cross-dataset).

    Writes:
      <output_dir>/records/<experiment_id>.json   full record
      <output_dir>/summary.jsonl                  summary
    """

    def __init__(self, output_dir: str, benchmark_type: str):
        assert benchmark_type in ("cross_user", "cross_dataset")
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._summary_path = self._dir / "summary.jsonl"
        self._type = benchmark_type
        self._t0 = time.perf_counter()
        self._wall: Dict[str, float] = {}
        self._rec: Dict[str, Any] = {
            "benchmark_type": benchmark_type,
            "experiment_id": None,
            "timestamp_utc": _utc(),
            "setting": {},
            "preprocessing": {},
            "split_stats": {},
            "hpo": {},
            "training": {},
            "metrics": {},
            "runtime": {**_device_info(), "hpo_wall_s": None, "train_wall_s": None,
                        "eval_wall_s": None, "total_wall_s": None,
                        "peak_gpu_memory_mb": None, "peak_cpu_memory_mb": None},
        }
        if _TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def set_setting(self, **kwargs):
        """
        Cross-user keys: dataset, label, model, backbone, seed, fold_id, n_folds,
                         val_ratio, hpo_trials, hpo_mode, max_epochs, patience
        Cross-dataset keys: label, model, backbone, seed, val_ratio, hpo_trials,
                            max_epochs, patience, train_datasets, test_dataset,
                            setting_type, n_common_features, common_feature_list
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        if self._type == "cross_user":
            eid = (f"{kwargs.get('model')}_{kwargs.get('dataset')}"
                   f"_{kwargs.get('label')}_fold{kwargs.get('fold_id')}"
                   f"_seed{kwargs.get('seed')}_{ts}")
        else:
            trains = "_".join(kwargs.get("train_datasets", []))
            eid = (f"{kwargs.get('model')}_{kwargs.get('label')}"
                   f"_{trains}_test{kwargs.get('test_dataset')}"
                   f"_seed{kwargs.get('seed')}_{ts}")
        self._rec["experiment_id"] = eid
        self._rec["setting"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                for k, v in kwargs.items()}

    def set_preprocessing(self, clip_value: float,
                          excluded_columns: Optional[List[str]] = None):
        self._rec["preprocessing"] = {
            "normalization": "per_user_zscore",
            "clip_method": "train_percentile_99.9",
            "clip_value": round(float(clip_value), 4),
            "excluded_columns": excluded_columns or ["timestamp", "participant", "label"],
        }

    def set_split_stats(self,
                        y_tr: np.ndarray, users_tr: Optional[np.ndarray],
                        y_va: np.ndarray, users_va: Optional[np.ndarray],
                        y_te: np.ndarray, users_te: Optional[np.ndarray]):
        self._rec["split_stats"] = {
            "train": _split_stats(y_tr, users_tr),
            "val":   _split_stats(y_va, users_va),
            "test":  _split_stats(y_te, users_te),
        }

    @contextmanager
    def time_hpo(self):
        with _timer(self._wall, "hpo"):
            yield
        self._rec["runtime"]["hpo_wall_s"] = self._wall["hpo"]

    @contextmanager
    def time_train(self):
        with _timer(self._wall, "train"):
            yield
        self._rec["runtime"]["train_wall_s"] = self._wall["train"]

    @contextmanager
    def time_eval(self):
        with _timer(self._wall, "eval"):
            yield
        self._rec["runtime"]["eval_wall_s"] = self._wall["eval"]

    def record_hpo(self, study, best_params: Dict):
        trials = []
        try:
            for t in study.trials:
                trials.append({
                    "trial_idx": t.number,
                    "params": {k: (v.item() if hasattr(v, "item") else v) for k, v in t.params.items()},
                    "value": float(t.value) if t.value is not None else None,
                    "duration_s": t.duration.total_seconds() if t.duration else None,
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
        }

    def record_hpo_no_study(self, best_params: Dict):
        self._rec["hpo"] = {
            "method": "none",
            "n_trials": 0,
            "metric": "val_auroc",
            "best_trial_idx": None,
            "best_value": None,
            "best_params": best_params,
            "all_trials": [],
        }

    def record_training(self, info: Optional[Dict] = None):
        info = info or {}
        self._rec["training"] = {
            "optimizer": info.get("optimizer", "adam"),
            "best_epoch": info.get("best_epoch"),
            "early_stopped": info.get("early_stopped", False),
            "early_stop_epoch": info.get("early_stop_epoch"),
            "model_selection_metric": "val_auroc",
            "epoch_history": info.get("epoch_history", []),
        }

    def record_metrics(self, model,
                       X_tr: np.ndarray, y_tr: np.ndarray,
                       X_va: np.ndarray, y_va: np.ndarray,
                       X_te: np.ndarray, y_te: np.ndarray):
        self._rec["metrics"] = {
            "train": _metrics(y_tr, _get_probs(model, X_tr)),
            "val":   _metrics(y_va, _get_probs(model, X_va)),
            "test":  _metrics(y_te, _get_probs(model, X_te)),
        }

    def finalize(self) -> Dict:
        self._rec["runtime"]["total_wall_s"] = round(time.perf_counter() - self._t0, 3)
        self._rec["runtime"]["peak_gpu_memory_mb"] = _peak_gpu_mb()
        self._rec["runtime"]["peak_cpu_memory_mb"] = _peak_cpu_mb()

        eid = self._rec["experiment_id"] or "unknown"
        record_path = self._dir / f"{eid}.json"
        with open(record_path, "w") as f:
            json.dump(self._rec, f, indent=2, default=_json_default)

        with open(self._summary_path, "a") as f:
            f.write(json.dumps(_flat_summary(self._rec), default=_json_default) + "\n")

        return self._rec
