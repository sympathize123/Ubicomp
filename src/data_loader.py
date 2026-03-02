import pickle
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
import os


class BenchmarkDataset:
    def __init__(self, dataset_name: str, file_path: str, min_samples: int = 100, min_class_samples: int = 10):
        self.dataset_name = dataset_name
        self.file_path = file_path
        self.min_samples = min_samples
        self.min_class_samples = min_class_samples

        self.X = None
        self.y = None
        self.users = None
        self.timestamps = None
        self.feature_names = None

        self.load_data()

    def load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        print(f"Loading {self.dataset_name} from {self.file_path}...")
        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)

        if isinstance(data, (tuple, list)) and len(data) >= 5:
            self.X = data[0]
            self.y = data[1]
            self.users = data[2]
            self.timestamps = data[4]

            if isinstance(self.X, pd.DataFrame):
                self.feature_names = list(self.X.columns)
                self.X = self.X.values
            else:
                if self.feature_names is None or len(self.feature_names) != self.X.shape[1]:
                    self.feature_names = [f"feature_{i}" for i in range(self.X.shape[1])]

            print(f"Feature Names Sample: {self.feature_names[:5]}")

            cols_to_drop = []
            for i, name in enumerate(self.feature_names):
                name_lower = str(name).lower()
                if 'timestamp' in name_lower or 'participant' in name_lower or 'label' in name_lower:
                    cols_to_drop.append(i)

            if cols_to_drop:
                print(f"Dropping {len(cols_to_drop)} non-feature columns: {[self.feature_names[i] for i in cols_to_drop[:5]]}...")
                keep_mask = np.ones(self.X.shape[1], dtype=bool)
                keep_mask[cols_to_drop] = False
                self.X = self.X[:, keep_mask]
                self.feature_names = [name for i, name in enumerate(self.feature_names) if keep_mask[i]]

            self.X = np.array(self.X, dtype=np.float32)
            self.y = np.array(self.y, dtype=np.int64)
            self.users = np.array(self.users)
            self.timestamps = np.array(self.timestamps)

            print(f"Loaded {self.X.shape[0]} samples, {self.X.shape[1]} features.")
        else:
            raise ValueError(f"Unexpected data format in {self.file_path}")

    def filter_one_month(self):
        print("Filtering usage data to first 1 month per user...")
        unique_users = np.unique(self.users)
        valid_indices = []

        for user in unique_users:
            user_mask = (self.users == user)
            user_global_indices = np.where(user_mask)[0]

            if len(user_global_indices) == 0:
                continue

            user_timestamps = self.timestamps[user_global_indices]
            start_date = user_timestamps.min()
            cutoff_date = start_date + pd.DateOffset(months=1)
            keep_mask = (user_timestamps <= cutoff_date)
            valid_indices.extend(user_global_indices[keep_mask])

        valid_indices = np.array(valid_indices)
        valid_indices.sort()

        original_count = len(self.X)
        self.X = self.X[valid_indices]
        self.y = self.y[valid_indices]
        self.users = self.users[valid_indices]
        self.timestamps = self.timestamps[valid_indices]

        print(f"Filtered data to 1 month. Samples: {original_count} -> {len(self.X)} (Removed {original_count - len(self.X)})")

    def filter_users(self):
        unique_users = np.unique(self.users)
        valid_indices = []
        dropped_users = []

        for user in unique_users:
            user_mask = (self.users == user)
            user_y = self.y[user_mask]

            if len(user_y) < self.min_samples:
                dropped_users.append(user)
                continue

            classes, counts = np.unique(user_y, return_counts=True)
            if len(classes) < 2 or np.min(counts) < self.min_class_samples:
                dropped_users.append(user)
                continue

            valid_indices.extend(np.where(user_mask)[0])

        if dropped_users:
            print(f"Dropping {len(dropped_users)} users due to insufficient data: {dropped_users}")

        valid_indices = np.array(valid_indices)
        self.X = self.X[valid_indices]
        self.y = self.y[valid_indices]
        self.users = self.users[valid_indices]
        self.timestamps = self.timestamps[valid_indices]
        print(f"Taking {len(valid_indices)} samples after filtering.")

    def normalize_features(self, train_idx, val_idx=None, test_idx=None):
        print("Normalizing features per user (using Train-only statistics)...")
        unique_users = np.unique(self.users)

        for user in unique_users:
            user_mask = (self.users == user)
            user_indices = np.where(user_mask)[0]
            user_train_indices = np.intersect1d(user_indices, train_idx)

            if len(user_train_indices) == 0:
                user_X_all = self.X[user_indices]
                mean = np.mean(user_X_all, axis=0)
                std = np.std(user_X_all, axis=0)
            else:
                user_X_train = self.X[user_train_indices]
                mean = np.mean(user_X_train, axis=0)
                std = np.std(user_X_train, axis=0)

            std[std < 1e-6] = 1.0
            self.X[user_mask] = ((self.X[user_mask] - mean) / std).astype(np.float32)

        clip_percentile = 99.9
        clip_min = 10.0
        train_vals = self.X[train_idx].reshape(-1)
        sample_size = min(1_000_000, train_vals.size)
        rng = np.random.default_rng(0)
        if train_vals.size > sample_size:
            sample_idx = rng.choice(train_vals.size, size=sample_size, replace=False)
            sample = np.abs(train_vals[sample_idx])
        else:
            sample = np.abs(train_vals)
        clip_value = float(np.percentile(sample, clip_percentile))
        if clip_value < clip_min:
            clip_value = clip_min
        print(f"Clipping normalized features to ±{clip_value:.4f} (p{clip_percentile}, min {clip_min})")
        self.X = np.clip(self.X, -clip_value, clip_value).astype(np.float32)

    def get_temporal_splits(self, train_ratio: float = 0.6, val_ratio: float = 0.2):
        train_indices = []
        val_indices = []
        test_indices = []

        unique_users = np.unique(self.users)

        for user in unique_users:
            user_global_indices = np.where(self.users == user)[0]
            user_timestamps = self.timestamps[user_global_indices]
            sorted_order = np.argsort(user_timestamps)
            sorted_indices = user_global_indices[sorted_order]

            n_samples = len(sorted_indices)
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)

            train_idx = sorted_indices[:n_train]
            val_idx = sorted_indices[n_train:n_train + n_val]
            test_idx = sorted_indices[n_train + n_val:]

            train_indices.extend(train_idx)
            val_indices.extend(val_idx)
            test_indices.extend(test_idx)

        return np.array(train_indices), np.array(val_indices), np.array(test_indices)
