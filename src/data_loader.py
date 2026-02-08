import pickle
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
import os

class StressDataset:
    def __init__(self, dataset_name: str, file_path: str, min_samples: int = 100, min_class_samples: int = 10):
        """
        Initialize the StressDataset.

        Args:
            dataset_name (str): Name of the dataset (e.g., 'D-1', 'D-2', 'D-3').
            file_path (str): Path to the pickle file.
            min_samples (int): Minimum number of samples required for a user to be included.
            min_class_samples (int): Minimum number of samples per class required for a user.
        """
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
        """Loads data from the pickle file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        print(f"Loading {self.dataset_name} from {self.file_path}...")
        with open(self.file_path, 'rb') as f:
            data = pickle.load(f)
            
        # Unpack based on expected structure (Tuple of 5)
        if isinstance(data, (tuple, list)) and len(data) >= 5:
            self.X = data[0]
            self.y = data[1]
            self.users = data[2]
            # data[3] is likely time offset, data[4] is timestamps (datetimes) based on EDA notebook
            # or maybe data[3] is timestamps?
            # Let's inspect types to be safe or rely on EDA notebook:
            # df_X_1, y_1, groups_1, t_1, datetimes_1 = pd.read_pickle(D_1)
            self.timestamps = data[4] 
            
            # Feature names extraction
            if isinstance(self.X, pd.DataFrame):
                 self.feature_names = list(self.X.columns)
                 self.X = self.X.values
            else:
                 # If no feature names provided, generate generic ones
                 # But self.feature_names is loaded from data[4] usually!
                 if self.feature_names is None or len(self.feature_names) != self.X.shape[1]:
                     self.feature_names = [f"feature_{i}" for i in range(self.X.shape[1])]
            
            print(f"Feature Names Sample: {self.feature_names[:5]}")
            
            # Identify columns to drop (timestamps, meta)
            # Heuristic: drop 'timestamp', 'time', 'participant', 'label', 'stress'
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
            
            # Ensure arrays
            
            # Ensure arrays
            self.X = np.array(self.X, dtype=np.float32)
            self.y = np.array(self.y, dtype=np.int64)
            self.users = np.array(self.users)
            self.timestamps = np.array(self.timestamps) # Ensure numpy array
            
            print(f"Loaded {self.X.shape[0]} samples, {self.X.shape[1]} features.")
        else:
            raise ValueError(f"Unexpected data format in {self.file_path}")

        self.filter_users()
        self.normalize_features()

    def filter_users(self):
        """Filters out users with insufficient data."""
        unique_users = np.unique(self.users)
        valid_indices = []
        dropped_users = []

        for user in unique_users:
            user_mask = (self.users == user)
            user_y = self.y[user_mask]
            
            if len(user_y) < self.min_samples:
                dropped_users.append(user)
                continue
                
            # Check class balance
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

    def normalize_features(self):
        """Performs Z-score normalization per user."""
        print("Normalizing features per user...")
        unique_users = np.unique(self.users)
        
        for user in unique_users:
            user_mask = (self.users == user)
            # Ensure user_mask is boolean array matching X length
            
            user_X = self.X[user_mask]
            
            mean = np.mean(user_X, axis=0)
            std = np.std(user_X, axis=0)
            
            # Avoid division by zero and near-zero std
            std[std < 1e-6] = 1.0
            
            normalized = (user_X - mean) / std
            self.X[user_mask] = normalized.astype(np.float32)
            
            # Debug for first user
            if user == unique_users[0]:
                print(f"User {user}:")
                print(f"  Original Mean (slice): {np.mean(user_X, axis=0)[:5]}")
                print(f"  New Mean (slice): {np.mean(normalized, axis=0)[:5]}")
                print(f"  Stored Mean (in X): {np.mean(self.X[user_mask], axis=0)[:5]}")

    def get_temporal_splits(self, train_ratio: float = 0.6, val_ratio: float = 0.2):
        """
        Generates indices for train, val, and test splits based on temporal order per user.
        
        Args:
            train_ratio (float): Proportion of data for training.
            val_ratio (float): Proportion of data for validation.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Train, Val, Test indices.
        """
        train_indices = []
        val_indices = []
        test_indices = []
        
        unique_users = np.unique(self.users)
        
        for user in unique_users:
            # Get indices for this user
            user_global_indices = np.where(self.users == user)[0]
            
            # Sort by timestamp
            user_timestamps = self.timestamps[user_global_indices]
            sorted_order = np.argsort(user_timestamps)
            sorted_indices = user_global_indices[sorted_order]
            
            n_samples = len(sorted_indices)
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            
            # Ensure at least 1 sample in each if possible (though filtering should handle this)
            # Standard slicing
            train_idx = sorted_indices[:n_train]
            val_idx = sorted_indices[n_train:n_train + n_val]
            test_idx = sorted_indices[n_train + n_val:]
            
            train_indices.extend(train_idx)
            val_indices.extend(val_idx)
            test_indices.extend(test_idx)
            
        return np.array(train_indices), np.array(val_indices), np.array(test_indices)

if __name__ == "__main__":
    # Test with D-1
    dataset_path = "/home/iclab/minseo/CHI/data/Archived/stress_binary_personal-full.pkl"
    if os.path.exists(dataset_path):
        ds = StressDataset("D-1", dataset_path)
        train_idx, val_idx, test_idx = ds.get_temporal_splits()
        print(f"Train/Val/Test sizes: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    else:
        print("D-1 file not found for testing.")
