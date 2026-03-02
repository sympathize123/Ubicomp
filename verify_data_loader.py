import numpy as np
import pandas as pd
from src.data_loader import BenchmarkDataset

def verify_loader():
    dataset_path = "/home/iclab/minseo/CHI/data/Archived/stress_binary_personal-full.pkl"
    print(f"Testing with {dataset_path}")
    
    ds = BenchmarkDataset("D-1", dataset_path)
    
    # Verify Normalization
    print("\nVerifying Normalization per user...")
    users = np.unique(ds.users)
    for user in users[:5]: # Check first 5 users
        mask = (ds.users == user)
        user_X = ds.X[mask]
        means = np.mean(user_X, axis=0)
        stds = np.std(user_X, axis=0)
        
        # Check if mean close to 0 and std close to 1 (where std was not 0)
        mean_check = np.allclose(means, 0, atol=1e-4)
        
        # std is 1 unless it was 0 initially (constant feature). 
        # But if we divide by 1.0 (when std=0), the new std is still 0.
        # So std should be 1 or 0.
        std_check = np.all(np.isclose(stds, 1, atol=1e-4) | np.isclose(stds, 0, atol=1e-4))
        
        print(f"User {user}: Mean~0: {mean_check}, Std~1 or 0: {std_check}")
        if not mean_check:
             print(f"  Max Mean Abs: {np.max(np.abs(means))}")
        if not std_check:
             print(f"  Max Std deviation from 1: {np.max(np.abs(stds - 1))}")

    # Verify Splits
    print("\nVerifying Splits...")
    train_idx, val_idx, test_idx = ds.get_temporal_splits()
    
    # Check disjoint
    intersect_tv = np.intersect1d(train_idx, val_idx)
    intersect_vt = np.intersect1d(val_idx, test_idx)
    intersect_tt = np.intersect1d(train_idx, test_idx)
    print(f"Intersections: T-V: {len(intersect_tv)}, V-T: {len(intersect_vt)}, T-T: {len(intersect_tt)}")
    
    # Check temporal order per user
    print("Checking temporal order...")
    
    split_correct = True
    for user in users:
        # Get global indices for this user
        u_global_indices = np.where(ds.users == user)[0]
        
        # Identify which of these global indices are in train/val/test
        # Intersection of user indices and split indices
        u_train_idx = np.intersect1d(u_global_indices, train_idx)
        u_val_idx = np.intersect1d(u_global_indices, val_idx)
        u_test_idx = np.intersect1d(u_global_indices, test_idx)
        
        # Check timestamps
        # Use simple max/min check. 
        # Note: if user has very few samples, skips might occur.
        
        if len(u_train_idx) > 0 and len(u_val_idx) > 0:
            if np.max(ds.timestamps[u_train_idx]) > np.min(ds.timestamps[u_val_idx]):
                print(f"User {user} Train overlaps/after Val")
                split_correct = False
        
        if len(u_val_idx) > 0 and len(u_test_idx) > 0:
             if np.max(ds.timestamps[u_val_idx]) > np.min(ds.timestamps[u_test_idx]):
                print(f"User {user} Val overlaps/after Test")
                split_correct = False
                
    print(f"Temporal Split Correct: {split_correct}")

if __name__ == "__main__":
    verify_loader()
