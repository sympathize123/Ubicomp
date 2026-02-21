import pickle
import numpy as np

# Method 1: Ubicomp implementation
def ubicomp_norm(X, users):
    X_new = X.copy()
    unique_users = np.unique(users)
    for u in unique_users:
        mask = (users == u)
        user_X = X_new[mask]
        mean = np.mean(user_X, axis=0)
        std = np.std(user_X, axis=0)
        std[std < 1e-6] = 1.0
        X_new[mask] = (user_X - mean) / std
    return X_new

# Method 2: CrossUserDataset implementation (using train_only_stats=True as seen in temporal)
def crossuser_norm_temporal(X, users, timestamps, test_size=0.2):
    X_new = X.copy()
    unique_users = np.unique(users)
    for u in unique_users:
        mask = (users == u)
        user_X = X_new[mask]
        user_times = timestamps[mask]
        
        # Temporal split inside user
        order = np.argsort(user_times)
        split_idx = int(len(order) * (1.0 - test_size))
        train_idx_user = order[:split_idx]
        
        # Calculate stats ONLY on train set!
        train_X = user_X[train_idx_user]
        mean = np.mean(train_X, axis=0)
        std = np.std(train_X, axis=0)
        std[std == 0] = 1.0
        
        X_new[mask] = (user_X - mean) / std
    return X_new

path = '/home/iclab/minseo/Ubicomp/data/valence_personal-full_D#2.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)

X, users, timestamps = data[0], data[2], data[4]
X = np.array(X, dtype=float)
users = np.array(users)
timestamps = np.array(timestamps)

X_ubi = ubicomp_norm(X, users)
X_cross = crossuser_norm_temporal(X, users, timestamps)

diff = np.abs(X_ubi - X_cross).mean()
print(f"Average difference in normalization outputs: {diff}")
if diff > 1e-5:
    print("WARNING: The normalization methods are mathematically different!")
