import pickle
import numpy as np
import pandas as pd
import os

base_path = '/home/iclab/minseo/CHI/data/Archived'
files = {
    'D1_full': 'stress_binary_personal-full.pkl', # Candidate for D-1 Full
    'D1_fixed_full': 'features_stress_fixed-full_D#1.pkl', # Another candidate
    'D2_full': 'stress_binary_personal-full_D#2.pkl',
    'D3_full': 'stress_binary_personal-full_D#3.pkl'
}

for name, filename in files.items():
    filepath = os.path.join(base_path, filename)
    print(f"\n--- Checking {name} ({filepath}) ---")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Assuming list/tuple structure like before
            if isinstance(data, (list, tuple)):
                print(f"Structure: Tuple/List of length {len(data)}")
                if len(data) >= 1:
                     X = data[0]
                     if hasattr(X, 'shape'):
                         print(f"X shape: {X.shape}")
                     if len(data) >= 3:
                         users = data[2]
                         print(f"Users Sample: {np.unique(users)[:5]}")
                         print(f"Total Users: {len(np.unique(users))}")
            elif isinstance(data, pd.DataFrame):
                print(f"Structure: DataFrame {data.shape}")
            else:
                 print(f"Structure: {type(data)}")

        except Exception as e:
            print(f"Error loading: {e}")
    else:
        print("File not found.")
