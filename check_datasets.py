files = {
    'D-3_new.pkl': '/home/iclab/minseo/CHI/data/Archived/stress_binary_personal-full.pkl'
}

for name, path in files.items():
    print(f"--- Loading {name} ---")
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Assume data structure is dict with 'X', 'y', 'ids' (users) or dataframe
        if isinstance(data, dict):
            keys = data.keys()
            print(f"Keys: {keys}")
            if 'ids' in data:
                users = np.unique(data['ids'])
                print(f"Users: {len(users)}")
            elif 'participants' in data:
                users = np.unique(data['participants'])
                print(f"Users: {len(users)}")
            
            if 'X' in data:
                print(f"X shape: {data['X'].shape}")
        elif isinstance(data, tuple):
            print(f"Tuple length: {len(data)}")
            # Looking at StressDataset (src/data_loader.py), it usually loads X, y, users, etc.
            # Usually: X, y, sub, ...
            if len(data) >= 3:
                X = data[0]
                y = data[1]
                sub = data[2]
                print(f"X shape: {X.shape}")
                print(f"y shape: {y.shape}")
                print(f"Sub shape: {sub.shape}")
                print(f"Unique Users: {len(np.unique(sub))}")
        else:
            print("Data is not a dict or tuple, inspecting type...")
            print(type(data))
    except Exception as e:
        print(f"Error loading {name}: {e}")
    print("\n")
