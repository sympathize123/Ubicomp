import pickle
import numpy as np

path = '/home/iclab/minseo/Ubicomp/data/valence_personal-full_D#2.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)

y, users, t2 = data[1], data[2], data[4]
y = np.array(y)
users = np.array(users)
t2 = np.array(t2)

print('Total Samples:', len(y))

u = 'P001'
mask = (users == u)
y_u = y[mask]
t_u = t2[mask]
idx = np.argsort(t_u)
y_u = y_u[idx]

n = len(y_u)
n_train = int(n * 0.6)
n_val = int(n * 0.2)
train_y, val_y, test_y = y_u[:n_train], y_u[n_train:n_train+n_val], y_u[n_train+n_val:]
print(f'User {u} Label Means -> Train: {np.mean(train_y):.3f}, Val: {np.mean(val_y):.3f}, Test: {np.mean(test_y):.3f}')
