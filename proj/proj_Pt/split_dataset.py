"""Split dataset/Pt_ads_new.traj into train.traj (90%) and test.traj (10%)."""
import os
import numpy as np
from ase.io import read, Trajectory

SEED = 42
TEST_RATIO = 0.1

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
dataset_dir = os.path.join(path, 'dataset')

frames = read(os.path.join(dataset_dir, 'Pt_ads_new.traj'), index=':')
n = len(frames)
n_test = int(round(n * TEST_RATIO))
n_train = n - n_test

rng = np.random.default_rng(SEED)
indices = rng.permutation(n)
train_idx = indices[n_test:]
test_idx = indices[:n_test]

for split, idx, fname in [('train', train_idx, 'train.traj'), ('test', test_idx, 'test.traj')]:
    out = os.path.join(dataset_dir, fname)
    traj = Trajectory(out, 'w')
    for i in idx:
        traj.write(frames[i])
    traj.close()
    print(f'{split}: {len(idx)} frames -> {out}')
