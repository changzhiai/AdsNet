import os
import torch
import pandas as pd
from ase.io import read, write

from adsnet.models import AdsNet
from adsnet.calculators import MLCalculator

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

name = 'test'
traj_path = f'/pscratch/sd/c/changzhi/adsnet/extra_dataset/refine_1/train/output/predict/{name}_refined.traj'
model_path = f'{path}/../model.pt'
out_path   = f'{path}/{name}_outliers.csv'

state_dict = torch.load(model_path, map_location='cpu')
model = AdsNet(
	num_layers=state_dict["num_layers"],
	num_channels=state_dict["num_channels"],
	cutoff=state_dict["cutoff"],
)
model.load_state_dict(state_dict['model'])
model.eval()

calc = MLCalculator(model)

images = read(traj_path, ':')
rows = []
for i, atoms in enumerate(images):
    formula    = atoms.get_chemical_formula(mode='hill')
    dft_energy = atoms.info['data']['ads_energy']
    nnp_energy = calc.get_binding_energy(atoms)
    print(f'[{i:4d}] {formula:30s}  DFT={dft_energy:.4f}  NNP={nnp_energy:.4f}')
    rows.append({'formula': formula, 'dft_energies': dft_energy, 'nnp_energies': nnp_energy})

df = pd.DataFrame(rows)
df['error'] = df['nnp_energies'] - df['dft_energies']
df['abs_error'] = df['error'].abs()

outlier_mask = (df['abs_error'] > 0.4) | (df['dft_energies'] > 5)
outlier_images  = [images[i] for i in df.index[outlier_mask]]
refined_images  = [images[i] for i in df.index[~outlier_mask]]

os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)

write(f'{path}/{name}_outliers.traj',  outlier_images)
write(f'{path}/{name}_refined.traj',   refined_images)

print(f'\nTotal: {len(images)}  |  Outliers (|err|>0.2): {len(outlier_images)}  |  Refined: {len(refined_images)}')
print(f'Saved CSV     -> {out_path}')
print(f'Saved outliers -> {path}/outliers.traj')
print(f'Saved refined  -> {path}/refined.traj')

