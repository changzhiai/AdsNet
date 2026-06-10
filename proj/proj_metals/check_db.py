"""Export total.traj to a CSV with formula, ads_idx, and ads_energy."""
import os
import csv
from ase.io import read

path      = os.path.abspath(os.path.join(os.path.dirname(__file__)))
traj_path = f'{path}/dataset/total.traj'
out_csv   = f'{path}/dataset/total.csv'

print('Loading trajectory...')
traj = read(traj_path, ':')

with open(out_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['formula', 'ads_idx', 'ads_energy'])
    for atoms in traj:
        formula    = atoms.get_chemical_formula()
        ads_idx    = atoms.info['data']['ads_idx']
        ads_energy = atoms.info['data']['ads_energy']
        writer.writerow([formula, ads_idx, ads_energy])

print(f'Saved {out_csv}  ({len(traj)} rows)')
