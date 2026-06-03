import os
from ase.db import connect
from ase.io import read
import numpy as np

db_path = "metals_with_adss.db"
db = connect(db_path)

slab_root = "/sdf/data/suncat/suncatlustre/changzhi/dataset/run_dft"
gas_root = "/sdf/data/suncat/suncatlustre/changzhi/dataset/run_dft/ref"

refs = {}
refs['H2'] = read(os.path.join(gas_root, 'H2', 'OUTCAR')).get_potential_energy()
refs['CO'] = read(os.path.join(gas_root, 'CO', 'OUTCAR')).get_potential_energy()
refs['CO2'] = read(os.path.join(gas_root, 'CO2', 'OUTCAR')).get_potential_energy()
refs['H2O'] = read(os.path.join(gas_root, 'H2O', 'OUTCAR')).get_potential_energy()
refs['H'] = 0.5 * refs['H2']
refs['CO'] = refs['CO']
refs['HOCO'] = refs['CO2'] + 0.5 * refs['H2']
refs['OH'] = refs['H2O'] - 0.5 * refs['H2']

def read_outcar_energy(path):
    outcar = os.path.join(path, "OUTCAR")
    atoms = read(outcar)
    return atoms.get_potential_energy(), atoms.get_forces()


def get_ads_idx(atoms):
    # assume adsorbate atoms are tagged or highest z
    z = atoms.positions[:, 2]
    slab_z = np.sort(z)[:-4]  # crude guess
    threshold = slab_z.max() + 0.5
    ads_idx = np.where(z > threshold)[0]
    return ads_idx.tolist()


def get_adsorption_energy(atoms, slab_energy, gas_energy):
    e_ads_slab = atoms.get_potential_energy()
    return e_ads_slab - slab_energy - gas_energy


for row in db.select():

    atoms = db.get_atoms(row.id)
    path = row.path
    uid = row.uid

    metal = uid.split("_")[0]
    ads = uid.split("_")[1]

    slab_path = os.path.join(slab_root, metal, "slab")

    try:
        slab_e, slab_f = read_outcar_energy(slab_path)

        ads_idx = get_ads_idx(atoms)
        gas_e = refs[ads]

        ads_energy = get_adsorption_energy(atoms, slab_e, gas_e)

        forces = atoms.get_forces()
        delta_forces = forces.copy()

        # store in atoms.info
        if "data" not in atoms.info:
            atoms.info["data"] = {}

        atoms.info["data"]["ads_idx"] = ads_idx
        atoms.info["data"]["ads_energy"] = ads_energy
        atoms.info["data"]["delta_forces"] = delta_forces

        db.update(
            row.id,
            atoms=atoms,
            data={
                "ads_idx": ads_idx,
                "ads_energy": ads_energy,
                "delta_forces": delta_forces,
                },
            adsidx=str(ads_idx),
            ads_e=ads_energy,
            slabe=slab_e,
            adse=gas_e
        )

        print("processed", row.id)

    except Exception as e:
        print("failed", row.id, e)