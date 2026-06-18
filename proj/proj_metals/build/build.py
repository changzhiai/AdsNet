from ase.build import fcc111, bcc110, hcp0001, add_adsorbate
from ase.visualize import view
from ase import Atom, Atoms
from contextlib import contextmanager
from ase.io import read, write
import numpy as np
import os
import subprocess
import time

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    try:
        os.makedirs(newdir)
    except OSError:
        pass
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

def build_slab(metal, info):
    struct = info[0]
    if struct == "fcc":
        a = info[1]
        return fcc111(metal, size=(3, 3, 3), a=a, vacuum=10.0)
    elif struct == "bcc":
        a = info[1]
        return bcc110(metal, size=(3, 3, 3), a=a, vacuum=10.0)
    elif struct == "hcp":
        a, c = info[1], info[2]
        return hcp0001(metal, size=(3, 3, 3), a=a, c=c, vacuum=10.0)
    else:
        return None

def fix_bottom_layer(slab):
    from ase.constraints import FixAtoms
    z_positions = slab.positions[:, 2]
    z_cutoff = sorted(set(z_positions))[1]  # keeps bottom 2 layers fixed
    constraint = FixAtoms(mask=[z <= z_cutoff for z in z_positions])
    slab.set_constraint(constraint)
    return slab

def get_adsorption_site(struct, ads_name):
    if ads_name == "H":
        if struct == "fcc":
            return "fcc"
        elif struct == "hcp":
            return "hcp"
        elif struct == "bcc":
            return "ontop"   # safe fallback
    elif ads_name == "CO":
        return "ontop"
    elif ads_name == "OH":
        if struct == "fcc":
            return "bridge"
        else:
            return "ontop"   # hcp + bcc safe fallback
    elif ads_name == "HOCO":
        return "ontop"
    return "ontop"

def get_sites(gs):
    supported_sites = {
    "fcc": ["ontop", "bridge", "fcc", "hcp"],
    "bcc": ["ontop", "longbridge", "shortbridge", "hollow"],
    "hcp": ["ontop", "bridge", "fcc", "hcp"],
    }
    return supported_sites[gs]

def get_metals():

    metals = [
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Ir", "Pt", "Au"
    ]
    return metals

def get_adsobates():
    h = Atoms([Atom('H', (0., 0., 0.))])
    c = Atoms([Atom('C', (0., 0., 0.))])
    o = Atoms([Atom('O', (0., 0., 0.))])
    
    co = Atoms([Atom('C', (0., 0., 0.)),
                 Atom('O', (0., 0., 1.14))])
    
    oh = Atoms([Atom('O', (0., 0., 0.)),
             Atom('H', (0., 0., 0.97))])
    
    hoco = Atoms([Atom('C', (0.,  0.00000,  0.00000)),
                  Atom('H', (0., -1.89485, -0.10328)),
                  Atom('O', (0.,  1.05786,  0.55283)),
                  Atom('O', (0., -1.21428,  0.59825))])
    
    hoco.center(vacuum=8.0)
    
    adsorbates = {
        "C": c,
        "O": o,
        "H": h,
        "CO": co,
        "OH": oh,
        "HOCO": hoco,
    }
    
    return adsorbates

ground_state = {
    "Sc": ("hcp", 3.31, 5.27),
    "Ti": ("hcp", 2.95, 4.68),
    "V":  ("bcc", 3.03),
    "Cr": ("bcc", 2.88),
    "Mn": ("fcc", 3.62),  # approximate metastable
    "Fe": ("bcc", 2.86),
    "Co": ("hcp", 2.51, 4.07),
    "Ni": ("fcc", 3.52),
    "Cu": ("fcc", 3.61),
    "Zn": ("hcp", 2.66, 4.95),
    "Y":  ("hcp", 3.65, 5.73),
    "Zr": ("hcp", 3.23, 5.15),
    "Nb": ("bcc", 3.30),
    "Mo": ("bcc", 3.15),
    "Ru": ("hcp", 2.71, 4.28),
    "Rh": ("fcc", 3.80),
    "Pd": ("fcc", 3.89),
    "Ag": ("fcc", 4.09),
    "Cd": ("hcp", 2.98, 5.62),
    "Hf": ("hcp", 3.19, 5.05),
    "Ta": ("bcc", 3.30),
    "W":  ("bcc", 3.16),
    "Re": ("hcp", 2.76, 4.46),
    "Ir": ("fcc", 3.84),
    "Pt": ("fcc", 3.92),
    "Au": ("fcc", 4.08),
}

if __name__ == "__main__":
    name = 'metals_with_adss.traj'
    # name = None
    images = []
    metals = get_metals()
    adsorbates = get_adsobates()
    path = os.getcwd()
    with cd('run_dft'):
        for metal in metals[:]:
            with cd(f'{metal}'):
                info = ground_state.get(metal, None)
                if info is None or info[0] == "complex":
                    print(f"Skipping {metal} (no simple crystal structure)")
                    continue
                slab = build_slab(metal, info)
                slab = fix_bottom_layer(slab)
                # with cd('slab'):
                    # write('start.traj', slab)
                    # os.system(f'cp {path}/dft.py .')
                    # os.system(f'cp {path}/arguments.toml .')
                    # os.system(f'cp {path}/submit.sh .')
                    # os.system('sbatch submit.sh')

                sites = get_sites(info[0])
                for ads_name, ads in adsorbates.items():
                    with cd(f'{ads_name}'):
                        for site in sites:
                            with cd(f'{site}'):
                                slab_with_ads = slab.copy()
                                height = {"H": 1.0, "C": 1.5, "O": 1.5, "CO": 1.8, "OH": 1.5, "HOCO": 2.0}[ads_name]
                                add_adsorbate(slab_with_ads, ads, height=height, position=site)
                                images.append(slab_with_ads)
                                write('start.traj', slab_with_ads)
                                os.system(f'cp {path}/dft.py .')
                                os.system(f'cp {path}/arguments.toml .')
                                os.system(f'cp {path}/submit.sh .')
                                os.system('sbatch submit.sh')
                                # assert False