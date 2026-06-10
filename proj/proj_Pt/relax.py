import torch
import os
from ase.io import read
from ase.visualize import view

from adsnet.models import AdsNet
from adsnet.calculators import MLCalculator
from adsnet.tools import relax_structure

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def main():
    # Load model
    state_dict = torch.load(f'{path}/output/model.pt', map_location='cpu')
    model = AdsNet(
        num_layers=state_dict["num_layers"],
        num_channels=state_dict["num_channels"],
        cutoff=state_dict["cutoff"],
        compute_forces=True,
    )
    model.load_state_dict(state_dict["model"])

    # Load structure
    images = read(f'{path}/Pt_ads_forces.traj', ':')
    atoms = images[120]

    # Relax using tools/relax.py
    result = relax_structure(
        atoms,
        model,
        method="pgd",
        max_step_xy=0.2,
        max_step_z=0.4,
        force_tol=1e-3,
        max_iter=80,
        ads_z_perturb=0.4,
    )

    print(f"success={result['success']}")
    print(f"energy={result['energy']:.6f} eV")
    print(f"n_evals={result['n_evals']}")
    print(f"message={result['message']}")
    print(f"trajectory frames: {len(result['trajectory'])}")

    if result['trajectory']:
        view(result['trajectory'])


if __name__ == "__main__":
    main()
