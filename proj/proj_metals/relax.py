import os
import torch
from ase.io import read, write
from ase.visualize import view

from adsnet.models import AdsNet
from adsnet.tools import relax_structure

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def main():
    # Load model
    state_dict = torch.load(f'{path}/output/model.pt', map_location='cpu')
    model = AdsNet(
        num_interactions=state_dict["num_layer"],
        hidden_state_size=state_dict["node_size"],
        cutoff=state_dict["cutoff"],
        compute_forces=True,
    )
    model.load_state_dict(state_dict["model"])

    # Load structures
    # images = read(f'{path}/legacy_dataset/images_refined.traj', ':')
    images = read(f'{path}/dataset/total.traj', ':')
    structure_index = 2000

    atoms = images[structure_index]

    # Relax
    result = relax_structure(
        atoms,
        model,
        method="pgd",
        max_step_xy=0.15,
        max_step_z=0.55,
        force_tol=1e-3,
        max_iter=80,
        maxfun=80,
        energy_plateau_tol=1e-5,
        energy_plateau_patience=30,
        ads_z_perturb=0.,
        bounds_from_ref=True,
    )

    print(f"success={result['success']}")
    print(f"energy={result['energy']:.6f} eV")
    print(f"n_evals={result['n_evals']}")
    print(f"message={result['message']}")
    print(f"trajectory frames: {len(result['trajectory'])}")

    # Save trajectory
    if result['trajectory']:
        traj_path = f'{path}/relax_output/relax.traj'
        os.makedirs(f'{path}/relax_output', exist_ok=True)
        write(traj_path, result['trajectory'])
        print(f'Wrote {len(result["trajectory"])} frames to {traj_path}')
        view(result['trajectory'])


if __name__ == "__main__":
    main()
