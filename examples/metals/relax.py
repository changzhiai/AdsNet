import os
import torch
from ase.io import read, write

from adsnet.models import AdsNet
from adsnet.tools import relax_structure

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def main():
    state_dict = torch.load(f"{path}/output/model.pt", map_location="cpu")
    model = AdsNet(
        num_layers=state_dict["num_layers"],
        num_channels=state_dict["num_channels"],
        cutoff=state_dict["cutoff"],
        compute_forces=True,
        rbf_type=state_dict.get("rbf_type", "sinc"),
        edge_embed=state_dict.get("edge_embed", "linear"),
    )
    model.load_state_dict(state_dict["model"])

    images = read(f"{path}/dataset/test.traj", ":")
    atoms = images[0]

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
        bounds_from_ref=True,
    )

    print(f"success={result['success']}")
    print(f"energy={result['energy']:.6f} eV")
    print(f"n_evals={result['n_evals']}")
    print(f"message={result['message']}")
    print(f"trajectory frames: {len(result['trajectory'])}")

    if result["trajectory"]:
        os.makedirs(f"{path}/output", exist_ok=True)
        traj_path = f"{path}/output/relax.traj"
        write(traj_path, result["trajectory"])
        print(f"Wrote {len(result['trajectory'])} frames to {traj_path}")


if __name__ == "__main__":
    main()
