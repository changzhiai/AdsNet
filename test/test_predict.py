import torch
import numpy as np
import os
from ase.io import read

from adsnet.models import AdsNet
from adsnet.calculators import MLCalculator

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def main():
    # Load model
    state_dict = torch.load(f'{path}/model_output/best_model.pth', map_location='cpu')
    model = AdsNet(
        num_interactions=state_dict["num_layer"],
        hidden_state_size=state_dict["node_size"],
        cutoff=state_dict["cutoff"],
    )
    model.load_state_dict(state_dict["model"])
    model.eval()

    calc = MLCalculator(model)

    # Load dataset
    images = read(f'{path}/Pt_ads.traj', ':')
    print(f'Dataset: {len(images)} structures')

    # Predict
    dft_energies = []
    nnp_energies = []
    for atoms in images:
        dft_energies.append(atoms.info['data']['ads_energy'])
        nnp_energies.append(calc.get_binding_energy(atoms))

    # Report
    dft = np.array(dft_energies)
    nnp = np.array(nnp_energies)
    mae = np.mean(np.abs(dft - nnp))
    rmse = np.sqrt(np.mean((dft - nnp) ** 2))
    print(f'MAE:  {mae*1000:.1f} meV')
    print(f'RMSE: {rmse*1000:.1f} meV')


if __name__ == "__main__":
    main()
