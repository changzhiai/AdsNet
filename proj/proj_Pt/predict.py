import torch
import numpy as np
import os
import pandas as pd
from ase.io import read, write
import matplotlib.pyplot as plt

from adsnet.models import AdsNet
from adsnet.calculators import MLCalculator

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def plot_fitting(dft_energies, nnp_energies, fig_name='fitting.png'):
    dft = np.array(dft_energies)
    nnp = np.array(nnp_energies)
    rmse = np.sqrt(np.mean((dft - nnp) ** 2))
    mae = np.mean(np.abs(dft - nnp))

    fig, ax = plt.subplots()
    ax.plot(dft, nnp, 'ob', mfc='none')

    e_range = max(np.append(dft, nnp)) - min(np.append(dft, nnp))
    rmin = min(np.append(dft, nnp)) - 0.05 * e_range
    rmax = max(np.append(dft, nnp)) + 0.05 * e_range
    ax.plot([rmin, rmax], [rmin, rmax], 'r')
    ax.set_xlim(rmin, rmax)
    ax.set_ylim(rmin, rmax)

    ax.set_xlabel("$E_b$(DFT) (eV)")
    ax.set_ylabel("$E_b$(NNP) (eV)")
    ax.set_title(f"Parity plot ({len(dft)} structures)")
    ax.text(0.95, 0.05,
            f"MAE = {mae*1000:.1f} meV\nRMSE = {rmse*1000:.1f} meV",
            verticalalignment='bottom',
            horizontalalignment='right',
            transform=ax.transAxes,
            fontsize=11)
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(f'{path}/{fig_name}', dpi=150)
    plt.show()
    print(f'Saved figure to {path}/{fig_name}')


def main():
    # Load model
    state_dict = torch.load(f'{path}/output/model.pt', map_location='cpu')
    model = AdsNet(
        num_interactions=state_dict["num_layer"],
        hidden_state_size=state_dict["node_size"],
        cutoff=state_dict["cutoff"],
    )
    model.load_state_dict(state_dict["model"])
    model.eval()

    calc = MLCalculator(model)

    # Load dataset
    images = read(f'{path}/Pt_ads_new.traj', ':')
    print(f'Dataset: {len(images)} structures')

    # Predict
    formulas = []
    dft_energies = []
    nnp_energies = []
    new_images = []
    for atoms in images:
        formulas.append(atoms.get_chemical_formula(mode='hill'))
        dft_energy = atoms.info['data']['ads_energy']
        nnp_energy = calc.get_binding_energy(atoms)
        dft_energies.append(dft_energy)
        nnp_energies.append(nnp_energy)
        if dft_energy > -7:
            new_images.append(atoms)

    # Report
    dft = np.array(dft_energies)
    nnp = np.array(nnp_energies)
    mae = np.mean(np.abs(dft - nnp))
    rmse = np.sqrt(np.mean((dft - nnp) ** 2))
    print(f'MAE:  {mae*1000:.1f} meV')
    print(f'RMSE: {rmse*1000:.1f} meV')

    # Save to CSV
    df = pd.DataFrame({
        'formula': formulas,
        'dft_energies': dft_energies,
        'nnp_energies': nnp_energies,
    })
    csv_path = f'{path}/predict.csv'
    df.to_csv(csv_path, index=False)
    print(f'Saved predictions to {csv_path}')

    # Plot
    plot_fitting(dft_energies, nnp_energies)
    write(f"{path}/Pt_ads_new.traj", new_images)
    print(f'Saved {len(new_images)} structures with DFT energy < -7 eV to {path}/Pt_ads_new.traj')


if __name__ == "__main__":
    main()
