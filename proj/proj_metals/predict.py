import torch
import numpy as np
import os
import pandas as pd
from ase.io import read
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

from adsnet.models import AdsNet
from adsnet.calculators import MLCalculator

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def plot_fitting_joint(train_dft, train_nnp, test_dft, test_nnp, fig_name='fitting_joint.png'):
    train_dft = np.array(train_dft)
    train_nnp = np.array(train_nnp)
    test_dft  = np.array(test_dft)
    test_nnp  = np.array(test_nnp)

    all_vals = np.concatenate([train_dft, train_nnp, test_dft, test_nnp])
    e_range  = all_vals.max() - all_vals.min()
    rmin = all_vals.min() - 0.05 * e_range
    rmax = all_vals.max() + 0.05 * e_range
    grid = np.linspace(rmin, rmax, 300)

    fig = plt.figure(figsize=(6, 6))
    gs  = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                   hspace=0.04, wspace=0.04)
    ax_top   = fig.add_subplot(gs[0, 0])
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    ax_main.plot(train_dft, train_nnp, 'o', color='#6096BA',
                 mfc='none', ms=5, lw=0.8, label=f'Train ({len(train_dft)})')
    ax_main.plot(test_dft,  test_nnp,  's', color='#E07A5F',
                 mfc='none', ms=6, lw=1.0, label=f'Test ({len(test_dft)})')
    ax_main.plot([rmin, rmax], [rmin, rmax], 'k--', lw=1)
    ax_main.set_xlim(rmin, rmax)
    ax_main.set_ylim(rmin, rmax)
    ax_main.set_xlabel("$E_b$(DFT) (eV)")
    ax_main.set_ylabel("$E_b$(NNP) (eV)")
    ax_main.set_aspect('equal')

    def mae_rmse(a, b):
        return np.mean(np.abs(a - b)), np.sqrt(np.mean((a - b) ** 2))

    tr_mae, tr_rmse = mae_rmse(train_dft, train_nnp)
    te_mae, te_rmse = mae_rmse(test_dft,  test_nnp)
    ax_main.text(0.97, 0.03,
                #  f"Train  MAE={tr_mae*1000:.1f} meV  RMSE={tr_rmse*1000:.1f} meV\n"
                #  f"Test   MAE={te_mae*1000:.1f} meV  RMSE={te_rmse*1000:.1f} meV",
                 f"Train RMSE={tr_rmse*1000:.1f} meV\n"
                 f"Test  RMSE={te_rmse*1000:.1f} meV",
                 va='bottom', ha='right', transform=ax_main.transAxes, fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75))
    ax_main.legend(loc='upper left', fontsize=8, framealpha=0.75)

    for vals, color, alpha in [(train_dft, '#6096BA', 0.4), (test_dft, '#E07A5F', 0.4)]:
        kde = gaussian_kde(vals, bw_method=0.3)
        ax_top.fill_between(grid, kde(grid), alpha=alpha, color=color)
        ax_top.plot(grid, kde(grid), color=color, lw=1)
    ax_top.set_xlim(rmin, rmax)
    ax_top.set_xticklabels([])
    ax_top.set_yticks([])
    ax_top.set_ylabel('KDE', fontsize=7)
    ax_top.tick_params(bottom=False)
    for spine in ['top', 'right', 'left']:
        ax_top.spines[spine].set_visible(False)

    for vals, color, alpha in [(train_nnp, '#6096BA', 0.4), (test_nnp, '#E07A5F', 0.4)]:
        kde = gaussian_kde(vals, bw_method=0.3)
        ax_right.fill_betweenx(grid, kde(grid), alpha=alpha, color=color)
        ax_right.plot(kde(grid), grid, color=color, lw=1)
    ax_right.set_ylim(rmin, rmax)
    ax_right.set_yticklabels([])
    ax_right.set_xticks([])
    ax_right.set_xlabel('KDE', fontsize=7)
    ax_right.tick_params(left=False)
    for spine in ['top', 'right', 'bottom']:
        ax_right.spines[spine].set_visible(False)

    fig.savefig(f'{path}/output/{fig_name}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved figure to {path}/output/{fig_name}')


def plot_fitting(train_dft, train_nnp, test_dft, test_nnp, fig_name='fitting.png'):
    train_dft = np.array(train_dft)
    train_nnp = np.array(train_nnp)
    test_dft  = np.array(test_dft)
    test_nnp  = np.array(test_nnp)

    def metrics(dft, nnp):
        return np.mean(np.abs(dft - nnp)), np.sqrt(np.mean((dft - nnp) ** 2))

    train_mae, train_rmse = metrics(train_dft, train_nnp)
    test_mae,  test_rmse  = metrics(test_dft,  test_nnp)

    all_vals = np.concatenate([train_dft, train_nnp, test_dft, test_nnp])
    e_range = all_vals.max() - all_vals.min()
    rmin = all_vals.min() - 0.05 * e_range
    rmax = all_vals.max() + 0.05 * e_range

    fig, ax = plt.subplots()
    ax.plot(train_dft, train_nnp, 'ob', mfc='none', label=f'Train ({len(train_dft)})')
    ax.plot(test_dft,  test_nnp,  'sr', mfc='none', label=f'Test ({len(test_dft)})')
    ax.plot([rmin, rmax], [rmin, rmax], 'k--', lw=1)
    ax.set_xlim(rmin, rmax)
    ax.set_ylim(rmin, rmax)
    ax.set_xlabel("$E_b$(DFT) (eV)")
    ax.set_ylabel("$E_b$(NNP) (eV)")
    ax.set_title("Parity plot")
    ax.text(0.97, 0.05,
            f"Train  MAE={train_mae*1000:.1f} meV  RMSE={train_rmse*1000:.1f} meV\n"
            f"Test   MAE={test_mae*1000:.1f} meV  RMSE={test_rmse*1000:.1f} meV",
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    ax.legend(loc='upper left', fontsize=9)
    ax.set_aspect('equal')
    plt.tight_layout()
    fig.savefig(f'{path}/output/{fig_name}', dpi=300)
    plt.close()
    print(f'Saved figure to {path}/output/{fig_name}')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Predict and plot AdsNet results.')
    parser.add_argument('--csv', default=True,
                        help='Skip inference and load predictions from an existing CSV '
                             '(must have columns: split, dft_energies, nnp_energies)')
    args = parser.parse_args()

    csv_path = f'{path}/output/predict/predict.csv'

    if args.csv:
        print(f'Loading predictions from {csv_path}...')
        df = pd.read_csv(csv_path)
        train = df[df['split'] == 'train']
        test  = df[df['split'] == 'test']
        train_dft = train['dft_energies'].tolist()
        train_nnp = train['nnp_energies'].tolist()
        test_dft  = test['dft_energies'].tolist()
        test_nnp  = test['nnp_energies'].tolist()
        print(f'Train: {len(train_dft)} structures  |  Test: {len(test_dft)} structures')
    else:
        state_dict = torch.load(f'{path}/output/model.pt', map_location='cpu')
        model = AdsNet(
            num_layers=state_dict["num_layers"],
            num_channels=state_dict["num_channels"],
            cutoff=state_dict["cutoff"],
            rbf_type=state_dict.get("rbf_type", "sinc"),
            edge_embed=state_dict.get("edge_embed", "linear"),
        )
        model.load_state_dict(state_dict["model"])
        model.eval()

        calc = MLCalculator(model)

        def predict_set(traj_path):
            images = read(traj_path, ':')
            formulas, dft_energies, nnp_energies = [], [], []
            for i, atoms in enumerate(images):
                print(f'Predicting {i}: {atoms.info["id"]}')
                formulas.append(atoms.get_chemical_formula(mode='hill'))
                dft_energies.append(atoms.info['data']['ads_energy'])
                nnp_energies.append(calc.get_binding_energy(atoms))
            return formulas, dft_energies, nnp_energies

        train_formulas, train_dft, train_nnp = predict_set(f'{path}/dataset/train.traj')
        test_formulas,  test_dft,  test_nnp  = predict_set(f'{path}/dataset/test.traj')
        print(f'Train: {len(train_dft)} structures  |  Test: {len(test_dft)} structures')

        df = pd.DataFrame({
            'split':        ['train'] * len(train_dft) + ['test'] * len(test_dft),
            'formula':      train_formulas + test_formulas,
            'dft_energies': train_dft + test_dft,
            'nnp_energies': train_nnp + test_nnp,
        })
        df.to_csv(csv_path, index=False)
        print(f'Saved predictions to {csv_path}')

    for label, dft, nnp in [('Train', train_dft, train_nnp), ('Test', test_dft, test_nnp)]:
        dft_a, nnp_a = np.array(dft), np.array(nnp)
        mae  = np.mean(np.abs(dft_a - nnp_a))
        rmse = np.sqrt(np.mean((dft_a - nnp_a) ** 2))
        print(f'{label}  MAE: {mae*1000:.1f} meV  RMSE: {rmse*1000:.1f} meV')

    plot_fitting(train_dft, train_nnp, test_dft, test_nnp)
    plot_fitting_joint(train_dft, train_nnp, test_dft, test_nnp)


if __name__ == "__main__":
    main()
