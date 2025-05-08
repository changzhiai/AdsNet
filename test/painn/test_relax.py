from ase.io import read, write, Trajectory
import torch
import toml
import argparse
import logging
from adsnet.painn import PainnModel
from adsnet.calculator import MLCalculator, EnsembleCalculator
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.db import connect
import numpy as np
import pandas as pd
import os

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

class EnergyObservor:
    def __init__(self, atoms):
        self.atoms = atoms
        print("Energy observor")

    def __call__(self, threshold=0):
        energy = self.atoms.get_potential_energy()
        forces = self.atoms.get_forces()
        print(f'energy: {energy}')
        if energy > threshold or energy<-10000:
            raise ValueError('energy is too large or too low')

def plot_fitting(dft_energies, nnp_energies, fig_name='prediction.png', output_dir=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    rmse = mean_squared_error(dft_energies, nnp_energies, squared=False)
    m, b = np.polyfit(dft_energies, nnp_energies, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.array(dft_energies), m * np.array(dft_energies) + b)
    plt.plot(dft_energies, nnp_energies, 'ob',mfc='none')
    plt.xlabel("$E_b$(DFT) (eV)")
    plt.ylabel("$E_b$(NNP) (eV)")
    plt.title("fit using {} data points.".format(len(nnp_energies)))
    X = dft_energies
    Y = nnp_energies
    if np.size(X) and np.size(Y) != 0:
        e_range = max(np.append(X, Y)) - min(np.append(X, Y))
        rmin = min(np.append(X, Y)) - 0.05 * e_range
        rmax = max(np.append(X, Y)) + 0.05 * e_range
    else:
        rmin = -10
        rmax = 10
    linear_fit = np.arange(rmin - 10, rmax + 10, 1)
    ax.plot(linear_fit, linear_fit, 'r')
    ax.axis([rmin, rmax, rmin, rmax])
    ax.text(0.95,
            0.01,
            "RMSE = {:.3f} meV".format(rmse*1000),
            verticalalignment='bottom',
            horizontalalignment='right',
            transform=ax.transAxes,
            fontsize=12)
    ticks = False
    if ticks:
        plt.xticks(np.arange(min(X), max(Y), 0.2))
        plt.yticks(np.arange(min(X), max(Y), 0.2))
    plt.show()
    fig.savefig(f'{path}/{output_dir}/{fig_name}')

def setup_seed(seed):
     torch.manual_seed(seed)
     if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(
        description="graph neural networks", fromfile_prefix_chars="+"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=f"{path}/arguments_predict.toml",
        help="Path to config file. e.g. 'arguments.toml'"
    )

    return parser.parse_args(arg_list)

def update_namespace(ns, d):
    for k, v in d.items():
        ns.__dict__[k] = v

class CallsCounter:
    def __init__(self, func):
        self.calls = 0
        self.func = func
    def __call__(self, *args, **kwargs):
        self.calls += 1
        self.func(*args, **kwargs)

def generate_run_config(params):
    # with open(os.path.join(params['run_dir'], "run.toml"), 'w') as f:
    with open(os.path.join(path, "run.toml"), 'w') as f:
        toml.dump(params, f)
    return params

def read_df(args):
    df = pd.read_csv(f'{path}/{args.save_csv}')
    # df = df.loc[df['converged']==True]
    dft_energies = df['dft_energies']
    nnp_energies = df['nnp_energies']
    fig_name = args.fig_name
    return dft_energies, nnp_energies, fig_name

def get_init_bad_structs(args):
    df = pd.read_csv(args.save_csv)
    df = df[(df['dft_energies']-df['nnp_energies']>0.02) | (df['converged']==False)]
    images = read(args.all_init_dft_dir, ':')
    lists = list(df.index.values)
    images_bad = [images[i] for i in lists]
    print(len(images_bad))
    print([images[i].get_chemical_formula() for i in lists])
    write('images_bad.traj', images_bad)
    return df

def relax(read_csv=False):
    args = get_arguments()
    if args.cfg:
        with open(args.cfg, 'r') as f:
            params = toml.load(f)
    update_namespace(args, params)
    if read_csv:
        print('Reading csv')
        # get_init_bad_structs(args)
        return read_df(args)

    _ = generate_run_config(params)
    setup_seed(args.random_seed)
    # set logger
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)
    runHandler = logging.FileHandler(f'{path}/{args.output_dir}/args.predict_log', mode='w')
    runHandler.setLevel(logging.DEBUG)
    runHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    errorHandler = logging.FileHandler(f'{path}/{args.output_dir}/args.error_log', mode='w')
    errorHandler.setLevel(logging.WARNING)
    errorHandler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)7s - %(message)s"))
    logger.addHandler(runHandler)
    logger.addHandler(errorHandler)
    logger.addHandler(logging.StreamHandler())
    logger.warning = CallsCounter(logger.warning)
    logger.info = CallsCounter(logger.info)
    # load model
    state_dict = torch.load(f'{path}/{args.load_model}', map_location=torch.device(args.device)) 
    model = PainnModel(
        num_interactions=state_dict["num_layer"],
        hidden_state_size=state_dict["node_size"],
        cutoff=state_dict["cutoff"],
        compute_forces = bool(args.forces_weight),
    )
    model.to(args.device)
    model.load_state_dict(state_dict["model"])
    calc = MLCalculator(model)

    images = read(f'{path}/{args.dataset}', ':')
    count = 0
    formulas = []
    dft_energies = []
    nnp_energies = []
    E_per_atoms = args.E_per_atoms
    for i, atoms in enumerate(images):
        import copy
        from ase.visualize import view
        if i != 120:
            continue
        ads_idx = atoms.info['data']['ads_idx']
        if len(ads_idx) == 1:
            atoms[ads_idx[0]].z += 0.4
        traj = []
        def compute_properties(ads_pos):
            atoms_temp = copy.deepcopy(atoms)
            ads_idx = atoms.info['data']['ads_idx']
            if len(ads_idx) == 1:
                atoms_temp[ads_idx[0]].position = ads_pos

            atoms_temp.calc = copy.deepcopy(calc)
            binding_energy = atoms_temp.get_potential_energy()
            ads_forces = atoms_temp.calc.results['ads_forces']
            local_idx = atoms_temp.calc.results['local_idx']
            local_energies = atoms_temp.calc.results['local_energies']
            ads_idx_in_local = list(local_idx).index(ads_idx)
            ads_force = ads_forces[ads_idx_in_local]
            return binding_energy, ads_force, local_idx, local_energies, atoms_temp

        def objective(ads_pos):
            binding_energy, _, _, _, atoms_new = compute_properties(ads_pos)
            # view(atoms_new)
            ads_idx = atoms.info['data']['ads_idx']
            print(f"Objective function evaluated at ads_pos: {ads_pos}, ads_idx: {ads_idx}")
            print(atoms_new[ads_idx].positions)
            traj.append(atoms_new)
            return binding_energy
        
        def gradient(ads_pos):
            _, ads_forces, _, _, _ = compute_properties(ads_pos)
            return -np.ascontiguousarray(ads_forces, dtype=np.float64).ravel()
        

        atoms_init = copy.deepcopy(atoms)
        ads_pos = atoms_init[atoms.info['data']['ads_idx']].positions.flatten()
        print(f"ads_pos shape: {ads_pos.shape}, dtype: {ads_pos.dtype}")
        print(f'ads_pos: {ads_pos}')

        max_step = 0.2
        bounds = [(x - max_step, x + max_step) for x in ads_pos]
        from scipy.optimize import minimize
        result = minimize(
            objective, ads_pos.copy(), 
            jac=gradient, 
            # method='BFGS', 
            method='L-BFGS-B', 
            # method='TNC',
            tol=1e-4,
            bounds=bounds,
            )
        print(result)
        view(traj)
        # break


if __name__ == "__main__":
    relax(read_csv=False) # relax one atom adsorbate

