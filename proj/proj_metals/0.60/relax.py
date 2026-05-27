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
import copy
from ase.visualize import view
from scipy.optimize import minimize

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
        default=f"{path}/arguments_relax.toml",
        help="Path to config file. e.g. 'arguments_relax.toml'"
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

def build_local_bounds(ads_pos, local_idx, ads_idx, max_step_xy, max_step_z):
    """Box bounds for 3D relaxation of all atoms in local_idx."""
    pos = ads_pos.reshape(-1, 3)
    ads_set = set(ads_idx)
    bounds = []
    for i, atom_idx in enumerate(local_idx):
        for dim in range(3):
            coord = pos[i, dim]
            if dim == 2 and atom_idx in ads_set:
                step = max_step_z
            else:
                step = max_step_xy
            bounds.append((coord - step, coord + step))
    return bounds

def bounds_to_arrays(bounds):
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    return lo, hi

def optimize_local_coords(
    x0,
    fun_and_grad,
    bounds,
    *,
    method="pgd",
    tol=1e-4,
    force_tol=1e-3,
    max_iter=80,
    maxfun=80,
    # alphas=(0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001),
    alphas=(0.5, 0.3, 0.1, 0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001),
    on_step=None,
    coord_search_on_stuck=True,
    energy_plateau_tol=1e-5,
    energy_plateau_patience=30,
    n_evals=None,
):
    """Relax local coordinates using PGD (default) or L-BFGS-B."""
    x0 = np.asarray(x0, dtype=np.float64)
    lo, hi = bounds_to_arrays(bounds)
    force_tol = float(force_tol)
    energy_plateau_tol = float(energy_plateau_tol)
    energy_plateau_patience = int(energy_plateau_patience)
    use_maxfun = maxfun is not None and maxfun > 0

    if method.lower() == "lbfgsb":
        def objective(x):
            energy, _ = fun_and_grad(x)
            if on_step is not None:
                on_step(energy, x)
            return energy

        def gradient(x):
            _, grad = fun_and_grad(x)
            return grad

        return minimize(
            objective,
            x0.copy(),
            jac=gradient,
            method="L-BFGS-B",
            tol=tol,
            bounds=bounds,
            options={"maxfun": maxfun, "maxls": 10},
        )

    x = x0.copy()
    energy, grad = fun_and_grad(x)
    if use_maxfun and n_evals is not None and n_evals[0] >= maxfun:
        return {
            "success": False,
            "x": x,
            "fun": energy,
            "nit": 0,
            "message": f"MAXFUN (|grad|_inf={np.linalg.norm(grad, ord=np.inf):.4g})",
        }
    if on_step is not None:
        on_step(energy, x)

    plateau_count = 0

    for iteration in range(max_iter):
        if use_maxfun and n_evals is not None and n_evals[0] >= maxfun:
            return {
                "success": True,
                "x": x,
                "fun": energy,
                "nit": iteration,
                "message": (
                    f"CONVERGED: ML eval budget (maxfun={maxfun}), "
                    f"|grad|_inf={np.linalg.norm(grad, ord=np.inf):.4g}"
                ),
            }

        grad_norm = np.linalg.norm(grad, ord=np.inf)
        if grad_norm < force_tol:
            return {
                "success": True,
                "x": x,
                "fun": energy,
                "nit": iteration,
                "message": f"CONVERGED: |grad|_inf={grad_norm:.4g} <= {force_tol}",
            }

        energy_at_iter_start = energy

        improved = False
        for alpha in alphas:
            if use_maxfun and n_evals is not None and n_evals[0] >= maxfun:
                break
            x_try = np.clip(x - alpha * grad, lo, hi)
            energy_try, grad_try = fun_and_grad(x_try)
            if np.isfinite(energy_try) and energy_try < energy - 1e-6:
                x, energy, grad = x_try, energy_try, grad_try
                improved = True
                if on_step is not None:
                    on_step(energy, x)
                break

        if not improved:
            if coord_search_on_stuck:
                # 1D line search on each coordinate with largest |grad|
                order = np.argsort(-np.abs(grad))
                for j in order:
                    for sign in (-1, 1):
                        for delta in (0.05, 0.02, 0.01, 0.005, 0.001):
                            if use_maxfun and n_evals is not None and n_evals[0] >= maxfun:
                                break
                            x_try = x.copy()
                            x_try[j] = np.clip(x[j] + sign * delta, lo[j], hi[j])
                            if np.allclose(x_try, x):
                                continue
                            energy_try, grad_try = fun_and_grad(x_try)
                            if np.isfinite(energy_try) and energy_try < energy - 1e-6:
                                x, energy, grad = x_try, energy_try, grad_try
                                improved = True
                                if on_step is not None:
                                    on_step(energy, x)
                                break
                        if improved:
                            break
                    if improved:
                        break

        # Energy plateau: |ΔE| per outer iteration tiny (flat curve) even if |grad| is large (ML).
        if abs(energy - energy_at_iter_start) < energy_plateau_tol:
            plateau_count += 1
        else:
            plateau_count = 0
        if plateau_count >= energy_plateau_patience:
            return {
                "success": True,
                "x": x,
                "fun": energy,
                "nit": iteration + 1,
                "message": (
                    f"CONVERGED: energy plateau (|dE| < {energy_plateau_tol} for "
                    f"{energy_plateau_patience} iters), |grad|_inf={grad_norm:.4g}"
                ),
            }

        if not improved:
            return {
                "success": False,
                "x": x,
                "fun": energy,
                "nit": iteration,
                "message": (
                    f"NO_PROGRESS (|grad|_inf={grad_norm:.4g}, force_tol={force_tol})"
                ),
            }

    grad_norm = np.linalg.norm(grad, ord=np.inf)
    return {
        "success": False,
        "x": x,
        "fun": energy,
        "nit": max_iter,
        "message": (
            f"MAX_ITER ({max_iter}): |grad|_inf={grad_norm:.4g} > force_tol={force_tol}. "
            "Energy may be flat; raise relax_force_tol or enable relax_energy_plateau_*."
        ),
    }

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
    os.makedirs(f'{path}/{args.output_dir}', exist_ok=True)
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

    images = Trajectory(f'{args.dataset}')
    # view(images)

    structure_index = getattr(args, 'structure_index', 100)
    ads_z_perturb = getattr(args, 'ads_z_perturb', 0.0)
    max_step_xy = getattr(args, 'relax_max_step_xy', 0.15)
    max_step_z = getattr(args, 'relax_max_step_z', 0.55)
    relax_tol = getattr(args, 'relax_tol', 1e-4)
    relax_method = getattr(args, 'relax_method', 'pgd')
    relax_max_iter = getattr(args, 'relax_max_iter', 80)
    relax_force_tol = float(getattr(args, 'relax_force_tol', 1e-3))
    relax_energy_plateau_tol = float(getattr(args, 'relax_energy_plateau_tol', 1e-5))
    relax_energy_plateau_patience = int(getattr(args, 'relax_energy_plateau_patience', 30))
    relax_maxfun = int(getattr(args, 'relax_maxfun', 0))
    print(
        f"relax: force_tol={relax_force_tol}  max_iter={relax_max_iter}  "
        f"maxfun={'unlimited' if relax_maxfun <= 0 else relax_maxfun}  "
        f"energy_plateau: |dE|<{relax_energy_plateau_tol} for {relax_energy_plateau_patience} iters"
    )

    for i, atoms in enumerate(images):
        if i != structure_index:
            continue

        ads_idx = atoms.info['data']['ads_idx']
        atoms_ref = copy.deepcopy(atoms)
        if ads_z_perturb:
            for j in ads_idx:
                atoms[j].z += ads_z_perturb

        traj = []
        n_evals = [0]
        eval_cache = {}

        atoms_work = copy.deepcopy(atoms)
        atoms_work.calc = calc
        _ = atoms_work.get_potential_energy()
        local_idx = np.asarray(atoms_work.calc.results['local_idx'], dtype=int)
        ads_pos = atoms_work.positions[local_idx].flatten()
        bounds_anchor = (
            atoms_ref.positions[local_idx].flatten()
            if getattr(args, 'relax_bounds_from_ref', True)
            else ads_pos
        )
        print(f"local_idx ({len(local_idx)} atoms): {local_idx.tolist()}")
        print(f"ads_pos shape: {ads_pos.shape}, dtype: {ads_pos.dtype}")

        def forces_on_local_idx(ads_forces, current_local_idx):
            """Map model forces onto the fixed local_idx used as DOFs."""
            force_map = {
                int(current_local_idx[i]): ads_forces[i]
                for i in range(len(current_local_idx))
            }
            mapped = np.zeros((len(local_idx), 3), dtype=np.float64)
            for i, atom_id in enumerate(local_idx):
                if atom_id in force_map:
                    mapped[i] = force_map[atom_id]
            return mapped

        def compute_properties(ads_pos_flat):
            pos = ads_pos_flat.reshape(-1, 3)
            atoms_work.positions[local_idx] = pos
            binding_energy = atoms_work.get_potential_energy()
            if not np.isfinite(binding_energy) or binding_energy < -50 or binding_energy > 50:
                return None, None
            current_local_idx = np.asarray(
                atoms_work.calc.results['local_idx'], dtype=int
            )
            ads_forces = forces_on_local_idx(
                atoms_work.calc.results['ads_forces'], current_local_idx
            )
            return binding_energy, ads_forces

        def fun_and_grad(ads_pos_flat):
            key = ads_pos_flat.tobytes()
            if key not in eval_cache:
                n_evals[0] += 1
                binding_energy, ads_forces = compute_properties(ads_pos_flat)
                if binding_energy is None:
                    gradient = np.zeros(len(ads_pos_flat), dtype=np.float64)
                    eval_cache[key] = (np.inf, gradient)
                else:
                    gradient = (-np.ascontiguousarray(ads_forces, dtype=np.float64)).ravel()
                    eval_cache[key] = (binding_energy, gradient)
            return eval_cache[key]

        def record_step(energy, _x):
            if not np.isfinite(energy) or energy > 50:
                return
            print(f'eval {n_evals[0]}: binding energy = {energy:.6f} eV')
            traj.append(copy.deepcopy(atoms_work))

        bounds = build_local_bounds(
            bounds_anchor, local_idx, ads_idx, max_step_xy, max_step_z
        )
        ref_energy = None
        aw_ref = copy.deepcopy(atoms_ref)
        aw_ref.calc = calc
        ref_energy = aw_ref.get_potential_energy()
        print(f"reference binding energy (unperturbed): {ref_energy:.6f} eV")
        result = optimize_local_coords(
            ads_pos,
            fun_and_grad,
            bounds,
            method=relax_method,
            tol=relax_tol,
            force_tol=relax_force_tol,
            max_iter=relax_max_iter,
            maxfun=relax_maxfun,
            energy_plateau_tol=relax_energy_plateau_tol,
            energy_plateau_patience=relax_energy_plateau_patience,
            on_step=record_step,
            n_evals=n_evals,
        )
        if isinstance(result, dict):
            atoms_work.positions[local_idx] = result["x"].reshape(-1, 3)
            print(
                f"success={result['success']}  E={result['fun']:.6f} eV  "
                f"iter={result['nit']}  {result['message']}"
            )
        else:
            atoms_work.positions[local_idx] = result.x.reshape(-1, 3)
            print(result)
        print(f'ML evaluations: {n_evals[0]}')
        traj_path = f'{path}/{args.output_dir}/relax.traj'
        write(traj_path, traj)
        print(f'Wrote {len(traj)} frames to {traj_path}')
        local_energies = atoms_work.calc.results['local_energies']
        print(f"local_idx: {local_idx.tolist()}, local_energies: {local_energies.tolist()}")
        if getattr(args, 'show_view', True):
            view(traj)
        # break

if __name__ == "__main__":
    relax(read_csv=False)

