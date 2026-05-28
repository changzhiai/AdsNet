import numpy as np
import copy
from scipy.optimize import minimize


def build_local_bounds(ads_pos, local_idx, ads_idx, max_step_xy=0.15, max_step_z=0.55):
    """Box bounds for 3D relaxation of all atoms in local_idx.

    Args:
        ads_pos: Flattened positions of local atoms (N*3,)
        local_idx: Atom indices in the original structure
        ads_idx: Indices of adsorbate atoms
        max_step_xy: Max displacement in x,y directions
        max_step_z: Max displacement in z for adsorbate atoms
    """
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
    alphas=(0.5, 0.3, 0.1, 0.02, 0.01, 0.005, 0.001, 0.0005, 0.0001),
    on_step=None,
    coord_search_on_stuck=True,
    energy_plateau_tol=1e-5,
    energy_plateau_patience=30,
    n_evals=None,
):
    """Relax local coordinates using PGD (default) or L-BFGS-B.

    Args:
        x0: Initial flat coordinate vector
        fun_and_grad: Callable(x) -> (energy, gradient)
        bounds: Box bounds per DOF
        method: "pgd" or "lbfgsb"
        tol: Convergence tolerance for scipy
        force_tol: Max gradient norm for convergence
        max_iter: Maximum PGD outer iterations
        maxfun: Maximum function evaluations budget
        alphas: Step sizes to try in PGD line search
        on_step: Callback(energy, x) called on improvement
        coord_search_on_stuck: Try coordinate-wise search when stuck
        energy_plateau_tol: Threshold for energy plateau detection
        energy_plateau_patience: Steps before declaring plateau convergence
        n_evals: Mutable [count] for tracking total evaluations

    Returns:
        dict with keys: success, x, fun, nit, message
    """
    x0 = np.asarray(x0, dtype=np.float64)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
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

        if not improved and coord_search_on_stuck:
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
            "Energy may be flat; raise force_tol or enable energy_plateau_*."
        ),
    }


def make_fun_and_grad(atoms_work, local_idx, calculator=None, cache=None, n_evals=None):
    """Create a fun_and_grad closure for use with optimize_local_coords.

    Args:
        atoms_work: ASE Atoms object (modified in place during optimization)
        local_idx: Indices of atoms being relaxed
        calculator: MLCalculator instance (if None, uses atoms_work.calc)
        cache: Optional evaluation cache dict (if None, creates one)
        n_evals: Optional mutable [count] for evaluation tracking

    Returns:
        Callable(x_flat) -> (energy, gradient_flat)
    """
    if cache is None:
        cache = {}
    if n_evals is None:
        n_evals = [0]

    def forces_on_local_idx(ads_forces, current_local_idx):
        force_map = {
            int(current_local_idx[i]): ads_forces[i]
            for i in range(len(current_local_idx))
        }
        mapped = np.zeros((len(local_idx), 3), dtype=np.float64)
        for i, atom_id in enumerate(local_idx):
            if atom_id in force_map:
                mapped[i] = force_map[atom_id]
        return mapped

    def fun_and_grad(ads_pos_flat):
        key = ads_pos_flat.tobytes()
        if key not in cache:
            n_evals[0] += 1
            pos = ads_pos_flat.reshape(-1, 3)
            atoms_work.positions[local_idx] = pos
            binding_energy = atoms_work.get_potential_energy()
            if not np.isfinite(binding_energy) or binding_energy < -50 or binding_energy > 50:
                gradient = np.zeros(len(ads_pos_flat), dtype=np.float64)
                cache[key] = (np.inf, gradient)
            else:
                current_local_idx = np.asarray(
                    atoms_work.calc.results['local_idx'], dtype=int
                )
                ads_forces = forces_on_local_idx(
                    atoms_work.calc.results['ads_forces'], current_local_idx
                )
                gradient = (-np.ascontiguousarray(ads_forces, dtype=np.float64)).ravel()
                cache[key] = (binding_energy, gradient)
        return cache[key]

    return fun_and_grad


def relax_structure(
    atoms,
    model,
    *,
    method="pgd",
    max_step_xy=0.15,
    max_step_z=0.55,
    force_tol=1e-3,
    max_iter=80,
    maxfun=0,
    energy_plateau_tol=1e-5,
    energy_plateau_patience=30,
    ads_z_perturb=0.0,
    bounds_from_ref=True,
    on_step=None,
):
    """High-level API: relax adsorbate on a surface structure.

    Args:
        atoms: ASE Atoms object with atoms.info['data']['ads_idx']
        model: Trained AdsNet model (or MLCalculator)
        method: "pgd" or "lbfgsb"
        max_step_xy: Bounds in xy plane
        max_step_z: Bounds in z direction for adsorbate
        force_tol: Force convergence criterion
        max_iter: Max optimization steps
        maxfun: Max model evaluations (0 = unlimited)
        energy_plateau_tol: Plateau detection threshold
        energy_plateau_patience: Steps for plateau
        ads_z_perturb: Optional z-perturbation before relaxation
        bounds_from_ref: Use unperturbed structure as bounds anchor
        on_step: Optional progress callback(energy, x)

    Returns:
        dict with: success, atoms (relaxed), energy, trajectory, n_evals, message
    """
    from adsnet.calculators import MLCalculator

    ads_idx = atoms.info['data']['ads_idx']
    atoms_ref = copy.deepcopy(atoms)

    if ads_z_perturb:
        for j in ads_idx:
            atoms[j].z += ads_z_perturb

    if hasattr(model, 'calculate'):
        calc = model
    else:
        calc = MLCalculator(model)

    atoms_work = copy.deepcopy(atoms)
    atoms_work.calc = calc
    _ = atoms_work.get_potential_energy()
    local_idx = np.asarray(atoms_work.calc.results['local_idx'], dtype=int)

    ads_pos = atoms_work.positions[local_idx].flatten()
    if bounds_from_ref:
        bounds_anchor = atoms_ref.positions[local_idx].flatten()
    else:
        bounds_anchor = ads_pos

    traj = []
    n_evals = [0]
    n_steps = [0]
    cache = {}

    fun_grad = make_fun_and_grad(atoms_work, local_idx, cache=cache, n_evals=n_evals)

    def record_and_callback(energy, x):
        if np.isfinite(energy) and energy <= 50:
            n_steps[0] += 1
            traj.append(copy.deepcopy(atoms_work))
            print(f'step {n_steps[0]}: binding energy = {energy:.6f} eV')
        if on_step is not None:
            on_step(energy, x)

    bounds = build_local_bounds(bounds_anchor, local_idx, ads_idx, max_step_xy, max_step_z)

    result = optimize_local_coords(
        ads_pos,
        fun_grad,
        bounds,
        method=method,
        force_tol=force_tol,
        max_iter=max_iter,
        maxfun=maxfun,
        energy_plateau_tol=energy_plateau_tol,
        energy_plateau_patience=energy_plateau_patience,
        on_step=record_and_callback,
        n_evals=n_evals,
    )

    if isinstance(result, dict):
        atoms_work.positions[local_idx] = result["x"].reshape(-1, 3)
        return {
            "success": result["success"],
            "atoms": atoms_work,
            "energy": result["fun"],
            "trajectory": traj,
            "n_steps": n_steps[0],
            "n_evals": n_evals[0],
            "message": result["message"],
        }
    else:
        atoms_work.positions[local_idx] = result.x.reshape(-1, 3)
        return {
            "success": result.success,
            "atoms": atoms_work,
            "energy": result.fun,
            "trajectory": traj,
            "n_steps": n_steps[0],
            "n_evals": n_evals[0],
            "message": str(result.message),
        }
