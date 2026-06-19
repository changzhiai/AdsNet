Relaxation Guide
================

This guide explains how to use trained AdsNet models for structure relaxation of adsorbates on surfaces.


Basic Usage
-----------

The high-level ``relax_structure`` function performs constrained local optimization:

.. code-block:: python

   from adsnet.tools import relax_structure
   from adsnet.models import AdsNet
   from ase.io import read
   import torch

   # Load model
   state_dict = torch.load("output/model.pt", map_location='cpu')
   model = AdsNet(
       num_layers=state_dict["num_layers"],
       num_channels=state_dict["num_channels"],
       cutoff=state_dict["cutoff"],
       compute_forces=True,
       rbf_type=state_dict.get("rbf_type", "sinc"),
       edge_embed=state_dict.get("edge_embed", "linear"),
   )
   model.load_state_dict(state_dict["model"])

   # Load structure (must have atoms.info['data']['ads_idx'])
   atoms = read("structure.traj")

   # Relax
   result = relax_structure(atoms, model, force_tol=1e-3)

   print(f"Converged: {result['success']}")
   print(f"Energy: {result['energy']:.6f} eV")
   print(f"Steps: {result['n_steps']}, Evaluations: {result['n_evals']}")


Using the MLCalculator
----------------------

You can also pass an ``MLCalculator`` instance directly:

.. code-block:: python

   from adsnet.calculators import MLCalculator
   from adsnet.tools import relax_structure
   from ase.io import read

   atoms = read("structure.traj")
   calc = MLCalculator("output/model.pt")

   result = relax_structure(atoms, calc, fmax=0.05)


Optimization Methods
--------------------

Two optimization methods are available:

**Projected Gradient Descent (PGD)** (default):

Performs a line search over multiple step sizes with box constraints. Falls back to coordinate-wise search when stuck.

.. code-block:: python

   result = relax_structure(
       atoms, model,
       method="pgd",
       max_iter=80,
       force_tol=1e-3,
   )

**L-BFGS-B**:

Uses scipy's L-BFGS-B optimizer with box bounds:

.. code-block:: python

   result = relax_structure(
       atoms, model,
       method="lbfgsb",
       maxfun=80,
   )


Controlling Bounds
------------------

The relaxation constrains atomic displacements within a box around the initial positions:

- ``max_step_xy``: Maximum displacement in x and y directions (default: 0.15 A)
- ``max_step_z``: Maximum displacement in z direction for adsorbate atoms (default: 0.55 A)

.. code-block:: python

   result = relax_structure(
       atoms, model,
       max_step_xy=0.2,
       max_step_z=0.8,
   )


Convergence Criteria
--------------------

Relaxation terminates when any of these conditions is met:

1. **Force convergence**: Maximum gradient component drops below ``force_tol``
2. **Energy plateau**: Energy change stays below ``energy_plateau_tol`` for ``energy_plateau_patience`` consecutive steps
3. **Evaluation budget**: Number of model evaluations reaches ``maxfun`` (if > 0)
4. **Max iterations**: Number of optimization steps reaches ``max_iter``

.. code-block:: python

   result = relax_structure(
       atoms, model,
       force_tol=1e-3,
       max_iter=80,
       maxfun=80,
       energy_plateau_tol=1e-5,
       energy_plateau_patience=30,
   )


Full Example
------------

A complete relaxation workflow with trajectory output:

.. code-block:: python

   import os
   import torch
   from ase.io import read, write
   from adsnet.models import AdsNet
   from adsnet.tools import relax_structure

   # Load model
   state_dict = torch.load("output/model.pt", map_location='cpu')
   model = AdsNet(
       num_layers=state_dict["num_layers"],
       num_channels=state_dict["num_channels"],
       cutoff=state_dict["cutoff"],
       compute_forces=True,
       rbf_type=state_dict.get("rbf_type", "sinc"),
       edge_embed=state_dict.get("edge_embed", "linear"),
   )
   model.load_state_dict(state_dict["model"])

   # Load and relax
   atoms = read("dataset/total.traj", index=2000)

   result = relax_structure(
       atoms, model,
       method="pgd",
       max_step_xy=0.15,
       max_step_z=0.55,
       force_tol=1e-3,
       max_iter=80,
       maxfun=80,
       energy_plateau_tol=1e-5,
       energy_plateau_patience=30,
   )

   print(f"success={result['success']}")
   print(f"energy={result['energy']:.6f} eV")
   print(f"n_evals={result['n_evals']}")
   print(f"message={result['message']}")

   # Save trajectory
   if result['trajectory']:
       os.makedirs("relax_output", exist_ok=True)
       write("relax_output/relax.traj", result['trajectory'])


Return Value
------------

``relax_structure`` returns a dictionary with:

- ``success`` (bool): Whether convergence was achieved
- ``atoms`` (Atoms): The relaxed structure
- ``energy`` (float): Final adsorption energy in eV
- ``trajectory`` (list): List of Atoms snapshots at each improvement step
- ``n_steps`` (int): Number of improvement steps taken
- ``n_evals`` (int): Total number of model evaluations
- ``message`` (str): Description of termination reason


SLURM Job Script
----------------

Example SLURM submission script for relaxation on a GPU node:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=adsnet-relax
   #SBATCH --partition=gpu
   #SBATCH --nodes=1
   #SBATCH --ntasks=1
   #SBATCH --gpus-per-node=1
   #SBATCH --cpus-per-task=4
   #SBATCH --mem=16G
   #SBATCH --time=4:00:00
   #SBATCH --output=relax_%j.out

   module load cuda pytorch

   python relax.py

For relaxing many structures in parallel on CPU (useful when GPU is not needed):

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=adsnet-relax-batch
   #SBATCH --partition=normal
   #SBATCH --nodes=1
   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=16
   #SBATCH --mem=64G
   #SBATCH --time=8:00:00
   #SBATCH --output=relax_batch_%j.out

   module load pytorch

   python relax_batch.py

Where ``relax_batch.py`` uses multiprocessing to relax structures in parallel:

.. code-block:: python

   import torch
   from concurrent.futures import ProcessPoolExecutor
   from ase.io import read, write
   from adsnet.models import AdsNet
   from adsnet.tools import relax_structure

   def relax_one(idx):
       state_dict = torch.load("output/model.pt", map_location='cpu')
       model = AdsNet(
           num_layers=state_dict["num_layers"],
           num_channels=state_dict["num_channels"],
           cutoff=state_dict["cutoff"],
           compute_forces=True,
           rbf_type=state_dict.get("rbf_type", "sinc"),
           edge_embed=state_dict.get("edge_embed", "linear"),
       )
       model.load_state_dict(state_dict["model"])
       model.eval()

       atoms = read("structures.traj", index=idx)
       result = relax_structure(atoms, model, force_tol=1e-3, max_iter=80)
       return result['atoms'], result['energy'], result['success']

   if __name__ == "__main__":
       images = read("structures.traj", ":")
       n_workers = 16

       with ProcessPoolExecutor(max_workers=n_workers) as executor:
           results = list(executor.map(relax_one, range(len(images))))

       relaxed = [r[0] for r in results]
       write("relaxed_all.traj", relaxed)
       print(f"Relaxed {len(relaxed)} structures")


For more details on the API, see the :doc:`api` reference.
