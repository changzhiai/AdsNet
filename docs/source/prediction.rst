Prediction Guide
================

This guide explains how to use trained AdsNet models for making predictions.


Using the ML Calculator
-----------------------

The ``MLCalculator`` provides a convenient ASE calculator interface:

.. code-block:: python

   from adsnet.calculators import MLCalculator
   from ase.io import read

   # Create calculator with trained model
   calc = MLCalculator("output/model.pt")

   # Read structures
   images = read("test_structures.traj", ":")

   # Make predictions
   for atoms in images:
       atoms.calc = calc
       energy = atoms.get_potential_energy()
       print(f"Energy: {energy} eV")

The calculator automatically:

* Loads the model architecture from the saved state dict
* Moves the model to the appropriate device
* Handles batching and graph construction


Batch Prediction
----------------

For efficient batch prediction over a dataset:

.. code-block:: python

   import torch
   from adsnet.models import AdsNet
   from adsnet.data import AseData, collate_atomsdata
   from torch.utils.data import DataLoader

   state_dict = torch.load("output/model.pt", map_location='cpu')
   model = AdsNet(
       num_layers=state_dict["num_layers"],
       num_channels=state_dict["num_channels"],
       cutoff=state_dict["cutoff"],
   )
   model.load_state_dict(state_dict["model"])
   model.eval()

   dataset = AseData("test_structures.traj")
   dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_atomsdata)

   with torch.no_grad():
       for batch in dataloader:
           output = model(batch)
           energies = output['ads_energy']


SLURM Job Script
----------------

Example SLURM submission script for batch prediction on a GPU node:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=adsnet-predict
   #SBATCH --partition=gpu
   #SBATCH --nodes=1
   #SBATCH --ntasks=1
   #SBATCH --gpus-per-node=1
   #SBATCH --cpus-per-task=4
   #SBATCH --mem=16G
   #SBATCH --time=2:00:00
   #SBATCH --output=predict_%j.out

   module load cuda pytorch

   python predict.py

Where ``predict.py`` contains:

.. code-block:: python

   import torch
   from adsnet.models import AdsNet
   from adsnet.data import AseData, collate_atomsdata
   from torch.utils.data import DataLoader

   state_dict = torch.load("output/model.pt", map_location='cuda')
   model = AdsNet(
       num_layers=state_dict["num_layers"],
       num_channels=state_dict["num_channels"],
       cutoff=state_dict["cutoff"],
       compute_forces=True,
       rbf_type=state_dict.get("rbf_type", "sinc"),
       edge_embed=state_dict.get("edge_embed", "linear"),
   )
   model.load_state_dict(state_dict["model"])
   model.to("cuda")
   model.eval()

   dataset = AseData("test_structures.traj", cutoff=5.5, compute_forces=True)
   dataloader = DataLoader(dataset, batch_size=64, num_workers=4,
                           collate_fn=collate_atomsdata, pin_memory=True)

   with torch.no_grad():
       for batch in dataloader:
           batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
           output = model(batch)
           energies = output['ads_energy'].cpu().numpy()
           for e in energies:
               print(f"{e:.6f}")


For structure relaxation, see the :doc:`relaxation` guide. For more details on the API, see the :doc:`api` reference.
