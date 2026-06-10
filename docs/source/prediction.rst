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


Applications: Structure Relaxation
-----------------------------------

The ASE optimizers like ``BFGS`` can be used to relax a structure:

.. code-block:: python

   from adsnet.calculators import MLCalculator
   from adsnet.tools import relax_structure
   from ase.io import read

   atoms = read("atoms.traj")
   calc = MLCalculator("output/model.pt")

   result = relax_structure(atoms, calc, fmax=0.05)


For more details on the API, see the :doc:`api` reference.
