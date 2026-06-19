Troubleshooting
===============

This page covers common issues and their solutions when using AdsNet.


Installation Issues
-------------------

**PyTorch CUDA version mismatch**

If you get CUDA-related errors, ensure your PyTorch installation matches your CUDA driver version:

.. code-block:: bash

   python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"

Reinstall PyTorch with the correct CUDA version from https://pytorch.org/get-started/locally/.


Data Issues
-----------

**"Atoms info error: ... Please add ads. data into atoms.info!"**

Each ``Atoms`` object must have ``atoms.info['data']`` with at least ``ads_idx`` and ``ads_energy``:

.. code-block:: python

   atoms.info['data'] = {
       'ads_idx': [36, 37],
       'ads_energy': -1.23,
       'delta_forces': forces_array,  # optional
   }

**"Error collating key ..."**

This typically means inconsistent tensor shapes across structures in a batch. Check that all structures in your trajectory have consistent metadata keys and shapes.

**"No ads_idx_pairs found!"**

The neighbor list found no neighbors within the cutoff radius for the adsorbate atoms. Possible causes:

* The ``cutoff`` is too small — try increasing it (default: 5.5 A)
* The ``ads_idx`` points to isolated atoms far from the surface
* The structure has incorrect positions or cell


Training Issues
---------------

**NaN in loss during training**

"NaN detected in evaluation metrics" indicates numerical instability. Try:

* Reducing the learning rate (e.g., ``initial_lr: 1e-5``)
* Checking your dataset for outlier energies or unphysical structures
* Reducing ``forces_weight`` if force labels are noisy

**Training does not converge**

* Ensure your dataset is large enough (at least a few hundred structures)
* Try a smaller batch size for more frequent updates
* Verify that adsorption energy labels are in consistent units (eV)
* Check that ``ads_idx`` correctly identifies adsorbate atoms

**Out of memory (OOM) on GPU**

* Reduce ``batch_size``
* Use a smaller model (fewer ``num_channels`` or ``num_layers``)
* Reduce the cutoff radius to decrease graph size


Prediction Issues
-----------------

**Model architecture mismatch when loading checkpoint**

The model parameters (``num_channels``, ``num_layers``, ``cutoff``) must match the checkpoint:

.. code-block:: python

   state_dict = torch.load("model.pt", map_location='cpu')
   model = AdsNet(
       num_layers=state_dict["num_layers"],
       num_channels=state_dict["num_channels"],
       cutoff=state_dict["cutoff"],
       compute_forces=True,
       rbf_type=state_dict.get("rbf_type", "sinc"),
       edge_embed=state_dict.get("edge_embed", "linear"),
   )
   model.load_state_dict(state_dict["model"])

**Unreasonable energy predictions**

* Verify the input structure has correct ``atoms.info['data']['ads_idx']``
* Ensure the structure is similar to the training data distribution
* Check that the cell and periodic boundary conditions are set correctly


Relaxation Issues
-----------------

**Relaxation returns "NO_PROGRESS"**

The optimizer cannot reduce the energy further. Possible fixes:

* Increase ``max_step_xy`` or ``max_step_z`` to allow larger displacements
* Increase ``force_tol`` if the structure is already near a minimum
* Try ``method="lbfgsb"`` as an alternative optimizer

**Relaxation returns "MAX_ITER"**

The optimization did not converge within the iteration limit:

* Increase ``max_iter`` or ``maxfun``
* The potential energy surface may be very flat — try increasing ``energy_plateau_patience``
* Lower ``force_tol`` expectations for flat PES regions

**Energy diverges during relaxation (returns inf)**

The structure moved to an unphysical configuration:

* Reduce ``max_step_xy`` and ``max_step_z`` for tighter constraints
* Check that the initial structure is reasonable
* The model may be extrapolating outside its training domain


Multi-GPU Issues
----------------

**Distributed training hangs or crashes**

* Ensure all GPUs are visible: ``CUDA_VISIBLE_DEVICES=0,1,2,3``
* Use ``torchrun`` (not ``python``) to launch distributed training
* Check that ``distributed=True`` is set in the Trainer

.. code-block:: bash

   torchrun --nproc_per_node=4 train.py


Getting Help
------------

If your issue is not covered here:

* Check the `GitHub issues <https://github.com/changzhiai/AdsNet/issues>`_ for known problems
* Open a new issue with a minimal reproducible example
