Foundation Model
================

This guide describes the foundation model approach with AdsNet: pretraining on a large, diverse dataset and fine-tuning on specific target systems.


Overview
--------

The foundation model workflow follows two stages:

1. **Pretraining**: Train on a large multi-metal, multi-adsorbate dataset to learn general surface chemistry representations.
2. **Fine-tuning**: Adapt the pretrained model to a specific system (e.g., a particular metal or adsorbate) using a smaller, targeted dataset.

This transfer learning approach enables:

* Better performance on small datasets by leveraging knowledge from diverse systems
* Faster convergence when training on new surface/adsorbate combinations
* More robust models that generalize across similar chemical environments


Pretraining
-----------

Pretrain on a large dataset covering multiple metals, facets, and adsorbates:

.. code-block:: python

   from adsnet.trainer import Trainer

   trainer = Trainer(
       config={
           "device": "cuda",
           "num_channels": 128,
           "num_layers": 3,
           "cutoff": 5.5,
           "batch_size": 12,
           "initial_lr": 1e-4,
           "max_steps": 1000000,
           "forces_weight": 0.90,
           "stop_patience": 800,
           "output_dir": "output_foundation",
           "output_model": "foundation.pt",
       },
       distributed=False,
   )

   trainer.train("large_dataset.traj")

For large-scale pretraining, use multi-GPU training:

.. code-block:: bash

   torchrun --nproc_per_node=4 pretrain.py


Fine-tuning
-----------

Fine-tune the pretrained foundation model on a target system using ``load_model``:

.. code-block:: python

   from adsnet.trainer import Trainer

   trainer = Trainer(
       config={
           "device": "cuda",
           "num_channels": 128,
           "num_layers": 3,
           "cutoff": 5.5,
           "batch_size": 12,
           "initial_lr": 1e-5,        # lower learning rate for fine-tuning
           "max_steps": 200000,
           "forces_weight": 0.90,
           "stop_patience": 200,
           "load_model": "output_foundation/foundation.pt",
           "output_dir": "output_Pt",
           "output_model": "model.pt",
       },
       distributed=False,
   )

   trainer.train("Pt_dataset.traj")

Key considerations for fine-tuning:

* Use a **lower learning rate** (e.g., 1e-5) to avoid catastrophic forgetting
* The model architecture (``num_channels``, ``num_layers``, ``cutoff``) must match the pretrained checkpoint
* A smaller ``stop_patience`` is often sufficient since the model starts from a good initialization


Fine-tuning Strategies
----------------------

**Full fine-tuning**: All model parameters are updated (default behavior). Best when the target dataset is moderately sized.

.. code-block:: python

   config = {
       "load_model": "output_foundation/foundation.pt",
       "initial_lr": 1e-5,
       "max_steps": 200000,
   }

**Warm restart with scheduler**: Use a plateau scheduler to reduce the learning rate as the fine-tuning converges:

.. code-block:: python

   config = {
       "load_model": "output_foundation/foundation.pt",
       "initial_lr": 5e-5,
       "plateau_scheduler": True,
       "lr_decay_factor": 0.5,
       "lr_patience": 10,
   }


Dataset Guidelines
------------------

**Pretraining dataset**:

* Cover diverse metals (e.g., Pt, Pd, Cu, Ag, Au, Ni, Co, Fe, ...)
* Include multiple facets (111, 100, 110, stepped surfaces)
* Include various adsorbates (CO, OH, O, H, N, CHx, ...)
* Larger is generally better; thousands to hundreds of thousands of structures

**Fine-tuning dataset**:

* Focus on the target system of interest
* Can be much smaller (hundreds to thousands of structures)
* Should include representative configurations (different sites, coverages)


Example: Multi-metal to Single-metal Transfer
----------------------------------------------

.. code-block:: python

   import os
   from adsnet.trainer import Trainer

   # Stage 1: Pretrain on all metals
   trainer = Trainer(
       config={
           "device": "cuda",
           "num_channels": 128,
           "num_layers": 3,
           "cutoff": 5.5,
           "batch_size": 12,
           "initial_lr": 1e-4,
           "max_steps": 1000000,
           "forces_weight": 0.90,
           "stop_patience": 800,
           "output_dir": "output_foundation",
           "output_model": "foundation.pt",
       },
   )
   trainer.train("all_metals.traj")

   # Stage 2: Fine-tune on Pt only
   trainer = Trainer(
       config={
           "device": "cuda",
           "num_channels": 128,
           "num_layers": 3,
           "cutoff": 5.5,
           "batch_size": 12,
           "initial_lr": 1e-5,
           "max_steps": 200000,
           "forces_weight": 0.90,
           "stop_patience": 200,
           "load_model": "output_foundation/foundation.pt",
           "output_dir": "output_Pt",
           "output_model": "model.pt",
       },
   )
   trainer.train("Pt_dataset.traj")
