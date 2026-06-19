Training Guide
==============

This guide covers how to train AdsNet models for adsorption energy prediction.

Preparing Your Dataset
----------------------

AdsNet works with ASE trajectory (.traj) or database (.db) files. Your data should contain:

* Atomic structures (positions, atomic numbers, cell)
* Adsorption energy labels
* Force labels (optional, but recommended)


Running Training
----------------

Here is a simple example ``train.py`` of how to run training:

.. code-block:: python

   from adsnet.trainer import Trainer

   trainer = Trainer(
       config={
           'device': 'cuda',
           'output_dir': 'output',
           'output_log': 'output.log',
           'output_model': 'model.pt',
       },
       distributed=False,
   )

   trainer.train("dataset.traj")

Available configurations for ``config``:

.. code-block:: python

   config = {
       # Model parameters
       "num_channels": 128,     # feature dimension
       "num_layers": 3,         # number of interaction blocks
       "cutoff": 5.5,           # cutoff radius in Angstroms
       "num_basis": 20,         # number of radial basis functions
       # Training parameters
       "device": None,          # 'cpu' or 'cuda'
       "batch_size": 12,
       "initial_lr": 0.0001,
       "max_steps": 1000000,
       "forces_weight": 0.05,   # weight for force loss (0 to disable)
       # Scheduler
       "plateau_scheduler": False,
       "lr_decay_factor": 0.5,
       "lr_patience": 10,
       # Early stopping
       "stop_patience": 200,
       # Data
       "val_ratio": 0.1,
       "random_seed": 3407,
       "load_model": None,      # path to checkpoint to resume from
       # Output
       "output_dir": "output",
       "output_log": "output.log",
       "output_model": "model.pt",
       "log_interval": 10,
   }


Resuming Training
-----------------

To resume training from a checkpoint:

.. code-block:: python

   trainer = Trainer(
       config={
           'load_model': 'output/model.pt',
           'output_dir': 'output',
       },
   )
   trainer.train("dataset.traj")


Multi-GPU Training
------------------

AdsNet supports distributed data-parallel training across multiple GPUs:

.. code-block:: bash

   torchrun --nproc_per_node=4 train.py

Set ``distributed=True`` in the ``Trainer``:

.. code-block:: python

   trainer = Trainer(
       config={'device': 'cuda', 'output_dir': 'output'},
       distributed=True,
   )
   trainer.train("dataset.traj")


SLURM Job Script
----------------

Example SLURM submission script for training on a GPU cluster:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=adsnet-train
   #SBATCH --partition=gpu
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=1
   #SBATCH --gpus-per-node=1
   #SBATCH --cpus-per-task=4
   #SBATCH --mem=32G
   #SBATCH --time=24:00:00
   #SBATCH --output=train_%j.out

   module load cuda pytorch

   python train.py

For multi-GPU training:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=adsnet-train-ddp
   #SBATCH --partition=gpu
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=4
   #SBATCH --gpus-per-node=4
   #SBATCH --cpus-per-task=4
   #SBATCH --mem=128G
   #SBATCH --time=48:00:00
   #SBATCH --output=train_ddp_%j.out

   module load cuda pytorch

   torchrun --nproc_per_node=4 train.py
