Parallelization
===============

This guide covers strategies for scaling AdsNet workflows across multiple GPUs and CPUs.


Distributed Training (Multi-GPU)
---------------------------------

AdsNet supports distributed data-parallel (DDP) training across multiple GPUs using ``torchrun``.

**Setup:**

.. code-block:: python

   # train_distributed.py
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
           "output_dir": "output",
           "output_model": "model.pt",
       },
       distributed=True,
   )
   trainer.train("dataset.traj")

**Launch with torchrun:**

.. code-block:: bash

   # Single node, 4 GPUs
   torchrun --nproc_per_node=4 train_distributed.py

   # Multi-node (2 nodes, 4 GPUs each)
   torchrun --nnodes=2 --nproc_per_node=4 \
       --rdzv_backend=c10d --rdzv_endpoint=HOST:PORT \
       train_distributed.py

Key considerations:

* The effective batch size scales with the number of GPUs (``batch_size × nproc``)
* Consider reducing ``initial_lr`` or using a learning rate warmup when scaling to many GPUs
* Each process loads a full copy of the dataset; ensure sufficient memory


Parallel Prediction
-------------------

For predicting over many structures, use batch processing with DataLoader:

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
       compute_forces=True,
       rbf_type=state_dict.get("rbf_type", "sinc"),
       edge_embed=state_dict.get("edge_embed", "linear"),
   )
   model.load_state_dict(state_dict["model"])
   model.eval()
   model.to("cuda")

   dataset = AseData("structures.traj", cutoff=5.5, compute_forces=True)
   dataloader = DataLoader(
       dataset,
       batch_size=64,
       num_workers=4,           # parallel data loading
       collate_fn=collate_atomsdata,
       pin_memory=True,
   )

   results = []
   with torch.no_grad():
       for batch in dataloader:
           batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
           output = model(batch)
           results.append(output['ads_energy'].cpu())

   all_energies = torch.cat(results)

Increasing ``num_workers`` in the DataLoader speeds up graph construction (neighbor list computation), which can be a bottleneck for large structures.


Parallel Relaxation
-------------------

To relax multiple structures in parallel, use Python's ``multiprocessing`` or ``concurrent.futures``:

.. code-block:: python

   import torch
   from concurrent.futures import ProcessPoolExecutor
   from ase.io import read
   from adsnet.models import AdsNet
   from adsnet.tools import relax_structure

   def relax_one(args):
       model_path, atoms_idx, traj_path = args

       state_dict = torch.load(model_path, map_location='cpu')
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

       atoms = read(traj_path, index=atoms_idx)
       result = relax_structure(atoms, model, force_tol=1e-3, max_iter=80)
       return atoms_idx, result['energy'], result['success']

   # Relax structures in parallel
   images = read("structures.traj", ":")
   tasks = [("output/model.pt", i, "structures.traj") for i in range(len(images))]

   with ProcessPoolExecutor(max_workers=8) as executor:
       results = list(executor.map(relax_one, tasks))

   for idx, energy, success in results:
       print(f"Structure {idx}: energy={energy:.4f} eV, converged={success}")

.. note::

   Each worker process loads its own copy of the model. For GPU-based relaxation, manage device assignment carefully to avoid memory contention.


GPU-Parallel Relaxation
-----------------------

When multiple GPUs are available, assign each worker to a separate device:

.. code-block:: python

   import torch
   from concurrent.futures import ProcessPoolExecutor
   from ase.io import read
   from adsnet.models import AdsNet
   from adsnet.tools import relax_structure

   def relax_on_gpu(args):
       model_path, atoms_idx, traj_path, gpu_id = args

       device = torch.device(f"cuda:{gpu_id}")
       state_dict = torch.load(model_path, map_location=device)
       model = AdsNet(
           num_layers=state_dict["num_layers"],
           num_channels=state_dict["num_channels"],
           cutoff=state_dict["cutoff"],
           compute_forces=True,
           rbf_type=state_dict.get("rbf_type", "sinc"),
           edge_embed=state_dict.get("edge_embed", "linear"),
       )
       model.load_state_dict(state_dict["model"])
       model.to(device)
       model.eval()

       atoms = read(traj_path, index=atoms_idx)
       result = relax_structure(atoms, model, force_tol=1e-3, max_iter=80)
       return atoms_idx, result['energy'], result['success']

   # Distribute across 4 GPUs
   n_gpus = torch.cuda.device_count()
   images = read("structures.traj", ":")
   tasks = [
       ("output/model.pt", i, "structures.traj", i % n_gpus)
       for i in range(len(images))
   ]

   with ProcessPoolExecutor(max_workers=n_gpus) as executor:
       results = list(executor.map(relax_on_gpu, tasks))


HPC Job Scripts
---------------

**SLURM example** (multi-GPU training):

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=adsnet-train
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=4
   #SBATCH --gres=gpu:4
   #SBATCH --time=24:00:00

   module load cuda pytorch

   torchrun --nproc_per_node=4 train_distributed.py

**SLURM example** (parallel relaxation across nodes):

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=adsnet-relax
   #SBATCH --nodes=1
   #SBATCH --ntasks=16
   #SBATCH --cpus-per-task=2
   #SBATCH --time=4:00:00

   python parallel_relax.py --workers 16


Performance Tips
----------------

* **Data loading**: Set ``num_workers > 0`` in DataLoader to overlap data preparation with GPU computation
* **Pin memory**: Use ``pin_memory=True`` for faster CPU-to-GPU transfers
* **Batch size**: Larger batches improve GPU utilization but require more memory
* **Mixed precision**: For large models, consider ``torch.cuda.amp`` to reduce memory and speed up training
* **Cutoff radius**: Smaller cutoff means fewer edges per graph, reducing computation per structure
