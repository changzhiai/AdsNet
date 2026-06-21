# AdsNet (Adsorption Network)

[![Docs](https://img.shields.io/badge/Docs-available-blue)](https://adsnet.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Installation](#2-installation)
- [3. Quickstart: Examples](#3-quickstart-examples)
- [4. Training](#4-training)
- [5. Prediction](#5-prediction)
- [6. Foundation Models](#6-foundation-models)
- [7. Relaxation](#7-relaxation)
- [8. Parallelization](#8-parallelization)
- [9. Modules](#9-modules)
- [Troubleshooting](#troubleshooting)
- [Issues](#issues)
- [Maintainer](#maintainer)
- [Citation](#citation)

## 1. Introduction

AdsNet is an equivariant graph neural network framework for predicting adsorption energies on metal surfaces. It operates on a local adsorbate graph — containing only the adsorbate atoms and their neighboring surface atoms within a cutoff radius — to predict binding energies and local forces in a single forward pass. This local graph formulation decouples computational cost from total system size and encodes continuous geometric information (interatomic distances and directional vectors), enabling sensitivity to subtle variations in local bonding environments.

Key features:

- **Equivariant GNN architecture** — message passing with scalar and vector channels
- **Local interaction model** — computational cost independent of total system size
- **Multi-task learning** — jointly predicts adsorption energies and local forces
- **ASE calculator interface** — seamless integration with the Atomic Simulation Environment
- **Structure relaxation** — projected gradient descent optimizer with box constraints
- **Foundation model support** — pretrain on diverse datasets and fine-tune on specific systems
- **Distributed training** — multi-GPU training via PyTorch DDP

### Documentation

Full documentation: [https://adsnet.readthedocs.io](https://adsnet.readthedocs.io)

## 2. Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 1.12
- ASE >= 3.22
- asap3
- scipy >= 1.7

### Installing AdsNet

```bash
# Clone the repository
git clone https://github.com/changzhiai/AdsNet.git
cd AdsNet

# Install with pip
pip install -e .
```

### GPU Support

For GPU-accelerated training, install PyTorch with CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 3. Quickstart: Examples

Example training and prediction scripts are available in the `proj/` directory:

```
proj/
├── proj_metals/    # Multi-metal training and relaxation
│   ├── train.py
│   ├── predict.py
│   └── relax.py
└── proj_Pt/        # Single-metal (Pt) training
    ├── train.py
    ├── predict.py
    └── relax.py
```

## 4. Training

### Preparing your dataset

AdsNet uses ASE trajectory (`.traj`) files. Each structure must have `atoms.info['data']` containing:

- `ads_idx`: list of adsorbate atom indices
- `ads_energy`: adsorption energy (eV)
- `delta_forces`: force labels (optional, recommended)

### Training

```python
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
)
trainer.train("dataset.traj")
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_channels` | 128 | Feature dimension |
| `num_layers` | 3 | Message passing layers |
| `cutoff` | 5.5 | Neighbor cutoff (A) |
| `rbf_type` | "sinc" | Radial basis type ("sinc" or "gaussian") |
| `edge_embed` | "linear" | Edge embedding ("linear" or "mlp") |
| `batch_size` | 12 | Training batch size |
| `initial_lr` | 1e-4 | Learning rate |
| `max_steps` | 1000000 | Max training steps |
| `forces_weight` | 0.05 | Weight for forces in loss |
| `stop_patience` | 200 | Early stopping patience |
| `device` | None | "cpu" or "cuda" (auto-detected if None) |

## 5. Prediction

### Making Predictions with ASE calculator

```python
import torch
from adsnet.models import AdsNet
from adsnet.calculators import MLCalculator
from ase.io import read

# Load model
state_dict = torch.load("output/model.pt", map_location="cpu")
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

# Predict binding energy
calc = MLCalculator(model)
atoms = read("structure.traj")
atoms.calc = calc
energy = atoms.get_potential_energy()
```

### Calculator API

```python
calc = MLCalculator(model)

# Binding energy
energy = calc.get_binding_energy(atoms)

# Per-atom energy contributions
local_idx, local_energies = calc.get_be_distributions(atoms)

# Local forces (requires compute_forces=True)
local_idx, forces = calc.get_local_forces(atoms)
```

## 6. Foundation Models

### Fine-tuning Foundation Models

```python
from adsnet.trainer import Trainer

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
        "output_dir": "output_finetune",
        "output_model": "model.pt",
    },
)
trainer.train("target_dataset.traj")
```

## 7. Relaxation

```python
from adsnet.tools import relax_structure

result = relax_structure(
    atoms,
    model,
    method="pgd",
    max_step_xy=0.15,
    max_step_z=0.55,
    force_tol=1e-3,
    max_iter=80,
    maxfun=80,
)
print(f"Relaxed energy: {result['energy']:.4f} eV")
print(f"Converged: {result['success']}")
print(f"Message: {result['message']}")
```

## 8. Parallelization

### Multi-GPU Training

```python
trainer = Trainer(
    config={"device": "cuda", "output_dir": "output"},
    distributed=True,
)
trainer.train("dataset.traj")
```

Launch with torchrun:

```bash
torchrun --nproc_per_node=4 train.py
```

### SLURM Example

```bash
#!/bin/bash
#SBATCH --job-name=adsnet-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00

module load cuda pytorch

torchrun --nproc_per_node=4 train.py
```

## 9. Modules

### adsnet.models

The `AdsNet` model implementing equivariant message passing with scalar and vector node features, sinc/Gaussian RBF, and configurable edge embeddings.

### adsnet.data

`AseData` dataset class and `collate_atomsdata` collation function for graph construction from ASE trajectory files.

### adsnet.calculators

`MLCalculator` and `EnsembleCalculator` providing ASE calculator interfaces for energy prediction, force computation, and uncertainty quantification.

### adsnet.trainer

`Trainer` class for single-GPU and distributed training with early stopping, learning rate scheduling, and checkpoint management.

### adsnet.tools

Structure relaxation utilities: `relax_structure`, `optimize_local_coords`, `build_local_bounds`, `make_fun_and_grad`.

## Troubleshooting

See the [Troubleshooting Guide](https://adsnet.readthedocs.io/en/latest/troubleshooting.html) for common issues and solutions.

## Issues

For questions, issues, and contributions, please use the [GitHub issue tracker](https://github.com/changzhiai/AdsNet/issues).

## Maintainer

Dr. Changzhi Ai (changzhi@stanford.edu)
SUNCAT Center for Interface Science and Catalysis, Stanford University and SLAC National Accelerator Laboratory.

## Citation

If you use AdsNet in your research, please cite:

```
@article{ai2025adsnet,
  title={AdsNet: A Graph Neural Network for Adsorption Energy Prediction},
  author={Ai, Changzhi},
  year={2025},
}
```
