# AdsNet

**Ads**orption **Net**work: A message passing graph neural network for adsorption energy prediction.

## Introduction

AdsNet is a machine learning framework for predicting adsorption properties using graph neural networks. It uses efficient local interactions via local neighborlists to predict binding energies and local forces for adsorbate-surface systems.

## Installation

```bash
git clone https://github.com/changzhiai/AdsNet.git
cd AdsNet
pip install -e .
```

Requirements: Python >= 3.8, PyTorch >= 1.12, ASE >= 3.22, asap3, scipy

## Package Structure

```
adsnet/
├── models/        # AdsNet model architecture
├── data/          # Dataset and collation
├── calculators/   # ASE calculator interface
├── trainer/       # Training loop
└── tools/         # Relaxation utilities
```

## Quick Start

### Training

```python
from adsnet.trainer import Trainer

trainer = Trainer(
    config={
        "node_size": 128,
        "num_interactions": 3,
        "cutoff": 5.5,
        "batch_size": 12,
        "max_steps": 1000000,
        "forces_weight": 0.05,
        "device": "cuda",
        "output_dir": "model_output",
    },
    distributed=False
)
trainer.train("path/to/data.traj")
```

### Prediction

```python
import torch
from ase.io import read
from adsnet.models import AdsNet
from adsnet.calculators import MLCalculator

# Load model
state_dict = torch.load("model_output/best_model.pth", map_location="cpu")
model = AdsNet(
    num_interactions=state_dict["num_layer"],
    hidden_state_size=state_dict["node_size"],
    cutoff=state_dict["cutoff"],
)
model.load_state_dict(state_dict["model"])
model.eval()

# Predict binding energy
calc = MLCalculator(model)
atoms = read("structure.traj")
energy = calc.get_binding_energy(atoms)
```

### Relaxation

```python
from adsnet.tools import relax_structure

result = relax_structure(
    atoms,
    model,
    method="pgd",
    max_step_xy=0.15,
    max_step_z=0.55,
    force_tol=1e-3,
)
print(f"Relaxed energy: {result['energy']:.4f} eV")
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

## Model Outputs

### Binding Energy
- Predicts the adsorption binding energy of the system
- Units: eV
- Applications:
  - Adsorption site prediction
  - Stability analysis
  - Reaction pathway studies

### Local Forces
- Computes atomic forces for each atom in the local environment
- Units: eV/A
- Applications:
  - Local structure optimization
  - Transition state search

### BE Distributions
- Per-atom energy contributions to total binding energy
- Units: eV
- Applications:
  - Identify key contributing atoms
  - Understand bonding mechanisms

## Configuration

All training parameters with defaults (see `adsnet.trainer.DEFAULT_CONFIG`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `node_size` | 128 | Hidden state dimension |
| `num_interactions` | 3 | Message passing layers |
| `cutoff` | 5.5 | Neighbor cutoff (A) |
| `batch_size` | 12 | Training batch size |
| `initial_lr` | 1e-4 | Learning rate |
| `max_steps` | 1000000 | Max training steps |
| `forces_weight` | 0.05 | Weight for forces in loss |
| `stop_patience` | 200 | Early stopping patience |
| `device` | "cpu" | "cpu" or "cuda" |

