# AdsNet Architecture

## Overview

AdsNet is a graph neural network (GNN) framework for predicting adsorption energies and local forces on surfaces. It represents atomic structures as graphs and uses message passing to propagate geometric information, enabling accurate predictions across diverse adsorbate-surface systems.

## Directory Structure

```
AdsNet/
├── adsnet/                    # Core package
│   ├── models/adsnet.py       # GNN architecture
│   ├── data/data.py           # Dataset and graph construction
│   ├── calculators/calculator.py  # ASE calculator interface
│   ├── trainer/trainer.py     # Training pipeline
│   └── tools/relax.py         # Structure relaxation utilities
├── proj/                      # Application examples
│   ├── proj_Pt/               # Platinum system
│   └── proj_metals/           # Multi-element systems
├── test/                      # Unit tests and test data
└── docs/                      # Documentation
```

## Neural Network Architecture

![AdsNet Architecture](architecture.png)

### Overview

```
Input: atomic numbers Z, 3D coordinates, edge displacements (periodic/non-periodic)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Atom Embedding   Embedding(119, H)  →  Linear(H, H)    │
│  node_scalar: (N_local, H)                              │
│  node_vector: (N_local, 3, H)  ←  zeros                 │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │   Interaction Block     │  × num_interactions
          │                         │
          │  ┌─── Message ────────┐ │
          │  │ sinc_RBF(d) → Linear(E, 3H)             │ │
          │  │  × cosine_cutoff(d)                      │ │
          │  │ scalar_MLP(s_j) → Linear(H, 3H)         │ │
          │  │ hadamard → split into 3 × H gates        │ │
          │  │  gate_state  · v_j  →  Δv (state gate)  │ │
          │  │  gate_edge   · r̂_ij →  Δv (edge gate)   │ │
          │  │  message_scalar     →  Δs                │ │
          │  │ index_add (scatter) to target atoms      │ │
          │  └──────────────────────────────────────────┘ │
          │  ┌─── Update ─────────────────────────────┐ │
          │  │  U·v, V·v  (linear projections)         │ │
          │  │  a_vv, a_sv, a_ss = MLP([‖V·v‖, s])    │ │
          │  │  Δv = a_vv ⊙ U·v                        │ │
          │  │  Δs = a_sv·⟨U·v, V·v⟩ + a_ss           │ │
          │  └──────────────────────────────────────────┘ │
          └────────────┬────────────┘
                       │
         ┌─────────────▼──────────────┐
         │  Readout MLP               │
         │  Linear(H,H) → SiLU        │
         │  → Linear(H,1) per atom    │
         │  index_add → per-image sum │
         └─────────────┬──────────────┘
                       │
         ┌─────────────▼──────────────────────────────────┐
         │  (optional) Force via autograd                  │
         │  F = -∂E/∂r  computed from edge displacements  │
         └─────────────────────────────────────────────────┘

Output: ads_energy (eV), local_energies (eV/atom), ads_forces (eV/Å)
```

### Edge Features

Each edge `(i, j)` within the cutoff radius carries a scalar distance `d_ij` and a displacement vector `r_ij = R_j - R_i`. Both directions are included (`i→j` and `j→i`).

**Sinc RBF** expands the scalar distance into `edge_embedding_size` basis functions:

```
φ_n(d) = sin(n·π·d / d_cut) / d,   n = 1, 2, ..., E
```

**Cosine cutoff** weights the basis to enforce smooth decay to zero at the boundary:

```
f_cut(d) = 0.5 · (cos(π·d / d_cut) + 1)   for d < d_cut, else 0
```

The final edge embedding is `φ(d) · f_cut(d)`, shape `(N_edges, E)`.

### Interaction Block

Each block refines the node representations through two sequential phases.

#### Message Phase

```
filter_weight  = Linear(E, 3H)(edge_emb)            # shape (N_edges, 3H)
scalar_out     = MLP_scalar(s_j)                     # Linear(H,H) → SiLU → Linear(H,3H)
filter_out     = filter_weight ⊙ scalar_out[j]       # elementwise, (N_edges, 3H)

gate_state_vec, gate_edge_vec, msg_scalar = split(filter_out, H)

# Two sources contribute to the vector message:
msg_vector  = gate_state_vec · v_j                   # gates the neighbor's vector state
           +  gate_edge_vec  · r̂_ij                  # injects bond direction

# Scatter to target atom i:
Δs_i = Σ_{j∈N(i)} msg_scalar
Δv_i = Σ_{j∈N(i)} msg_vector
```

#### Update Phase

```
Uv = Linear(H, H)(v)          # (N, 3, H)
Vv = Linear(H, H)(v)          # (N, 3, H)

Vv_norm = ‖Vv‖                # (N, H)   — rotation-equivariant scalar
a_vv, a_sv, a_ss = MLP([Vv_norm ‖ s])   # (N, 3H)

Δv = a_vv ⊙ Uv
Δs = a_sv · ⟨Uv, Vv⟩ + a_ss  # inner product couples vector and scalar channels
```

This update is inspired by PaiNN. The inner product `⟨Uv, Vv⟩` is rotationally invariant, ensuring the scalar channel remains invariant while the vector channel remains equivariant.

### Readout

```python
atom_energies = Linear(H,1)(SiLU(Linear(H,H)(node_scalar)))  # (N_local,)
ads_energy    = scatter_sum(atom_energies, image_idx)          # (N_images,)
```

Only atoms in the local neighborhood (within cutoff of any adsorbate atom) contribute to the readout.

### Force Computation

Forces are obtained via reverse-mode autograd through the edge displacements:

```python
dE_ddiff = ∂E / ∂(R_j - R_i)   # (N_edges, 3)

# Newton's third law decomposition:
F_i += +dE_ddiff   # atom i: diff = R_j - R_i, so ∂E/∂R_i = -∂E/∂diff → sign flip handled in loss
F_j += -dE_ddiff
```

This avoids explicit coordinate differentiation and is compatible with periodic boundary conditions because forces are expressed through displacement vectors.

### Weight Initialization

All `Linear` layers use **Xavier uniform** initialization; the atom embedding uses a small normal distribution (`σ = 0.05`). Biases are initialized to zero.

---

## Core Components

### 1. Model (`adsnet/models/adsnet.py`)

The GNN model processes atomic structures via:

- **Atom embedding**: Learned embeddings for each of 119 elements.
- **Sinc RBF expansion**: Edge distances encoded as `sin(n·π·d/d_cut) / d`, providing a smooth basis for inter-atomic distances.
- **Cosine cutoff**: Behler-Parinello envelope that smoothly decays neighbor interactions to zero at `d_cut`.
- **Interaction blocks** (repeated `num_interactions` times): Each block performs a message-passing step maintaining both scalar and vector (3D Cartesian) node features.
  - *Message phase*: A filter network processes edge RBF features and gates information from neighboring nodes.
  - *Update phase*: An MLP combines vector norms and scalar states to update node representations.
- **Readout MLP**: Maps per-atom hidden states to per-atom energy contributions; the system energy is their sum.
- **Force computation**: Optionally computed as the negative gradient of the predicted energy via autograd.

Key hyperparameters:

| Parameter | Default | Description |
|---|---|---|
| `num_interactions` | 3 | Message-passing depth |
| `hidden_state_size` | 128 | Node feature dimension |
| `cutoff` | 5.5 Å | Neighbor interaction radius |
| `edge_embedding_size` | 20 | RBF basis size |

### 2. Data Pipeline (`adsnet/data/data.py`)

**`AseData`** — a PyTorch `Dataset` that reads ASE trajectory files. Each frame must carry `atoms.info['data']` with:

- `ads_idx`: indices of adsorbate atoms
- `ads_energy`: binding energy label (eV)
- `delta_forces` *(optional)*: per-atom force labels (eV/Å)

**Graph construction** per frame:

1. Build neighbor list within `cutoff` using asap3 (periodic) or scipy (non-periodic).
2. Emit both `i→j` and `j→i` edges with displacement vectors `r_ij` and `-r_ij`.
3. Compact local atom indices for efficient batch processing.

**`collate_atomsdata`** concatenates variable-size graphs into a single batched graph with offset edge indices.

### 3. Calculator (`adsnet/calculators/calculator.py`)

Wraps trained models behind the ASE `Calculator` interface.

- **`MLCalculator`**: single-model predictions.
  - `get_binding_energy(atoms)` — scalar adsorption energy
  - `get_be_distributions(atoms)` — per-atom energy contributions
  - `get_local_forces(atoms)` — forces on the local atomic neighborhood
  - Supports independent energy and force scaling factors.

- **`EnsembleCalculator`**: aggregates an ensemble of `MLCalculator` instances, returning mean predictions and uncertainty estimates (standard deviation across models).

### 4. Trainer (`adsnet/trainer/trainer.py`)

Manages the full supervised learning loop.

**Loss function**:

```
L = (1 - w_f) · HuberLoss(E_pred, E_true)
  + w_f · HuberLoss(||F_pred||, ||F_true||)
```

where `w_f` is `forces_weight` (0 disables force supervision).

**Training configuration** (selected defaults):

| Parameter | Default |
|---|---|
| `batch_size` | 10 |
| `initial_lr` | 5e-4 |
| `max_steps` | 3000 |
| `forces_weight` | 0 |
| `val_ratio` | 0.1 |

**Evaluation metrics**: energy MAE/RMSE; force L2-norm MAE/RMSE and component-wise MAE/RMSE.

**Checkpointing**: saves the best model together with optimizer and scheduler state; resumes from checkpoint on restart.

### 5. Relaxation Tools (`adsnet/tools/relax.py`)

Provides local structure optimization using the trained model as a surrogate potential.

- **`relax_structure(atoms, calculator, ...)`**: high-level API. Identifies the local neighborhood from `atoms.info['data']['ads_idx']`, optionally perturbs the adsorbate along z, then runs optimization and returns the relaxed structure, final energy, convergence message, and recorded trajectory.

- **`optimize_local_coords(...)`**: low-level optimizer supporting:
  - *PGD* (Projected Gradient Descent) with line search and adaptive step sizes.
  - *L-BFGS-B* via scipy, with box constraints.
  - Box constraints generated by `build_local_bounds()` (asymmetric: tighter xy, looser z for adsorbates).
  - Convergence criteria: force norm below threshold, energy plateau detection, or `maxfun` budget exhausted.

## Data Flow

```
Training
────────
Raw trajectory (.traj)
  └─ AseData  ──  neighbor list + graph tensors
       └─ collate_atomsdata  ──  batched graph
            └─ AdsNet.forward  ──  per-atom energies → system energy + forces
                 └─ Loss (energy + forces)  ──  Adam → weight update

Inference
─────────
ASE Atoms
  └─ MLCalculator / EnsembleCalculator
       └─ AseData (single structure)  →  AdsNet.forward
            └─ binding energy, per-atom contributions, forces

Relaxation
──────────
Initial Atoms + MLCalculator
  └─ relax_structure
       └─ optimize_local_coords (PGD or L-BFGS-B)
            ├─ repeated energy + force evaluations via calculator
            └─ relaxed Atoms + trajectory
```

## Design Decisions

**Scalar + vector features**: Maintaining 3D vector node features alongside scalars allows the model to propagate directional geometric information through the graph without relying solely on distance.

**Local neighborhood compaction**: Each structure uses a compact local index mapping, so batching variable-sized systems requires only an index offset rather than padding.

**ASE integration**: Implementing the `Calculator` interface means AdsNet models slot directly into ASE workflows (NEB, MD, etc.) without any adapter code.

**Configurable force supervision**: Setting `forces_weight > 0` adds force labels to training. This is off by default to support datasets without force labels.

**Ensemble uncertainty**: `EnsembleCalculator` aggregates independently trained models, giving a cheap uncertainty estimate useful for active learning or reliability filtering.

## Dependencies

| Package | Role |
|---|---|
| PyTorch ≥ 1.12 | Neural network and autograd |
| ASE ≥ 3.22 | Atomic structure representation |
| asap3 | Fast periodic neighbor lists |
| numpy / scipy | Numerical operations and L-BFGS-B optimizer |
| pandas / matplotlib | Analysis and plotting (optional) |
