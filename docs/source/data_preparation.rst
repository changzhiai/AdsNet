Data Preparation
================

This guide explains how to prepare datasets for training AdsNet models.


Data Format
-----------

AdsNet uses ASE trajectory (``.traj``) files. Each frame (``Atoms`` object) must contain:

* **Atomic structure**: positions, atomic numbers, and unit cell (if periodic)
* **Adsorption metadata**: stored in ``atoms.info['data']``

The required metadata fields in ``atoms.info['data']`` are:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Key
     - Type
     - Description
   * - ``ads_idx``
     - list of int
     - Indices of adsorbate atoms in the structure
   * - ``ads_energy``
     - float
     - Adsorption energy label (eV)
   * - ``delta_forces``
     - array (N, 3)
     - Force labels on local atoms (eV/A), optional but recommended


Creating a Dataset
------------------

Here is how to construct a trajectory file with the required metadata:

.. code-block:: python

   from ase import Atoms
   from ase.io import Trajectory
   import numpy as np

   traj = Trajectory("dataset.traj", "w")

   for slab, adsorbate_indices, energy, forces in your_data:
       atoms = slab.copy()
       atoms.info['data'] = {
           'ads_idx': adsorbate_indices,    # e.g., [36, 37] for a 2-atom adsorbate
           'ads_energy': energy,            # e.g., -1.23 eV
           'delta_forces': forces,          # shape (N_local, 3)
       }
       traj.write(atoms)

   traj.close()


Structure Requirements
----------------------

Each structure should represent an adsorbate on a metal surface:

* **Surface slab**: Typically 3-4 layers of metal atoms with periodic boundary conditions in x and y
* **Adsorbate**: One or more atoms adsorbed on the surface
* **Adsorbate indices** (``ads_idx``): Must correctly identify which atoms are the adsorbate

.. code-block:: python

   from ase.build import fcc111, add_adsorbate
   from ase import Atoms

   # Build a Pt(111) slab
   slab = fcc111('Pt', size=(3, 3, 4), vacuum=10.0)

   # Add CO adsorbate
   co = Atoms('CO', positions=[[0, 0, 0], [0, 0, 1.2]])
   add_adsorbate(slab, co, height=1.8, position='ontop')

   # Identify adsorbate indices (last 2 atoms added)
   n_slab = 36  # 3*3*4
   ads_idx = [n_slab, n_slab + 1]


Computing Adsorption Energies
-----------------------------

The adsorption energy is typically computed from DFT as:

.. math::

   E_{ads} = E_{slab+ads} - E_{slab} - E_{ads,gas}

.. code-block:: python

   # From DFT total energies
   E_slab_ads = -350.5   # total energy of slab + adsorbate
   E_slab = -345.2       # total energy of clean slab
   E_gas = -6.0          # total energy of gas-phase adsorbate

   ads_energy = E_slab_ads - E_slab - E_gas  # = 0.7 eV


Force Labels
------------

Force labels (``delta_forces``) correspond to the forces on atoms in the local environment around the adsorbate. These are used when ``forces_weight > 0`` during training.

.. code-block:: python

   from ase.calculators.emt import EMT

   # Compute forces from DFT or other calculator
   atoms.calc = EMT()
   all_forces = atoms.get_forces()

   # Store forces for local atoms
   atoms.info['data']['delta_forces'] = all_forces


Splitting a Dataset
-------------------

Split your data into training and test sets:

.. code-block:: python

   import numpy as np
   from ase.io import read, Trajectory

   frames = read("dataset.traj", ":")
   n = len(frames)
   n_test = int(round(n * 0.1))

   rng = np.random.default_rng(42)
   indices = rng.permutation(n)
   train_idx = indices[n_test:]
   test_idx = indices[:n_test]

   for split, idx, fname in [('train', train_idx, 'train.traj'),
                              ('test', test_idx, 'test.traj')]:
       traj = Trajectory(fname, 'w')
       for i in idx:
           traj.write(frames[i])
       traj.close()
       print(f'{split}: {len(idx)} frames')


Loading Data in AdsNet
-----------------------

The ``AseData`` class handles graph construction from trajectory files:

.. code-block:: python

   from adsnet.data import AseData, collate_atomsdata
   from torch.utils.data import DataLoader

   dataset = AseData("train.traj", cutoff=5.5, compute_forces=True)
   dataloader = DataLoader(
       dataset,
       batch_size=12,
       shuffle=True,
       collate_fn=collate_atomsdata,
   )

Key parameters:

* ``cutoff``: Neighbor list cutoff radius in Angstroms (should match model cutoff)
* ``compute_forces``: Enable gradient tracking for force prediction

The dataset automatically constructs neighbor lists and edge vectors for the graph neural network. Periodic boundary conditions are handled via ``asap3.FullNeighborList``.


Validating Your Dataset
------------------------

Check that your data is correctly formatted before training:

.. code-block:: python

   from ase.io import read

   images = read("dataset.traj", ":")

   for i, atoms in enumerate(images):
       assert 'data' in atoms.info, f"Frame {i}: missing atoms.info['data']"
       data = atoms.info['data']
       assert 'ads_idx' in data, f"Frame {i}: missing ads_idx"
       assert 'ads_energy' in data, f"Frame {i}: missing ads_energy"

       # Check ads_idx are valid
       n_atoms = len(atoms)
       for idx in data['ads_idx']:
           assert 0 <= idx < n_atoms, f"Frame {i}: ads_idx {idx} out of range"

   print(f"Validated {len(images)} frames successfully")
