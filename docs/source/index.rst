Welcome to AdsNet's documentation!
===================================

**AdsNet** (Adsorption Network) is a machine learning framework for predicting adsorption properties on metal surfaces using graph neural networks. It employs a PaiNN-style equivariant message-passing architecture with scalar and vector node features, operating on local neighbor graphs to efficiently predict binding energies and atomic forces for adsorbate–surface systems.

Key Features
------------

* **Equivariant GNN architecture** — message passing with scalar and vector channels that respect rotational symmetry
* **Local interaction model** — uses cutoff-based neighbor lists for efficient graph construction
* **Multi-task learning** — jointly predicts adsorption energies and local forces
* **ASE calculator interface** — seamless integration with the Atomic Simulation Environment
* **Structure relaxation** — built-in projected gradient descent optimizer with box constraints
* **Foundation model support** — pretrain on diverse datasets and fine-tune on specific systems
* **Distributed training** — multi-GPU training via PyTorch DDP

The code is available on `GitHub <https://github.com/changzhiai/AdsNet>`_. For questions or issues, please use the GitHub issue tracker.

.. note::

   AdsNet is developed and maintained by Dr. Changzhi Ai at the SUNCAT Center, Stanford University and SLAC National Accelerator Laboratory.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   data_preparation
   training
   prediction
   relaxation
   parallelization
   foundation
   api
   troubleshooting

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
