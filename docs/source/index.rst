Welcome to AdsNet's documentation!
===================================

**AdsNet** (Adsorption Network) is a machine learning framework for predicting adsorption properties on metal surfaces using graph neural networks. It employs an equivariant message-passing architecture with scalar and vector node features, operating on local neighbor graphs to efficiently predict binding energies and atomic forces for adsorbate–surface systems.

Key Features
------------

* **Single-pass adsorption energy** — predicts adsorption energies directly, no separate slab calculation needed
* **System-size independent** — operates on local adsorbate graphs, cost does not grow with size
* **Local structure relaxation** — built-in optimizer with projected gradient descent due to local forces supported
* **Equivariant architecture** — equivariant message passing with scalar and vector node features
* **Foundation model provided** — pretrain on diverse datasets, fine-tune on specific systems
* **Parallel training** — multi-GPU training via PyTorch DDP
* **ASE integration** — drop-in ASE calculator for use in existing workflows

AdsNet is developed and maintained by Dr. Changzhi Ai at the SUNCAT Center, Stanford University and SLAC National Accelerator Laboratory. The code is available on `GitHub <https://github.com/changzhiai/AdsNet>`_. For questions or issues, please use the GitHub issue tracker.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   about
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

