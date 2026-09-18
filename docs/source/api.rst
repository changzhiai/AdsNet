API Reference
=============

This section provides detailed documentation for the AdsNet package modules, classes, and functions.


Models
------

.. autoclass:: adsnet.models.adsnet.AdsNet
   :members: __init__, forward

.. autoclass:: adsnet.models.adsnet.Interaction
   :members: __init__, forward

.. autoclass:: adsnet.models.adsnet.SincRBF
   :members: __init__, forward

.. autoclass:: adsnet.models.adsnet.GaussianRBF
   :members: __init__, forward

.. autoclass:: adsnet.models.adsnet.EdgeEmbedding
   :members: __init__, forward

.. autofunction:: adsnet.models.adsnet.cosine_cutoff


Data
----

.. autoclass:: adsnet.data.AseData
   :members: __init__, __len__, __getitem__, get_atoms_data

.. autofunction:: adsnet.data.collate_atomsdata


Trainer
-------

.. autoclass:: adsnet.trainer.Trainer
   :members: __init__, train, evaluate

.. autoclass:: adsnet.trainer.EarlyStopping
   :members: __init__, __call__

.. autodata:: adsnet.trainer.DEFAULT_CONFIG


Calculators
-----------

.. autoclass:: adsnet.calculators.MLCalculator
   :members: __init__, calculate, get_binding_energy, get_local_forces, get_be_distributions

.. autoclass:: adsnet.calculators.EnsembleCalculator
   :members: __init__, calculate, get_binding_energy, get_local_forces, get_be_distributions, get_uncertainty


Foundation
----------

.. autodata:: adsnet.foundation.foundation_model


Tools
-----

.. autofunction:: adsnet.tools.relax_structure

.. autofunction:: adsnet.tools.optimize_local_coords

.. autofunction:: adsnet.tools.build_local_bounds

.. autofunction:: adsnet.tools.make_fun_and_grad
