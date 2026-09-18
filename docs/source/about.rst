About
============

AdsNet is an equivariant graph neural network that directly predicts adsorption energies on transition metal surfaces in a single forward pass. Unlike conventional machine-learning interatomic potentials (MLIPs) that construct full-system graphs, AdsNet operates on a local adsorbate graph containing only the adsorbate atoms and their neighboring surface atoms within a cutoff radius.

Adsorbate Graph
---------------

The key of AdsNet is the use of a local adsorbate graph instead of a full graph. As shown in the example below, the full graph of a Pt 4×4×4 slab with one CO* adsorbate contains 66 nodes and 1083 edges, while the corresponding adsorbate graph contains only 12 nodes and 17 edges. This dramatically reduces computational cost, which becomes independent of the total system size.

.. image:: _static/images/1_cost.png
   :width: 70%
   :align: center
   :alt: Comparison of full graph and adsorbate graph representations and their computational cost scaling.

**(a)** Full graph representation of Pt surface with CO* adsorbate. **(b)** Local adsorbate graph used by AdsNet. **(c)** Computational cost comparison showing that adsorbate graph cost is independent of system size.

Model Architecture
------------------

AdsNet implements an equivariant message-passing architecture with scalar and vector node features. The inputs are atomic numbers and interatomic vectors within the adsorbate graph. Interatomic distances are expanded using radial basis functions and modulated by a cosine cutoff envelope. After multiple interaction layers of message passing and equivariant updates, a readout MLP predicts adsorption energy contributions that are summed to obtain the total adsorption energy. Forces are computed via automatic differentiation.

.. image:: _static/images/2_arch.png
   :width: 80%
   :align: center
   :alt: Architecture of AdsNet showing equivariant message passing with scalar and vector node features.

The model uses equivariant message passing with scalar and vector channels over *L* interaction layers, followed by a readout MLP to predict adsorption energies.

Citation
--------

If you use AdsNet in your research, please cite::

    @article{AdsNet2026,
      author = {Ai, Changzhi and others},
      title = {AdsNet: A Graph Neural Network for Adsorption Energy Prediction},
      year = {2026},
    }

The AdsTraj dataset is available on `Hugging Face <https://huggingface.co/datasets/changzhiai/AdsTraj>`_.
