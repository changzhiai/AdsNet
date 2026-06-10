Installation
============

Prerequisites
-------------

Before installing AdsNet, ensure you have the following prerequisites:

* Python 3.7 or higher
* PyTorch 1.9 or higher

To install PyTorch, visit the `official PyTorch website <https://pytorch.org/get-started/locally/>`_ for installation instructions specific to your system.

Here is an example of how to install PyTorch on a Linux machine with CUDA support:

.. code-block:: bash

   # Choose Stable, Linux, Pip, Python, CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


Installing AdsNet
-----------------

You can install AdsNet using pip:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/changzhiai/AdsNet.git
   cd AdsNet

   # Install with pip
   pip install .


GPU Support
-----------

For GPU acceleration, make sure you have CUDA installed and PyTorch with CUDA support:

.. code-block:: bash

   # Check if PyTorch is using CUDA
   python -c "import torch; print(torch.cuda.is_available())"


Verifying Installation
----------------------

To verify your installation, you can run:

.. code-block:: python

   import adsnet

   # Print the version of AdsNet
   print(adsnet.__version__)

If no error occurs, AdsNet has been installed successfully.
