Quickstart
==========

This guide will help you get started with AdsNet quickly. We'll cover the basic usage of the package for training and prediction.

Basic Example
-------------

Here's a simple example that demonstrates how to use AdsNet:

.. code-block:: python

    from adsnet.trainer import Trainer
    from adsnet.calculators import MLCalculator
    from ase.io import read

    # Train a model
    trainer = Trainer(
        config={
            'device': 'cpu',
            'output_dir': 'output',
            'output_log': 'output.log',
            'output_model': 'model.pt',
        },
        distributed=False,
    )

    # Prepare dataset and train model
    trainer.train("dataset.traj")

    # Create calculator with trained model
    calc = MLCalculator("output/model.pt")

    # Read structures
    images = read("test_structures.traj", ":")

    # Make predictions
    for atoms in images:
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        print(f"Energy: {energy} eV")

Next Steps
----------

After running the quickstart example, you might want to:

1. Check out :doc:`training` for detailed training instructions
2. Learn about :doc:`prediction` for making predictions with trained models
3. Read the :doc:`api` reference for full API documentation
