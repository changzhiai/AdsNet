from ase.calculators.calculator import Calculator, all_changes
from adsnet.data import AseData
import numpy as np


class MLCalculator(Calculator):
    implemented_properties = ["energy",]

    def __init__(
        self,
        model,
        energy_scale=1.0,
        forces_scale=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.model = model
        self.model_device = next(model.parameters()).device
        self.cutoff = model.cutoff
        self.compute_forces = model.compute_forces
        self.asedata = AseData(cutoff=self.cutoff, compute_forces=self.compute_forces)
        self.energy_scale = energy_scale
        self.forces_scale = forces_scale
#        self.stress_scale = stress_scale

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """

        if atoms is not None:
            self.atoms = atoms.copy()       

        model_inputs = self.asedata.get_atoms_data(self.atoms)
        model_inputs = {
            k: v.to(self.model_device) for (k, v) in model_inputs.items()
        }

        model_results = self.model(model_inputs)

        results = {}
        results["energy"] = (
            model_results["ads_energy"][0].detach().cpu().numpy().item()
            * self.energy_scale
        )

        if self.compute_forces:
            results["ads_forces"] = (
                model_results["ads_forces"].detach().cpu().numpy() * self.forces_scale
            )
            
        results["local_idx"] = (
            model_results["local_idx"].detach().cpu().numpy() 
        )
        results["local_energies"] = (
            model_results["local_energies"].detach().cpu().numpy()
        )
    
        self.results = results

    def get_binding_energy(self, atoms):
        """Predict binding energy for atoms."""
        self.calculate(atoms)
        return self.results["energy"]

    def get_local_forces(self, atoms):
        """Get forces on local atoms (adsorbate + neighbors).

        Returns:
            (local_idx, ads_forces) — atom indices and corresponding force vectors
        """
        self.calculate(atoms)
        local_idx = self.results["local_idx"]
        ads_forces = self.results.get("ads_forces", None)
        return local_idx, ads_forces

    def get_be_distributions(self, atoms):
        """Get per-atom binding energy contributions.

        Returns:
            (local_idx, local_energies) — atom indices and their energy contributions
        """
        self.calculate(atoms)
        local_idx = self.results["local_idx"]
        local_energies = self.results["local_energies"]
        return local_idx, local_energies


class EnsembleCalculator(Calculator):
    implemented_properties = ["energy", "ensemble"]

    def __init__(self, models, energy_scale=1.0, forces_scale=1.0, **kwargs):
        super().__init__(**kwargs)

        self.models = models
        self.model_device = next(models[0].parameters()).device
        self.cutoff = models[0].cutoff
        self.compute_forces = models[0].compute_forces
        self.asedata = AseData(cutoff=self.cutoff, compute_forces=self.compute_forces)
        self.energy_scale = energy_scale
        self.forces_scale = forces_scale

    def calculate(self, atoms=None, properties=["energy", "ensemble"], system_changes=all_changes):
        if atoms is not None:
            self.atoms = atoms.copy()

        model_inputs = self.asedata.get_atoms_data(self.atoms)
        model_inputs = {
            k: v.to(self.model_device) for (k, v) in model_inputs.items()
        }

        energies = []
        all_local_energies = []
        all_forces = []
        for model in self.models:
            model_results = model(model_inputs)
            energies.append(model_results["ads_energy"][0].detach().cpu().numpy().item() * self.energy_scale)
            all_local_energies.append(model_results["local_energies"].detach().cpu().numpy())
            if self.compute_forces:
                all_forces.append(model_results["ads_forces"].detach().cpu().numpy() * self.forces_scale)

        results = {
            "energy": np.mean(energies),
            "local_idx": model_results["local_idx"].detach().cpu().numpy(),
            "local_energies": np.mean(all_local_energies, axis=0),
            "ensemble": {
                "energy_var": np.var(energies),
                "energies": energies,
            },
        }

        if self.compute_forces:
            results["ads_forces"] = np.mean(all_forces, axis=0)
            results["ensemble"]["forces_var"] = np.var(all_forces, axis=0)

        self.results = results

    def get_binding_energy(self, atoms):
        """Predict mean binding energy across ensemble."""
        self.calculate(atoms)
        return self.results["energy"]

    def get_local_forces(self, atoms):
        """Get mean forces on local atoms across ensemble."""
        self.calculate(atoms)
        local_idx = self.results["local_idx"]
        ads_forces = self.results.get("ads_forces", None)
        return local_idx, ads_forces

    def get_be_distributions(self, atoms):
        """Get mean per-atom binding energy contributions across ensemble."""
        self.calculate(atoms)
        local_idx = self.results["local_idx"]
        local_energies = self.results["local_energies"]
        return local_idx, local_energies

    def get_uncertainty(self, atoms):
        """Get energy variance across ensemble models."""
        self.calculate(atoms)
        return self.results["ensemble"]["energy_var"]
