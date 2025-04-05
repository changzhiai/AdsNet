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
#        stress_scale=1.0,
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
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)
        if atoms is not None:
            self.atoms = atoms.copy()       

        model_inputs = self.asedata.get_atoms_data(self.atoms)
        model_inputs = {
            k: v.to(self.model_device) for (k, v) in model_inputs.items()
        }

        model_results = self.model(model_inputs)

        # from adsnet.mace import AtomwiseReduce
        # m = AtomwiseReduce(output_key='ads_energy')
        # model_results = m(model_results)

        results = {}

        # Convert outputs to calculator format
        # results["forces"] = (
        #     model_results["forces"].detach().cpu().numpy() * self.forces_scale
        # )
        results["energy"] = (
            model_results["ads_energy"][0].detach().cpu().numpy().item()
            * self.energy_scale
        )

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

class EnsembleCalculator(Calculator):
    implemented_properties = ["energy", "ensemble"]

    def __init__(
        self,
        models,
        energy_scale=1.0,
        forces_scale=1.0,
#        stress_scale=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.models = models
        self.model_device = next(models[0].parameters()).device
        self.cutoff = models[0].cutoff
        self.asedata = AseData(self.cutoff)
        self.energy_scale = energy_scale
        self.forces_scale = forces_scale
#        self.stress_scale = stress_scale
    
    def get_ensemble(self):
        """Get the calculated ensemble"""
        try:
            return self.results["ensemble"]
        except:
            return None

    def calculate(self, atoms=None, properties=["energy", "ensemble"], system_changes=all_changes):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)
        if atoms is not None:
            self.atoms = atoms.copy()       

        model_inputs = self.asedata.get_atoms_data(self.atoms)
        model_inputs = {
            k: v.to(self.model_device) for (k, v) in model_inputs.items()
        }

        predictions = {'energy': [],}
        for model in self.models:
            model_results = model(model_inputs)
            predictions['energy'].append(model_results["ads_energy"][0].detach().cpu().numpy().item() * self.energy_scale)
            # predictions['forces'].append(model_results["forces"].detach().cpu().numpy() * self.forces_scale)

        results = {"energy": np.mean(predictions['energy'])}
        # results["forces"] = np.mean(np.stack(predictions['forces']), axis=0)

        ensemble = {
            'energy_var': np.var(predictions['energy']),
            # 'forces_var': np.var(np.stack(predictions['forces']), axis=0),
            # 'forces_l2_var': np.var(np.linalg.norm(predictions['forces'], axis=2), axis=0),
        }

        results['ensemble'] = ensemble

        self.results = results
