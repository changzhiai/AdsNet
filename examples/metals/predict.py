import torch
import numpy as np
import os
import pandas as pd
from ase.io import read

from adsnet.models import AdsNet
from adsnet.calculators import MLCalculator

path = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def main():
    state_dict = torch.load(f"{path}/output/model.pt", map_location="cpu")
    model = AdsNet(
        num_layers=state_dict["num_layers"],
        num_channels=state_dict["num_channels"],
        cutoff=state_dict["cutoff"],
        rbf_type=state_dict.get("rbf_type", "sinc"),
        edge_embed=state_dict.get("edge_embed", "linear"),
    )
    model.load_state_dict(state_dict["model"])
    model.eval()

    calc = MLCalculator(model)

    def predict_set(traj_path):
        images = read(traj_path, ":")
        formulas, dft_energies, nnp_energies = [], [], []
        for atoms in images:
            formulas.append(atoms.get_chemical_formula(mode="hill"))
            dft_energies.append(atoms.info["data"]["ads_energy"])
            nnp_energies.append(calc.get_binding_energy(atoms))
        return formulas, dft_energies, nnp_energies

    train_formulas, train_dft, train_nnp = predict_set(f"{path}/dataset/train.traj")
    test_formulas, test_dft, test_nnp = predict_set(f"{path}/dataset/test.traj")
    print(f"Train: {len(train_dft)} structures  |  Test: {len(test_dft)} structures")

    for label, dft, nnp in [("Train", train_dft, train_nnp), ("Test", test_dft, test_nnp)]:
        dft_a, nnp_a = np.array(dft), np.array(nnp)
        mae = np.mean(np.abs(dft_a - nnp_a))
        rmse = np.sqrt(np.mean((dft_a - nnp_a) ** 2))
        print(f"{label}  MAE: {mae*1000:.1f} meV  RMSE: {rmse*1000:.1f} meV")

    df = pd.DataFrame({
        "split": ["train"] * len(train_dft) + ["test"] * len(test_dft),
        "formula": train_formulas + test_formulas,
        "dft_energies": train_dft + test_dft,
        "nnp_energies": train_nnp + test_nnp,
    })
    csv_path = f"{path}/output/predict.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved predictions to {csv_path}")


if __name__ == "__main__":
    main()
