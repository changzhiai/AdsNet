import pandas as pd

df = pd.read_csv("predict.csv")

# Compute energy difference
df["energy_diff"] = df["nnp_energies"] - df["dft_energies"]

# Find rows with |ΔE| > 0.2 eV
outliers = df[df["energy_diff"].abs() > 0.5]

print(outliers)
print(f"\nNumber of outliers: {len(outliers)}")
