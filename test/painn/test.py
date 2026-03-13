from ase.io import read
  
atoms = read("Pt_ads_forces.traj", index=0)

print(atoms.info)
print(atoms.info['data'])
