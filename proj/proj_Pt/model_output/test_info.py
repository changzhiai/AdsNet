from ase.io import read
  
atoms = read("metals_with_adss.traj", index=0)

print(atoms.info)