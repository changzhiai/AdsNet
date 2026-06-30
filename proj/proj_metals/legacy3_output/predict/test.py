from ase.io import read

atoms = read("/pscratch/sd/c/changzhi/test/adsnet/forces/test/db/dataset_refine2/test.traj", index=2440)
print(atoms.info['data']['ads_energy'])
