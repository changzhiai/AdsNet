from ase.io import read,write
from ase.db import connect
from ase.visualize import view

import numpy as np
import copy,os

def get_adsorption_energy(atoms, slab):
    tote = atoms.get_potential_energy()
    slabe = slab.get_potential_energy()
    adse = 2.25547973
    ads_e = tote - slabe - adse
    return ads_e, slabe, adse

def get_forces(atoms, slab, final_slab):
    totf = atoms.get_forces()
    slabf = final_slab.get_forces()
    atoms = atoms.copy()
    slab = slab.copy()
    atoms.wrap()
    slab.wrap()
    totpos = atoms.get_positions()
    slabpos = slab.get_positions()
    new_forces = []
    ads_idx = []
    slab_list = []
    for i, tpos in enumerate(totpos):
        tolerance = 1e-3
        matches = np.all(np.abs(tpos - slabpos) <= tolerance, axis=1)
        if np.any(matches):
            j = np.where(matches)
            assert len(j) == 1, 'j match too much'
            j = j[0]
            assert np.allclose(totpos[i], slabpos[j], rtol=tolerance, atol=tolerance)
            delta_force = totf[i] - slabf[j]
            assert len(delta_force) == 1, 'delta forces match too much'
            delta_force = delta_force[0]
            slab_list.append(j)
        else:
            ads_idx.append(i)
            delta_force = totf[i]
            slab_list.append(i*1000)
        new_forces.append(delta_force)
    if len(ads_idx) != 1:
        print(totpos, slabpos)
        assert False
    print(ads_idx)
    new_forces = np.array(new_forces)
    # print(new_forces.shape, totf.shape)
    assert new_forces.shape == totf.shape
    return new_forces, ads_idx, slab_list

def get_force_costom(atoms, slab, final_slab):
    totf = atoms.get_forces()
    slabf = final_slab.get_forces()
    new_row = np.array([0, 0, 0])
    slabf_new = np.vstack([new_row, slabf])
    assert slabf_new.shape == totf.shape
    delta_force = totf - slabf_new
    ads_idx = [0]
    return delta_force, ads_idx 

def get_forces_except_init(atoms, slab, ads_idx, slab_list, final_slab):
    totf = atoms.get_forces()
    slabf = final_slab.get_forces()
    new_forces = np.zeros_like(totf)
    count = 0
    for i, tpos in enumerate(totf):
        if slab_list[i] == ads_idx[0]*1000:
            new_forces[i] = tpos
            count += 1
        else:
            new_forces[i] = slabf[slab_list[i]]
    assert count == 1
    delta_forces = totf - new_forces
    return delta_forces, ads_idx

os.system('cp ../Pt_all_traj.db Pt_all_traj_full.db')
db = connect('Pt_all_traj_full.db')

all_images = []
dir = '/fs/ddn/sdf/group/suncat/smandal/published/01-correlation/generation-of-alpha/for-purePt/100'
folders = ['CN1', 'CN2-new', 'CN4', 'CN5', 'CN6', 'CN7', 'CN8']
for folder in folders:
    path = dir+'/'+folder+'/OUTCAR'
    images = read(path, ':')
    path_slab = path.replace('for-purePt', 'ref-Pt')
    images_slab = read(path_slab, ':')
    atoms_slab = images_slab[0]

    print(folder, len(images), len(images_slab))
    print(path, path_slab)
    for i, atoms in enumerate(images):
        ads_e, slabe, adse = get_adsorption_energy(atoms, images_slab[-1])
        slab = copy.deepcopy(atoms_slab)
        if i == 0:
            delta_forces, ads_idx, slab_list = get_forces(atoms, slab, images_slab[-1])
        else:
            delta_forces, ads_idx = get_forces_except_init(atoms, slab, ads_idx, slab_list, images_slab[-1])

        if 'data' not in atoms.info:
            atoms.info['data'] = {}
            atoms.info['data']['ads_idx'] = ads_idx
            atoms.info['data']['ads_energy'] = ads_e
            # atoms.info['data']['delta_forces'] = delta_forces
            atoms.info['data']['delta_forces'] = atoms.get_forces()
        all_images.append(atoms)    
        db.write(atoms, path=path, path_slab=path_slab, ionic_step=i)

dir = '/fs/ddn/sdf/group/suncat/smandal/published/01-correlation/generation-of-alpha/for-purePt/111'
folders = ['CN1', 'CN2-new', 'CN3', 'CN4', 'CN5', 'CN6', 'CN7', 'CN8', 'CN9']
for folder in folders:
    path = dir+'/'+folder+'/OUTCAR'
    images = read(path, ':')
    path_slab = path.replace('for-purePt', 'ref-Pt')
    images_slab = read(path_slab, ':')
    atoms_slab = images_slab[0]

    print(folder, len(images), len(images_slab))
    print(path, path_slab)
    for i, atoms in enumerate(images):
        ads_e, slabe, adse = get_adsorption_energy(atoms, images_slab[-1])
        slab = copy.deepcopy(atoms_slab)
        if i == 0:
            delta_forces, ads_idx, slab_list = get_forces(atoms, slab, images_slab[-1])
        else:
            delta_forces, ads_idx = get_forces_except_init(atoms, slab, ads_idx, slab_list, images_slab[-1])

        if 'data' not in atoms.info:
            atoms.info['data'] = {}
            atoms.info['data']['ads_idx'] = ads_idx
            atoms.info['data']['ads_energy'] = ads_e
            # atoms.info['data']['delta_forces'] = delta_forces
            atoms.info['data']['delta_forces'] = atoms.get_forces()
        all_images.append(atoms)    
        db.write(atoms, path=path, path_slab=path_slab, ionic_step=i)    

dir = '/fs/ddn/sdf/group/suncat/smandal/published/01-correlation/generation-of-alpha/for-purePt/211'
# folders = [ 'CN7',]
folders = ['CN5', 'CN6', 'CN7',]
for folder in folders:
    path = dir+'/'+folder+'/OUTCAR'
    images = read(path, ':')
    path_slab = path.replace('for-purePt', 'ref-Pt')
    images_slab = read(path_slab, ':')
    atoms_slab = images_slab[0]

    print(folder, len(images), len(images_slab))
    print(path, path_slab)
    for i, atoms in enumerate(images):
        ads_e, slabe, adse = get_adsorption_energy(atoms, images_slab[-1])
        slab = copy.deepcopy(atoms_slab)
        if folder in ['CN5', 'CN6']:
            delta_forces, ads_idx = get_force_costom(atoms, slab, images_slab[-1])
        else:
            if i == 0:
                delta_forces, ads_idx, slab_list = get_forces(atoms, slab, images_slab[-1])
            else:
                delta_forces, ads_idx = get_forces_except_init(atoms, slab, ads_idx, slab_list, images_slab[-1])

        if 'data' not in atoms.info:
            atoms.info['data'] = {}
            atoms.info['data']['ads_idx'] = ads_idx
            atoms.info['data']['ads_energy'] = ads_e
            # atoms.info['data']['delta_forces'] = delta_forces
            atoms.info['data']['delta_forces'] = atoms.get_forces()
        all_images.append(atoms)    
        db.write(atoms, path=path, path_slab=path_slab, ionic_step=i)

folders = ['CN7']
for folder in folders:
    path = dir+'/'+folder+'/OUTCAR'
    images = read(path, ':')
    for i, atoms in enumerate(images):
        path_slab = '/sdf/home/c/changzhi/changzhi/ads_ML/spc/ref-Pt/211/CN7/rm2/OUTCAR'
        images_slab = read(path_slab, ':')
        atoms_slab = images_slab[-1]
        ads_e, slabe, adse = get_adsorption_energy(atoms, atoms_slab)
        
        totf = atoms.get_forces()
        slabf = atoms_slab.get_forces()
        new_row = np.array([0, 0, 0])
        slabf_new = np.insert(slabf, 3, new_row, axis=0)
        assert slabf_new.shape == totf.shape
        delta_force = totf - slabf_new
        ads_idx = [3]
        print(ads_idx)
        print(path_slab)
        if 'data' not in atoms.info:
            atoms.info['data'] = {}
            atoms.info['data']['ads_idx'] = ads_idx
            atoms.info['data']['ads_energy'] = ads_e
            # atoms.info['data']['delta_forces'] = delta_forces
            atoms.info['data']['delta_forces'] = atoms.get_forces()
        all_images.append(atoms)   
        db.write(atoms, path=path, path_slab=path_slab, ionic_step=i)

folders = ['CN7']
for folder in folders:
    path = dir+'/'+folder+'/OUTCAR'
    images = read(path, ':')
    for i, atoms in enumerate(images):
        path_slab = '/sdf/home/c/changzhi/changzhi/ads_ML/spc/ref-Pt/211/CN7/rm3/OUTCAR'
        images_slab = read(path_slab, ':')
        atoms_slab = images_slab[-1]
        ads_e, slabe, adse = get_adsorption_energy(atoms, atoms_slab)
        
        totf = atoms.get_forces()
        slabf = atoms_slab.get_forces()
        new_row = np.array([0, 0, 0])
        slabf_new = np.insert(slabf, 6, new_row, axis=0)
        assert slabf_new.shape == totf.shape
        delta_force = totf - slabf_new
        ads_idx = [6]
        print(ads_idx)
        print(path_slab)
        if 'data' not in atoms.info:
            atoms.info['data'] = {}
            atoms.info['data']['ads_idx'] = ads_idx
            atoms.info['data']['ads_energy'] = ads_e
            # atoms.info['data']['delta_forces'] = delta_forces
            atoms.info['data']['delta_forces'] = atoms.get_forces()
        all_images.append(atoms)   
        db.write(atoms, path=path, path_slab=path_slab, ionic_step=i)

new_images = []
skip_is = [0, 19, 37, 50, 60, 72, 84, 96, 114, 155, 167, 180, 196, 210, 224, 239, 251, 267, 281]
for i, atoms in enumerate(all_images):
    if i in skip_is:
        continue
    new_images.append(atoms)
write('Pt_ads_forces_full.traj', new_images)

