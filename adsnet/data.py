"""
Data for AdsNet
"""
from ase.io import read, Trajectory
import torch
import asap3
import numpy as np
from typing import List
from scipy.spatial import distance_matrix

class AseData(torch.utils.data.Dataset):
    def __init__(self, ase_data=None, cutoff=5.0, **kwargs):
        super().__init__(**kwargs)
        
        # To do: Add atoms.info filter (required included data)
        if ase_data == None:
            self.images = 'Images not defined!'
        elif isinstance(ase_data, str):
            self.images = Trajectory(ase_data)
        else:
            self.images = ase_data
        
        self.cutoff = cutoff
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        atoms = self.images[idx]
        atoms_data = self.get_atoms_data(atoms)
        return atoms_data
    
    def get_atoms_data(self, atoms):
        atoms_data = {}
        atoms_data['num_atoms'] = torch.tensor([atoms.get_global_number_of_atoms()])
        atoms_data['elems'] = torch.tensor(atoms.numbers)
        atoms_data['coords'] = torch.tensor(atoms.positions, dtype=torch.float)
        try:
            ads_data = atoms.info['data']
        except AttributeError as e:
            print(f'Atoms info error: {e}. Please add ads. data into atoms.info!')
        atoms_data.update(ads_data)
        atoms_data['ads_energy'] = torch.tensor(ads_data['ads_energy'], dtype=torch.float)
        atoms_data['ads_idx'] = torch.tensor(ads_data['ads_idx'])
        
        if atoms.pbc.any():
            ads_idx_pairs, ads_edge_diff = self._get_neighborlist(atoms)
        else:
            ads_idx_pairs, ads_edge_diff = self._get_neighborlist_simple(atoms)

        atoms_data['ads_idx_pairs'] = torch.from_numpy(ads_idx_pairs)
        atoms_data['ads_edge_diff'] = torch.from_numpy(ads_edge_diff).float()
        atoms_data['ads_num_pairs'] = torch.tensor([ads_idx_pairs.shape[0]])
        atoms_data['ads_num_atoms'] = torch.tensor([len(atoms_data['ads_idx'])])
        atoms_data['ads_atomic_num'] = torch.tensor([atoms.numbers[atoms_data['ads_idx']]])
        atoms_data['ads_coords'] = torch.tensor([atoms.positions[atoms_data['ads_idx']]], dtype=torch.float)
        atoms_data['ads_nbr_coords'] = torch.tensor(atoms.positions[ads_idx_pairs[:,1]], dtype=torch.float)
        # atoms_data['ads_forces'] = torch.tensor(atoms.get_forces()[ads_data['ads_idx']])
        
        return atoms_data

    def _get_neighborlist(self, atoms):        
        nl = asap3.FullNeighborList(self.cutoff, atoms)
        pair_i_idx = []
        pair_j_idx = []
        ads_edge_diff = []
        ads_idx = atoms.info['data']['ads_idx']
        for i in ads_idx:
            indices, diff, _ = nl.get_neighbors(i)
            pair_i_idx += [i] * len(indices)               # local index of pair i
            pair_j_idx.append(indices)   # local index of pair j
            ads_edge_diff.append(diff)

        pair_j_idx = np.concatenate(pair_j_idx)
        ads_idx_pairs = np.stack((pair_i_idx, pair_j_idx), axis=1)
        ads_edge_diff = np.concatenate(ads_edge_diff)
        
        return ads_idx_pairs, ads_edge_diff

    def _get_neighborlist_simple(self, atoms):
        pos = atoms.get_positions()
        dist_mat = distance_matrix(pos, pos)
        mask = dist_mat < self.cutoff
        np.fill_diagonal(mask, False)        
        ads_idx_pairs = np.argwhere(mask)
        ads_edge_diff = pos[ads_idx_pairs[:, 1]] - pos[ads_idx_pairs[:, 0]]
        
        return ads_idx_pairs, ads_edge_diff
    

def cat_tensors(tensors: List[torch.Tensor]):
    if tensors[0].shape:
        return torch.cat(tensors)
    return torch.stack(tensors)

def collate_atomsdata(atoms_data: List[dict], pin_memory=True):
    # convert from list of dicts to dict of lists
    dict_of_lists = {k: [dic[k] for dic in atoms_data] for k in atoms_data[0]}
    if pin_memory:
        pin = lambda x: x.pin_memory()
    else:
        pin = lambda x: x
        
    collated = {k: cat_tensors(v) for k, v in dict_of_lists.items()}
    return collated
