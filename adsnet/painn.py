import torch
from torch import nn

def sinc_expansion(edge_dist: torch.Tensor, edge_size: int, cutoff: float):
    """
    calculate sinc radial basis function:
    
    sin(n *pi*d/d_cut)/d
    """
    n = torch.arange(edge_size, device=edge_dist.device) + 1
    return torch.sin(edge_dist.unsqueeze(-1) * n * torch.pi / cutoff) / edge_dist.unsqueeze(-1)

def cosine_cutoff(edge_dist: torch.Tensor, cutoff: float):
    """
    Calculate cutoff value based on distance.
    This uses the cosine Behler-Parinello cutoff function:

    f(d) = 0.5*(cos(pi*d/d_cut)+1) for d < d_cut and 0 otherwise
    """

    return torch.where(
        edge_dist < cutoff,
        0.5 * (torch.cos(torch.pi * edge_dist / cutoff) + 1),
        torch.tensor(0.0, device=edge_dist.device, dtype=edge_dist.dtype),
    )

class PainnMessage(nn.Module):
    """Message function"""
    def __init__(self, node_size: int, edge_size: int, cutoff: float):
        super().__init__()
        
        self.edge_size = edge_size
        self.node_size = node_size
        self.cutoff = cutoff
        
        self.scalar_message_mlp = nn.Sequential(
            nn.Linear(node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )
        
        self.filter_layer = nn.Linear(edge_size, node_size * 3)
        
    def forward(self, node_scalar, node_vector, ads_idx_pairs, ads_edge_diff, edge_dist, all_scalar, all_vector):
        # remember to use v_j, s_j but not v_i, s_i        
        filter_weight = self.filter_layer(sinc_expansion(edge_dist, self.edge_size, self.cutoff))
        filter_weight = filter_weight * cosine_cutoff(edge_dist, self.cutoff).unsqueeze(-1)
        scalar_out = self.scalar_message_mlp(all_scalar[ads_idx_pairs[:, 1]])        
        filter_out = filter_weight * scalar_out 
        
        gate_state_vector, gate_edge_vector, message_scalar = torch.split(
            filter_out, 
            self.node_size,
            dim = 1,
        )
        
        message_vector =  all_vector[ads_idx_pairs[:, 1]] * gate_state_vector.unsqueeze(1) 
        edge_vector = gate_edge_vector.unsqueeze(1) * (ads_edge_diff / edge_dist.unsqueeze(-1)).unsqueeze(-1)
        message_vector = message_vector + edge_vector
        
        # sum message
        residual_scalar = torch.zeros_like(node_scalar)
        residual_vector = torch.zeros_like(node_vector)
        _, idx = torch.unique_consecutive(ads_idx_pairs[:, 0], return_inverse=True)
        residual_scalar.index_add_(0, idx, message_scalar)
        residual_vector.index_add_(0, idx, message_vector)
        
        # new node state
        new_node_scalar = node_scalar + residual_scalar
        new_node_vector = node_vector + residual_vector
        
        return new_node_scalar, new_node_vector

class PainnUpdate(nn.Module):
    """Update function"""
    def __init__(self, node_size: int):
        super().__init__()
        
        self.update_U = nn.Linear(node_size, node_size)
        self.update_V = nn.Linear(node_size, node_size)
        
        self.update_mlp = nn.Sequential(
            nn.Linear(node_size * 2, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )
        
    def forward(self, node_scalar, node_vector):
        Uv = self.update_U(node_vector)
        Vv = self.update_V(node_vector)
        
        Vv_norm = torch.linalg.norm(Vv, dim=1)
        mlp_input = torch.cat((Vv_norm, node_scalar), dim=1)
        mlp_output = self.update_mlp(mlp_input)
        
        a_vv, a_sv, a_ss = torch.split(
            mlp_output,                                        
            node_vector.shape[-1],                                       
            dim = 1,
        )
        
        delta_v = a_vv.unsqueeze(1) * Uv
        inner_prod = torch.sum(Uv * Vv, dim=1)
        delta_s = a_sv * inner_prod + a_ss
        
        return node_scalar + delta_s, node_vector + delta_v

class PainnModel(nn.Module):
    """PainnModel without ads_idx_pairs updating"""
    def __init__(
        self, 
        num_interactions, 
        hidden_state_size, 
        cutoff,
        normalization=True,
        target_mean=[0.0],
        target_stddev=[1.0],
        atomwise_normalization=True, 
        **kwargs,
    ):
        super().__init__()
        
        num_embedding = 119   # number of all elements
        self.cutoff = cutoff
        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.edge_embedding_size = 20
        
        # Setup atom embeddings
        self.atom_embedding = nn.Embedding(num_embedding, hidden_state_size)

        # Setup message-passing layers
        self.message_layers = nn.ModuleList(
            [
                PainnMessage(self.hidden_state_size, self.edge_embedding_size, self.cutoff)
                for _ in range(self.num_interactions)
            ]
        )
        self.update_layers = nn.ModuleList(
            [
                PainnUpdate(self.hidden_state_size)
                for _ in range(self.num_interactions)
            ]            
        )
        
        # Setup readout function
        self.readout_mlp = nn.Sequential(
            nn.Linear(self.hidden_state_size, self.hidden_state_size),
            nn.SiLU(),
            nn.Linear(self.hidden_state_size, 1),
        )

        # Normalisation constants
        self.normalization = torch.nn.Parameter(
            torch.tensor(normalization), requires_grad=False
        )
        self.atomwise_normalization = torch.nn.Parameter(
            torch.tensor(atomwise_normalization), requires_grad=False
        )
        self.normalize_stddev = torch.nn.Parameter(
            torch.tensor(target_stddev[0]), requires_grad=False
        )
        self.normalize_mean = torch.nn.Parameter(
            torch.tensor(target_mean[0]), requires_grad=False
        )
        
    def forward(self, input_dict, compute_forces=False):
        num_atoms = input_dict['num_atoms']
        ads_num_pairs = input_dict['ads_num_pairs']
        ads_atomic_num = input_dict['ads_atomic_num']
        ads_coords = input_dict['ads_coords']
        coords = input_dict['coords']
        ads_num_atoms = input_dict['ads_num_atoms']

        # ads_idx_pairs offset. Add offset to edges to get indices of pairs in a batch but not a structure
        ads_idx_pairs = input_dict['ads_idx_pairs']
        ads_idx_offset = torch.cumsum(
            torch.cat((torch.tensor([0], 
                                    device=num_atoms.device,
                                    dtype=num_atoms.dtype,                                    
                                   ), num_atoms[:-1])),
            dim=0
        )
        ads_idx_offset = torch.repeat_interleave(ads_idx_offset, ads_num_pairs)
        ads_idx_pairs = ads_idx_pairs + ads_idx_offset.unsqueeze(-1)        
        ads_edge_diff = input_dict['ads_edge_diff']
        if compute_forces:
            ads_edge_diff.requires_grad_()
        edge_dist = torch.linalg.norm(ads_edge_diff, dim=1)
        
        node_scalar = self.atom_embedding(ads_atomic_num)
        all_scalar = self.atom_embedding(input_dict['elems'])
        node_vector = torch.zeros((ads_coords.shape[0], 3, self.hidden_state_size),
                                  device=ads_coords.device,
                                  dtype=ads_coords.dtype,
                                 )
        all_vector = torch.zeros((coords.shape[0], 3, self.hidden_state_size),
                                  device=coords.device,
                                  dtype=coords.dtype,
                                 )
        
        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            node_scalar, node_vector = message_layer(node_scalar, node_vector, ads_idx_pairs, ads_edge_diff, edge_dist, all_scalar, all_vector)
            node_scalar, node_vector = update_layer(node_scalar, node_vector)
        
        node_scalar = self.readout_mlp(node_scalar)
        node_scalar.squeeze_()

        image_idx = torch.arange(ads_num_atoms.shape[0],
                                 device=ads_idx_pairs.device,
                                )
        image_idx = torch.repeat_interleave(image_idx, ads_num_atoms)
        
        ads_energy = torch.zeros_like(ads_num_atoms).float()        
        ads_energy.index_add_(0, image_idx, node_scalar)

        # Apply (de-)normalization
        # if self.normalization:
        #     normalizer = self.normalize_stddev
        #     energy = normalizer * energy
        #     mean_shift = self.normalize_mean
        #     if self.atomwise_normalization:
        #         mean_shift = input_dict["num_atoms"] * mean_shift
        #     energy = energy + mean_shift

        result_dict = {'ads_energy': ads_energy}
        
        # if compute_forces:
        #     dE_ddiff = torch.autograd.grad(
        #         energy,
        #         ads_edge_diff,
        #         grad_outputs=torch.ones_like(energy),
        #         retain_graph=True,
        #         create_graph=True,
        #     )[0]
            
            ## diff = R_j - R_i, so -dE/dR_j = -dE/ddiff, -dE/R_i = dE/ddiff  
            # i_forces = torch.zeros_like(input_dict['coord']).index_add(0, ads_idx_pairs[:, 0], dE_ddiff)
            # j_forces = torch.zeros_like(input_dict['coord']).index_add(0, ads_idx_pairs[:, 1], -dE_ddiff)
            # forces = i_forces + j_forces
            
            # result_dict['forces'] = forces
            
        return result_dict