import torch
from torch import nn


def cosine_cutoff(edge_dist: torch.Tensor, cutoff: float):
    """
    Cosine Behler-Parinello cutoff: f(d) = 0.5*(cos(pi*d/d_cut)+1) for d < d_cut, else 0
    """
    return torch.where(
        edge_dist < cutoff,
        0.5 * (torch.cos(torch.pi * edge_dist / cutoff) + 1),
        torch.tensor(0.0, device=edge_dist.device, dtype=edge_dist.dtype),
    )


class GaussianRBF(nn.Module):
    """Learnable Gaussian radial basis functions."""
    def __init__(self, num_rbf: int, cutoff: float):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer('initial_centers', centers.clone())
        self.centers = nn.Parameter(centers)
        self.widths = nn.Parameter(torch.full((num_rbf,), 10.0 / (cutoff / num_rbf) ** 2))

    def forward(self, edge_dist: torch.Tensor):
        return torch.exp(-self.widths * (edge_dist.unsqueeze(-1) - self.centers) ** 2)


class EdgeEmbedding(nn.Module):
    """Two-layer MLP to embed radial basis into filter space."""
    def __init__(self, edge_size: int, output_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_size, output_size),
            nn.SiLU(),
            nn.Linear(output_size, output_size),
        )

    def forward(self, rbf):
        return self.mlp(rbf)


class Interaction(nn.Module):
    """Interaction block: message passing + node update with layer norm."""
    def __init__(self, node_size: int, edge_size: int, cutoff: float):
        super().__init__()

        self.node_size = node_size
        self.cutoff = cutoff

        # Message
        self.scalar_message_mlp = nn.Sequential(
            nn.Linear(node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )
        self.edge_embedding = EdgeEmbedding(edge_size, node_size * 3)

        # Update
        self.update_U = nn.Linear(node_size, node_size)
        self.update_V = nn.Linear(node_size, node_size)
        self.update_mlp = nn.Sequential(
            nn.Linear(node_size * 2, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )

        # Layer normalization
        self.layer_norm_scalar = nn.LayerNorm(node_size)
        self.layer_norm_gate = nn.LayerNorm(node_size)

    def forward(self, node_scalar, node_vector, ads_idx_pairs, ads_edge_diff, edge_dist, rbf):
        # --- Message ---
        filter_weight = self.edge_embedding(rbf)
        filter_weight = filter_weight * cosine_cutoff(edge_dist, self.cutoff).unsqueeze(-1)
        scalar_out = self.scalar_message_mlp(node_scalar)

        filter_out = filter_weight * scalar_out[ads_idx_pairs[:, 1]]

        gate_state_vector, gate_edge_vector, message_scalar = torch.split(
            filter_out, self.node_size, dim=1,
        )

        message_vector = node_vector[ads_idx_pairs[:, 1]] * gate_state_vector.unsqueeze(1)
        edge_vector = gate_edge_vector.unsqueeze(1) * (ads_edge_diff / edge_dist.unsqueeze(-1)).unsqueeze(-1)
        message_vector = message_vector + edge_vector

        residual_scalar = torch.zeros_like(node_scalar)
        residual_vector = torch.zeros_like(node_vector)
        residual_scalar.index_add_(0, ads_idx_pairs[:, 0], message_scalar)
        residual_vector.index_add_(0, ads_idx_pairs[:, 0], message_vector)

        node_scalar = self.layer_norm_scalar(node_scalar + residual_scalar)
        node_vector = node_vector + residual_vector

        # --- Update ---
        Uv = self.update_U(node_vector)
        Vv = self.update_V(node_vector)

        Vv_norm = torch.linalg.norm(Vv, dim=1)
        mlp_input = torch.cat((Vv_norm, node_scalar), dim=1)
        mlp_output = self.update_mlp(mlp_input)

        a_vv, a_sv, a_ss = torch.split(
            mlp_output, node_vector.shape[-1], dim=1,
        )

        delta_v = a_vv.unsqueeze(1) * Uv
        inner_prod = torch.sum(Uv * Vv, dim=1)
        delta_s = a_sv * inner_prod + a_ss

        return self.layer_norm_gate(node_scalar + delta_s), node_vector + delta_v


class AdsNet(nn.Module):
    """AdsNet: Adsorption Network: GNN message passing model for adsorption energy prediction."""
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        edge_embedding_size=20,
        compute_forces=False,
        **kwargs,
    ):
        super().__init__()

        num_embedding = 119
        self.cutoff = cutoff
        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.edge_embedding_size = edge_embedding_size
        self.compute_forces = compute_forces

        self.atom_embedding = nn.Embedding(num_embedding, hidden_state_size)
        self.node_projection = nn.Linear(hidden_state_size, hidden_state_size)

        # Learnable radial basis
        self.rbf = GaussianRBF(edge_embedding_size, cutoff)

        self.interactions = nn.ModuleList(
            [
                Interaction(hidden_state_size, edge_embedding_size, cutoff)
                for _ in range(num_interactions)
            ]
        )

        # Gated equivariant readout: use vector channel to gate scalar
        self.readout_vector_proj = nn.Linear(hidden_state_size, hidden_state_size)
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_state_size * 2, hidden_state_size),
            nn.SiLU(),
            nn.Linear(hidden_state_size, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.atom_embedding.weight, mean=0.0, std=0.05)

    def _build_graph(self, input_dict):
        """Extract and offset graph indices for batched inputs."""
        num_atoms = input_dict['num_atoms']
        ads_num_pairs = input_dict['ads_num_pairs']

        ads_idx_pairs = input_dict['ads_idx_pairs']
        ads_idx_offset = torch.cumsum(
            torch.cat((torch.tensor([0], device=num_atoms.device, dtype=num_atoms.dtype), num_atoms[:-1])),
            dim=0,
        )
        ads_idx_offset = torch.repeat_interleave(ads_idx_offset, ads_num_pairs)
        ads_idx_pairs = ads_idx_pairs + ads_idx_offset.unsqueeze(-1)
        _, idx = torch.unique_consecutive(ads_idx_pairs[:, 0], return_inverse=True)

        mapper = torch.stack((ads_idx_pairs[:, 0], idx), dim=1)
        mapper = torch.unique_consecutive(mapper, dim=0)
        if ads_idx_pairs.numel() > 0:
            max_key = int(ads_idx_pairs.max().item())
            mapper_dict = torch.full((max_key + 1,), -1, device=ads_idx_pairs.device, dtype=ads_idx_pairs.dtype)
            mapper_dict[mapper[:, 0]] = mapper[:, 1]
            ads_idx_pairs_compact = mapper_dict[ads_idx_pairs]
        else:
            raise ValueError('No ads_idx_pairs found!')

        local_idx = torch.unique_consecutive(ads_idx_pairs[:, 0])
        return ads_idx_pairs, ads_idx_pairs_compact, local_idx

    def forward(self, input_dict):
        coords = input_dict['coords']
        ads_num_atoms = input_dict['ads_num_atoms']

        # Build graph
        ads_idx_pairs, ads_idx_pairs_compact, local_idx = self._build_graph(input_dict)

        ads_edge_diff = input_dict['ads_edge_diff']
        edge_dist = torch.linalg.norm(ads_edge_diff, dim=1)

        # Radial basis
        rbf = self.rbf(edge_dist)

        # Initialize node features
        all_scalar = self.atom_embedding(input_dict['elems'])
        node_scalar = self.node_projection(all_scalar[local_idx])

        node_vector = torch.zeros(
            (node_scalar.shape[0], 3, self.hidden_state_size),
            device=coords.device, dtype=coords.dtype,
        )

        # Message passing with residual connections across interactions
        for interaction in self.interactions:
            scalar_in = node_scalar
            node_scalar, node_vector = interaction(
                node_scalar, node_vector, ads_idx_pairs_compact, ads_edge_diff, edge_dist, rbf
            )
            node_scalar = node_scalar + scalar_in

        # Gated equivariant readout
        vector_norm = torch.linalg.norm(
            self.readout_vector_proj(node_vector), dim=1
        )
        readout_input = torch.cat((node_scalar, vector_norm), dim=-1)
        atom_energies = self.readout_mlp(readout_input).squeeze(-1)

        if not bool(atom_energies.shape):
            atom_energies = atom_energies.unsqueeze(0)

        image_idx = torch.arange(ads_num_atoms.shape[0], device=ads_idx_pairs.device)
        image_idx = torch.repeat_interleave(image_idx, ads_num_atoms)

        ads_energy = torch.zeros_like(ads_num_atoms).float()
        ads_energy.index_add_(0, image_idx, atom_energies)

        result_dict = {
            'ads_energy': ads_energy,
            'local_idx': local_idx,
            'local_energies': atom_energies,
        }

        # Forces via autograd
        if self.compute_forces:
            dE_ddiff = torch.autograd.grad(
                ads_energy,
                ads_edge_diff,
                grad_outputs=torch.ones_like(ads_energy),
                retain_graph=True,
                create_graph=True,
            )[0]

            i_forces = torch.zeros_like(input_dict['ads_coords']).index_add(0, ads_idx_pairs_compact[:, 0], dE_ddiff)
            j_forces = torch.zeros_like(input_dict['ads_coords']).index_add(0, ads_idx_pairs_compact[:, 1], -dE_ddiff)
            result_dict['ads_forces'] = i_forces + j_forces

        return result_dict
