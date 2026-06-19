from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class AnomalyDAE(nn.Module):
    """Minimal PyG AnomalyDAE.

    Structure AE:
        X -> Linear -> GAT -> Z_v -> sigmoid(Z_v Z_v^T)

    Attribute AE:
        X^T -> Linear -> Linear -> Z_a
        X_hat = Z_v Z_a^T
    """

    def __init__(
        self,
        num_nodes: int,
        in_dim: int,
        emb_dim: int = 128,
        hid_dim: int = 128,
        dropout: float = 0.0,
        heads: int = 1,
        gat_dropout: float | None = None,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.dropout = dropout

        if gat_dropout is None:
            gat_dropout = dropout

        self.structure_fc = nn.Linear(in_dim, emb_dim)
        self.structure_gat = GATConv(
            emb_dim,
            hid_dim,
            heads=heads,
            concat=False,
            dropout=gat_dropout,
            add_self_loops=True,
        )

        self.attribute_fc1 = nn.Linear(num_nodes, emb_dim)
        self.attribute_fc2 = nn.Linear(emb_dim, hid_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.structure_fc.weight)
        nn.init.zeros_(self.structure_fc.bias)
        nn.init.xavier_uniform_(self.attribute_fc1.weight)
        nn.init.zeros_(self.attribute_fc1.bias)
        nn.init.xavier_uniform_(self.attribute_fc2.weight)
        nn.init.zeros_(self.attribute_fc2.bias)
        self.structure_gat.reset_parameters()

    def encode_structure(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.structure_fc(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        z_v = self.structure_gat(h, edge_index)
        return z_v

    def encode_attribute(self, x: torch.Tensor) -> torch.Tensor:
        # Treat each attribute dimension as an object described by all nodes.
        z_a = self.attribute_fc1(x.t())
        z_a = F.relu(z_a)
        z_a = F.dropout(z_a, p=self.dropout, training=self.training)
        z_a = self.attribute_fc2(z_a)
        z_a = F.dropout(z_a, p=self.dropout, training=self.training)
        return z_a

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        z_v = self.encode_structure(x, edge_index)
        z_a = self.encode_attribute(x)

        adj_hat = torch.sigmoid(z_v @ z_v.t())
        x_hat = z_v @ z_a.t()
        return x_hat, adj_hat, z_v, z_a
