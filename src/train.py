from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch_geometric.utils import to_dense_adj
from tqdm import trange

from .loss import anomalydae_loss
from .model import AnomalyDAE


@dataclass
class TrainResult:
    scores: torch.Tensor
    struct_scores: torch.Tensor
    attr_scores: torch.Tensor
    embedding: torch.Tensor
    final_loss: float


def train_full_batch(
    data,
    emb_dim: int = 128,
    hid_dim: int = 128,
    heads: int = 1,
    dropout: float = 0.0,
    alpha: float = 0.7,
    eta: float = 5.0,
    theta: float = 40.0,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    epochs: int = 100,
    device: str = "cpu",
    log_every: int = 10,
    compile_model: bool = False,
) -> TrainResult:
    """Train AnomalyDAE in full-batch mode.

    AnomalyDAE reconstructs a dense adjacency matrix, so full-batch training is
    faithful to the paper but has O(N^2) memory cost. For very large graphs,
    use a sampled/minibatch variant instead.
    """
    device_obj = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    data = data.to(device_obj)
    x = data.x.float()
    edge_index = data.edge_index.long()
    num_nodes = data.num_nodes
    in_dim = x.size(1)

    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].float()
    adj.fill_diagonal_(0.0)
    adj = (adj > 0).float()

    model = AnomalyDAE(
        num_nodes=num_nodes,
        in_dim=in_dim,
        emb_dim=emb_dim,
        hid_dim=hid_dim,
        dropout=dropout,
        heads=heads,
    ).to(device_obj)

    if compile_model:
        model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    final_loss = 0.0

    iterator = trange(epochs, desc="Training", leave=True)
    for epoch in iterator:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        x_hat, adj_hat, _, _ = model(x, edge_index)
        loss, scores, _, _ = anomalydae_loss(
            x=x,
            x_hat=x_hat,
            adj=adj,
            adj_hat=adj_hat,
            alpha=alpha,
            eta=eta,
            theta=theta,
        )
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())

        if log_every > 0 and ((epoch + 1) % log_every == 0 or epoch == 0):
            iterator.set_postfix(loss=f"{final_loss:.6f}")

    model.eval()
    with torch.no_grad():
        x_hat, adj_hat, z_v, _ = model(x, edge_index)
        loss, scores, struct_scores, attr_scores = anomalydae_loss(
            x=x,
            x_hat=x_hat,
            adj=adj,
            adj_hat=adj_hat,
            alpha=alpha,
            eta=eta,
            theta=theta,
        )

    return TrainResult(
        scores=scores.detach().cpu(),
        struct_scores=struct_scores.detach().cpu(),
        attr_scores=attr_scores.detach().cpu(),
        embedding=z_v.detach().cpu(),
        final_loss=float(loss.detach().cpu()),
    )
