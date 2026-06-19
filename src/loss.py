from __future__ import annotations

import torch


def anomalydae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    adj: torch.Tensor,
    adj_hat: torch.Tensor,
    alpha: float = 0.7,
    eta: float = 5.0,
    theta: float = 40.0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """AnomalyDAE double reconstruction loss.

    The paper defines the objective as:
        alpha * ||(A - A_hat) * Theta||_F^2
      + (1 - alpha) * ||(X - X_hat) * Eta||_F^2

    For node-level scores, this implementation keeps the first dimension and
    returns one score for each node.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if eta < 1.0 or theta < 1.0:
        raise ValueError("eta and theta should be >= 1 according to the paper")

    attr_weight = torch.ones_like(x)
    attr_weight = torch.where(x != 0, attr_weight * eta, attr_weight)
    attr_diff = torch.pow((x - x_hat) * attr_weight, 2)
    attr_score = torch.sqrt(torch.sum(attr_diff, dim=1) + eps)

    struct_weight = torch.ones_like(adj)
    struct_weight = torch.where(adj != 0, struct_weight * theta, struct_weight)
    struct_diff = torch.pow((adj - adj_hat) * struct_weight, 2)
    struct_score = torch.sqrt(torch.sum(struct_diff, dim=1) + eps)

    score = alpha * struct_score + (1.0 - alpha) * attr_score
    loss = torch.mean(score)
    return loss, score, struct_score, attr_score
