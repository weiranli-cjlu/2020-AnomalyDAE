from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


PRESETS = {
    "blogcatalog": {"epochs": 100, "alpha": 0.7, "eta": 5.0, "theta": 40.0, "emb_dim": 128, "hid_dim": 128, "lr": 0.001},
    "flickr": {"epochs": 100, "alpha": 0.9, "eta": 8.0, "theta": 90.0, "emb_dim": 128, "hid_dim": 128, "lr": 0.001},
    "acm": {"epochs": 80, "alpha": 0.7, "eta": 3.0, "theta": 10.0, "emb_dim": 128, "hid_dim": 128, "lr": 0.001},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def evaluate_scores(y_true: torch.Tensor | np.ndarray, scores: torch.Tensor | np.ndarray) -> dict:
    y = np.asarray(y_true).reshape(-1)
    s = np.asarray(scores).reshape(-1)
    out = {}
    if len(np.unique(y)) >= 2:
        out["auroc"] = float(roc_auc_score(y, s))
        out["auprc"] = float(average_precision_score(y, s))
    return out


def topk_indices(scores: np.ndarray, k: Optional[int] = None, contamination: Optional[float] = None) -> np.ndarray:
    scores = np.asarray(scores).reshape(-1)
    n = scores.shape[0]
    if k is None:
        if contamination is None:
            contamination = 0.1
        k = max(1, int(round(n * contamination)))
    k = min(max(int(k), 1), n)
    return np.argsort(-scores)[:k]


def save_outputs(scores: np.ndarray, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = np.asarray(scores).reshape(-1)
    np.save(output_dir / "anomaly_scores.npy", scores)

    order = np.argsort(-scores)
    with open(output_dir / "ranking.csv", "w", encoding="utf-8") as f:
        f.write("rank,node_id,score\n")
        for rank, idx in enumerate(order, start=1):
            f.write(f"{rank},{int(idx)},{float(scores[idx])}\n")
