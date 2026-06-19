from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, remove_self_loops


ADJ_KEYS = ("Network", "A", "adj", "Adj", "network", "graph")
FEAT_KEYS = ("Attributes", "X", "features", "Features", "attrb", "attr")
LABEL_KEYS = ("Label", "label", "labels", "Labels", "y", "gnd", "GroundTruth", "anomaly_label")


def _first_existing(mat: dict, keys: Tuple[str, ...], kind: str):
    for key in keys:
        if key in mat:
            return mat[key], key
    visible = [k for k in mat.keys() if not k.startswith("__")]
    raise KeyError(f"Cannot find {kind}. Tried keys={keys}. Existing keys={visible}")


def _to_numpy_dense(x):
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x)


def _to_scipy_csr(a) -> sp.csr_matrix:
    if sp.issparse(a):
        return a.tocsr()
    return sp.csr_matrix(np.asarray(a))


def _adj_to_edge_index(adj: sp.csr_matrix, make_undirected: bool = True) -> torch.Tensor:
    adj = adj.tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64))
    col = torch.from_numpy(adj.col.astype(np.int64))
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = remove_self_loops(edge_index)
    if make_undirected:
        rev = edge_index.flip(0)
        edge_index = torch.cat([edge_index, rev], dim=1)
    edge_index = coalesce(edge_index)
    return edge_index


def load_mat_data(
    path: str | Path,
    make_undirected: bool = True,
    binarize_adj: bool = True,
    feature_normalize: Optional[str] = None,
) -> Data:
    """Load common graph anomaly detection .mat files as a PyG Data object.

    Parameters
    ----------
    path:
        Path to .mat file.
    make_undirected:
        Whether to symmetrize edge_index.
    binarize_adj:
        Whether to convert all non-zero adjacency values to 1.
    feature_normalize:
        None, "row", or "standard".
    """
    path = Path(path)
    mat = sio.loadmat(path)

    adj_raw, adj_key = _first_existing(mat, ADJ_KEYS, "adjacency matrix")
    feat_raw, feat_key = _first_existing(mat, FEAT_KEYS, "node feature matrix")

    adj = _to_scipy_csr(adj_raw)
    if binarize_adj:
        adj.data[:] = 1.0
    adj.setdiag(0)
    adj.eliminate_zeros()

    x_np = _to_numpy_dense(feat_raw).astype(np.float32)
    if feature_normalize == "row":
        denom = np.maximum(np.abs(x_np).sum(axis=1, keepdims=True), 1e-12)
        x_np = x_np / denom
    elif feature_normalize == "standard":
        mean = x_np.mean(axis=0, keepdims=True)
        std = x_np.std(axis=0, keepdims=True)
        std[std < 1e-12] = 1.0
        x_np = (x_np - mean) / std
    elif feature_normalize not in (None, "none"):
        raise ValueError("feature_normalize must be one of: none, row, standard")

    edge_index = _adj_to_edge_index(adj, make_undirected=make_undirected)
    x = torch.from_numpy(x_np).float()

    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = x.size(0)
    data.adj_key = adj_key
    data.feat_key = feat_key

    for key in LABEL_KEYS:
        if key in mat:
            y = np.asarray(mat[key]).reshape(-1)
            # Some datasets use {-1, 1}; convert to {0, 1} when needed.
            unique = set(np.unique(y).tolist())
            if unique == {-1, 1}:
                y = (y == 1).astype(np.int64)
            data.y = torch.from_numpy(y.astype(np.int64))
            data.label_key = key
            break

    return data
