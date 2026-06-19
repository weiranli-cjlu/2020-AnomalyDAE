from __future__ import annotations

import os
import argparse
from pathlib import Path

import numpy as np

from src.data import load_mat_data
from src.train import train_full_batch
from src.utils import PRESETS, evaluate_scores, save_outputs, set_seed, topk_indices


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AnomalyDAE PyG minimal reproduction")
    parser.add_argument("--dataset", type=str, default="Disney")
    parser.add_argument("--data_dir", type=str, default="~/datasets/GAD/mat")
    parser.add_argument("--preset", type=str, default=None, choices=[None, "blogcatalog", "flickr", "acm"], help="Use paper-like hyper-parameters")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--hid_dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.7, help="Structure reconstruction weight")
    parser.add_argument("--eta", type=float, default=5.0, help="Non-zero feature reconstruction penalty")
    parser.add_argument("--theta", type=float, default=40.0, help="Non-zero structure reconstruction penalty")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--feature_normalize", type=str, default="none", choices=["none", "row", "standard"])
    parser.add_argument("--directed", action="store_true", help="Do not symmetrize edges")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--contamination", type=float, default=0.1)
    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset is None:
        return args
    for key, value in PRESETS[args.preset].items():
        setattr(args, key, value)
    return args


def main() -> None:
    args = apply_preset(build_args())
    set_seed(args.seed)
    data = load_mat_data(
        os.path.join(os.path.expanduser(args.data_dir), args.dataset+".mat"),
        make_undirected=not args.directed,
        feature_normalize=None if args.feature_normalize == "none" else args.feature_normalize,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Nodes={data.num_nodes}, Edges={data.edge_index.size(1)}, Features={data.x.size(1)}")
    print(f"Feature key={getattr(data, 'feat_key', None)}, Adj key={getattr(data, 'adj_key', None)}")
    if hasattr(data, "y"):
        print(f"Label key={getattr(data, 'label_key', None)}, Anomalies={int(data.y.sum())}")

    result = train_full_batch(
        data=data,
        emb_dim=args.emb_dim,
        hid_dim=args.hid_dim,
        heads=args.heads,
        dropout=args.dropout,
        alpha=args.alpha,
        eta=args.eta,
        theta=args.theta,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=args.device,
        log_every=args.log_every,
        compile_model=args.compile,
    )

    scores = result.scores.numpy()
    print(f"Final loss: {result.final_loss:.6f}")

    if hasattr(data, "y"):
        metrics = evaluate_scores(data.y.cpu().numpy(), scores)
        if metrics:
            print(f"AUROC: {metrics['auroc']:.4f}")
            print(f"AUPRC: {metrics['auprc']:.4f}")
        else:
            print("Labels contain only one class; AUROC/AUPRC skipped.")

    top_idx = topk_indices(scores, k=args.topk, contamination=args.contamination)
    print("Top anomaly node ids:", top_idx[:20].tolist())

    save_outputs(scores, args.output_dir)
    np.save(Path(args.output_dir) / "structure_scores.npy", result.struct_scores.numpy())
    np.save(Path(args.output_dir) / "attribute_scores.npy", result.attr_scores.numpy())
    np.save(Path(args.output_dir) / "embedding.npy", result.embedding.numpy())
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
