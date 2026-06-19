from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from src.data import load_mat_data
from src.train import train_full_batch
from src.utils import PRESETS, evaluate_scores, set_seed


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AnomalyDAE PyG multi-trial evaluation")
    parser.add_argument("--dataset", type=str, default="Disney", help="Dataset name, or comma-separated names such as ACM,Flickr")
    parser.add_argument("--data_dir", type=str, default="~/datasets/GAD/mat")
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=[None, "blogcatalog", "flickr", "acm"],
        help="Use paper-like hyper-parameters",
    )
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
    parser.add_argument("--seed", type=int, default=42, help="Base seed; trial i uses seed + i")
    parser.add_argument("--trials", type=int, default=1, help="Number of repeated trials")
    parser.add_argument("--feature_normalize", type=str, default="none", choices=["none", "row", "standard"])
    parser.add_argument("--directed", action="store_true", help="Do not symmetrize edges")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--result_csv", type=str, default="outputs/results.csv", help="CSV path for summarized results")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite result_csv instead of appending")
    parser.add_argument("--verbose", action="store_true", help="Print trial progress and enable training progress bar")
    parser.add_argument("--dis_tqdm", action="store_true", help="Disable training progress bar")
    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.preset is None:
        return args
    for key, value in PRESETS[args.preset].items():
        setattr(args, key, value)
    return args


def parse_datasets(dataset_arg: str) -> list[str]:
    datasets = [item.strip() for item in dataset_arg.split(",") if item.strip()]
    if not datasets:
        raise ValueError("--dataset cannot be empty")
    return datasets


def format_metric(values: Iterable[float]) -> str:
    arr = np.asarray(list(values), dtype=float) * 100.0
    if arr.size == 0:
        raise ValueError("metric values cannot be empty")
    return f"{arr.mean():.2f}±{arr.std():.2f}({arr.max():.2f})"


def append_summary_row(csv_path: str | Path, row: dict[str, str], overwrite: bool = False) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    mode = "w" if overwrite else "a"
    write_header = overwrite or not file_exists

    with csv_path.open(mode, newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["datetime", "dataset", "trial", "auc", "auprc"])
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def evaluate_one_dataset(args: argparse.Namespace, dataset: str) -> dict[str, str]:
    data_path = os.path.join(os.path.expanduser(args.data_dir), dataset + ".mat")
    data = load_mat_data(
        data_path,
        make_undirected=not args.directed,
        feature_normalize=None if args.feature_normalize == "none" else args.feature_normalize,
    )

    if not hasattr(data, "y"):
        raise ValueError(f"Dataset {dataset} has no label field; cannot compute AUC/AUPRC.")

    auc_values: list[float] = []
    auprc_values: list[float] = []

    for trial in range(args.trials):
        trial_seed = args.seed + trial
        set_seed(trial_seed)
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
            log_every=0,
            compile_model=args.compile,
            show_progress=args.verbose,
            tqdm=not args.dis_tqdm,
        )
        metrics = evaluate_scores(data.y.cpu().numpy(), result.scores.numpy())
        if "auroc" not in metrics or "auprc" not in metrics:
            raise ValueError(f"Dataset {dataset} labels contain only one class; cannot compute AUC/AUPRC.")
        auc_values.append(metrics["auroc"])
        auprc_values.append(metrics["auprc"])
        if args.verbose:
            print(
                f"{dataset} trial {trial + 1}/{args.trials}: "
                f"auc={metrics['auroc'] * 100:.2f}, auprc={metrics['auprc'] * 100:.2f}"
            )

    return {
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset,
        "trial": str(args.trials),
        "auc": format_metric(auc_values),
        "auprc": format_metric(auprc_values),
    }


def main() -> None:
    args = apply_preset(build_args())
    if args.trials <= 0:
        raise ValueError("--trials must be a positive integer")

    rows: list[dict[str, str]] = []
    for idx, dataset in enumerate(parse_datasets(args.dataset)):
        row = evaluate_one_dataset(args, dataset)
        append_summary_row(args.result_csv, row, overwrite=args.overwrite and idx == 0)
        rows.append(row)

    for row in rows:
        print(
            f"{row['datetime']},{row['dataset']},{row['trial']},"
            f"{row['auc']},{row['auprc']}"
        )
    print(f"Saved results to: {args.result_csv}")


if __name__ == "__main__":
    main()
