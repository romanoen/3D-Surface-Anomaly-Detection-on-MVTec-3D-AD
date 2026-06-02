"""Train the autoencoder baseline and save figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.training.autoencoder_trainer import train_autoencoder_model
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the compact convolutional autoencoder on normal depth patches."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help="Path to the base config file. Defaults to configs/base.yaml.",
    )
    parser.add_argument(
        "--autoencoder-config",
        type=Path,
        default=None,
        help="Path to the autoencoder config file. Defaults to configs/autoencoder.yaml.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the maximum number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the patch batch size.",
    )
    parser.add_argument(
        "--max-train-patches",
        type=int,
        default=None,
        help="Optionally subsample the normal training patches.",
    )
    parser.add_argument(
        "--max-val-patches",
        type=int,
        default=None,
        help="Optionally subsample the normal validation patches.",
    )
    return parser.parse_args()


def resolve_repo_root() -> Path:
    """Return the repository root based on this script location."""
    return Path(__file__).resolve().parents[1]


def main() -> None:
    """Run autoencoder training."""
    args = parse_args()
    repo_root = resolve_repo_root()
    base_config_path = args.base_config or repo_root / "configs" / "base.yaml"
    autoencoder_config_path = args.autoencoder_config or repo_root / "configs" / "autoencoder.yaml"
    cfg = load_config(base_config_path, autoencoder_config_path)

    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = int(args.batch_size)
    if args.max_train_patches is not None:
        cfg["training"]["max_train_patches"] = int(args.max_train_patches)
    if args.max_val_patches is not None:
        cfg["training"]["max_val_patches"] = int(args.max_val_patches)

    splits_root = repo_root / cfg["paths"]["splits_root"]
    train_df = pd.read_csv(splits_root / "train.csv", keep_default_na=False)
    val_df = pd.read_csv(splits_root / "val.csv", keep_default_na=False)

    result = train_autoencoder_model(train_df, val_df, cfg, repo_root=repo_root)

    print(f"Saved per-category models under {result.model_root.relative_to(repo_root)}")
    print(f"Saved summary to {result.summary_path.relative_to(repo_root)}")
    print(f"Saved category summary to {result.category_summary_path.relative_to(repo_root)}")
    print(
        f"Trained {len(result.category_results)} category-specific autoencoders on "
        f"{result.train_patch_count} normal patches and validated on {result.val_patch_count} patches."
    )


if __name__ == "__main__":
    main()
