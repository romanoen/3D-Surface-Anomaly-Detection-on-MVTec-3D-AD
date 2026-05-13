"""Train the classical anomaly detection baseline and save figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.training.classical_trainer import train_classical_model
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the classical One-Class SVM baseline on normal patches."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help="Path to the base config file. Defaults to configs/base.yaml.",
    )
    parser.add_argument(
        "--classical-config",
        type=Path,
        default=None,
        help="Path to the classical config file. Defaults to configs/classical.yaml.",
    )
    parser.add_argument(
        "--max-train-patches",
        type=int,
        default=None,
        help="Override the maximum number of training patches used per category.",
    )
    parser.add_argument(
        "--max-val-patches",
        type=int,
        default=None,
        help="Override the maximum number of validation patches scored per category.",
    )
    return parser.parse_args()


def resolve_repo_root() -> Path:
    """Return the repository root based on this script location."""
    return Path(__file__).resolve().parents[1]


def main() -> None:
    """Run classical training."""
    args = parse_args()
    repo_root = resolve_repo_root()
    base_config_path = args.base_config or repo_root / "configs" / "base.yaml"
    classical_config_path = args.classical_config or repo_root / "configs" / "classical.yaml"
    cfg = load_config(base_config_path, classical_config_path)

    if args.max_train_patches is not None:
        cfg["training"]["max_train_patches"] = int(args.max_train_patches)
    if args.max_val_patches is not None:
        cfg["training"]["max_val_patches"] = int(args.max_val_patches)

    splits_root = repo_root / cfg["paths"]["splits_root"]
    train_df = pd.read_csv(splits_root / "train.csv", keep_default_na=False)
    val_df = pd.read_csv(splits_root / "val.csv", keep_default_na=False)

    result = train_classical_model(train_df, val_df, cfg, repo_root=repo_root)

    print(f"Saved per-category models under {result.model_root.relative_to(repo_root)}")
    print(f"Saved summary to {result.summary_path.relative_to(repo_root)}")
    print(f"Saved category summary to {result.category_summary_path.relative_to(repo_root)}")
    print(
        f"Trained {len(result.category_results)} category-specific OCSVMs on "
        f"{result.train_patch_count} patches with {len(result.selected_feature_names)} model features; "
        f"scored {result.val_patch_count} validation patches."
    )


if __name__ == "__main__":
    main()
