"""Run classical model inference and save image scores plus heatmaps."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.inference.classical_inference import run_classical_inference
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Score a split with the trained classical One-Class SVM baseline."
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split to score. Defaults to test.",
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
        "--max-images",
        type=int,
        default=None,
        help="Optionally score only the first N images from the split.",
    )
    parser.add_argument(
        "--max-heatmap-figures",
        type=int,
        default=None,
        help="Override how many qualitative heatmap figures are saved per category.",
    )
    return parser.parse_args()


def resolve_repo_root() -> Path:
    """Return the repository root based on this script location."""
    return Path(__file__).resolve().parents[1]


def main() -> None:
    """Run classical inference."""
    args = parse_args()
    repo_root = resolve_repo_root()
    base_config_path = args.base_config or repo_root / "configs" / "base.yaml"
    classical_config_path = args.classical_config or repo_root / "configs" / "classical.yaml"
    cfg = load_config(base_config_path, classical_config_path)

    if args.max_heatmap_figures is not None:
        cfg["inference"]["max_heatmap_figures_per_category"] = int(args.max_heatmap_figures)

    split_path = repo_root / cfg["paths"]["splits_root"] / f"{args.split}.csv"
    split_df = pd.read_csv(split_path, keep_default_na=False)

    result = run_classical_inference(
        split_df,
        cfg,
        repo_root=repo_root,
        split_name=args.split,
        max_images=args.max_images,
    )

    print(f"Saved image scores to {result.image_scores_path.relative_to(repo_root)}")
    print(f"Saved patch scores to {result.patch_scores_path.relative_to(repo_root)}")
    print(f"Saved heatmaps under {result.heatmap_root.relative_to(repo_root)}")
    print(
        f"Scored {result.image_count} images and {result.patch_count} patches "
        f"from the {args.split} split."
    )


if __name__ == "__main__":
    main()
