"""Benchmark both anomaly detectors and save metrics plus result figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.benchmark import run_benchmark
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the shared classical-vs-autoencoder benchmark on one split."
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split to benchmark. Defaults to test.",
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
        "--autoencoder-config",
        type=Path,
        default=None,
        help="Path to the autoencoder config file. Defaults to configs/autoencoder.yaml.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optionally benchmark only the first N images from the split.",
    )
    parser.add_argument(
        "--max-heatmap-figures",
        type=int,
        default=None,
        help="Override the per-category qualitative heatmap budget for both methods.",
    )
    parser.add_argument(
        "--max-comparison-figures",
        type=int,
        default=None,
        help="Override the number of side-by-side comparison figures saved per category.",
    )
    return parser.parse_args()


def resolve_repo_root() -> Path:
    """Return the repository root based on this script location."""
    return Path(__file__).resolve().parents[1]


def main() -> None:
    """Run the shared benchmark."""
    args = parse_args()
    repo_root = resolve_repo_root()
    base_config_path = args.base_config or repo_root / "configs" / "base.yaml"
    classical_config_path = args.classical_config or repo_root / "configs" / "classical.yaml"
    autoencoder_config_path = args.autoencoder_config or repo_root / "configs" / "autoencoder.yaml"

    classical_cfg = load_config(base_config_path, classical_config_path)
    autoencoder_cfg = load_config(base_config_path, autoencoder_config_path)

    if args.max_heatmap_figures is not None:
        classical_cfg.setdefault("inference", {})
        autoencoder_cfg.setdefault("inference", {})
        classical_cfg["inference"]["max_heatmap_figures_per_category"] = int(args.max_heatmap_figures)
        autoencoder_cfg["inference"]["max_heatmap_figures_per_category"] = int(args.max_heatmap_figures)

    split_path = repo_root / classical_cfg["paths"]["splits_root"] / f"{args.split}.csv"
    split_df = pd.read_csv(split_path, keep_default_na=False)
    result = run_benchmark(
        split_df,
        classical_cfg,
        autoencoder_cfg,
        repo_root=repo_root,
        split_name=args.split,
        max_images=args.max_images,
        max_comparison_figures_per_category=(
            int(args.max_comparison_figures)
            if args.max_comparison_figures is not None
            else int(autoencoder_cfg.get("inference", {}).get("max_heatmap_figures_per_category", 4))
        ),
    )

    print(f"Saved per-image comparison table to {result.per_image_path.relative_to(repo_root)}")
    print(f"Saved per-category metrics to {result.per_category_path.relative_to(repo_root)}")
    print(f"Saved summary metrics to {result.summary_path.relative_to(repo_root)}")
    print(
        f"Benchmarked {result.image_count} images across {result.category_count} categories; "
        f"comparison figures live under {result.figure_root.relative_to(repo_root)}."
    )


if __name__ == "__main__":
    main()
