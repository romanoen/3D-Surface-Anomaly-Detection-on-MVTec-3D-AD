"""Prepare dataset indices, split files, and dataset summary figures."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "mvtec_3d_ad_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataset_index import build_index, save_index, split_index, summarize_index


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build dataset index CSV files and a dataset summary figure."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the base config file. Defaults to configs/base.yaml.",
    )
    return parser.parse_args()


def resolve_repo_root() -> Path:
    """Return the repository root based on this script location."""
    return Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict:
    """Load the project config file."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_summary_figure(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Save a simple grouped bar chart for dataset counts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pivot_df = (
        summary_df.groupby(["category", "split"], dropna=False)["count"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=["train", "val", "test"], fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    pivot_df.plot(kind="bar", ax=ax, width=0.85)
    ax.set_title("Dataset sample counts by category and split")
    ax.set_xlabel("Category")
    ax.set_ylabel("Number of samples")
    ax.legend(title="Split")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Build the first data-preparation artifacts for the project."""
    args = parse_args()
    repo_root = resolve_repo_root()
    config_path = args.config or repo_root / "configs" / "base.yaml"
    config = load_config(config_path)

    dataset_root = repo_root / config["paths"]["dataset_root"]
    splits_root = repo_root / config["paths"]["splits_root"]
    figure_root = repo_root / config["paths"]["fig_root"] / "01_dataset"
    categories = config["data"]["categories"]

    index_df = build_index(dataset_root, categories)
    summary_df = summarize_index(index_df)

    save_index(index_df, splits_root / "index.csv")
    save_index(summary_df, splits_root / "index_summary.csv")

    split_tables = split_index(index_df)
    save_index(split_tables["train"], splits_root / "train.csv")
    save_index(split_tables["val"], splits_root / "val.csv")
    save_index(split_tables["test"], splits_root / "test.csv")

    save_summary_figure(summary_df, figure_root / "index_summary.png")

    print(f"Saved {len(index_df)} rows to {(splits_root / 'index.csv').relative_to(repo_root)}")
    print(f"Saved split tables to {splits_root.relative_to(repo_root)}")
    print(f"Saved summary table to {(splits_root / 'index_summary.csv').relative_to(repo_root)}")
    print(f"Saved summary figure to {(figure_root / 'index_summary.png').relative_to(repo_root)}")


if __name__ == "__main__":
    main()
