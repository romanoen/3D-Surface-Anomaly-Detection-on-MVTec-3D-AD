"""Prepare dataset indices, split files, processed maps, and summary figures."""

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
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from src.data.dataset_index import build_index, save_index, split_index, summarize_index
from src.data.patching import (
    compute_patch_coverage,
    extract_patches_with_mask,
)
from src.data.preprocessing import (
    build_processed_output_paths,
    build_valid_mask,
    extract_depth_map,
    load_xyz_map,
    preprocess_xyz_path,
    save_processed_sample,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build dataset indices, processed maps, split CSV files, and figures."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the base config file. Defaults to configs/base.yaml.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of samples to preprocess for quick checks.",
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


def normalize_for_display(depth_map: np.ndarray, valid_mask: np.ndarray) -> np.ma.MaskedArray:
    """Normalize a depth map for display only on valid pixels."""
    if not np.any(valid_mask):
        return np.ma.array(np.zeros_like(depth_map, dtype=np.float32), mask=np.ones_like(valid_mask, dtype=bool))

    valid_values = depth_map[valid_mask]
    low = float(valid_values.min())
    high = float(valid_values.max())
    scale = max(high - low, 1e-6)
    normalized = np.zeros_like(depth_map, dtype=np.float32)
    normalized[valid_mask] = (depth_map[valid_mask] - low) / scale
    return np.ma.array(normalized, mask=~valid_mask)


def save_preprocessing_figure(
    category: str,
    sample_id: str,
    raw_depth: np.ndarray,
    raw_mask: np.ndarray,
    processed_depth: np.ndarray,
    processed_mask: np.ndarray,
    crop_box: tuple[int, int, int, int],
    output_path: Path,
) -> None:
    """Save a side-by-side raw vs processed preprocessing figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")

    raw_preview = normalize_for_display(raw_depth, raw_mask)
    processed_preview = np.ma.array(processed_depth, mask=~processed_mask)

    processed_aspect = processed_depth.shape[1] / float(processed_depth.shape[0])
    if processed_aspect >= 2.5:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), layout="constrained")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    axes = np.ravel(axes)

    axes[0].imshow(raw_preview, cmap=cmap, vmin=0.0, vmax=1.0)
    crop_top, crop_bottom, crop_left, crop_right = crop_box
    crop_rectangle = Rectangle(
        (crop_left, crop_top),
        crop_right - crop_left,
        crop_bottom - crop_top,
        fill=False,
        edgecolor="red",
        linewidth=2.0,
    )
    axes[0].add_patch(crop_rectangle)
    axes[0].set_title("Raw masked depth")
    axes[0].axis("off")

    axes[1].imshow(processed_preview, cmap=cmap, vmin=0.0, vmax=1.0)
    axes[1].set_title(
        f"Processed crop ({processed_depth.shape[0]}x{processed_depth.shape[1]})"
    )
    axes[1].axis("off")

    fig.suptitle(f"{category} preprocessing preview ({sample_id})")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_patch_overview_figure(
    category: str,
    sample_id: str,
    processed_depth: np.ndarray,
    processed_mask: np.ndarray,
    patch_size: tuple[int, int],
    stride: tuple[int, int],
    output_path: Path,
) -> None:
    """Save one patch-grid, sliding-order, and coverage-count figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")

    _, _, coords = extract_patches_with_mask(
        processed_depth,
        processed_mask,
        patch_size=patch_size,
        stride=stride,
    )
    coverage_map = compute_patch_coverage(coords, processed_depth.shape)

    depth_preview = np.ma.array(processed_depth, mask=~processed_mask)
    patch_centers_x = (coords[:, 2] + coords[:, 3]) / 2.0
    patch_centers_y = (coords[:, 0] + coords[:, 1]) / 2.0
    patch_order = np.arange(len(coords), dtype=np.float32)

    image_aspect = processed_depth.shape[1] / float(processed_depth.shape[0])
    if image_aspect >= 2.5:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), layout="constrained")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), layout="constrained")
    axes = np.ravel(axes)

    axes[0].imshow(depth_preview, cmap=cmap, vmin=0.0, vmax=1.0)
    for top, bottom, left, right in coords:
        axes[0].add_patch(
            Rectangle(
                (left, top),
                right - left,
                bottom - top,
                fill=False,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.35,
            )
        )
    axes[0].set_title(
        f"Patch grid ({len(coords)} patches, image={processed_depth.shape[0]}x{processed_depth.shape[1]})"
    )
    axes[0].axis("off")

    axes[1].imshow(depth_preview, cmap=cmap, vmin=0.0, vmax=1.0)
    axes[1].plot(
        patch_centers_x,
        patch_centers_y,
        color="white",
        linewidth=0.8,
        alpha=0.6,
    )
    order_artist = axes[1].scatter(
        patch_centers_x,
        patch_centers_y,
        c=patch_order,
        cmap="plasma",
        s=9,
        edgecolors="none",
    )
    for patch_index in (0, len(coords) // 2, len(coords) - 1):
        top, bottom, left, right = coords[patch_index]
        axes[1].add_patch(
            Rectangle(
                (left, top),
                right - left,
                bottom - top,
                fill=False,
                edgecolor="white",
                linewidth=1.6,
            )
        )
    axes[1].set_title("Sliding order")
    axes[1].axis("off")
    fig.colorbar(order_artist, ax=axes[1], fraction=0.046, pad=0.04)

    coverage_artist = axes[2].imshow(
        coverage_map,
        cmap="magma",
        vmin=1.0,
        vmax=max(1.0, float(coverage_map.max())),
    )
    axes[2].set_title("Coverage count")
    axes[2].axis("off")
    fig.colorbar(coverage_artist, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{category} patch overview ({sample_id}) | patch={patch_size} stride={stride}"
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Build the first data-preparation artifacts for the project."""
    args = parse_args()
    repo_root = resolve_repo_root()
    config_path = args.config or repo_root / "configs" / "base.yaml"
    config = load_config(config_path)

    dataset_root = repo_root / config["paths"]["dataset_root"]
    processed_root = repo_root / config["paths"]["processed_root"]
    splits_root = repo_root / config["paths"]["splits_root"]
    dataset_figure_root = repo_root / config["paths"]["fig_root"] / "01_dataset"
    preprocessing_figure_root = repo_root / config["paths"]["fig_root"] / "02_preprocessing"
    patch_figure_root = repo_root / config["paths"]["fig_root"] / "03_patches"
    categories = config["data"]["categories"]
    patch_size = tuple(int(value) for value in config["patches"]["size"])
    patch_stride = tuple(int(value) for value in config["patches"]["stride"])
    save_step_figures = bool(config.get("artifacts", {}).get("save_step_figures", True))

    index_df = build_index(dataset_root, categories)
    summary_df = summarize_index(index_df)

    processed_df = index_df.copy()
    processed_df["processed_depth_path"] = ""
    processed_df["processed_mask_path"] = ""
    processed_df["processed_valid_fraction"] = np.nan
    processed_df["processed_height"] = np.nan
    processed_df["processed_width"] = np.nan
    processed_df["foreground_fraction"] = np.nan
    processed_df["crop_top"] = np.nan
    processed_df["crop_bottom"] = np.nan
    processed_df["crop_left"] = np.nan
    processed_df["crop_right"] = np.nan

    rows_to_process = processed_df if args.limit is None else processed_df.head(args.limit)
    preview_done: set[str] = set()
    patch_preview_done: set[str] = set()

    for row_index, row in tqdm(
        rows_to_process.iterrows(),
        total=len(rows_to_process),
        desc="Preprocessing depth maps",
    ):
        xyz_path = repo_root / row["xyz_path"]
        processed_depth, processed_mask, stats = preprocess_xyz_path(xyz_path, config)
        depth_output_path, mask_output_path = build_processed_output_paths(
            processed_root,
            row["category"],
            row["split"],
            row["defect_type"],
            row["sample_id"],
        )
        save_processed_sample(processed_depth, processed_mask, depth_output_path, mask_output_path)

        processed_df.at[row_index, "processed_depth_path"] = depth_output_path.relative_to(repo_root).as_posix()
        processed_df.at[row_index, "processed_mask_path"] = mask_output_path.relative_to(repo_root).as_posix()
        processed_df.at[row_index, "processed_valid_fraction"] = stats["processed_valid_fraction"]
        processed_df.at[row_index, "processed_height"] = stats["processed_height"]
        processed_df.at[row_index, "processed_width"] = stats["processed_width"]
        processed_df.at[row_index, "foreground_fraction"] = stats["foreground_fraction"]
        processed_df.at[row_index, "crop_top"] = stats["crop_top"]
        processed_df.at[row_index, "crop_bottom"] = stats["crop_bottom"]
        processed_df.at[row_index, "crop_left"] = stats["crop_left"]
        processed_df.at[row_index, "crop_right"] = stats["crop_right"]

        if save_step_figures and row["category"] not in preview_done:
            raw_depth = extract_depth_map(load_xyz_map(xyz_path))
            raw_mask = build_valid_mask(raw_depth)
            preview_path = preprocessing_figure_root / f"{row['category']}_raw_vs_processed.png"
            save_preprocessing_figure(
                row["category"],
                row["sample_id"],
                raw_depth,
                raw_mask,
                processed_depth,
                processed_mask,
                (
                    int(stats["crop_top"]),
                    int(stats["crop_bottom"]),
                    int(stats["crop_left"]),
                    int(stats["crop_right"]),
                ),
                preview_path,
            )
            preview_done.add(row["category"])

        if save_step_figures and row["category"] not in patch_preview_done:
            patch_preview_path = patch_figure_root / f"{row['category']}_patch_overview.png"
            save_patch_overview_figure(
                row["category"],
                row["sample_id"],
                processed_depth,
                processed_mask,
                patch_size=patch_size,
                stride=patch_stride,
                output_path=patch_preview_path,
            )
            patch_preview_done.add(row["category"])

    save_index(processed_df, splits_root / "index.csv")
    save_index(summary_df, splits_root / "index_summary.csv")

    split_tables = split_index(processed_df)
    save_index(split_tables["train"], splits_root / "train.csv")
    save_index(split_tables["val"], splits_root / "val.csv")
    save_index(split_tables["test"], splits_root / "test.csv")

    save_summary_figure(summary_df, dataset_figure_root / "index_summary.png")

    print(f"Saved {len(processed_df)} rows to {(splits_root / 'index.csv').relative_to(repo_root)}")
    print(f"Saved split tables to {splits_root.relative_to(repo_root)}")
    print(f"Saved summary table to {(splits_root / 'index_summary.csv').relative_to(repo_root)}")
    print(f"Saved summary figure to {(dataset_figure_root / 'index_summary.png').relative_to(repo_root)}")
    print(f"Saved processed files for {len(rows_to_process)} samples under {processed_root.relative_to(repo_root)}")
    if preview_done:
        print(f"Saved preprocessing previews to {preprocessing_figure_root.relative_to(repo_root)}")
    if patch_preview_done:
        print(f"Saved patch previews to {patch_figure_root.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
