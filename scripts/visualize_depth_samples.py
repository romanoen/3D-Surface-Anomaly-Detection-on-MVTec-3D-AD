"""Save one masked depth-map preview per selected category."""

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
import numpy as np
import tifffile
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Save one sample raw depth-map preview per selected category."
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


def find_sample_xyz(category_root: Path) -> Path:
    """Return one raw XYZ TIFF sample from train/good for a category."""
    xyz_dir = category_root / "train" / "good" / "xyz"
    samples = sorted(xyz_dir.glob("*.tiff"))
    if not samples:
        raise FileNotFoundError(f"No TIFF samples found in {xyz_dir}")
    return samples[0]


def extract_depth_map(xyz_path: Path) -> np.ndarray:
    """Load an XYZ TIFF and return the Z channel as a float32 depth map."""
    xyz = tifffile.imread(xyz_path)
    if xyz.ndim != 3 or xyz.shape[-1] < 3:
        raise ValueError(
            f"Expected an HxWx3 XYZ tensor in {xyz_path}, got shape {xyz.shape}"
        )
    return xyz[..., 2].astype(np.float32)


def build_valid_mask(depth_map: np.ndarray) -> np.ndarray:
    """Return a mask for finite, positive depth values."""
    return np.isfinite(depth_map) & (depth_map > 0.0)


def normalize_for_display(
    depth_map: np.ndarray, valid_mask: np.ndarray
) -> np.ma.MaskedArray:
    """Normalize a depth map for visualization only using robust percentiles."""
    if not np.any(valid_mask):
        raise ValueError("Depth map does not contain any finite values.")

    valid_values = depth_map[valid_mask]
    low, high = np.percentile(valid_values, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Computed invalid visualization percentiles.")
    if high <= low:
        high = low + 1e-6

    normalized = np.zeros_like(depth_map, dtype=np.float32)
    normalized[valid_mask] = np.clip(
        (depth_map[valid_mask] - low) / (high - low), 0.0, 1.0
    )
    return np.ma.array(normalized, mask=~valid_mask)


def save_depth_figure(
    category: str,
    sample_path: Path,
    depth_map: np.ma.MaskedArray,
    output_path: Path,
) -> None:
    """Save one depth preview figure for a category."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    image = ax.imshow(depth_map, cmap=cmap, vmin=0.0, vmax=1.0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.set_ticklabels(["0\ncloser", "0.5", "1\nfarther"])
    ax.set_title(f"{category} masked depth sample")
    ax.set_xlabel(sample_path.name)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Load one sample per category and save raw depth preview figures."""
    args = parse_args()
    repo_root = resolve_repo_root()
    config_path = args.config or repo_root / "configs" / "base.yaml"
    config = load_config(config_path)

    dataset_root = repo_root / config["paths"]["dataset_root"]
    fig_root = repo_root / config["paths"]["fig_root"] / "01_dataset"
    categories = config["data"]["categories"]

    for category in categories:
        category_root = dataset_root / category
        sample_path = find_sample_xyz(category_root)
        depth_map = extract_depth_map(sample_path)
        valid_mask = build_valid_mask(depth_map)
        depth_preview = normalize_for_display(depth_map, valid_mask)
        output_path = fig_root / f"{category}_sample_depth.png"
        save_depth_figure(
            category,
            sample_path,
            depth_preview,
            output_path,
        )
        print(f"Saved {output_path.relative_to(repo_root)} from {sample_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
