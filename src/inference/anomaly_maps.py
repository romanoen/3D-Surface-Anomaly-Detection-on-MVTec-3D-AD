"""Anomaly-map aggregation helpers shared across inference pipelines."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

from src.data.patching import aggregate_patch_values

_CACHE_ROOT = Path(tempfile.gettempdir()) / "mvtec_3d_ad_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def image_score_from_patch_scores(
    patch_scores: np.ndarray,
    method: str = "topk_mean",
    topk_fraction: float = 0.05,
) -> float:
    """Aggregate patch anomaly scores into one image-level score."""
    scores = np.asarray(patch_scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        raise ValueError("At least one patch score is required.")
    if not np.isfinite(scores).all():
        raise ValueError("Patch scores contain NaN or infinite values.")

    method = method.lower()
    if method == "max":
        return float(scores.max())
    if method == "mean":
        return float(scores.mean())
    if method == "topk_mean":
        fraction = float(topk_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"topk_fraction must be in (0, 1], got {topk_fraction}")
        k = max(1, int(np.ceil(scores.size * fraction)))
        top_scores = np.partition(scores, scores.size - k)[-k:]
        return float(top_scores.mean())

    raise ValueError(f"Unsupported image score method: {method}")


def aggregate_patch_scores(
    patch_scores: np.ndarray,
    coords: np.ndarray,
    image_shape: tuple[int, int] | Sequence[int],
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Aggregate scalar patch anomaly scores into a full-image heatmap."""
    heatmap = aggregate_patch_values(patch_scores, coords, image_shape).astype(np.float32)
    if valid_mask is None:
        return heatmap

    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.shape != heatmap.shape:
        raise ValueError(f"Mask shape {valid_mask.shape} does not match heatmap {heatmap.shape}")

    masked_heatmap = heatmap.copy()
    masked_heatmap[~valid_mask] = 0.0
    return masked_heatmap


def normalize_map_for_display(
    values: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Min-max normalize a map for visualization only."""
    values = np.asarray(values, dtype=np.float32)
    finite_mask = np.isfinite(values)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != values.shape:
            raise ValueError(f"Mask shape {valid_mask.shape} does not match values {values.shape}")
        finite_mask &= valid_mask

    output = np.zeros_like(values, dtype=np.float32)
    if not np.any(finite_mask):
        return output

    selected = values[finite_mask]
    min_value = float(selected.min())
    max_value = float(selected.max())
    if max_value <= min_value:
        return output

    output[finite_mask] = (values[finite_mask] - min_value) / (max_value - min_value)
    return np.clip(output, 0.0, 1.0)


def _draw_ground_truth_contour(axis: plt.Axes, ground_truth_mask: np.ndarray | None) -> None:
    """Draw a ground-truth mask contour on an axis when annotations are present."""
    if ground_truth_mask is None:
        return

    ground_truth_mask = np.asarray(ground_truth_mask, dtype=bool)
    if not np.any(ground_truth_mask):
        return

    if np.all(ground_truth_mask):
        height, width = ground_truth_mask.shape
        axis.plot(
            [0, width - 1, width - 1, 0, 0],
            [0, 0, height - 1, height - 1, 0],
            color="cyan",
            linewidth=1.2,
        )
        return

    axis.contour(
        ground_truth_mask.astype(np.float32),
        levels=[0.5],
        colors=["cyan"],
        linewidths=1.2,
    )


def save_heatmap_figure(
    depth: np.ndarray,
    heatmap: np.ndarray,
    output_path: str | Path,
    valid_mask: np.ndarray | None = None,
    ground_truth_mask: np.ndarray | None = None,
    title: str | None = None,
) -> None:
    """Save a compact depth/heatmap/overlay figure for qualitative inspection."""
    depth = np.asarray(depth, dtype=np.float32)
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if depth.shape != heatmap.shape:
        raise ValueError(f"Depth shape {depth.shape} does not match heatmap {heatmap.shape}")
    if ground_truth_mask is not None:
        ground_truth_mask = np.asarray(ground_truth_mask, dtype=bool)
        if ground_truth_mask.shape != depth.shape:
            raise ValueError(
                "Ground-truth mask shape does not match depth shape, "
                f"got {ground_truth_mask.shape} and {depth.shape}"
            )

    depth_display = normalize_map_for_display(depth, valid_mask)
    heatmap_display = normalize_map_for_display(heatmap, valid_mask)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
    if title:
        fig.suptitle(title)

    axes[0].imshow(depth_display, cmap="gray", interpolation="nearest")
    _draw_ground_truth_contour(axes[0], ground_truth_mask)
    axes[0].set_title("depth")
    axes[1].imshow(heatmap_display, cmap="magma", interpolation="nearest")
    _draw_ground_truth_contour(axes[1], ground_truth_mask)
    axes[1].set_title("heatmap")
    axes[2].imshow(depth_display, cmap="gray", interpolation="nearest")
    axes[2].imshow(heatmap_display, cmap="magma", alpha=0.55, interpolation="nearest")
    _draw_ground_truth_contour(axes[2], ground_truth_mask)
    axes[2].set_title("overlay")

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
