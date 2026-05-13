"""Raw normalized depth-patch feature extraction for classical models."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _normalize_patch_size(patch_size: int | Sequence[int]) -> tuple[int, int]:
    """Normalize an integer or 2D patch size into `(height, width)`."""
    if isinstance(patch_size, int):
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        return patch_size, patch_size

    if len(patch_size) != 2:
        raise ValueError(f"patch_size must have exactly two values, got {patch_size}")

    height, width = int(patch_size[0]), int(patch_size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"patch_size values must be positive, got {patch_size}")
    return height, width


def raw_patch_feature_names(patch_size: int | Sequence[int]) -> tuple[str, ...]:
    """Return stable feature names for flattened depth pixels."""
    height, width = _normalize_patch_size(patch_size)
    return tuple(f"height_r{row:02d}_c{col:02d}" for row in range(height) for col in range(width))


def _valid_mask_for_patch(patch: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Return finite pixels that are inside the object mask."""
    patch = np.asarray(patch, dtype=np.float32)
    valid_mask = np.isfinite(patch)
    if mask is None:
        return valid_mask

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != patch.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match patch shape {patch.shape}")
    return valid_mask & mask


def _fill_value(
    patch: np.ndarray,
    valid_mask: np.ndarray,
    fill_strategy: str,
) -> float:
    """Return the scalar used for invalid or outside-object pixels."""
    strategy = str(fill_strategy).lower()
    if strategy == "zero":
        return 0.0
    if strategy == "patch_mean":
        if not np.any(valid_mask):
            return 0.0
        return float(np.asarray(patch, dtype=np.float32)[valid_mask].mean())
    raise ValueError(f"Unsupported raw patch fill strategy: {fill_strategy}")


def extract_patch_features(
    patch: np.ndarray,
    mask: np.ndarray | None = None,
    fill_strategy: str = "patch_mean",
) -> np.ndarray:
    """Flatten one normalized depth patch into a mask-aware feature vector."""
    patch = np.asarray(patch, dtype=np.float32)
    if patch.ndim != 2:
        raise ValueError(f"Expected a 2D patch, got shape {patch.shape}")

    valid_mask = _valid_mask_for_patch(patch, mask)
    fill_value = _fill_value(patch, valid_mask, fill_strategy)
    filled_patch = np.asarray(patch, dtype=np.float32).copy()
    filled_patch[~valid_mask] = fill_value
    filled_patch = np.nan_to_num(
        filled_patch,
        nan=fill_value,
        posinf=fill_value,
        neginf=fill_value,
    )
    return filled_patch.reshape(-1).astype(np.float32)


def extract_batch_features(
    patches: np.ndarray,
    masks: np.ndarray | None = None,
    fill_strategy: str = "patch_mean",
) -> np.ndarray:
    """Flatten a batch of normalized depth patches into feature rows."""
    patches = np.asarray(patches, dtype=np.float32)
    if patches.ndim == 2:
        patches = patches[None, ...]
    if patches.ndim != 3:
        raise ValueError(f"Expected patches with shape (N, H, W), got {patches.shape}")

    if masks is not None:
        masks = np.asarray(masks, dtype=bool)
        if masks.ndim == 2:
            masks = masks[None, ...]
        if masks.shape != patches.shape:
            raise ValueError(f"Masks shape {masks.shape} does not match patches {patches.shape}")

    feature_count = int(np.prod(patches.shape[1:]))
    features = np.zeros((len(patches), feature_count), dtype=np.float32)
    for index, patch in enumerate(patches):
        patch_mask = None if masks is None else masks[index]
        features[index] = extract_patch_features(
            patch,
            patch_mask,
            fill_strategy=fill_strategy,
        )
    return features
