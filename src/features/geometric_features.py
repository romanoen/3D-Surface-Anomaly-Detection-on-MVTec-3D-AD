"""Handcrafted geometric feature extraction for local surface patches."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

FEATURE_NAMES = (
    "mean",
    "std",
    "min",
    "max",
    "range",
    "gradient_mean",
    "gradient_std",
    "roughness",
    "laplacian_var",
)


def _valid_mask_for_patch(patch: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Return the finite valid-pixel mask for one patch."""
    patch = np.asarray(patch, dtype=np.float32)
    finite_mask = np.isfinite(patch)
    if mask is None:
        return finite_mask

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != patch.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match patch shape {patch.shape}")
    return finite_mask & mask


def _fill_invalid_values(patch: np.ndarray, valid_mask: np.ndarray, fill_value: float) -> np.ndarray:
    """Return a copy where invalid pixels are replaced for spatial filters."""
    filled = np.asarray(patch, dtype=np.float32).copy()
    filled[~valid_mask] = fill_value
    return filled


def extract_patch_features(
    patch: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Extract a small fixed feature vector from one depth patch."""
    patch = np.asarray(patch, dtype=np.float32)
    if patch.ndim != 2:
        raise ValueError(f"Expected a 2D patch, got shape {patch.shape}")

    valid_mask = _valid_mask_for_patch(patch, mask)
    if not np.any(valid_mask):
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    values = patch[valid_mask].astype(np.float32)
    mean_value = float(values.mean())
    min_value = float(values.min())
    max_value = float(values.max())
    filled_patch = _fill_invalid_values(patch, valid_mask, mean_value)

    gradient_y, gradient_x = np.gradient(filled_patch)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2).astype(np.float32)
    gradient_values = gradient_magnitude[valid_mask]

    local_mean = ndimage.uniform_filter(filled_patch, size=3, mode="nearest")
    roughness_values = np.abs(filled_patch - local_mean)[valid_mask]

    laplacian = ndimage.laplace(filled_patch, mode="nearest")
    laplacian_values = laplacian[valid_mask]

    return np.asarray(
        [
            mean_value,
            float(values.std()),
            min_value,
            max_value,
            max_value - min_value,
            float(gradient_values.mean()),
            float(gradient_values.std()),
            float(roughness_values.mean()),
            float(laplacian_values.var()),
        ],
        dtype=np.float32,
    )


def extract_batch_features(
    patches: np.ndarray,
    masks: np.ndarray | None = None,
) -> np.ndarray:
    """Extract feature vectors for a batch of depth patches."""
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

    features = np.zeros((len(patches), len(FEATURE_NAMES)), dtype=np.float32)
    for index, patch in enumerate(patches):
        patch_mask = None if masks is None else masks[index]
        features[index] = extract_patch_features(patch, patch_mask)
    return features
