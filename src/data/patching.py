"""Patch extraction and overlap-aware aggregation helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _normalize_2d_size(name: str, value: int | Sequence[int]) -> tuple[int, int]:
    """Normalize an integer or 2D sequence into a `(height, width)` tuple."""
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value, value

    if len(value) != 2:
        raise ValueError(f"{name} must have exactly two elements, got {value}")

    height = int(value[0])
    width = int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} values must be positive, got {value}")
    return height, width


def _compute_patch_starts(axis_length: int, patch_length: int, step: int) -> list[int]:
    """Return deterministic start positions that always cover the full axis."""
    if patch_length > axis_length:
        raise ValueError(
            f"Patch length {patch_length} cannot exceed axis length {axis_length}."
        )

    starts = list(range(0, axis_length - patch_length + 1, step))
    final_start = axis_length - patch_length
    if not starts:
        return [0]
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """Validate the input image shape and return it as an ndarray."""
    image = np.asarray(image)
    if image.ndim < 2:
        raise ValueError(f"Expected at least 2 image dimensions, got shape {image.shape}")
    return image


def extract_patches(
    image: np.ndarray,
    patch_size: int | Sequence[int],
    stride: int | Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract spatial patches and their `(top, bottom, left, right)` coordinates."""
    image = _normalize_image(image)
    patch_height, patch_width = _normalize_2d_size("patch_size", patch_size)
    stride_height, stride_width = _normalize_2d_size("stride", stride)

    image_height, image_width = image.shape[:2]
    row_starts = _compute_patch_starts(image_height, patch_height, stride_height)
    col_starts = _compute_patch_starts(image_width, patch_width, stride_width)

    patches: list[np.ndarray] = []
    coords: list[tuple[int, int, int, int]] = []

    for top in row_starts:
        bottom = top + patch_height
        for left in col_starts:
            right = left + patch_width
            patches.append(np.asarray(image[top:bottom, left:right]))
            coords.append((top, bottom, left, right))

    return np.stack(patches, axis=0), np.asarray(coords, dtype=np.int32)


def extract_patches_with_mask(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int | Sequence[int],
    stride: int | Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract aligned image and mask patches with shared reconstruction coordinates."""
    image = _normalize_image(image)
    mask = np.asarray(mask, dtype=bool)

    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")
    if image.shape[:2] != mask.shape:
        raise ValueError(
            "Image and mask spatial shapes must match, "
            f"got {image.shape[:2]} and {mask.shape}"
        )

    image_patches, coords = extract_patches(image, patch_size, stride)
    mask_patches, _ = extract_patches(mask, patch_size, stride)
    return image_patches, mask_patches.astype(bool), coords


def _validate_coords(coords: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Validate coordinate layout and bounds for aggregation."""
    coords = np.asarray(coords, dtype=np.int32)
    if coords.size == 0:
        return coords.reshape(0, 4)
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"Expected coords with shape (N, 4), got {coords.shape}")

    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    for top, bottom, left, right in coords:
        if not (0 <= top < bottom <= image_height and 0 <= left < right <= image_width):
            raise ValueError(
                "Patch coordinates must stay inside the image bounds, "
                f"got {(top, bottom, left, right)} for {image_shape}"
            )
    return coords


def aggregate_patch_values(
    values: np.ndarray,
    coords: np.ndarray,
    image_shape: tuple[int, int] | Sequence[int],
) -> np.ndarray:
    """Aggregate scalar patch values back into image space via overlap averaging."""
    image_height, image_width = _normalize_2d_size("image_shape", image_shape)
    coords = _validate_coords(coords, (image_height, image_width))
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    if len(values) != len(coords):
        raise ValueError(f"Expected {len(coords)} values, got {len(values)}")

    aggregated = np.zeros((image_height, image_width), dtype=np.float32)
    weights = np.zeros((image_height, image_width), dtype=np.float32)

    for value, (top, bottom, left, right) in zip(values, coords, strict=True):
        aggregated[top:bottom, left:right] += float(value)
        weights[top:bottom, left:right] += 1.0

    valid_weights = weights > 0.0
    aggregated[valid_weights] /= weights[valid_weights]
    return aggregated


def compute_patch_coverage(
    coords: np.ndarray,
    image_shape: tuple[int, int] | Sequence[int],
) -> np.ndarray:
    """Count how many extracted patches cover each image pixel."""
    image_height, image_width = _normalize_2d_size("image_shape", image_shape)
    coords = _validate_coords(coords, (image_height, image_width))
    coverage = np.zeros((image_height, image_width), dtype=np.float32)

    for top, bottom, left, right in coords:
        coverage[top:bottom, left:right] += 1.0
    return coverage


def aggregate_patch_maps(
    patch_maps: np.ndarray,
    coords: np.ndarray,
    image_shape: tuple[int, int] | Sequence[int],
) -> np.ndarray:
    """Aggregate per-patch maps back into image space via overlap averaging."""
    image_height, image_width = _normalize_2d_size("image_shape", image_shape)
    coords = _validate_coords(coords, (image_height, image_width))
    patch_maps = np.asarray(patch_maps, dtype=np.float32)

    if patch_maps.ndim < 3:
        raise ValueError(
            f"Expected patch maps with shape (N, patch_h, patch_w, ...), got {patch_maps.shape}"
        )
    if len(patch_maps) != len(coords):
        raise ValueError(f"Expected {len(coords)} patch maps, got {len(patch_maps)}")

    trailing_shape = patch_maps.shape[3:]
    aggregated = np.zeros((image_height, image_width) + trailing_shape, dtype=np.float32)
    weights = np.zeros((image_height, image_width), dtype=np.float32)

    for patch_map, (top, bottom, left, right) in zip(patch_maps, coords, strict=True):
        expected_shape = (bottom - top, right - left) + trailing_shape
        if patch_map.shape != expected_shape:
            raise ValueError(
                "Patch map shape does not match its coordinates, "
                f"expected {expected_shape}, got {patch_map.shape}"
            )
        aggregated[top:bottom, left:right] += patch_map
        weights[top:bottom, left:right] += 1.0

    if trailing_shape:
        expanded_weights = weights.reshape(weights.shape + (1,) * len(trailing_shape))
    else:
        expanded_weights = weights

    np.divide(
        aggregated,
        expanded_weights,
        out=aggregated,
        where=expanded_weights > 0.0,
    )
    return aggregated
