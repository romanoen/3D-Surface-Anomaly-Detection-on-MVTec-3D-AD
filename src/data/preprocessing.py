"""Preprocessing utilities for depth or height maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage


def load_xyz_map(xyz_path: str | Path) -> np.ndarray:
    """Load one XYZ TIFF sample as a float32 array."""
    xyz_path = Path(xyz_path)
    xyz_map = tifffile.imread(xyz_path).astype(np.float32)
    if xyz_map.ndim != 3 or xyz_map.shape[-1] < 3:
        raise ValueError(f"Expected HxWx3 XYZ data in {xyz_path}, got {xyz_map.shape}")
    return xyz_map


def extract_depth_map(xyz_map: np.ndarray) -> np.ndarray:
    """Extract the Z channel from an XYZ tensor."""
    if xyz_map.ndim != 3 or xyz_map.shape[-1] < 3:
        raise ValueError(f"Expected HxWx3 XYZ data, got {xyz_map.shape}")
    return xyz_map[..., 2].astype(np.float32)


def build_valid_mask(depth_map: np.ndarray) -> np.ndarray:
    """Return a boolean mask for finite positive depth values."""
    return np.isfinite(depth_map) & (depth_map > 0.0)


def load_binary_mask(mask_path: str | Path) -> np.ndarray:
    """Load a binary annotation mask from an image file."""
    mask_path = Path(mask_path)
    with Image.open(mask_path) as mask_image:
        return np.asarray(mask_image.convert("L"), dtype=np.uint8) > 0


def compute_otsu_threshold(values: np.ndarray, num_bins: int = 256) -> float:
    """Compute a 1D Otsu threshold for a set of valid depth values."""
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        raise ValueError("Cannot compute an Otsu threshold without input values.")

    value_min = float(values.min())
    value_max = float(values.max())
    if np.isclose(value_min, value_max):
        return value_min

    hist, bin_edges = np.histogram(values, bins=num_bins, range=(value_min, value_max))
    hist = hist.astype(np.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    weight1 = np.cumsum(hist)
    weight2 = hist.sum() - weight1
    cumulative_mean = np.cumsum(hist * bin_centers)
    global_mean = cumulative_mean[-1]

    numerator = (global_mean * weight1 - cumulative_mean) ** 2
    denominator = np.maximum(weight1 * weight2, 1e-12)
    between_class_variance = numerator / denominator
    between_class_variance[weight2 <= 0] = -np.inf

    threshold_index = int(np.argmax(between_class_variance))
    return float(bin_centers[threshold_index])


def estimate_background_plane(depth_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Estimate a smooth background plane from valid border pixels."""
    image_height, image_width = depth_map.shape
    border_band = max(1, min(image_height, image_width) // 10)

    grid_y, grid_x = np.indices(depth_map.shape)
    border_mask = (
        (grid_y < border_band)
        | (grid_y >= image_height - border_band)
        | (grid_x < border_band)
        | (grid_x >= image_width - border_band)
    ) & valid_mask

    if int(border_mask.sum()) < 3:
        border_mask = valid_mask

    design_matrix = np.column_stack(
        [
            grid_y[border_mask].astype(np.float32),
            grid_x[border_mask].astype(np.float32),
            np.ones(int(border_mask.sum()), dtype=np.float32),
        ]
    )
    coefficients, *_ = np.linalg.lstsq(design_matrix, depth_map[border_mask].astype(np.float32), rcond=None)
    return (
        coefficients[0] * grid_y.astype(np.float32)
        + coefficients[1] * grid_x.astype(np.float32)
        + coefficients[2]
    ).astype(np.float32)


def infer_foreground_mask(depth_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Infer a rough object foreground mask from closer valid depth values."""
    if not np.any(valid_mask):
        return np.zeros_like(valid_mask, dtype=bool)

    background_plane = estimate_background_plane(depth_map, valid_mask)
    residual_map = background_plane - depth_map

    image_height, image_width = depth_map.shape
    border_band = max(1, min(image_height, image_width) // 10)
    grid_y, grid_x = np.indices(depth_map.shape)
    border_mask = (
        (grid_y < border_band)
        | (grid_y >= image_height - border_band)
        | (grid_x < border_band)
        | (grid_x >= image_width - border_band)
    ) & valid_mask
    if not np.any(border_mask):
        border_mask = valid_mask

    border_residuals = residual_map[border_mask]
    threshold = float(border_residuals.mean() + 3.0 * max(border_residuals.std(), 1e-6))
    candidate_mask = valid_mask & (residual_map >= threshold)
    if not np.any(candidate_mask):
        fallback_threshold = compute_otsu_threshold(depth_map[valid_mask])
        candidate_mask = valid_mask & (depth_map <= fallback_threshold)
        if not np.any(candidate_mask):
            return np.zeros_like(valid_mask, dtype=bool)

    candidate_mask = ndimage.binary_closing(candidate_mask, structure=np.ones((5, 5), dtype=bool))
    candidate_mask = ndimage.binary_fill_holes(candidate_mask)

    labeled_mask, num_labels = ndimage.label(candidate_mask)
    if num_labels == 0:
        return candidate_mask.astype(bool)

    image_height, image_width = depth_map.shape
    image_center = np.array([(image_height - 1) / 2.0, (image_width - 1) / 2.0], dtype=np.float32)
    component_slices = ndimage.find_objects(labeled_mask)

    non_border_components: list[tuple[int, int, float]] = []
    all_components: list[tuple[int, int, float]] = []

    for label_index, component_slice in enumerate(component_slices, start=1):
        if component_slice is None:
            continue

        row_slice, col_slice = component_slice
        component = labeled_mask[component_slice] == label_index
        area = int(component.sum())
        if area == 0:
            continue

        component_coords = np.argwhere(component)
        centroid = np.array(
            [
                component_coords[:, 0].mean() + row_slice.start,
                component_coords[:, 1].mean() + col_slice.start,
            ],
            dtype=np.float32,
        )
        center_distance = float(np.linalg.norm(centroid - image_center))
        touches_border = (
            row_slice.start == 0
            or col_slice.start == 0
            or row_slice.stop == image_height
            or col_slice.stop == image_width
        )

        component_info = (label_index, area, center_distance)
        all_components.append(component_info)
        if not touches_border:
            non_border_components.append(component_info)

    if non_border_components:
        best_label = max(non_border_components, key=lambda item: (item[1], -item[2]))[0]
    else:
        best_label = max(all_components, key=lambda item: item[1] / (1.0 + item[2]))[0]

    foreground_mask = labeled_mask == best_label
    foreground_mask = ndimage.binary_fill_holes(foreground_mask)
    return foreground_mask.astype(bool)


def compute_crop_box(
    mask: np.ndarray,
    image_shape: tuple[int, int],
    margin: int = 0,
) -> tuple[int, int, int, int]:
    """Return a bounding box with margin as (top, bottom, left, right)."""
    image_height, image_width = image_shape
    if not np.any(mask):
        return 0, image_height, 0, image_width

    rows, cols = np.where(mask)
    top = max(int(rows.min()) - margin, 0)
    bottom = min(int(rows.max()) + 1 + margin, image_height)
    left = max(int(cols.min()) - margin, 0)
    right = min(int(cols.max()) + 1 + margin, image_width)
    return top, bottom, left, right


def crop_depth_and_mask(
    depth_map: np.ndarray,
    valid_mask: np.ndarray,
    crop_box: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Crop depth and mask arrays using (top, bottom, left, right) bounds."""
    top, bottom, left, right = crop_box
    cropped_depth = depth_map[top:bottom, left:right].astype(np.float32)
    cropped_mask = valid_mask[top:bottom, left:right].astype(bool)
    return cropped_depth, cropped_mask


def _normalize_2d_tuple(name: str, value: tuple[int, int] | list[int]) -> tuple[int, int]:
    """Return a validated positive `(height, width)` tuple."""
    if len(value) != 2:
        raise ValueError(f"{name} must have exactly two elements, got {value}")

    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} values must be positive, got {value}")
    return height, width


def _ceil_length_to_patch_grid(length: int, patch_length: int, step: int) -> int:
    """Round an axis length up so fixed patches land exactly on the far edge."""
    if patch_length <= 0 or step <= 0:
        raise ValueError(
            f"patch_length and step must be positive, got {patch_length}, {step}"
        )

    snapped_length = max(int(length), int(patch_length))
    remainder = (snapped_length - patch_length) % step
    if remainder:
        snapped_length += step - remainder
    return snapped_length


def resize_depth_and_mask(
    depth_map: np.ndarray,
    valid_mask: np.ndarray,
    image_size: tuple[int, int] | list[int] | None,
    mode: str = "pad",
    min_size: tuple[int, int] | list[int] | None = None,
    patch_size: tuple[int, int] | list[int] | None = None,
    stride: tuple[int, int] | list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize the depth map and mask according to the configured geometry mode."""
    if image_size is None:
        return depth_map.astype(np.float32), valid_mask.astype(bool)

    target_height, target_width = int(image_size[0]), int(image_size[1])
    if depth_map.shape == (target_height, target_width):
        return depth_map.astype(np.float32), valid_mask.astype(bool)

    if mode == "stretch":
        zoom_factors = (
            target_height / float(depth_map.shape[0]),
            target_width / float(depth_map.shape[1]),
        )
        depth_resized = ndimage.zoom(depth_map, zoom_factors, order=0).astype(np.float32)
        mask_resized = ndimage.zoom(valid_mask.astype(np.uint8), zoom_factors, order=0) > 0
        depth_resized[~mask_resized] = 0.0
        return depth_resized, mask_resized

    if mode == "preserve_area":
        source_height, source_width = depth_map.shape
        target_area = float(target_height * target_width)
        source_area = float(source_height * source_width)
        scale = np.sqrt(target_area / source_area)

        resized_height = max(1, int(round(source_height * scale)))
        resized_width = max(1, int(round(source_width * scale)))

        if min_size is not None:
            min_height, min_width = _normalize_2d_tuple("min_size", min_size)
            min_scale = max(
                min_height / float(resized_height),
                min_width / float(resized_width),
                1.0,
            )
            resized_height = max(1, int(round(resized_height * min_scale)))
            resized_width = max(1, int(round(resized_width * min_scale)))

        if patch_size is not None or stride is not None:
            if patch_size is None or stride is None:
                raise ValueError("patch_size and stride must be provided together.")
            patch_height, patch_width = _normalize_2d_tuple("patch_size", patch_size)
            stride_height, stride_width = _normalize_2d_tuple("stride", stride)
            resized_height = _ceil_length_to_patch_grid(
                resized_height,
                patch_height,
                stride_height,
            )
            resized_width = _ceil_length_to_patch_grid(
                resized_width,
                patch_width,
                stride_width,
            )

        zoom_factors = (
            resized_height / float(source_height),
            resized_width / float(source_width),
        )
        depth_resized = ndimage.zoom(depth_map, zoom_factors, order=0).astype(np.float32)
        mask_resized = ndimage.zoom(valid_mask.astype(np.uint8), zoom_factors, order=0) > 0
        depth_resized = depth_resized[:resized_height, :resized_width]
        mask_resized = mask_resized[:resized_height, :resized_width]
        depth_resized[~mask_resized] = 0.0
        return depth_resized, mask_resized

    if mode != "pad":
        raise ValueError(f"Unsupported resize mode: {mode}")

    scale = min(
        target_height / float(depth_map.shape[0]),
        target_width / float(depth_map.shape[1]),
    )
    resized_height = max(1, int(round(depth_map.shape[0] * scale)))
    resized_width = max(1, int(round(depth_map.shape[1] * scale)))
    zoom_factors = (
        resized_height / float(depth_map.shape[0]),
        resized_width / float(depth_map.shape[1]),
    )

    depth_resized = ndimage.zoom(depth_map, zoom_factors, order=0).astype(np.float32)
    mask_resized = ndimage.zoom(valid_mask.astype(np.uint8), zoom_factors, order=0) > 0

    depth_resized = depth_resized[:resized_height, :resized_width]
    mask_resized = mask_resized[:resized_height, :resized_width]

    padded_depth = np.zeros((target_height, target_width), dtype=np.float32)
    padded_mask = np.zeros((target_height, target_width), dtype=bool)

    top = (target_height - resized_height) // 2
    left = (target_width - resized_width) // 2
    bottom = top + resized_height
    right = left + resized_width

    padded_depth[top:bottom, left:right] = depth_resized
    padded_mask[top:bottom, left:right] = mask_resized
    padded_depth[~padded_mask] = 0.0
    return padded_depth, padded_mask


def transform_binary_mask_like_processed(
    mask: np.ndarray,
    crop_box: tuple[int, int, int, int],
    cfg: dict,
) -> np.ndarray:
    """Apply the configured crop and resize geometry to a binary annotation mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D binary mask, got shape {mask.shape}")

    top, bottom, left, right = crop_box
    cropped_mask = mask[top:bottom, left:right].astype(bool)
    dummy_depth = cropped_mask.astype(np.float32)
    _, resized_mask = resize_depth_and_mask(
        dummy_depth,
        cropped_mask,
        cfg["data"].get("image_size"),
        mode=cfg["data"].get("resize_mode", "pad"),
        min_size=cfg.get("patches", {}).get("size"),
        patch_size=cfg.get("patches", {}).get("size"),
        stride=cfg.get("patches", {}).get("stride"),
    )
    return resized_mask.astype(bool)


def normalize_depth_map(
    depth_map: np.ndarray,
    valid_mask: np.ndarray,
    mode: str = "per_image",
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Normalize depth values while ignoring invalid pixels."""
    if not np.any(valid_mask):
        raise ValueError("Cannot normalize a depth map without valid pixels.")

    normalized = np.zeros_like(depth_map, dtype=np.float32)
    stats: dict[str, float | int] = {}
    valid_values = depth_map[valid_mask]

    if mode == "none":
        normalized[valid_mask] = valid_values.astype(np.float32)
        stats["valid_min"] = float(valid_values.min())
        stats["valid_max"] = float(valid_values.max())
        return normalized, stats

    if mode != "per_image":
        raise ValueError(f"Unsupported normalization mode: {mode}")

    valid_min = float(valid_values.min())
    valid_max = float(valid_values.max())
    scale = max(valid_max - valid_min, 1e-6)
    normalized[valid_mask] = ((valid_values - valid_min) / scale).astype(np.float32)
    stats["valid_min"] = valid_min
    stats["valid_max"] = valid_max
    stats["scale"] = scale
    return normalized, stats


def preprocess_depth_map(
    depth_map: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    """Build a processed depth representation and its object foreground mask."""
    raw_valid_mask = build_valid_mask(depth_map)
    foreground_mask = infer_foreground_mask(depth_map, raw_valid_mask)
    object_mask = foreground_mask & raw_valid_mask
    mask_source = "foreground" if np.any(object_mask) else "raw_valid"
    if not np.any(object_mask):
        object_mask = raw_valid_mask

    crop_margin = int(cfg["data"].get("crop_margin", 0))
    crop_box = compute_crop_box(object_mask, depth_map.shape, margin=crop_margin)
    cropped_depth, cropped_raw_valid_mask = crop_depth_and_mask(
        depth_map,
        raw_valid_mask,
        crop_box,
    )
    _, cropped_object_mask = crop_depth_and_mask(depth_map, object_mask, crop_box)

    resized_depth, resized_raw_valid_mask = resize_depth_and_mask(
        cropped_depth,
        cropped_raw_valid_mask,
        cfg["data"].get("image_size"),
        mode=cfg["data"].get("resize_mode", "pad"),
        min_size=cfg.get("patches", {}).get("size"),
        patch_size=cfg.get("patches", {}).get("size"),
        stride=cfg.get("patches", {}).get("stride"),
    )
    _, resized_object_mask = resize_depth_and_mask(
        cropped_depth,
        cropped_object_mask,
        cfg["data"].get("image_size"),
        mode=cfg["data"].get("resize_mode", "pad"),
        min_size=cfg.get("patches", {}).get("size"),
        patch_size=cfg.get("patches", {}).get("size"),
        stride=cfg.get("patches", {}).get("stride"),
    )
    if resized_depth.shape != resized_object_mask.shape:
        raise ValueError(
            "Resized depth and object mask shapes must match, "
            f"got {resized_depth.shape} and {resized_object_mask.shape}"
        )

    resized_object_mask &= resized_raw_valid_mask
    normalized_depth, stats = normalize_depth_map(
        resized_depth,
        resized_object_mask,
        mode=cfg["data"].get("normalization", "per_image"),
    )
    normalized_depth[~resized_object_mask] = 0.0

    crop_top, crop_bottom, crop_left, crop_right = crop_box
    stats["raw_valid_fraction"] = float(raw_valid_mask.mean())
    stats["foreground_fraction"] = float(object_mask.mean())
    stats["processed_valid_fraction"] = float(resized_object_mask.mean())
    stats["processed_mask_source"] = mask_source
    stats["raw_height"] = float(depth_map.shape[0])
    stats["raw_width"] = float(depth_map.shape[1])
    stats["crop_top"] = int(crop_top)
    stats["crop_bottom"] = int(crop_bottom)
    stats["crop_left"] = int(crop_left)
    stats["crop_right"] = int(crop_right)
    stats["cropped_height"] = float(cropped_depth.shape[0])
    stats["cropped_width"] = float(cropped_depth.shape[1])
    stats["processed_height"] = float(normalized_depth.shape[0])
    stats["processed_width"] = float(normalized_depth.shape[1])

    return normalized_depth.astype(np.float32), resized_object_mask.astype(bool), stats


def preprocess_xyz_path(
    xyz_path: str | Path, cfg: dict
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    """Load an XYZ TIFF file and preprocess its Z channel."""
    xyz_map = load_xyz_map(xyz_path)
    depth_map = extract_depth_map(xyz_map)
    return preprocess_depth_map(depth_map, cfg)


def build_processed_output_paths(
    processed_root: str | Path,
    category: str,
    split: str,
    defect_type: str,
    sample_id: str,
) -> tuple[Path, Path]:
    """Return output paths for one processed sample."""
    sample_root = Path(processed_root) / category / split / defect_type
    depth_output_path = sample_root / f"{sample_id}_depth.npy"
    mask_output_path = sample_root / f"{sample_id}_mask.npy"
    return depth_output_path, mask_output_path


def save_processed_sample(
    depth_map: np.ndarray,
    valid_mask: np.ndarray,
    depth_output_path: str | Path,
    mask_output_path: str | Path,
) -> None:
    """Save processed depth and mask arrays as NPY files."""
    depth_output_path = Path(depth_output_path)
    mask_output_path = Path(mask_output_path)
    depth_output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(depth_output_path, depth_map.astype(np.float32))
    np.save(mask_output_path, valid_mask.astype(bool))
