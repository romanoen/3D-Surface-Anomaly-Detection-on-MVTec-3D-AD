"""Dataset helpers for patch-level training and full-image inference."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.patching import extract_patches, extract_patches_with_mask


@dataclass(frozen=True)
class PatchEntry:
    """Metadata for one patch within one processed sample."""

    sample_position: int
    patch_position: int
    coords: tuple[int, int, int, int]
    valid_fraction: float


def _normalize_2d_size(name: str, value: int | Sequence[int]) -> tuple[int, int]:
    """Normalize an integer or 2D sequence into a `(height, width)` tuple."""
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value, value

    if len(value) != 2:
        raise ValueError(f"{name} must have exactly two elements, got {value}")

    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} values must be positive, got {value}")
    return height, width


def _resolve_path(path_value: str | Path, repo_root: str | Path) -> Path:
    """Resolve a path stored in split metadata."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(repo_root) / path


def _row_metadata(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """Return the stable metadata fields shared by patch and image datasets."""
    metadata: dict[str, Any] = {}
    for key in (
        "category",
        "split",
        "defect_type",
        "label",
        "sample_id",
        "rgb_path",
        "xyz_path",
        "mask_path",
        "processed_depth_path",
        "processed_mask_path",
        "crop_top",
        "crop_bottom",
        "crop_left",
        "crop_right",
    ):
        if key in row:
            value = row[key]
            metadata[key] = int(value) if key == "label" else value
    return metadata


class ImageDataset:
    """Lazy dataset for full processed maps used during image-level inference."""

    def __init__(self, split_df: pd.DataFrame, repo_root: str | Path = ".") -> None:
        self.rows = split_df.reset_index(drop=True).copy()
        self.repo_root = Path(repo_root)

    def __len__(self) -> int:
        """Return the number of processed images."""
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one processed depth map, mask, and its metadata."""
        row = self.rows.iloc[index]
        depth_path = _resolve_path(row["processed_depth_path"], self.repo_root)
        mask_path = _resolve_path(row["processed_mask_path"], self.repo_root)

        depth = np.load(depth_path).astype(np.float32, copy=False)
        mask = np.load(mask_path).astype(bool, copy=False)
        if depth.shape[:2] != mask.shape:
            raise ValueError(
                "Processed depth and mask shapes must match, "
                f"got {depth.shape[:2]} and {mask.shape} for {depth_path}"
            )

        sample = _row_metadata(row)
        sample.update(
            {
                "depth": depth,
                "mask": mask,
                "image_shape": depth.shape[:2],
                "sample_position": index,
            }
        )
        return sample


class PatchDataset:
    """Lazy dataset for fixed-size patches extracted from processed maps."""

    def __init__(
        self,
        split_df: pd.DataFrame,
        patch_size: int | Sequence[int],
        stride: int | Sequence[int],
        repo_root: str | Path = ".",
        min_valid_fraction: float = 0.0,
        normal_only: bool = False,
        cache_size: int = 4,
    ) -> None:
        if not 0.0 <= float(min_valid_fraction) <= 1.0:
            raise ValueError(
                f"min_valid_fraction must be between 0 and 1, got {min_valid_fraction}"
            )
        if cache_size < 0:
            raise ValueError(f"cache_size must be non-negative, got {cache_size}")

        rows = split_df.reset_index(drop=True).copy()
        if normal_only:
            rows = rows.loc[rows["label"].astype(int) == 0].reset_index(drop=True)

        self.rows = rows
        self.patch_size = _normalize_2d_size("patch_size", patch_size)
        self.stride = _normalize_2d_size("stride", stride)
        self.repo_root = Path(repo_root)
        self.min_valid_fraction = float(min_valid_fraction)
        self.cache_size = int(cache_size)
        self.entries = self._build_entries()
        self._sample_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def _build_entries(self) -> list[PatchEntry]:
        """Create patch coordinate metadata without keeping patch arrays in memory."""
        entries: list[PatchEntry] = []
        for sample_position, row in self.rows.iterrows():
            mask_path = _resolve_path(row["processed_mask_path"], self.repo_root)
            mask = np.load(mask_path).astype(bool, copy=False)
            mask_patches, coords = extract_patches(mask, self.patch_size, self.stride)
            valid_fractions = mask_patches.reshape(len(mask_patches), -1).mean(axis=1)

            for patch_position, (coord, valid_fraction) in enumerate(
                zip(coords, valid_fractions, strict=True)
            ):
                valid_fraction = float(valid_fraction)
                if valid_fraction >= self.min_valid_fraction:
                    entries.append(
                        PatchEntry(
                            sample_position=int(sample_position),
                            patch_position=int(patch_position),
                            coords=tuple(int(value) for value in coord),
                            valid_fraction=valid_fraction,
                        )
                    )
        return entries

    def __len__(self) -> int:
        """Return the number of retained patches."""
        return len(self.entries)

    def _load_sample_arrays(self, sample_position: int) -> tuple[np.ndarray, np.ndarray]:
        """Load and cache one processed sample."""
        if sample_position in self._sample_cache:
            self._sample_cache.move_to_end(sample_position)
            return self._sample_cache[sample_position]

        row = self.rows.iloc[sample_position]
        depth_path = _resolve_path(row["processed_depth_path"], self.repo_root)
        mask_path = _resolve_path(row["processed_mask_path"], self.repo_root)
        depth = np.load(depth_path).astype(np.float32, copy=False)
        mask = np.load(mask_path).astype(bool, copy=False)
        if depth.shape[:2] != mask.shape:
            raise ValueError(
                "Processed depth and mask shapes must match, "
                f"got {depth.shape[:2]} and {mask.shape} for {depth_path}"
            )

        if self.cache_size > 0:
            self._sample_cache[sample_position] = (depth, mask)
            self._sample_cache.move_to_end(sample_position)
            while len(self._sample_cache) > self.cache_size:
                self._sample_cache.popitem(last=False)
        return depth, mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one patch, its mask, reconstruction coordinates, and metadata."""
        entry = self.entries[index]
        row = self.rows.iloc[entry.sample_position]
        depth, mask = self._load_sample_arrays(entry.sample_position)
        top, bottom, left, right = entry.coords

        patch = depth[top:bottom, left:right].astype(np.float32, copy=False)
        mask_patch = mask[top:bottom, left:right].astype(bool, copy=False)
        expected_shape = self.patch_size
        if patch.shape[:2] != expected_shape:
            raise ValueError(
                f"Expected patch shape {expected_shape}, got {patch.shape[:2]}"
            )

        sample = _row_metadata(row)
        sample.update(
            {
                "patch": patch,
                "mask": mask_patch,
                "coords": np.asarray(entry.coords, dtype=np.int32),
                "image_shape": depth.shape[:2],
                "sample_position": entry.sample_position,
                "patch_position": entry.patch_position,
                "valid_fraction": entry.valid_fraction,
            }
        )
        return sample

    def patches_for_image(self, sample_position: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract all patches for one sample position in deterministic order."""
        depth, mask = self._load_sample_arrays(sample_position)
        return extract_patches_with_mask(depth, mask, self.patch_size, self.stride)


def build_patch_dataset(
    split_df: pd.DataFrame,
    cfg: dict,
    repo_root: str | Path = ".",
    normal_only: bool = True,
) -> PatchDataset:
    """Build a patch dataset from split metadata and project config."""
    patch_cfg = cfg["patches"]
    return PatchDataset(
        split_df,
        patch_size=patch_cfg["size"],
        stride=patch_cfg["stride"],
        repo_root=repo_root,
        min_valid_fraction=float(patch_cfg.get("min_valid_fraction", 0.0)),
        normal_only=normal_only,
    )


def build_image_dataset(
    split_df: pd.DataFrame,
    cfg: dict | None = None,
    repo_root: str | Path = ".",
) -> ImageDataset:
    """Build a full-image dataset from split metadata."""
    del cfg
    return ImageDataset(split_df, repo_root=repo_root)
