"""Dataset indexing utilities for selected MVTec 3D-AD categories."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

_RAW_SPLIT_MAP = {
    "train": "train",
    "val": "validation",
    "test": "test",
}

_SPLIT_ORDER = {
    "train": 0,
    "val": 1,
    "test": 2,
}

_INDEX_COLUMNS = [
    "category",
    "split",
    "defect_type",
    "label",
    "sample_id",
    "rgb_path",
    "xyz_path",
    "mask_path",
]


def _to_index_path(path: Path, base_path: Path) -> str:
    """Convert a filesystem path into a stable index string."""
    try:
        return path.relative_to(base_path).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _build_sample_maps(sample_root: Path) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
    """Collect sample files for one category/split/defect directory."""
    rgb_dir = sample_root / "rgb"
    xyz_dir = sample_root / "xyz"
    gt_dir = sample_root / "gt"

    if not xyz_dir.exists():
        raise FileNotFoundError(f"Missing xyz directory: {xyz_dir}")
    if not rgb_dir.exists():
        raise FileNotFoundError(f"Missing rgb directory: {rgb_dir}")

    rgb_map = {path.stem: path for path in sorted(rgb_dir.glob("*.png"))}
    xyz_map = {path.stem: path for path in sorted(xyz_dir.glob("*.tiff"))}
    mask_map = {path.stem: path for path in sorted(gt_dir.glob("*.png"))} if gt_dir.exists() else {}

    missing_rgb = sorted(set(xyz_map) - set(rgb_map))
    if missing_rgb:
        raise ValueError(f"Missing RGB files for samples {missing_rgb} in {sample_root}")

    return rgb_map, xyz_map, mask_map


def build_index(dataset_root: str | Path, categories: Iterable[str]) -> pd.DataFrame:
    """Build a dataset index for the selected categories."""
    dataset_root = Path(dataset_root)
    base_path = dataset_root.parent.parent
    rows: list[dict[str, str | int]] = []

    for category in categories:
        category_root = dataset_root / category
        if not category_root.exists():
            raise FileNotFoundError(f"Category directory does not exist: {category_root}")

        for split_name, raw_split_name in _RAW_SPLIT_MAP.items():
            split_root = category_root / raw_split_name
            if not split_root.exists():
                raise FileNotFoundError(f"Split directory does not exist: {split_root}")

            defect_dirs = sorted(path for path in split_root.iterdir() if path.is_dir())
            for defect_dir in defect_dirs:
                defect_type = defect_dir.name
                label = 0 if defect_type == "good" else 1
                rgb_map, xyz_map, mask_map = _build_sample_maps(defect_dir)

                for sample_id in sorted(xyz_map):
                    rows.append(
                        {
                            "category": category,
                            "split": split_name,
                            "defect_type": defect_type,
                            "label": label,
                            "sample_id": sample_id,
                            "rgb_path": _to_index_path(rgb_map[sample_id], base_path),
                            "xyz_path": _to_index_path(xyz_map[sample_id], base_path),
                            "mask_path": _to_index_path(mask_map[sample_id], base_path)
                            if sample_id in mask_map
                            else "",
                        }
                    )

    index_df = pd.DataFrame(rows, columns=_INDEX_COLUMNS)
    index_df["_split_order"] = index_df["split"].map(_SPLIT_ORDER)
    index_df["_label_order"] = index_df["label"]
    index_df = (
        index_df.sort_values(
            by=["category", "_split_order", "_label_order", "defect_type", "sample_id"],
            kind="stable",
        )
        .drop(columns=["_split_order", "_label_order"])
        .reset_index(drop=True)
    )
    return index_df


def save_index(index_df: pd.DataFrame, output_path: str | Path) -> None:
    """Persist an index dataframe as CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(output_path, index=False)


def load_index(index_path: str | Path) -> pd.DataFrame:
    """Load an index CSV."""
    index_path = Path(index_path)
    return pd.read_csv(index_path, keep_default_na=False)


def split_index(index_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return per-split index tables using the normalized split names."""
    split_tables: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        split_tables[split_name] = (
            index_df.loc[index_df["split"] == split_name].copy().reset_index(drop=True)
        )
    return split_tables


def summarize_index(index_df: pd.DataFrame) -> pd.DataFrame:
    """Return a simple count table for documentation and sanity checks."""
    summary_df = (
        index_df.groupby(["category", "split", "label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(by=["category", "split", "label"], kind="stable")
        .reset_index(drop=True)
    )
    return summary_df
