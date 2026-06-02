"""Benchmark orchestration for consistent evaluation across both methods."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_per_category, summarize_benchmark_results
from src.evaluation.visualization import (
    save_method_comparison_figure,
    save_per_category_metric_figure,
    select_representative_rows,
)
from src.inference.autoencoder_inference import run_autoencoder_inference
from src.inference.classical_inference import run_classical_inference
from src.data.preprocessing import load_binary_mask, transform_binary_mask_like_processed


@dataclass(frozen=True)
class BenchmarkResult:
    """Paths and summary values produced by the shared benchmark."""

    per_image_path: Path
    per_category_path: Path
    summary_path: Path
    figure_root: Path
    image_count: int
    category_count: int


def _metric_output_path(metrics_root: Path, base_name: str, split_name: str) -> Path:
    """Return a split-aware metric output path."""
    filename = f"{base_name}.csv" if split_name == "test" else f"{base_name}_{split_name}.csv"
    return metrics_root / filename


def _shared_pipeline_signature(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the shared-pipeline config values that must match across methods."""
    return {
        "paths": cfg.get("paths", {}),
        "data": {
            "categories": cfg.get("data", {}).get("categories"),
            "image_size": cfg.get("data", {}).get("image_size"),
            "resize_mode": cfg.get("data", {}).get("resize_mode"),
            "normalization": cfg.get("data", {}).get("normalization"),
        },
        "patches": cfg.get("patches", {}),
    }


def _validate_shared_pipeline_configs(
    classical_cfg: dict[str, Any],
    autoencoder_cfg: dict[str, Any],
) -> None:
    """Ensure both detectors are evaluated on the same shared pipeline settings."""
    if _shared_pipeline_signature(classical_cfg) != _shared_pipeline_signature(autoencoder_cfg):
        raise ValueError(
            "Classical and autoencoder configs must share the same paths, preprocessing, "
            "and patch extraction settings for a fair benchmark."
        )


def _prepare_inference_configs(
    classical_cfg: dict[str, Any],
    autoencoder_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return local config copies with heatmap export forced on for the benchmark."""
    classical_cfg = deepcopy(classical_cfg)
    autoencoder_cfg = deepcopy(autoencoder_cfg)
    classical_cfg.setdefault("inference", {})
    autoencoder_cfg.setdefault("inference", {})
    classical_cfg["inference"]["save_heatmaps"] = True
    autoencoder_cfg["inference"]["save_heatmaps"] = True
    return classical_cfg, autoencoder_cfg


def _merge_per_image_scores(
    split_df: pd.DataFrame,
    classical_scores_df: pd.DataFrame,
    autoencoder_scores_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge detector-specific per-image outputs into one comparison table."""
    split_df = split_df.reset_index(drop=True).copy()
    split_df["sample_position"] = np.arange(len(split_df), dtype=np.int64)
    merge_keys = ["category", "split", "defect_type", "label", "sample_position"]

    metadata_columns = merge_keys + [
        "sample_id",
        "processed_depth_path",
        "processed_mask_path",
        "mask_path",
        "crop_top",
        "crop_bottom",
        "crop_left",
        "crop_right",
    ]
    metadata_df = split_df.loc[:, [column for column in metadata_columns if column in split_df.columns]].copy()
    string_keys = ("category", "split", "defect_type", "sample_id")
    integer_keys = ("label", "sample_position")
    for dataframe in (metadata_df, classical_scores_df, autoencoder_scores_df):
        for key in string_keys:
            if key in dataframe.columns:
                dataframe[key] = dataframe[key].astype(str)
        for key in integer_keys:
            if key in dataframe.columns:
                dataframe[key] = dataframe[key].astype(int)

    classical_df = classical_scores_df.rename(
        columns={
            "image_score": "classical_image_score",
            "patch_count": "classical_patch_count",
            "heatmap_path": "classical_heatmap_path",
            "figure_path": "classical_figure_path",
            "ground_truth_pixel_count": "classical_ground_truth_pixel_count",
        }
    )
    autoencoder_df = autoencoder_scores_df.rename(
        columns={
            "image_score": "autoencoder_image_score",
            "patch_count": "autoencoder_patch_count",
            "heatmap_path": "autoencoder_heatmap_path",
            "figure_path": "autoencoder_figure_path",
            "ground_truth_pixel_count": "autoencoder_ground_truth_pixel_count",
        }
    )

    merged = metadata_df.merge(
        classical_df[merge_keys + [column for column in classical_df.columns if column.startswith("classical_")]],
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )
    merged = merged.merge(
        autoencoder_df[merge_keys + [column for column in autoencoder_df.columns if column.startswith("autoencoder_")]],
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(metadata_df):
        raise ValueError(
            "Merged benchmark table does not cover every split row, "
            f"got {len(merged)} rows for {len(metadata_df)} inputs."
        )

    merged["ground_truth_pixel_count"] = merged[
        ["classical_ground_truth_pixel_count", "autoencoder_ground_truth_pixel_count"]
    ].max(axis=1)
    merged["comparison_figure_path"] = ""
    return merged.sort_values(["category", "sample_position"]).reset_index(drop=True)


def _resolve_path(path_value: object, repo_root: Path) -> Path:
    """Resolve a potentially relative metadata path."""
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return repo_root / path


def _load_processed_ground_truth_mask(
    row: pd.Series,
    cfg: dict[str, Any],
    repo_root: Path,
) -> np.ndarray | None:
    """Load and transform a raw defect mask into processed-image coordinates."""
    mask_path_value = str(row.get("mask_path", "")).strip()
    if not mask_path_value:
        return None

    mask_path = _resolve_path(mask_path_value, repo_root)
    if not mask_path.exists():
        return None

    crop_keys = ("crop_top", "crop_bottom", "crop_left", "crop_right")
    if not all(key in row.index for key in crop_keys):
        return None

    crop_box = tuple(int(float(row[key])) for key in crop_keys)
    raw_mask = load_binary_mask(mask_path)
    return transform_binary_mask_like_processed(raw_mask, crop_box, cfg)


def _save_comparison_figures(
    per_image_df: pd.DataFrame,
    cfg: dict[str, Any],
    repo_root: Path,
    figure_root: Path,
    max_figures_per_category: int,
) -> pd.DataFrame:
    """Save representative side-by-side comparison panels and return the updated table."""
    per_image_df = per_image_df.copy()
    selected_rows = select_representative_rows(per_image_df, max_figures_per_category)
    if selected_rows.empty:
        return per_image_df

    for _, row in selected_rows.iterrows():
        depth_path = _resolve_path(row["processed_depth_path"], repo_root)
        mask_path = _resolve_path(row["processed_mask_path"], repo_root)
        classical_heatmap_path = _resolve_path(row["classical_heatmap_path"], repo_root)
        autoencoder_heatmap_path = _resolve_path(row["autoencoder_heatmap_path"], repo_root)

        depth = np.load(depth_path).astype(np.float32, copy=False)
        valid_mask = np.load(mask_path).astype(bool, copy=False)
        classical_heatmap = np.load(classical_heatmap_path).astype(np.float32, copy=False)
        autoencoder_heatmap = np.load(autoencoder_heatmap_path).astype(np.float32, copy=False)
        ground_truth_mask = _load_processed_ground_truth_mask(row, cfg, repo_root)

        output_path = (
            figure_root
            / str(row["category"])
            / str(row["defect_type"])
            / f"{row['sample_id']}_comparison.png"
        )
        save_method_comparison_figure(
            depth,
            classical_heatmap,
            autoencoder_heatmap,
            output_path,
            valid_mask=valid_mask,
            ground_truth_mask=ground_truth_mask,
            title=f"{row['category']} / {row['defect_type']} / {row['sample_id']}",
            classical_score=float(row["classical_image_score"]),
            autoencoder_score=float(row["autoencoder_image_score"]),
        )
        comparison_path = output_path.relative_to(repo_root).as_posix()
        row_mask = (
            (per_image_df["category"] == row["category"])
            & (per_image_df["sample_id"] == row["sample_id"])
            & (per_image_df["sample_position"].astype(int) == int(row["sample_position"]))
        )
        per_image_df.loc[row_mask, "comparison_figure_path"] = comparison_path

    return per_image_df


def run_benchmark(
    split_df: pd.DataFrame,
    classical_cfg: dict[str, Any],
    autoencoder_cfg: dict[str, Any],
    *,
    repo_root: str | Path = ".",
    split_name: str = "test",
    max_images: int | None = None,
    max_comparison_figures_per_category: int = 4,
) -> BenchmarkResult:
    """Run both detectors on the same split and save shared comparison artifacts."""
    _validate_shared_pipeline_configs(classical_cfg, autoencoder_cfg)
    repo_root = Path(repo_root)
    classical_cfg, autoencoder_cfg = _prepare_inference_configs(classical_cfg, autoencoder_cfg)

    if max_images is not None:
        split_df = split_df.head(int(max_images)).copy()

    classical_result = run_classical_inference(
        split_df,
        classical_cfg,
        repo_root=repo_root,
        split_name=split_name,
        max_images=max_images,
    )
    autoencoder_result = run_autoencoder_inference(
        split_df,
        autoencoder_cfg,
        repo_root=repo_root,
        split_name=split_name,
        max_images=max_images,
    )

    classical_scores_df = pd.read_csv(classical_result.image_scores_path, keep_default_na=False)
    autoencoder_scores_df = pd.read_csv(autoencoder_result.image_scores_path, keep_default_na=False)
    per_image_df = _merge_per_image_scores(split_df, classical_scores_df, autoencoder_scores_df)

    score_columns = {
        "classical": "classical_image_score",
        "autoencoder": "autoencoder_image_score",
    }
    per_category_df = evaluate_per_category(per_image_df, score_columns)
    summary_df = summarize_benchmark_results(per_image_df, per_category_df, score_columns)

    metrics_root = repo_root / classical_cfg["paths"]["outputs_root"] / "metrics"
    figure_root = repo_root / classical_cfg["paths"]["fig_root"] / "06_results"
    if split_name != "test":
        figure_root = figure_root / split_name

    per_image_df = _save_comparison_figures(
        per_image_df,
        classical_cfg,
        repo_root,
        figure_root,
        max_figures_per_category=max_comparison_figures_per_category,
    )
    save_per_category_metric_figure(per_category_df, "auroc", figure_root / "auroc_by_category.png")
    save_per_category_metric_figure(per_category_df, "ap", figure_root / "ap_by_category.png")

    per_image_path = _metric_output_path(metrics_root, "per_image", split_name)
    per_category_path = _metric_output_path(metrics_root, "per_category", split_name)
    summary_path = _metric_output_path(metrics_root, "summary", split_name)
    metrics_root.mkdir(parents=True, exist_ok=True)
    per_image_df.to_csv(per_image_path, index=False)
    per_category_df.to_csv(per_category_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    return BenchmarkResult(
        per_image_path=per_image_path,
        per_category_path=per_category_path,
        summary_path=summary_path,
        figure_root=figure_root,
        image_count=len(per_image_df),
        category_count=int(per_image_df["category"].nunique()),
    )
