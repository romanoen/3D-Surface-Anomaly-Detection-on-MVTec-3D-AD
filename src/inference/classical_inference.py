"""Inference utilities for classical patch scoring and image-level aggregation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.loaders import build_image_dataset
from src.data.patching import extract_patches_with_mask
from src.data.preprocessing import load_binary_mask, transform_binary_mask_like_processed
from src.features.raw_patch_features import extract_batch_features
from src.inference.anomaly_maps import (
    aggregate_patch_scores,
    image_score_from_patch_scores,
    save_heatmap_figure,
)
from src.models.ocsvm import OneClassSVMModel


@dataclass(frozen=True)
class ClassicalInferenceResult:
    """Paths and summary values produced by classical inference."""

    image_scores_path: Path
    patch_scores_path: Path
    heatmap_root: Path
    figure_root: Path
    image_count: int
    patch_count: int


def _safe_path_part(value: object) -> str:
    """Return a filesystem-friendly path fragment."""
    text = str(value).strip() or "unknown"
    for character in ("/", "\\", ":", " "):
        text = text.replace(character, "_")
    return text


def _load_feature_metadata(model_root: Path) -> dict[str, Any]:
    """Load raw-patch feature metadata saved during training."""
    feature_path = model_root / "features.json"
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Missing feature metadata: {feature_path}. Run classical training first."
        )

    payload = json.loads(feature_path.read_text(encoding="utf-8"))
    if str(payload.get("feature_mode", "")).lower() != "raw_depth":
        raise ValueError(f"Unsupported classical feature metadata in {feature_path}")
    if int(payload.get("input_feature_count", 0)) <= 0:
        raise ValueError(f"Invalid input feature count in {feature_path}")
    if int(payload.get("model_feature_count", 0)) <= 0:
        raise ValueError(f"Invalid model feature count in {feature_path}")
    return payload


def _load_classical_artifacts(
    cfg: dict[str, Any],
    repo_root: Path,
    category: str,
) -> tuple[OneClassSVMModel, Any, Any, dict[str, Any]]:
    """Load category-specific model, scaler, PCA, and feature metadata."""
    model_root = (
        repo_root
        / cfg["paths"]["outputs_root"]
        / "models"
        / "classical"
        / _safe_path_part(category)
    )
    model_path = model_root / "ocsvm.joblib"
    scaler_path = model_root / "scaler.joblib"
    pca_path = model_root / "pca.joblib"
    if not model_path.exists() or not scaler_path.exists() or not pca_path.exists():
        raise FileNotFoundError(
            "Missing category-specific classical model artifacts for "
            f"{category!r}. Run scripts/train_classical.py again."
        )

    model = OneClassSVMModel.load(model_path)
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    feature_metadata = _load_feature_metadata(model_root)
    return model, scaler, pca, feature_metadata


def _filter_valid_patches(
    patches: np.ndarray,
    masks: np.ndarray,
    coords: np.ndarray,
    min_valid_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep patches with enough valid pixels and return original patch positions."""
    valid_fractions = masks.reshape(len(masks), -1).mean(axis=1).astype(np.float32)
    keep_mask = valid_fractions >= float(min_valid_fraction)
    if not np.any(keep_mask):
        raise ValueError("No patches passed the minimum valid-fraction threshold.")

    patch_positions = np.arange(len(coords), dtype=np.int64)
    return (
        patches[keep_mask],
        masks[keep_mask],
        coords[keep_mask],
        valid_fractions[keep_mask],
        patch_positions[keep_mask],
    )


def _score_image_patches(
    depth: np.ndarray,
    mask: np.ndarray,
    cfg: dict[str, Any],
    model: OneClassSVMModel,
    scaler: Any,
    pca: Any,
    feature_metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract patches from one image and return scores plus patch metadata."""
    patch_cfg = cfg["patches"]
    patches, masks, coords = extract_patches_with_mask(
        depth,
        mask,
        patch_size=patch_cfg["size"],
        stride=patch_cfg["stride"],
    )
    patches, masks, coords, valid_fractions, patch_positions = _filter_valid_patches(
        patches,
        masks,
        coords,
        min_valid_fraction=float(patch_cfg.get("min_valid_fraction", 0.0)),
    )

    features = extract_batch_features(
        patches,
        masks,
        fill_strategy=str(feature_metadata.get("fill_strategy", "patch_mean")),
    )
    expected_feature_count = int(feature_metadata["input_feature_count"])
    if features.shape[1] != expected_feature_count:
        raise ValueError(
            "Raw patch feature count does not match the trained model metadata, "
            f"got {features.shape[1]} and expected {expected_feature_count}"
        )
    if scaler is not None:
        features = scaler.transform(features).astype(np.float32)
    if pca is not None:
        features = pca.transform(features).astype(np.float32)
    scores = model.score(features)
    return scores, coords, valid_fractions, patch_positions


def _heatmap_output_path(root: Path, sample: dict[str, Any]) -> Path:
    """Return the deterministic raw heatmap path for one sample."""
    return (
        root
        / _safe_path_part(sample["category"])
        / _safe_path_part(sample["defect_type"])
        / f"{_safe_path_part(sample['sample_id'])}_heatmap.npy"
    )


def _resolve_path(path_value: object, repo_root: Path) -> Path:
    """Resolve a metadata path against the repository root."""
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return repo_root / path


def _load_processed_ground_truth_mask(
    sample: dict[str, Any],
    cfg: dict[str, Any],
    repo_root: Path,
) -> np.ndarray | None:
    """Load and transform a raw defect mask into processed-image coordinates."""
    mask_path_value = str(sample.get("mask_path", "")).strip()
    if not mask_path_value:
        return None

    mask_path = _resolve_path(mask_path_value, repo_root)
    if not mask_path.exists():
        return None

    crop_keys = ("crop_top", "crop_bottom", "crop_left", "crop_right")
    if not all(key in sample for key in crop_keys):
        return None

    crop_box = tuple(int(float(sample[key])) for key in crop_keys)
    raw_mask = load_binary_mask(mask_path)
    processed_mask = transform_binary_mask_like_processed(raw_mask, crop_box, cfg)
    if processed_mask.shape != tuple(sample["image_shape"]):
        raise ValueError(
            "Processed ground-truth mask shape does not match image shape, "
            f"got {processed_mask.shape} and {sample['image_shape']} for {mask_path}"
        )
    return processed_mask


def _should_save_figure(
    label: int,
    saved_normal_figures: int,
    saved_anomaly_figures: int,
    max_figures_per_category: int,
) -> bool:
    """Decide whether to save a qualitative figure for this sample."""
    if max_figures_per_category <= 0:
        return False
    normal_budget = max(1, max_figures_per_category // 4)
    anomaly_budget = max_figures_per_category - normal_budget
    if label == 0:
        return saved_normal_figures < normal_budget
    return saved_anomaly_figures < anomaly_budget


def run_classical_inference(
    split_df: pd.DataFrame,
    cfg: dict[str, Any],
    repo_root: str | Path = ".",
    split_name: str = "test",
    max_images: int | None = None,
) -> ClassicalInferenceResult:
    """Score a split with the trained classical model and save inference artifacts."""
    repo_root = Path(repo_root)
    if max_images is not None:
        split_df = split_df.head(int(max_images)).copy()

    dataset = build_image_dataset(split_df, cfg, repo_root=repo_root)
    paths_cfg = cfg["paths"]
    inference_cfg = cfg.get("inference", {})
    image_score_method = str(inference_cfg.get("image_score", "topk_mean"))
    topk_fraction = float(inference_cfg.get("topk_fraction", 0.05))
    save_heatmaps = bool(inference_cfg.get("save_heatmaps", True))
    max_heatmap_figures_per_category = int(
        inference_cfg.get(
            "max_heatmap_figures_per_category",
            inference_cfg.get("max_heatmap_figures", 40),
        )
    )

    metrics_root = repo_root / paths_cfg["outputs_root"] / "metrics"
    heatmap_root = metrics_root / "classical_heatmaps" / split_name
    figure_root = repo_root / paths_cfg["fig_root"] / "04_classical" / f"{split_name}_heatmaps"
    image_scores_path = metrics_root / f"classical_image_scores_{split_name}.csv"
    patch_scores_path = metrics_root / f"classical_patch_scores_{split_name}.csv"

    artifact_cache: dict[str, tuple[OneClassSVMModel, Any, Any, dict[str, Any]]] = {}
    figure_counts: dict[str, dict[str, int]] = {}
    image_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []

    for sample_position in tqdm(range(len(dataset)), desc=f"Scoring {split_name} images"):
        sample = dataset[sample_position]
        category = str(sample["category"])
        label = int(sample["label"])
        if category not in artifact_cache:
            artifact_cache[category] = _load_classical_artifacts(cfg, repo_root, category)
        model, scaler, pca, feature_metadata = artifact_cache[category]

        scores, coords, valid_fractions, patch_positions = _score_image_patches(
            sample["depth"],
            sample["mask"],
            cfg,
            model,
            scaler,
            pca,
            feature_metadata,
        )
        heatmap = aggregate_patch_scores(
            scores,
            coords,
            sample["image_shape"],
            valid_mask=sample["mask"],
        )
        image_score = image_score_from_patch_scores(
            scores,
            method=image_score_method,
            topk_fraction=topk_fraction,
        )

        heatmap_path = _heatmap_output_path(heatmap_root, sample)
        if save_heatmaps:
            heatmap_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(heatmap_path, heatmap.astype(np.float32))

        ground_truth_mask = _load_processed_ground_truth_mask(sample, cfg, repo_root)
        figure_path = (
            figure_root
            / _safe_path_part(sample["category"])
            / _safe_path_part(sample["defect_type"])
            / f"{_safe_path_part(sample['sample_id'])}_heatmap.png"
        )
        saved_figure_path = ""
        category_counts = figure_counts.setdefault(category, {"normal": 0, "anomaly": 0})
        if _should_save_figure(
            label,
            category_counts["normal"],
            category_counts["anomaly"],
            max_heatmap_figures_per_category,
        ):
            save_heatmap_figure(
                sample["depth"],
                heatmap,
                figure_path,
                valid_mask=sample["mask"],
                ground_truth_mask=ground_truth_mask,
                title=(
                    f"{sample['category']} / {sample['defect_type']} / "
                    f"{sample['sample_id']} | score={image_score:.4f}"
                ),
            )
            if label == 0:
                category_counts["normal"] += 1
            else:
                category_counts["anomaly"] += 1
            saved_figure_path = figure_path.relative_to(repo_root).as_posix()

        image_rows.append(
            {
                "category": sample["category"],
                "split": sample["split"],
                "defect_type": sample["defect_type"],
                "label": label,
                "sample_id": sample["sample_id"],
                "sample_position": int(sample["sample_position"]),
                "image_height": int(sample["image_shape"][0]),
                "image_width": int(sample["image_shape"][1]),
                "patch_count": int(len(scores)),
                "image_score": float(image_score),
                "heatmap_path": heatmap_path.relative_to(repo_root).as_posix()
                if save_heatmaps
                else "",
                "figure_path": saved_figure_path,
                "ground_truth_pixel_count": int(ground_truth_mask.sum())
                if ground_truth_mask is not None
                else 0,
            }
        )

        for score, coord, valid_fraction, patch_position in zip(
            scores,
            coords,
            valid_fractions,
            patch_positions,
            strict=True,
        ):
            top, bottom, left, right = [int(value) for value in coord]
            patch_rows.append(
                {
                    "category": sample["category"],
                    "split": sample["split"],
                    "defect_type": sample["defect_type"],
                    "label": label,
                    "sample_id": sample["sample_id"],
                    "sample_position": int(sample["sample_position"]),
                    "patch_position": int(patch_position),
                    "top": top,
                    "bottom": bottom,
                    "left": left,
                    "right": right,
                    "valid_fraction": float(valid_fraction),
                    "score": float(score),
                }
            )

    metrics_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(image_rows).to_csv(image_scores_path, index=False)
    pd.DataFrame(patch_rows).to_csv(patch_scores_path, index=False)

    return ClassicalInferenceResult(
        image_scores_path=image_scores_path,
        patch_scores_path=patch_scores_path,
        heatmap_root=heatmap_root,
        figure_root=figure_root,
        image_count=len(image_rows),
        patch_count=len(patch_rows),
    )
