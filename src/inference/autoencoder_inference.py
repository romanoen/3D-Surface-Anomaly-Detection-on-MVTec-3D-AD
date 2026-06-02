"""Inference utilities for autoencoder reconstruction and residual scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.loaders import build_image_dataset
from src.data.patching import extract_patches_with_mask
from src.data.preprocessing import load_binary_mask, transform_binary_mask_like_processed
from src.inference.anomaly_maps import (
    aggregate_patch_residuals,
    image_score_from_patch_scores,
    save_heatmap_figure,
)
from src.models.autoencoder import ConvAutoencoder, load_autoencoder_checkpoint


@dataclass(frozen=True)
class AutoencoderInferenceResult:
    """Paths and summary values produced by autoencoder inference."""

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


def _resolve_path(path_value: object, repo_root: Path) -> Path:
    """Resolve a metadata path against the repository root."""
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return repo_root / path


def _select_device(training_cfg: dict[str, Any]) -> torch.device:
    """Return the device used for autoencoder inference."""
    requested = str(training_cfg.get("device", "auto")).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


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


def _load_autoencoder_model(
    cfg: dict[str, Any],
    repo_root: Path,
    category: str,
) -> tuple[ConvAutoencoder, torch.device]:
    """Load the trained autoencoder checkpoint for one category."""
    checkpoint_name = str(
        cfg.get("training", {}).get("checkpoint_name", "conv_autoencoder.pt")
    ).strip() or "conv_autoencoder.pt"
    checkpoint_path = (
        repo_root
        / cfg["paths"]["outputs_root"]
        / "models"
        / "autoencoder"
        / _safe_path_part(category)
        / checkpoint_name
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Missing category-specific autoencoder checkpoint for "
            f"{category!r}: {checkpoint_path}. Run scripts/train_autoencoder.py first."
        )

    device = _select_device(cfg.get("training", {}))
    model, _ = load_autoencoder_checkpoint(checkpoint_path, map_location=device)
    model = model.to(device)
    model.eval()
    return model, device


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


def _residual_maps_and_scores(
    patches: np.ndarray,
    masks: np.ndarray,
    model: ConvAutoencoder,
    device: torch.device,
    batch_size: int,
    patch_score_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the autoencoder on patches and return residual maps plus scalar scores."""
    patch_score_method = patch_score_method.lower()
    patch_tensors = torch.from_numpy(patches[:, None, ...].astype(np.float32, copy=False))
    mask_tensors = torch.from_numpy(masks[:, None, ...].astype(np.float32, copy=False))

    residual_batches: list[np.ndarray] = []
    score_batches: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            end = min(start + batch_size, len(patches))
            patch_batch = patch_tensors[start:end].to(device)
            mask_batch = mask_tensors[start:end].to(device)
            reconstruction = model(patch_batch)
            residual = torch.abs(reconstruction - patch_batch) * mask_batch

            if patch_score_method == "mean_residual":
                denominator = mask_batch.sum(dim=(1, 2, 3)).clamp_min(1.0)
                patch_scores = residual.sum(dim=(1, 2, 3)) / denominator
            elif patch_score_method == "max_residual":
                patch_scores = residual.amax(dim=(1, 2, 3))
            else:
                raise ValueError(f"Unsupported autoencoder patch score method: {patch_score_method}")

            residual_batches.append(residual.squeeze(1).cpu().numpy().astype(np.float32))
            score_batches.append(patch_scores.cpu().numpy().astype(np.float32))

    return np.concatenate(residual_batches, axis=0), np.concatenate(score_batches, axis=0)


def _score_image_patches(
    depth: np.ndarray,
    mask: np.ndarray,
    cfg: dict[str, Any],
    model: ConvAutoencoder,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract one image's patches and return residual maps, scores, and metadata."""
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
    residual_maps, patch_scores = _residual_maps_and_scores(
        patches,
        masks,
        model,
        device,
        batch_size=int(cfg.get("training", {}).get("batch_size", 64)),
        patch_score_method=str(cfg.get("inference", {}).get("patch_score", "mean_residual")),
    )
    return residual_maps, patch_scores, coords, valid_fractions, patch_positions


def _heatmap_output_path(root: Path, sample: dict[str, Any]) -> Path:
    """Return the deterministic raw heatmap path for one sample."""
    return (
        root
        / _safe_path_part(sample["category"])
        / _safe_path_part(sample["defect_type"])
        / f"{_safe_path_part(sample['sample_id'])}_heatmap.npy"
    )


def run_autoencoder_inference(
    split_df: pd.DataFrame,
    cfg: dict[str, Any],
    repo_root: str | Path = ".",
    split_name: str = "test",
    max_images: int | None = None,
) -> AutoencoderInferenceResult:
    """Score a split with the trained autoencoder and save inference artifacts."""
    repo_root = Path(repo_root)
    if max_images is not None:
        split_df = split_df.head(int(max_images)).copy()

    dataset = build_image_dataset(split_df, cfg, repo_root=repo_root)
    device = _select_device(cfg.get("training", {}))
    model_cache: dict[str, ConvAutoencoder] = {}
    paths_cfg = cfg["paths"]
    inference_cfg = cfg.get("inference", {})
    image_score_method = str(inference_cfg.get("image_score", "topk_mean"))
    topk_fraction = float(inference_cfg.get("topk_fraction", 0.05))
    save_heatmaps = bool(inference_cfg.get("save_heatmaps", True))
    max_heatmap_figures_per_category = int(
        inference_cfg.get("max_heatmap_figures_per_category", inference_cfg.get("max_heatmap_figures", 4))
    )

    metrics_root = repo_root / paths_cfg["outputs_root"] / "metrics"
    heatmap_root = metrics_root / "autoencoder_heatmaps" / split_name
    figure_root = repo_root / paths_cfg["fig_root"] / "05_autoencoder" / f"{split_name}_heatmaps"
    image_scores_path = metrics_root / f"autoencoder_image_scores_{split_name}.csv"
    patch_scores_path = metrics_root / f"autoencoder_patch_scores_{split_name}.csv"

    figure_counts: dict[str, dict[str, int]] = {}
    image_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []

    for sample_position in tqdm(range(len(dataset)), desc=f"Scoring {split_name} images (autoencoder)"):
        sample = dataset[sample_position]
        category = str(sample["category"])
        if category not in model_cache:
            model_cache[category], _ = _load_autoencoder_model(cfg, repo_root, category)
        label = int(sample["label"])
        residual_maps, patch_scores, coords, valid_fractions, patch_positions = _score_image_patches(
            sample["depth"],
            sample["mask"],
            cfg,
            model_cache[category],
            device,
        )
        heatmap = aggregate_patch_residuals(
            residual_maps,
            coords,
            sample["image_shape"],
            valid_mask=sample["mask"],
        )
        image_score = image_score_from_patch_scores(
            patch_scores,
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
        category_counts = figure_counts.setdefault(str(sample["category"]), {"normal": 0, "anomaly": 0})
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
                "patch_count": int(len(patch_scores)),
                "image_score": float(image_score),
                "heatmap_path": heatmap_path.relative_to(repo_root).as_posix() if save_heatmaps else "",
                "figure_path": saved_figure_path,
                "ground_truth_pixel_count": int(ground_truth_mask.sum()) if ground_truth_mask is not None else 0,
            }
        )

        for score, coord, valid_fraction, patch_position in zip(
            patch_scores,
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

    return AutoencoderInferenceResult(
        image_scores_path=image_scores_path,
        patch_scores_path=patch_scores_path,
        heatmap_root=heatmap_root,
        figure_root=figure_root,
        image_count=len(image_rows),
        patch_count=len(patch_rows),
    )
