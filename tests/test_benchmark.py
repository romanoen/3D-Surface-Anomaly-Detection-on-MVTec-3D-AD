"""Tests for the shared benchmark orchestration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.evaluation.benchmark import run_benchmark
from src.training.autoencoder_trainer import train_autoencoder_model
from src.training.classical_trainer import train_classical_model


def _save_sample(root: Path, relative_root: str, value: float) -> tuple[str, str]:
    """Save one synthetic processed depth/mask pair."""
    sample_root = Path(relative_root)
    depth_path = sample_root / "000_depth.npy"
    mask_path = sample_root / "000_mask.npy"
    full_depth_path = root / depth_path
    full_mask_path = root / mask_path
    full_depth_path.parent.mkdir(parents=True, exist_ok=True)
    depth = np.full((8, 8), value, dtype=np.float32)
    depth[2:6, 2:6] += np.linspace(0.0, 0.2, 16, dtype=np.float32).reshape(4, 4)
    mask = np.ones((8, 8), dtype=bool)
    np.save(full_depth_path, depth)
    np.save(full_mask_path, mask)
    return depth_path.as_posix(), mask_path.as_posix()


def _save_gt_mask(root: Path, relative_path: str, has_defect: bool) -> str:
    """Save one raw binary ground-truth mask image."""
    gt_path = Path(relative_path)
    full_gt_path = root / gt_path
    full_gt_path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((8, 8), dtype=np.uint8)
    if has_defect:
        mask[3:6, 3:6] = 255
    Image.fromarray(mask).save(full_gt_path)
    return gt_path.as_posix()


def _row(
    split: str,
    defect_type: str,
    label: int,
    depth_path: str,
    mask_path: str,
    sample_id: str,
    raw_gt_path: str = "",
) -> dict[str, object]:
    """Return one split row for synthetic processed files."""
    return {
        "category": "demo",
        "split": split,
        "defect_type": defect_type,
        "label": label,
        "sample_id": sample_id,
        "rgb_path": "",
        "xyz_path": "",
        "mask_path": raw_gt_path,
        "processed_depth_path": depth_path,
        "processed_mask_path": mask_path,
        "crop_top": 0,
        "crop_bottom": 8,
        "crop_left": 0,
        "crop_right": 8,
    }


class BenchmarkTestCase(unittest.TestCase):
    """Verify the shared benchmark writes comparison outputs."""

    def test_run_benchmark_writes_combined_metrics_and_figures(self) -> None:
        """The benchmark should run both methods and export merged result files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_depth_0, train_mask_0 = _save_sample(root, "data/processed/demo/train/good/000", 0.1)
            train_depth_1, train_mask_1 = _save_sample(root, "data/processed/demo/train/good/001", 0.2)
            val_depth, val_mask = _save_sample(root, "data/processed/demo/val/good/000", 0.15)
            test_depth_0, test_mask_0 = _save_sample(root, "data/processed/demo/test/good/000", 0.12)
            test_depth_1, test_mask_1 = _save_sample(root, "data/processed/demo/test/bump/001", 0.9)
            good_gt = _save_gt_mask(root, "data/raw/demo/test/good/gt/000.png", has_defect=False)
            bump_gt = _save_gt_mask(root, "data/raw/demo/test/bump/gt/001.png", has_defect=True)

            train_df = pd.DataFrame(
                [
                    _row("train", "good", 0, train_depth_0, train_mask_0, "000"),
                    _row("train", "good", 0, train_depth_1, train_mask_1, "001"),
                ]
            )
            val_df = pd.DataFrame([_row("val", "good", 0, val_depth, val_mask, "000")])
            test_df = pd.DataFrame(
                [
                    _row("test", "good", 0, test_depth_0, test_mask_0, "000", raw_gt_path=good_gt),
                    _row("test", "bump", 1, test_depth_1, test_mask_1, "001", raw_gt_path=bump_gt),
                ]
            )

            classical_cfg = {
                "project": {"seed": 7},
                "paths": {
                    "outputs_root": "outputs",
                    "fig_root": "fig",
                },
                "data": {
                    "categories": ["demo"],
                    "image_size": [8, 8],
                    "resize_mode": "preserve_area",
                    "normalization": "per_image",
                },
                "patches": {
                    "size": [4, 4],
                    "stride": [4, 4],
                    "min_valid_fraction": 0.01,
                },
                "features": {
                    "mode": "raw_depth",
                    "fill_strategy": "patch_mean",
                    "pca_components": 4,
                },
                "training": {
                    "scaler": "standard",
                    "ocsvm_kernel": "rbf",
                    "ocsvm_nu": 0.25,
                    "ocsvm_gamma": "scale",
                    "max_train_patches": 8,
                    "max_val_patches": 4,
                },
                "inference": {
                    "image_score": "topk_mean",
                    "topk_fraction": 0.5,
                    "save_heatmaps": True,
                    "max_heatmap_figures_per_category": 2,
                },
            }
            autoencoder_cfg = {
                "project": {"seed": 7},
                "paths": {
                    "outputs_root": "outputs",
                    "fig_root": "fig",
                },
                "data": {
                    "categories": ["demo"],
                    "image_size": [8, 8],
                    "resize_mode": "preserve_area",
                    "normalization": "per_image",
                },
                "patches": {
                    "size": [4, 4],
                    "stride": [4, 4],
                    "min_valid_fraction": 0.01,
                },
                "model": {
                    "name": "conv_autoencoder",
                    "in_channels": 1,
                    "encoder_channels": [4, 8],
                    "latent_channels": 8,
                },
                "training": {
                    "per_category": True,
                    "device": "cpu",
                    "batch_size": 4,
                    "num_workers": 0,
                    "epochs": 3,
                    "learning_rate": 0.01,
                    "loss": "mse",
                    "early_stopping_patience": 2,
                    "max_train_patches": 8,
                    "max_val_patches": 4,
                    "checkpoint_name": "conv_autoencoder.pt",
                    "reconstruction_examples": 2,
                },
                "inference": {
                    "patch_score": "mean_residual",
                    "image_score": "topk_mean",
                    "topk_fraction": 0.5,
                    "save_heatmaps": True,
                    "max_heatmap_figures_per_category": 2,
                },
            }

            train_classical_model(train_df, val_df, classical_cfg, repo_root=root)
            train_autoencoder_model(train_df, val_df, autoencoder_cfg, repo_root=root)
            result = run_benchmark(
                test_df,
                classical_cfg,
                autoencoder_cfg,
                repo_root=root,
                split_name="test",
                max_comparison_figures_per_category=2,
            )

            self.assertTrue(result.per_image_path.exists())
            self.assertTrue(result.per_category_path.exists())
            self.assertTrue(result.summary_path.exists())
            self.assertTrue((result.figure_root / "auroc_by_category.png").exists())
            self.assertTrue((result.figure_root / "ap_by_category.png").exists())
            self.assertTrue((result.figure_root / "demo" / "good" / "000_comparison.png").exists())
            self.assertTrue((result.figure_root / "demo" / "bump" / "001_comparison.png").exists())

            per_image_df = pd.read_csv(result.per_image_path, keep_default_na=False)
            per_category_df = pd.read_csv(result.per_category_path, keep_default_na=False)
            summary_df = pd.read_csv(result.summary_path, keep_default_na=False)

            self.assertEqual(len(per_image_df), 2)
            self.assertEqual(set(per_category_df["method"]), {"classical", "autoencoder"})
            self.assertEqual(set(summary_df["method"]), {"classical", "autoencoder"})
            self.assertIn("comparison_figure_path", per_image_df.columns)


if __name__ == "__main__":
    unittest.main()
