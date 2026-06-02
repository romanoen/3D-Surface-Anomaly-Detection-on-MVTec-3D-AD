"""Tests for autoencoder inference."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.inference.autoencoder_inference import run_autoencoder_inference
from src.training.autoencoder_trainer import train_autoencoder_model


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
    category: str = "demo",
) -> dict[str, object]:
    """Return one split row for synthetic processed files."""
    return {
        "category": category,
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


class AutoencoderInferenceTestCase(unittest.TestCase):
    """Verify autoencoder inference writes scores and heatmaps."""

    def test_run_autoencoder_inference_writes_artifacts(self) -> None:
        """A tiny fitted autoencoder should score normal and anomalous samples."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_depth_0, train_mask_0 = _save_sample(root, "data/processed/demo/train/good/000", 0.1)
            train_depth_1, train_mask_1 = _save_sample(root, "data/processed/demo/train/good/001", 0.2)
            val_depth, val_mask = _save_sample(root, "data/processed/demo/val/good/000", 0.15)
            other_train_depth_0, other_train_mask_0 = _save_sample(root, "data/processed/other/train/good/000", 0.4)
            other_train_depth_1, other_train_mask_1 = _save_sample(root, "data/processed/other/train/good/001", 0.5)
            other_val_depth, other_val_mask = _save_sample(root, "data/processed/other/val/good/000", 0.45)
            test_depth_0, test_mask_0 = _save_sample(root, "data/processed/demo/test/good/000", 0.12)
            test_depth_1, test_mask_1 = _save_sample(root, "data/processed/demo/test/bump/001", 0.9)
            other_test_depth, other_test_mask = _save_sample(root, "data/processed/other/test/good/000", 0.42)
            good_gt = _save_gt_mask(root, "data/raw/demo/test/good/gt/000.png", has_defect=False)
            bump_gt = _save_gt_mask(root, "data/raw/demo/test/bump/gt/001.png", has_defect=True)
            other_good_gt = _save_gt_mask(root, "data/raw/other/test/good/gt/000.png", has_defect=False)

            train_df = pd.DataFrame(
                [
                    _row("train", "good", 0, train_depth_0, train_mask_0, "000"),
                    _row("train", "good", 0, train_depth_1, train_mask_1, "001"),
                    _row("train", "good", 0, other_train_depth_0, other_train_mask_0, "000", category="other"),
                    _row("train", "good", 0, other_train_depth_1, other_train_mask_1, "001", category="other"),
                ]
            )
            val_df = pd.DataFrame(
                [
                    _row("val", "good", 0, val_depth, val_mask, "000"),
                    _row("val", "good", 0, other_val_depth, other_val_mask, "000", category="other"),
                ]
            )
            test_df = pd.DataFrame(
                [
                    _row("test", "good", 0, test_depth_0, test_mask_0, "000", raw_gt_path=good_gt),
                    _row("test", "bump", 1, test_depth_1, test_mask_1, "001", raw_gt_path=bump_gt),
                    _row(
                        "test",
                        "good",
                        0,
                        other_test_depth,
                        other_test_mask,
                        "000",
                        raw_gt_path=other_good_gt,
                        category="other",
                    ),
                ]
            )
            cfg = {
                "project": {"seed": 7},
                "paths": {
                    "outputs_root": "outputs",
                    "fig_root": "fig",
                },
                "data": {
                    "image_size": [8, 8],
                    "resize_mode": "preserve_area",
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

            train_autoencoder_model(train_df, val_df, cfg, repo_root=root)
            result = run_autoencoder_inference(test_df, cfg, repo_root=root, split_name="test")

            self.assertEqual(result.image_count, 3)
            self.assertEqual(result.patch_count, 12)
            self.assertTrue(result.image_scores_path.exists())
            self.assertTrue(result.patch_scores_path.exists())

            image_scores = pd.read_csv(result.image_scores_path)
            patch_scores = pd.read_csv(result.patch_scores_path)

            self.assertEqual(len(image_scores), 3)
            self.assertEqual(len(patch_scores), 12)
            self.assertEqual(set(image_scores["label"].astype(int)), {0, 1})
            self.assertEqual(int(image_scores.loc[0, "ground_truth_pixel_count"]), 0)
            self.assertGreater(int(image_scores.loc[1, "ground_truth_pixel_count"]), 0)
            self.assertEqual(set(image_scores["category"].astype(str)), {"demo", "other"})

            heatmap_path = root / image_scores.loc[0, "heatmap_path"]
            heatmap = np.load(heatmap_path)
            self.assertEqual(heatmap.shape, (8, 8))
            self.assertTrue(bool(np.isfinite(heatmap).all()))
            self.assertTrue((result.figure_root / "demo" / "good" / "000_heatmap.png").exists())
            self.assertTrue((result.figure_root / "demo" / "bump" / "001_heatmap.png").exists())
            self.assertTrue((result.figure_root / "other" / "good" / "000_heatmap.png").exists())


if __name__ == "__main__":
    unittest.main()
