"""Tests for the autoencoder training loop."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

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


def _row(
    split: str,
    depth_path: str,
    mask_path: str,
    sample_id: str,
    category: str = "demo",
) -> dict[str, object]:
    """Return one split row for synthetic processed files."""
    return {
        "category": category,
        "split": split,
        "defect_type": "good",
        "label": 0,
        "sample_id": sample_id,
        "rgb_path": "",
        "xyz_path": "",
        "mask_path": "",
        "processed_depth_path": depth_path,
        "processed_mask_path": mask_path,
    }


class AutoencoderTrainerTestCase(unittest.TestCase):
    """Verify autoencoder training writes the expected artifacts."""

    def test_train_autoencoder_model_writes_artifacts(self) -> None:
        """A tiny normal-only dataset should train and save autoencoder outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_depth_0, train_mask_0 = _save_sample(root, "data/processed/demo/train/good/000", 0.1)
            train_depth_1, train_mask_1 = _save_sample(root, "data/processed/demo/train/good/001", 0.2)
            val_depth, val_mask = _save_sample(root, "data/processed/demo/val/good/000", 0.15)
            other_train_depth_0, other_train_mask_0 = _save_sample(root, "data/processed/other/train/good/000", 0.4)
            other_train_depth_1, other_train_mask_1 = _save_sample(root, "data/processed/other/train/good/001", 0.5)
            other_val_depth, other_val_mask = _save_sample(root, "data/processed/other/val/good/000", 0.45)

            train_df = pd.DataFrame(
                [
                    _row("train", train_depth_0, train_mask_0, "000"),
                    _row("train", train_depth_1, train_mask_1, "001"),
                    _row("train", other_train_depth_0, other_train_mask_0, "000", category="other"),
                    _row("train", other_train_depth_1, other_train_mask_1, "001", category="other"),
                ]
            )
            val_df = pd.DataFrame(
                [
                    _row("val", val_depth, val_mask, "000"),
                    _row("val", other_val_depth, other_val_mask, "000", category="other"),
                ]
            )
            cfg = {
                "project": {"seed": 7},
                "paths": {
                    "outputs_root": "outputs",
                    "fig_root": "fig",
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
            }

            result = train_autoencoder_model(train_df, val_df, cfg, repo_root=root)

            self.assertEqual(result.train_patch_count, 16)
            self.assertEqual(result.val_patch_count, 8)
            self.assertEqual(len(result.category_results), 2)
            self.assertEqual({item.category for item in result.category_results}, {"demo", "other"})
            self.assertTrue(result.model_root.exists())
            self.assertTrue(result.summary_path.exists())
            self.assertTrue(result.category_summary_path.exists())
            for category_result in result.category_results:
                self.assertGreaterEqual(category_result.epochs_completed, 1)
                self.assertTrue(category_result.checkpoint_path.exists())
                self.assertTrue(category_result.history_path.exists())
                self.assertTrue((category_result.figure_root / "loss_curve.png").exists())
                self.assertTrue((category_result.figure_root / "reconstruction_examples.png").exists())


if __name__ == "__main__":
    unittest.main()
