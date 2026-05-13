"""Tests for the classical training loop."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

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


class ClassicalTrainerTestCase(unittest.TestCase):
    """Verify classical training writes the expected artifacts."""

    def test_train_classical_model_writes_artifacts(self) -> None:
        """A tiny normal-only dataset should train and save model outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_depth_0, train_mask_0 = _save_sample(root, "data/processed/demo/train/good/000", 0.1)
            train_depth_1, train_mask_1 = _save_sample(root, "data/processed/demo/train/good/001", 0.2)
            val_depth, val_mask = _save_sample(root, "data/processed/demo/val/good/000", 0.15)
            other_train_depth_0, other_train_mask_0 = _save_sample(
                root,
                "data/processed/other/train/good/000",
                0.4,
            )
            other_train_depth_1, other_train_mask_1 = _save_sample(
                root,
                "data/processed/other/train/good/001",
                0.5,
            )
            other_val_depth, other_val_mask = _save_sample(
                root,
                "data/processed/other/val/good/000",
                0.45,
            )

            train_df = pd.DataFrame(
                [
                    _row("train", train_depth_0, train_mask_0, "000"),
                    _row("train", train_depth_1, train_mask_1, "001"),
                    _row(
                        "train",
                        other_train_depth_0,
                        other_train_mask_0,
                        "000",
                        category="other",
                    ),
                    _row(
                        "train",
                        other_train_depth_1,
                        other_train_mask_1,
                        "001",
                        category="other",
                    ),
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
                "features": {
                    "use_mean": True,
                    "use_std": True,
                    "use_min": True,
                    "use_max": True,
                    "use_range": True,
                    "use_gradient_mean": True,
                    "use_gradient_std": True,
                    "use_roughness": True,
                    "use_laplacian_var": True,
                },
                "training": {
                    "scaler": "standard",
                    "ocsvm_kernel": "rbf",
                    "ocsvm_nu": 0.25,
                    "ocsvm_gamma": "scale",
                    "max_train_patches": 8,
                    "max_val_patches": 4,
                },
            }

            result = train_classical_model(train_df, val_df, cfg, repo_root=root)

            self.assertEqual(result.train_patch_count, 16)
            self.assertEqual(result.val_patch_count, 8)
            self.assertEqual(len(result.category_results), 2)
            self.assertEqual({result.category for result in result.category_results}, {"demo", "other"})
            self.assertTrue(result.model_root.exists())
            for category_result in result.category_results:
                self.assertTrue(category_result.model_path.exists())
                self.assertTrue(category_result.scaler_path.exists())
                self.assertTrue(category_result.feature_metadata_path.exists())
            self.assertTrue(result.summary_path.exists())
            self.assertTrue(result.category_summary_path.exists())
            self.assertTrue(result.train_score_path.exists())
            self.assertTrue(result.val_score_path.exists())
            self.assertTrue(
                (root / "fig" / "04_classical" / "training" / "demo" / "score_distributions.png").exists()
            )
            self.assertTrue(
                (root / "fig" / "04_classical" / "training" / "other" / "score_distributions.png").exists()
            )


if __name__ == "__main__":
    unittest.main()
