"""Tests for patch and image dataset helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loaders import ImageDataset, PatchDataset, build_image_dataset, build_patch_dataset


class LoaderTestCase(unittest.TestCase):
    """Verify processed maps become deterministic patch and image samples."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        self.depth_path = Path("data/processed/demo/train/good/000_depth.npy")
        self.mask_path = Path("data/processed/demo/train/good/000_mask.npy")
        self.anomaly_depth_path = Path("data/processed/demo/test/hole/001_depth.npy")
        self.anomaly_mask_path = Path("data/processed/demo/test/hole/001_mask.npy")

        depth = np.arange(16, dtype=np.float32).reshape(4, 4)
        mask = np.ones((4, 4), dtype=bool)
        mask[0, 0] = False
        anomaly_depth = np.full((4, 4), 5.0, dtype=np.float32)
        anomaly_mask = np.ones((4, 4), dtype=bool)

        for relative_path, array in (
            (self.depth_path, depth),
            (self.mask_path, mask),
            (self.anomaly_depth_path, anomaly_depth),
            (self.anomaly_mask_path, anomaly_mask),
        ):
            output_path = self.root / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, array)

        self.split_df = pd.DataFrame(
            [
                {
                    "category": "demo",
                    "split": "train",
                    "defect_type": "good",
                    "label": 0,
                    "sample_id": "000",
                    "rgb_path": "",
                    "xyz_path": "",
                    "mask_path": "",
                    "processed_depth_path": self.depth_path.as_posix(),
                    "processed_mask_path": self.mask_path.as_posix(),
                },
                {
                    "category": "demo",
                    "split": "test",
                    "defect_type": "hole",
                    "label": 1,
                    "sample_id": "001",
                    "rgb_path": "",
                    "xyz_path": "",
                    "mask_path": "",
                    "processed_depth_path": self.anomaly_depth_path.as_posix(),
                    "processed_mask_path": self.anomaly_mask_path.as_posix(),
                },
            ]
        )
        self.cfg = {
            "patches": {
                "size": [2, 2],
                "stride": [2, 2],
                "min_valid_fraction": 0.5,
            }
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_patch_dataset_keeps_normal_patches_and_metadata(self) -> None:
        """PatchDataset should filter normal samples and return aligned patches."""
        dataset = build_patch_dataset(
            self.split_df,
            self.cfg,
            repo_root=self.root,
            normal_only=True,
        )

        self.assertIsInstance(dataset, PatchDataset)
        self.assertEqual(len(dataset), 4)

        sample = dataset[0]
        self.assertEqual(sample["category"], "demo")
        self.assertEqual(sample["label"], 0)
        self.assertEqual(sample["patch"].shape, (2, 2))
        self.assertEqual(sample["mask"].shape, (2, 2))
        np.testing.assert_array_equal(
            sample["coords"],
            np.array([0, 2, 0, 2], dtype=np.int32),
        )
        self.assertAlmostEqual(float(sample["valid_fraction"]), 0.75)

    def test_patch_dataset_can_return_all_patches_for_one_image(self) -> None:
        """Per-image patch extraction should preserve deterministic order."""
        dataset = PatchDataset(
            self.split_df.iloc[:1],
            patch_size=(2, 2),
            stride=(2, 2),
            repo_root=self.root,
        )

        patches, mask_patches, coords = dataset.patches_for_image(0)

        self.assertEqual(patches.shape, (4, 2, 2))
        self.assertEqual(mask_patches.shape, (4, 2, 2))
        np.testing.assert_array_equal(coords[-1], np.array([2, 4, 2, 4], dtype=np.int32))

    def test_image_dataset_loads_full_processed_maps(self) -> None:
        """ImageDataset should keep full image tensors for inference."""
        dataset = build_image_dataset(self.split_df, self.cfg, repo_root=self.root)

        self.assertIsInstance(dataset, ImageDataset)
        self.assertEqual(len(dataset), 2)

        sample = dataset[1]
        self.assertEqual(sample["label"], 1)
        self.assertEqual(sample["image_shape"], (4, 4))
        self.assertTrue(bool(sample["mask"].all()))
        self.assertAlmostEqual(float(sample["depth"].mean()), 5.0)


if __name__ == "__main__":
    unittest.main()
