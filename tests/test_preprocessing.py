"""Tests for preprocessing utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile

from src.data.preprocessing import (
    build_processed_output_paths,
    build_valid_mask,
    compute_crop_box,
    compute_otsu_threshold,
    crop_depth_and_mask,
    extract_depth_map,
    infer_foreground_mask,
    load_xyz_map,
    normalize_depth_map,
    preprocess_depth_map,
    preprocess_xyz_path,
    resize_depth_and_mask,
    save_processed_sample,
)


class PreprocessingTestCase(unittest.TestCase):
    """Verify preprocessing stays explicit and testable."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.xyz_path = self.root / "sample_xyz.tiff"

        xyz_map = np.zeros((4, 4, 3), dtype=np.float32)
        xyz_map[..., 2] = np.array(
            [
                [0.0, 1.0, 2.0, 0.0],
                [1.0, 2.0, 3.0, 4.0],
                [0.0, 3.0, 4.0, 5.0],
                [0.0, 0.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        )
        tifffile.imwrite(self.xyz_path, xyz_map, photometric="rgb")

        self.cfg = {
            "data": {
                "image_size": [2, 2],
                "normalization": "per_image",
                "crop_margin": 0,
            }
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_extract_and_mask(self) -> None:
        """XYZ loading should expose the Z channel and valid pixels correctly."""
        xyz_map = load_xyz_map(self.xyz_path)
        depth_map = extract_depth_map(xyz_map)
        valid_mask = build_valid_mask(depth_map)

        self.assertEqual(xyz_map.shape, (4, 4, 3))
        self.assertEqual(depth_map.shape, (4, 4))
        self.assertEqual(int(valid_mask.sum()), 11)
        self.assertFalse(bool(valid_mask[0, 0]))
        self.assertTrue(bool(valid_mask[1, 1]))

    def test_resize_and_normalize(self) -> None:
        """Resizing and normalization should preserve valid-only semantics."""
        xyz_map = load_xyz_map(self.xyz_path)
        depth_map = extract_depth_map(xyz_map)
        valid_mask = build_valid_mask(depth_map)

        resized_depth, resized_mask = resize_depth_and_mask(depth_map, valid_mask, (2, 2))
        normalized_depth, stats = normalize_depth_map(resized_depth, resized_mask, mode="per_image")

        self.assertEqual(resized_depth.shape, (2, 2))
        self.assertEqual(resized_mask.shape, (2, 2))
        self.assertEqual(normalized_depth.shape, (2, 2))
        self.assertGreaterEqual(float(normalized_depth[resized_mask].min()), 0.0)
        self.assertLessEqual(float(normalized_depth[resized_mask].max()), 1.0)
        self.assertIn("valid_min", stats)
        self.assertIn("valid_max", stats)

    def test_resize_preserves_aspect_ratio_with_padding(self) -> None:
        """Aspect-ratio preserving resize should letterbox wide crops instead of stretching them."""
        depth_map = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            dtype=np.float32,
        )
        valid_mask = np.ones_like(depth_map, dtype=bool)

        resized_depth, resized_mask = resize_depth_and_mask(
            depth_map,
            valid_mask,
            (4, 4),
            mode="pad",
        )

        self.assertEqual(resized_depth.shape, (4, 4))
        self.assertEqual(resized_mask.shape, (4, 4))
        self.assertFalse(bool(resized_mask[0, 0]))
        self.assertFalse(bool(resized_mask[3, 3]))
        self.assertTrue(bool(resized_mask[1, 0]))
        self.assertTrue(bool(resized_mask[2, 3]))
        np.testing.assert_array_equal(resized_depth[1:3, :], depth_map)

    def test_resize_preserves_area_without_padding(self) -> None:
        """Area-preserving resize should keep rectangular crops rectangular."""
        depth_map = np.ones((2, 8), dtype=np.float32)
        valid_mask = np.ones_like(depth_map, dtype=bool)

        resized_depth, resized_mask = resize_depth_and_mask(
            depth_map,
            valid_mask,
            (4, 4),
            mode="preserve_area",
        )

        self.assertEqual(resized_depth.shape, (2, 8))
        self.assertEqual(resized_mask.shape, (2, 8))
        self.assertTrue(bool(resized_mask.all()))

    def test_area_preserving_resize_respects_minimum_size(self) -> None:
        """Very thin crops should still be large enough for configured patches."""
        depth_map = np.ones((1, 64), dtype=np.float32)
        valid_mask = np.ones_like(depth_map, dtype=bool)

        resized_depth, resized_mask = resize_depth_and_mask(
            depth_map,
            valid_mask,
            (8, 8),
            mode="preserve_area",
            min_size=(4, 4),
        )

        self.assertGreaterEqual(resized_depth.shape[0], 4)
        self.assertGreaterEqual(resized_depth.shape[1], 4)
        self.assertEqual(resized_depth.shape, resized_mask.shape)

    def test_area_preserving_resize_snaps_to_patch_grid(self) -> None:
        """Processed shapes should allow fixed-size patches without edge extras."""
        depth_map = np.ones((95, 900), dtype=np.float32)
        valid_mask = np.ones_like(depth_map, dtype=bool)

        resized_depth, resized_mask = resize_depth_and_mask(
            depth_map,
            valid_mask,
            (256, 256),
            mode="preserve_area",
            min_size=(32, 32),
            patch_size=(32, 32),
            stride=(16, 16),
        )

        self.assertEqual(resized_depth.shape, resized_mask.shape)
        self.assertEqual((resized_depth.shape[0] - 32) % 16, 0)
        self.assertEqual((resized_depth.shape[1] - 32) % 16, 0)
        self.assertGreaterEqual(resized_depth.shape[0], 32)
        self.assertGreaterEqual(resized_depth.shape[1], 32)

    def test_preprocess_and_save(self) -> None:
        """End-to-end preprocessing should save depth and mask arrays."""
        processed_depth, processed_mask, stats = preprocess_xyz_path(self.xyz_path, self.cfg)
        self.assertEqual(processed_depth.shape, (2, 2))
        self.assertEqual(processed_mask.shape, (2, 2))
        self.assertTrue(np.all((processed_depth[processed_mask] >= 0.0) & (processed_depth[processed_mask] <= 1.0)))
        self.assertIn("processed_valid_fraction", stats)
        self.assertIn("crop_top", stats)
        self.assertIn("crop_bottom", stats)

        depth_output_path, mask_output_path = build_processed_output_paths(
            self.root / "processed",
            "demo",
            "train",
            "good",
            "000",
        )
        save_processed_sample(processed_depth, processed_mask, depth_output_path, mask_output_path)
        self.assertTrue(depth_output_path.exists())
        self.assertTrue(mask_output_path.exists())

        saved_depth = np.load(depth_output_path)
        saved_mask = np.load(mask_output_path)
        self.assertEqual(saved_depth.shape, (2, 2))
        self.assertEqual(saved_mask.shape, (2, 2))

    def test_preprocess_depth_map_without_path(self) -> None:
        """Direct depth preprocessing should also work without TIFF loading."""
        depth_map = np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)
        processed_depth, processed_mask, stats = preprocess_depth_map(depth_map, self.cfg)
        self.assertEqual(processed_depth.shape, (2, 2))
        self.assertEqual(processed_mask.shape, (2, 2))
        self.assertFalse(bool(processed_mask[0, 0]))
        self.assertAlmostEqual(float(processed_depth[0, 1]), 0.0)
        self.assertAlmostEqual(float(processed_depth[1, 1]), 1.0)
        self.assertAlmostEqual(float(stats["processed_valid_fraction"]), 0.75)

    def test_foreground_mask_and_crop(self) -> None:
        """Foreground inference should isolate the object and return a stable crop."""
        depth_map = np.full((8, 8), 5.0, dtype=np.float32)
        depth_map[2:6, 2:6] = 2.0
        depth_map[0, 0] = 0.0
        valid_mask = build_valid_mask(depth_map)

        threshold = compute_otsu_threshold(depth_map[valid_mask])
        self.assertGreater(threshold, 2.0)
        self.assertLess(threshold, 5.0)

        foreground_mask = infer_foreground_mask(depth_map, valid_mask)
        self.assertTrue(bool(foreground_mask[3, 3]))
        self.assertFalse(bool(foreground_mask[0, 7]))

        crop_box = compute_crop_box(foreground_mask, depth_map.shape, margin=1)
        self.assertEqual(crop_box, (1, 7, 1, 7))

        cropped_depth, cropped_mask = crop_depth_and_mask(depth_map, valid_mask, crop_box)
        self.assertEqual(cropped_depth.shape, (6, 6))
        self.assertEqual(cropped_mask.shape, (6, 6))
        self.assertTrue(bool(cropped_mask[2, 2]))


if __name__ == "__main__":
    unittest.main()
