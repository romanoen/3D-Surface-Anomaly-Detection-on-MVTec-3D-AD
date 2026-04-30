"""Tests for handcrafted patch feature extraction."""

from __future__ import annotations

import unittest

import numpy as np

from src.features.geometric_features import (
    FEATURE_NAMES,
    extract_batch_features,
    extract_patch_features,
)


class GeometricFeatureTestCase(unittest.TestCase):
    """Verify feature extraction returns stable fixed-length vectors."""

    def test_constant_patch_has_zero_spatial_features(self) -> None:
        """A constant patch should have no gradient, roughness, or Laplacian variance."""
        patch = np.full((4, 4), 2.0, dtype=np.float32)

        features = extract_patch_features(patch)

        self.assertEqual(features.shape, (len(FEATURE_NAMES),))
        self.assertAlmostEqual(float(features[0]), 2.0)
        self.assertAlmostEqual(float(features[1]), 0.0)
        self.assertAlmostEqual(float(features[4]), 0.0)
        self.assertAlmostEqual(float(features[5]), 0.0)
        self.assertAlmostEqual(float(features[7]), 0.0)
        self.assertAlmostEqual(float(features[8]), 0.0)

    def test_mask_excludes_invalid_depth_values(self) -> None:
        """Masked pixels should not affect scalar statistics."""
        patch = np.array(
            [
                [1.0, 2.0],
                [3.0, 100.0],
            ],
            dtype=np.float32,
        )
        mask = np.array(
            [
                [True, True],
                [True, False],
            ],
            dtype=bool,
        )

        features = extract_patch_features(patch, mask)

        self.assertAlmostEqual(float(features[0]), 2.0)
        self.assertAlmostEqual(float(features[2]), 1.0)
        self.assertAlmostEqual(float(features[3]), 3.0)
        self.assertAlmostEqual(float(features[4]), 2.0)

    def test_batch_features_keep_expected_shape(self) -> None:
        """Batch extraction should produce one fixed vector per patch."""
        patches = np.stack(
            [
                np.zeros((3, 3), dtype=np.float32),
                np.ones((3, 3), dtype=np.float32),
            ],
            axis=0,
        )

        features = extract_batch_features(patches)

        self.assertEqual(features.shape, (2, len(FEATURE_NAMES)))
        self.assertAlmostEqual(float(features[0, 0]), 0.0)
        self.assertAlmostEqual(float(features[1, 0]), 1.0)

    def test_empty_mask_returns_zero_features(self) -> None:
        """A patch with no valid pixels should produce a neutral feature vector."""
        patch = np.ones((3, 3), dtype=np.float32)
        mask = np.zeros((3, 3), dtype=bool)

        features = extract_patch_features(patch, mask)

        np.testing.assert_array_equal(features, np.zeros(len(FEATURE_NAMES), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
