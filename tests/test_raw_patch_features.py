"""Tests for raw normalized depth-patch features."""

from __future__ import annotations

import unittest

import numpy as np

from src.features.raw_patch_features import (
    extract_batch_features,
    extract_patch_features,
    raw_patch_feature_names,
)


class RawPatchFeatureTestCase(unittest.TestCase):
    """Verify raw patch flattening keeps spatial depth values."""

    def test_patch_values_are_flattened_row_major(self) -> None:
        """A valid patch should become one row-major height vector."""
        patch = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

        features = extract_patch_features(patch)

        np.testing.assert_allclose(features, np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))

    def test_masked_pixels_use_patch_mean_by_default(self) -> None:
        """Outside-object pixels should not inject artificial zero-depth structure."""
        patch = np.array([[0.1, 0.2], [0.3, 0.9]], dtype=np.float32)
        mask = np.array([[True, True], [True, False]], dtype=bool)

        features = extract_patch_features(patch, mask)

        expected = np.array([0.1, 0.2, 0.3, 0.2], dtype=np.float32)
        np.testing.assert_allclose(features, expected)

    def test_zero_fill_strategy_is_available(self) -> None:
        """A zero fill can be selected for ablations."""
        patch = np.array([[0.1, 0.2], [0.3, 0.9]], dtype=np.float32)
        mask = np.array([[True, True], [True, False]], dtype=bool)

        features = extract_patch_features(patch, mask, fill_strategy="zero")

        expected = np.array([0.1, 0.2, 0.3, 0.0], dtype=np.float32)
        np.testing.assert_allclose(features, expected)

    def test_batch_features_keep_expected_shape(self) -> None:
        """Batch extraction should return one feature row per patch."""
        patches = np.stack(
            [
                np.zeros((2, 3), dtype=np.float32),
                np.ones((2, 3), dtype=np.float32),
            ],
            axis=0,
        )

        features = extract_batch_features(patches)

        self.assertEqual(features.shape, (2, 6))
        self.assertEqual(len(raw_patch_feature_names((2, 3))), 6)
        self.assertAlmostEqual(float(features[1].mean()), 1.0)

    def test_empty_mask_returns_neutral_zero_vector(self) -> None:
        """A patch without valid object pixels should become a neutral vector."""
        patch = np.ones((2, 2), dtype=np.float32)
        mask = np.zeros((2, 2), dtype=bool)

        features = extract_patch_features(patch, mask)

        np.testing.assert_array_equal(features, np.zeros(4, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
