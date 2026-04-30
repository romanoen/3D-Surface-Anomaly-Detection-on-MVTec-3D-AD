"""Tests for patch extraction and overlap-aware aggregation."""

from __future__ import annotations

import unittest

import numpy as np

from src.data.patching import (
    aggregate_patch_maps,
    aggregate_patch_values,
    compute_patch_coverage,
    extract_patches,
    extract_patches_with_mask,
)


class PatchingTestCase(unittest.TestCase):
    """Verify the shared patch helpers behave deterministically."""

    def test_extract_patches_covers_edges_in_row_major_order(self) -> None:
        """Patch coordinates should stay deterministic and cover the image border."""
        image = np.arange(30, dtype=np.float32).reshape(5, 6)

        patches, coords = extract_patches(image, patch_size=(3, 4), stride=(2, 3))

        expected_coords = np.array(
            [
                [0, 3, 0, 4],
                [0, 3, 2, 6],
                [2, 5, 0, 4],
                [2, 5, 2, 6],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(coords, expected_coords)
        np.testing.assert_array_equal(patches[0], image[0:3, 0:4])
        np.testing.assert_array_equal(patches[-1], image[2:5, 2:6])

    def test_extract_patches_with_mask_keeps_alignment(self) -> None:
        """Image and mask patches should share the same coordinates."""
        image = np.arange(16, dtype=np.float32).reshape(4, 4)
        mask = np.array(
            [
                [True, True, False, False],
                [True, True, False, False],
                [False, False, True, True],
                [False, False, True, True],
            ],
            dtype=bool,
        )

        image_patches, mask_patches, coords = extract_patches_with_mask(
            image,
            mask,
            patch_size=(2, 2),
            stride=(2, 2),
        )

        self.assertEqual(image_patches.shape, (4, 2, 2))
        self.assertEqual(mask_patches.shape, (4, 2, 2))
        self.assertTrue(np.array_equal(mask_patches[0], mask[0:2, 0:2]))
        self.assertTrue(np.array_equal(mask_patches[-1], mask[2:4, 2:4]))
        np.testing.assert_array_equal(coords[0], np.array([0, 2, 0, 2], dtype=np.int32))
        np.testing.assert_array_equal(coords[-1], np.array([2, 4, 2, 4], dtype=np.int32))

    def test_patch_map_aggregation_reconstructs_original_image(self) -> None:
        """Averaging overlapping patches should recover the source image exactly."""
        image = np.arange(42, dtype=np.float32).reshape(6, 7)

        patches, coords = extract_patches(image, patch_size=(4, 3), stride=(3, 2))
        reconstructed = aggregate_patch_maps(patches, coords, image.shape)

        np.testing.assert_allclose(reconstructed, image)

    def test_patch_value_aggregation_averages_overlap_counts(self) -> None:
        """Scalar patch scores should be averaged where patches overlap."""
        coords = np.array(
            [
                [0, 2, 0, 2],
                [0, 2, 1, 3],
            ],
            dtype=np.int32,
        )
        values = np.array([2.0, 6.0], dtype=np.float32)

        aggregated = aggregate_patch_values(values, coords, image_shape=(2, 3))
        expected = np.array(
            [
                [2.0, 4.0, 6.0],
                [2.0, 4.0, 6.0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(aggregated, expected)

    def test_patch_coverage_counts_actual_overlaps(self) -> None:
        """Coverage should count how often patches touch each pixel."""
        image = np.zeros((4, 4), dtype=np.float32)
        _, coords = extract_patches(image, patch_size=(2, 2), stride=(1, 1))

        coverage = compute_patch_coverage(coords, image.shape)
        expected = np.array(
            [
                [1.0, 2.0, 2.0, 1.0],
                [2.0, 4.0, 4.0, 2.0],
                [2.0, 4.0, 4.0, 2.0],
                [1.0, 2.0, 2.0, 1.0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(coverage, expected)


if __name__ == "__main__":
    unittest.main()
