"""Tests for anomaly-map and image-score aggregation helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.inference.anomaly_maps import (
    aggregate_patch_scores,
    image_score_from_patch_scores,
    normalize_map_for_display,
    save_heatmap_figure,
)


class AnomalyMapsTestCase(unittest.TestCase):
    """Verify shared inference aggregation helpers."""

    def test_image_score_topk_mean_uses_largest_scores(self) -> None:
        """Top-k mean should average the most anomalous patch scores."""
        scores = np.array([-3.0, 2.0, 7.0, 1.0], dtype=np.float32)

        self.assertAlmostEqual(
            image_score_from_patch_scores(scores, method="topk_mean", topk_fraction=0.5),
            4.5,
        )
        self.assertAlmostEqual(image_score_from_patch_scores(scores, method="max"), 7.0)

    def test_aggregate_patch_scores_masks_invalid_pixels(self) -> None:
        """Patch scores should be averaged by overlap and zeroed outside the object mask."""
        coords = np.array(
            [
                [0, 2, 0, 2],
                [0, 2, 1, 3],
            ],
            dtype=np.int32,
        )
        scores = np.array([2.0, 6.0], dtype=np.float32)
        valid_mask = np.array(
            [
                [True, True, False],
                [True, True, True],
            ],
            dtype=bool,
        )

        heatmap = aggregate_patch_scores(scores, coords, (2, 3), valid_mask=valid_mask)
        expected = np.array(
            [
                [2.0, 4.0, 0.0],
                [2.0, 4.0, 6.0],
            ],
            dtype=np.float32,
        )

        np.testing.assert_allclose(heatmap, expected)

    def test_display_normalization_and_figure_save(self) -> None:
        """Display helpers should normalize valid regions and write a PNG figure."""
        values = np.array([[1.0, 3.0], [5.0, 9.0]], dtype=np.float32)
        valid_mask = np.array([[True, True], [True, False]], dtype=bool)

        display = normalize_map_for_display(values, valid_mask)

        self.assertAlmostEqual(float(display[0, 0]), 0.0)
        self.assertAlmostEqual(float(display[1, 0]), 1.0)
        self.assertAlmostEqual(float(display[1, 1]), 0.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "heatmap.png"
            save_heatmap_figure(
                values,
                values,
                output_path,
                valid_mask=valid_mask,
                ground_truth_mask=valid_mask,
            )
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
