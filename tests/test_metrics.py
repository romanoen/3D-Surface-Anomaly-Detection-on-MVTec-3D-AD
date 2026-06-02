"""Tests for shared benchmark metric helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from src.evaluation.metrics import (
    compute_binary_classification_metrics,
    evaluate_per_category,
    summarize_benchmark_results,
)


class MetricsTestCase(unittest.TestCase):
    """Verify image-level benchmark metrics are computed consistently."""

    def test_compute_binary_classification_metrics_returns_expected_counts(self) -> None:
        """Binary metric helper should report counts and valid scalar metrics."""
        metrics = compute_binary_classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])

        self.assertEqual(metrics["image_count"], 4)
        self.assertEqual(metrics["positive_count"], 2)
        self.assertEqual(metrics["negative_count"], 2)
        self.assertAlmostEqual(float(metrics["auroc"]), 1.0)
        self.assertAlmostEqual(float(metrics["ap"]), 1.0)

    def test_grouped_category_and_summary_metrics_follow_long_format(self) -> None:
        """Per-category and summary outputs should provide one row per method."""
        per_image_df = pd.DataFrame(
            [
                {
                    "category": "demo",
                    "label": 0,
                    "classical_image_score": 0.1,
                    "autoencoder_image_score": 0.2,
                },
                {
                    "category": "demo",
                    "label": 1,
                    "classical_image_score": 0.9,
                    "autoencoder_image_score": 0.8,
                },
                {
                    "category": "other",
                    "label": 0,
                    "classical_image_score": 0.2,
                    "autoencoder_image_score": 0.3,
                },
                {
                    "category": "other",
                    "label": 1,
                    "classical_image_score": 0.7,
                    "autoencoder_image_score": 0.9,
                },
            ]
        )
        score_columns = {
            "classical": "classical_image_score",
            "autoencoder": "autoencoder_image_score",
        }

        per_category_df = evaluate_per_category(per_image_df, score_columns)
        summary_df = summarize_benchmark_results(per_image_df, per_category_df, score_columns)

        self.assertEqual(len(per_category_df), 4)
        self.assertEqual(set(per_category_df["method"]), {"classical", "autoencoder"})
        self.assertEqual(len(summary_df), 2)
        self.assertEqual(set(summary_df["method"]), {"classical", "autoencoder"})


if __name__ == "__main__":
    unittest.main()
