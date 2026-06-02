"""Tests for evaluation visualization helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation.visualization import (
    save_metric_delta_figure,
    save_overall_roc_pr_figure,
    save_rank_histogram_figure,
    select_extreme_cases,
    select_method_advantage_cases,
)


class VisualizationSelectionTests(unittest.TestCase):
    """Verify selection helpers choose stable representative cases."""

    def setUp(self) -> None:
        self.per_image_df = pd.DataFrame(
            [
                {
                    "category": "alpha",
                    "label": 0,
                    "sample_position": 0,
                    "classical_image_score": 0.95,
                    "autoencoder_image_score": 0.10,
                },
                {
                    "category": "beta",
                    "label": 0,
                    "sample_position": 1,
                    "classical_image_score": 0.90,
                    "autoencoder_image_score": 0.20,
                },
                {
                    "category": "alpha",
                    "label": 1,
                    "sample_position": 2,
                    "classical_image_score": 0.20,
                    "autoencoder_image_score": 0.90,
                },
                {
                    "category": "beta",
                    "label": 1,
                    "sample_position": 3,
                    "classical_image_score": 0.85,
                    "autoencoder_image_score": 0.40,
                },
                {
                    "category": "gamma",
                    "label": 1,
                    "sample_position": 4,
                    "classical_image_score": 0.10,
                    "autoencoder_image_score": 0.80,
                },
            ]
        )

    def test_select_extreme_cases_diversifies_categories(self) -> None:
        """False-positive selection should spread across categories when possible."""
        selected = select_extreme_cases(
            self.per_image_df,
            "classical_image_score",
            label_value=0,
            ascending=False,
            top_k=2,
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(set(selected["category"].astype(str)), {"alpha", "beta"})

    def test_select_method_advantage_cases_prefers_large_rank_gaps(self) -> None:
        """Advantage selection should surface cases where one method clearly wins."""
        selected = select_method_advantage_cases(
            self.per_image_df,
            "autoencoder_image_score",
            "classical_image_score",
            top_k=2,
        )
        self.assertEqual(len(selected), 2)
        self.assertIn("alpha", set(selected["category"].astype(str)))
        self.assertIn("gamma", set(selected["category"].astype(str)))


class VisualizationFigureTests(unittest.TestCase):
    """Verify figure helpers write their outputs."""

    def setUp(self) -> None:
        self.per_image_df = pd.DataFrame(
            [
                {
                    "category": "alpha",
                    "label": 0,
                    "classical_image_score": 0.10,
                    "autoencoder_image_score": 0.05,
                },
                {
                    "category": "alpha",
                    "label": 0,
                    "classical_image_score": 0.20,
                    "autoencoder_image_score": 0.15,
                },
                {
                    "category": "alpha",
                    "label": 1,
                    "classical_image_score": 0.60,
                    "autoencoder_image_score": 0.75,
                },
                {
                    "category": "beta",
                    "label": 0,
                    "classical_image_score": 0.30,
                    "autoencoder_image_score": 0.25,
                },
                {
                    "category": "beta",
                    "label": 1,
                    "classical_image_score": 0.80,
                    "autoencoder_image_score": 0.95,
                },
            ]
        )
        self.per_category_df = pd.DataFrame(
            [
                {"category": "alpha", "method": "classical", "auroc": 0.55, "ap": 0.70},
                {"category": "alpha", "method": "autoencoder", "auroc": 0.75, "ap": 0.82},
                {"category": "beta", "method": "classical", "auroc": 0.65, "ap": 0.78},
                {"category": "beta", "method": "autoencoder", "auroc": 0.60, "ap": 0.74},
            ]
        )
        self.score_columns = {
            "classical": "classical_image_score",
            "autoencoder": "autoencoder_image_score",
        }

    def test_figure_helpers_write_pngs(self) -> None:
        """Curve and metric plots should be written to disk."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            roc_path = tmp_root / "overall_roc_pr.png"
            ranks_path = tmp_root / "score_rank_histograms.png"
            delta_path = tmp_root / "auroc_delta_by_category.png"

            save_overall_roc_pr_figure(self.per_image_df, self.score_columns, roc_path)
            save_rank_histogram_figure(self.per_image_df, self.score_columns, ranks_path)
            save_metric_delta_figure(self.per_category_df, "auroc", delta_path)

            self.assertTrue(roc_path.exists())
            self.assertTrue(ranks_path.exists())
            self.assertTrue(delta_path.exists())


if __name__ == "__main__":
    unittest.main()
