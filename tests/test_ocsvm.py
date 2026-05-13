"""Tests for the One-Class SVM wrapper."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.models.ocsvm import OneClassSVMModel


class OneClassSVMModelTestCase(unittest.TestCase):
    """Verify the classical model wrapper exposes stable anomaly scores."""

    def test_fit_score_save_and_load(self) -> None:
        """A fitted wrapper should score feature rows and round-trip via joblib."""
        features = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [0.0, 0.1],
                [0.1, 0.1],
            ],
            dtype=np.float32,
        )

        model = OneClassSVMModel(nu=0.25, gamma="scale").fit(features)
        scores = model.score(features)

        self.assertEqual(scores.shape, (4,))
        self.assertTrue(bool(np.isfinite(scores).all()))

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "ocsvm.joblib"
            model.save(model_path)
            loaded_model = OneClassSVMModel.load(model_path)

        np.testing.assert_allclose(loaded_model.score(features), scores)


if __name__ == "__main__":
    unittest.main()
