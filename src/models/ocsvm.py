"""One-Class SVM model wrapper for patch-level anomaly scoring."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import OneClassSVM


def _as_feature_matrix(features: np.ndarray) -> np.ndarray:
    """Return features as a validated 2D float32 matrix."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {features.shape}")
    if features.shape[0] == 0:
        raise ValueError("Feature matrix must contain at least one row.")
    if not np.isfinite(features).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")
    return features


class OneClassSVMModel:
    """Thin wrapper that exposes higher-is-more-anomalous patch scores."""

    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.05,
        gamma: str | float = "scale",
    ) -> None:
        self.model = OneClassSVM(kernel=kernel, nu=float(nu), gamma=gamma)

    def fit(self, features: np.ndarray) -> "OneClassSVMModel":
        """Fit the One-Class SVM on normal training feature vectors."""
        self.model.fit(_as_feature_matrix(features))
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """Return anomaly scores where larger values indicate stronger anomalies."""
        features = _as_feature_matrix(features)
        decision_scores = self.model.decision_function(features).reshape(-1)
        return (-decision_scores).astype(np.float32)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return sklearn's inlier/outlier labels for feature vectors."""
        return self.model.predict(_as_feature_matrix(features))

    def save(self, output_path: str | Path) -> None:
        """Persist the fitted wrapper."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)

    @classmethod
    def load(cls, input_path: str | Path) -> "OneClassSVMModel":
        """Load a persisted wrapper."""
        model = joblib.load(input_path)
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(model).__name__}")
        return model
