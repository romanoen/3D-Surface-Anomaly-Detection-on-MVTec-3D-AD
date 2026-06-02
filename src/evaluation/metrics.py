"""Evaluation metrics for anomaly detection and optional localization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _as_binary_arrays(labels: np.ndarray | pd.Series, scores: np.ndarray | pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return validated binary labels and score arrays."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if labels.size == 0 or scores.size == 0:
        raise ValueError("labels and scores must contain at least one value.")
    if labels.shape != scores.shape:
        raise ValueError(f"labels and scores must have the same shape, got {labels.shape} and {scores.shape}")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or infinite values.")
    unique_labels = set(np.unique(labels).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"labels must be binary 0/1 values, got {sorted(unique_labels)}")
    return labels, scores


def compute_binary_classification_metrics(
    labels: np.ndarray | pd.Series,
    scores: np.ndarray | pd.Series,
) -> dict[str, float | int]:
    """Compute safe image-level anomaly metrics for one score vector."""
    labels, scores = _as_binary_arrays(labels, scores)
    positive_count = int(labels.sum())
    image_count = int(labels.size)
    negative_count = int(image_count - positive_count)

    metrics: dict[str, float | int] = {
        "image_count": image_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "auroc": float("nan"),
        "ap": float("nan"),
    }

    if positive_count > 0 and negative_count > 0:
        metrics["auroc"] = float(roc_auc_score(labels, scores))
        metrics["ap"] = float(average_precision_score(labels, scores))
    return metrics


def evaluate_per_category(
    per_image_df: pd.DataFrame,
    score_columns: Mapping[str, str],
    category_column: str = "category",
    label_column: str = "label",
) -> pd.DataFrame:
    """Compute one row of image-level metrics per category and method."""
    rows: list[dict[str, Any]] = []
    grouped = per_image_df.groupby(category_column, sort=True, dropna=False)
    for category, group_df in grouped:
        for method_name, score_column in score_columns.items():
            metrics = compute_binary_classification_metrics(group_df[label_column], group_df[score_column])
            rows.append(
                {
                    "category": category,
                    "method": method_name,
                    **metrics,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "category",
                "method",
                "image_count",
                "positive_count",
                "negative_count",
                "score_mean",
                "score_std",
                "score_min",
                "score_max",
                "auroc",
                "ap",
            ]
        )

    return pd.DataFrame(rows).sort_values(["category", "method"]).reset_index(drop=True)


def summarize_benchmark_results(
    per_image_df: pd.DataFrame,
    per_category_df: pd.DataFrame,
    score_columns: Mapping[str, str],
    label_column: str = "label",
) -> pd.DataFrame:
    """Compute overall and macro image-level metrics for each method."""
    rows: list[dict[str, Any]] = []
    for method_name, score_column in score_columns.items():
        overall = compute_binary_classification_metrics(per_image_df[label_column], per_image_df[score_column])
        category_rows = per_category_df.loc[per_category_df["method"] == method_name]
        macro_auroc = float(category_rows["auroc"].dropna().mean()) if not category_rows["auroc"].dropna().empty else float("nan")
        macro_ap = float(category_rows["ap"].dropna().mean()) if not category_rows["ap"].dropna().empty else float("nan")
        rows.append(
            {
                "method": method_name,
                "image_count": overall["image_count"],
                "positive_count": overall["positive_count"],
                "negative_count": overall["negative_count"],
                "overall_auroc": overall["auroc"],
                "overall_ap": overall["ap"],
                "macro_auroc": macro_auroc,
                "macro_ap": macro_ap,
                "category_count": int(per_image_df["category"].nunique()),
                "evaluated_category_count": int(category_rows["auroc"].notna().sum()),
            }
        )

    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
