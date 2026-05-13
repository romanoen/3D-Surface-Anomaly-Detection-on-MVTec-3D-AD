"""Training utilities for the classical anomaly detection baseline."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir()) / "mvtec_3d_ad_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loaders import PatchDataset, build_patch_dataset
from src.features.raw_patch_features import (
    extract_patch_features,
    raw_patch_feature_names,
)
from src.models.ocsvm import OneClassSVMModel


@dataclass(frozen=True)
class ClassicalCategoryTrainingResult:
    """Paths and summary values produced by one category-specific model."""

    category: str
    model_path: Path
    scaler_path: Path
    pca_path: Path
    feature_metadata_path: Path
    selected_feature_names: tuple[str, ...]
    train_patch_count: int
    val_patch_count: int


@dataclass(frozen=True)
class ClassicalTrainingResult:
    """Paths and summary values produced by classical training."""

    model_root: Path
    summary_path: Path
    category_summary_path: Path
    train_score_path: Path
    val_score_path: Path
    category_results: tuple[ClassicalCategoryTrainingResult, ...]
    selected_feature_names: tuple[str, ...]
    train_patch_count: int
    val_patch_count: int


def _feature_mode(feature_cfg: dict[str, Any]) -> str:
    """Return the configured classical feature mode."""
    mode = str(feature_cfg.get("mode", "raw_depth")).lower()
    if mode != "raw_depth":
        raise ValueError(
            f"Unsupported classical feature mode: {mode}. "
            "The main baseline expects mode='raw_depth'."
        )
    return mode


def _fill_strategy(feature_cfg: dict[str, Any]) -> str:
    """Return the mask fill strategy for raw-patch features."""
    strategy = str(feature_cfg.get("fill_strategy", "patch_mean")).lower()
    if strategy not in {"patch_mean", "zero"}:
        raise ValueError(f"Unsupported raw patch fill strategy: {strategy}")
    return strategy


def _requested_pca_components(feature_cfg: dict[str, Any]) -> int:
    """Return the requested PCA dimensionality, or zero when disabled."""
    value = feature_cfg.get("pca_components", 0)
    if value is None:
        return 0
    components = int(value)
    if components < 0:
        raise ValueError(f"pca_components must be non-negative, got {components}")
    return components


def _model_feature_names(pca: PCA | None, input_feature_names: tuple[str, ...]) -> tuple[str, ...]:
    """Return the feature names seen by the one-class model."""
    if pca is None:
        return input_feature_names
    return tuple(f"pca_{index:03d}" for index in range(int(pca.n_components_)))


def _sample_patch_indices(
    patch_count: int,
    max_patches: int | None,
    seed: int,
) -> np.ndarray:
    """Return deterministic patch indices, optionally sampled without replacement."""
    if max_patches is None or max_patches <= 0 or patch_count <= max_patches:
        return np.arange(patch_count, dtype=np.int64)

    rng = np.random.default_rng(seed)
    indices = rng.choice(patch_count, size=int(max_patches), replace=False)
    return np.sort(indices.astype(np.int64))


def extract_dataset_features(
    dataset: PatchDataset,
    feature_cfg: dict[str, Any],
    max_patches: int | None = None,
    seed: int = 42,
    desc: str = "Extracting features",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract raw normalized height vectors and patch metadata from a dataset."""
    _feature_mode(feature_cfg)
    fill_strategy = _fill_strategy(feature_cfg)
    patch_indices = _sample_patch_indices(len(dataset), max_patches, seed)
    feature_count = int(np.prod(dataset.patch_size))
    features = np.zeros((len(patch_indices), feature_count), dtype=np.float32)
    metadata_rows: list[dict[str, Any]] = []

    for output_index, patch_index in enumerate(tqdm(patch_indices, desc=desc, leave=False)):
        sample = dataset[int(patch_index)]
        features[output_index] = extract_patch_features(
            sample["patch"],
            sample["mask"],
            fill_strategy=fill_strategy,
        )

        top, bottom, left, right = [int(value) for value in sample["coords"]]
        metadata_rows.append(
            {
                "category": sample["category"],
                "split": sample["split"],
                "defect_type": sample["defect_type"],
                "label": int(sample["label"]),
                "sample_id": sample["sample_id"],
                "sample_position": int(sample["sample_position"]),
                "patch_position": int(sample["patch_position"]),
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
                "valid_fraction": float(sample["valid_fraction"]),
            }
        )

    return features, pd.DataFrame(metadata_rows)


def _build_scaler(training_cfg: dict[str, Any]) -> StandardScaler | None:
    """Create the configured feature scaler."""
    scaler_name = str(training_cfg.get("scaler", "standard")).lower()
    if scaler_name == "standard":
        return StandardScaler()
    if scaler_name in {"none", "identity"}:
        return None
    raise ValueError(f"Unsupported scaler: {scaler_name}")


def _transform_features(
    scaler: StandardScaler | None,
    train_features: np.ndarray,
    val_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the scaler on training features and transform train/val features."""
    if scaler is None:
        return train_features.astype(np.float32), val_features.astype(np.float32)

    train_scaled = scaler.fit_transform(train_features).astype(np.float32)
    val_scaled = scaler.transform(val_features).astype(np.float32)
    return train_scaled, val_scaled


def _fit_pca(
    train_features: np.ndarray,
    requested_components: int,
    seed: int,
) -> PCA | None:
    """Fit a PCA transform when requested and feasible."""
    if requested_components <= 0:
        return None

    effective_components = min(
        int(requested_components),
        int(train_features.shape[0]),
        int(train_features.shape[1]),
    )
    if effective_components <= 0:
        return None

    pca = PCA(n_components=effective_components, svd_solver="auto", random_state=seed)
    pca.fit(train_features)
    return pca


def _apply_pca(
    pca: PCA | None,
    train_features: np.ndarray,
    val_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an optional fitted PCA transform."""
    if pca is None:
        return train_features.astype(np.float32), val_features.astype(np.float32)

    train_transformed = pca.transform(train_features).astype(np.float32)
    val_transformed = pca.transform(val_features).astype(np.float32)
    return train_transformed, val_transformed


def _save_feature_distribution_figure(
    train_features: np.ndarray,
    val_features: np.ndarray,
    feature_names: tuple[str, ...],
    output_path: Path,
) -> None:
    """Save compact train/validation feature distribution plots."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(feature_names) > 40:
        rng = np.random.default_rng(0)
        max_values = 250_000
        train_values = train_features.reshape(-1)
        val_values = val_features.reshape(-1)
        if train_values.size > max_values:
            train_values = train_values[
                rng.choice(train_values.size, size=max_values, replace=False)
            ]
        if val_values.size > max_values:
            val_values = val_values[
                rng.choice(val_values.size, size=max_values, replace=False)
            ]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        axes[0].hist(train_values, bins=60, alpha=0.8, density=True)
        axes[0].set_title("Train raw height values")
        axes[0].set_xlabel("Normalized height")
        axes[0].set_ylabel("Density")
        axes[0].grid(axis="y", linestyle="--", alpha=0.3)

        axes[1].hist(val_values, bins=60, alpha=0.8, density=True)
        axes[1].set_title("Validation raw height values")
        axes[1].set_xlabel("Normalized height")
        axes[1].grid(axis="y", linestyle="--", alpha=0.3)

        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    axes[0].boxplot(train_features, tick_labels=feature_names, showfliers=False)
    axes[0].set_title("Train feature distribution")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].boxplot(val_features, tick_labels=feature_names, showfliers=False)
    axes[1].set_title("Validation feature distribution")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_score_distribution_figure(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    output_path: Path,
) -> None:
    """Save train/validation patch score distributions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.hist(train_scores, bins=50, alpha=0.65, label="train normal", density=True)
    ax.hist(val_scores, bins=50, alpha=0.65, label="val normal", density=True)
    ax.set_title("Classical patch anomaly scores")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Density")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_pca_variance_figure(pca: PCA | None, output_path: Path) -> None:
    """Save the explained-variance curve for the raw-patch PCA transform."""
    if pca is None:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    explained = np.asarray(pca.explained_variance_ratio_, dtype=np.float32)
    cumulative = np.cumsum(explained)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", markersize=3)
    ax.set_title("PCA cumulative explained variance")
    ax.set_xlabel("PCA components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _safe_path_part(value: object) -> str:
    """Return a filesystem-friendly path fragment."""
    text = str(value).strip() or "unknown"
    for character in ("/", "\\", ":", " "):
        text = text.replace(character, "_")
    return text


def _normal_rows_for_category(split_df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Return normal rows for one category."""
    rows = split_df.loc[
        (split_df["category"].astype(str) == category)
        & (split_df["label"].astype(int) == 0)
    ].copy()
    return rows.reset_index(drop=True)


def _train_one_category(
    category: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict[str, Any],
    repo_root: Path,
    input_feature_names: tuple[str, ...],
    seed: int,
) -> tuple[ClassicalCategoryTrainingResult, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Train and persist one category-specific classical model."""
    paths_cfg = cfg["paths"]
    training_cfg = cfg["training"]
    feature_cfg = cfg.get("features", {})
    feature_mode = _feature_mode(feature_cfg)
    fill_strategy = _fill_strategy(feature_cfg)
    requested_pca_components = _requested_pca_components(feature_cfg)
    category_path = _safe_path_part(category)

    train_dataset = build_patch_dataset(train_df, cfg, repo_root=repo_root, normal_only=True)
    val_dataset = build_patch_dataset(val_df, cfg, repo_root=repo_root, normal_only=True)
    if len(train_dataset) == 0:
        raise ValueError(f"No valid normal training patches found for category {category!r}.")
    if len(val_dataset) == 0:
        raise ValueError(f"No valid normal validation patches found for category {category!r}.")

    train_features, train_metadata = extract_dataset_features(
        train_dataset,
        feature_cfg,
        max_patches=training_cfg.get("max_train_patches"),
        seed=seed,
        desc=f"Extracting train features [{category}]",
    )
    val_features, val_metadata = extract_dataset_features(
        val_dataset,
        feature_cfg,
        max_patches=training_cfg.get("max_val_patches"),
        seed=seed + 1,
        desc=f"Extracting val features [{category}]",
    )

    scaler = _build_scaler(training_cfg)
    train_scaled, val_scaled = _transform_features(scaler, train_features, val_features)
    pca = _fit_pca(train_scaled, requested_pca_components, seed=seed)
    train_model_features, val_model_features = _apply_pca(pca, train_scaled, val_scaled)
    model_feature_names = _model_feature_names(pca, input_feature_names)

    model = OneClassSVMModel(
        kernel=str(training_cfg.get("ocsvm_kernel", "rbf")),
        nu=float(training_cfg.get("ocsvm_nu", 0.05)),
        gamma=training_cfg.get("ocsvm_gamma", "scale"),
    ).fit(train_model_features)

    train_scores = model.score(train_model_features)
    val_scores = model.score(val_model_features)

    model_root = repo_root / paths_cfg["outputs_root"] / "models" / "classical" / category_path
    fig_root = repo_root / paths_cfg["fig_root"] / "04_classical" / "training" / category_path
    model_path = model_root / "ocsvm.joblib"
    scaler_path = model_root / "scaler.joblib"
    pca_path = model_root / "pca.joblib"
    feature_metadata_path = model_root / "features.json"

    model.save(model_path)
    model_root.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    joblib.dump(pca, pca_path)
    feature_metadata_path.write_text(
        json.dumps(
            {
                "category": category,
                "feature_mode": feature_mode,
                "fill_strategy": fill_strategy,
                "patch_size": list(train_dataset.patch_size),
                "input_feature_count": int(train_features.shape[1]),
                "model_feature_count": int(train_model_features.shape[1]),
                "model_feature_names": list(model_feature_names),
                "pca_components_requested": int(requested_pca_components),
                "pca_components_effective": int(pca.n_components_) if pca is not None else 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _save_feature_distribution_figure(
        train_features,
        val_features,
        input_feature_names,
        fig_root / "feature_distributions.png",
    )
    _save_pca_variance_figure(pca, fig_root / "pca_explained_variance.png")
    _save_score_distribution_figure(
        train_scores,
        val_scores,
        fig_root / "score_distributions.png",
    )

    train_score_df = train_metadata.copy()
    train_score_df["score"] = train_scores.astype(np.float32)
    val_score_df = val_metadata.copy()
    val_score_df["score"] = val_scores.astype(np.float32)

    summary = {
        "category": category,
        "model": "ocsvm",
        "feature_mode": feature_mode,
        "fill_strategy": fill_strategy,
        "train_patch_count": int(len(train_features)),
        "val_patch_count": int(len(val_features)),
        "input_feature_count": int(train_features.shape[1]),
        "model_feature_count": int(train_model_features.shape[1]),
        "pca_components_requested": int(requested_pca_components),
        "pca_components_effective": int(pca.n_components_) if pca is not None else 0,
        "train_score_mean": float(train_scores.mean()),
        "train_score_std": float(train_scores.std()),
        "val_score_mean": float(val_scores.mean()),
        "val_score_std": float(val_scores.std()),
        "model_path": model_path.relative_to(repo_root).as_posix(),
        "scaler_path": scaler_path.relative_to(repo_root).as_posix(),
        "pca_path": pca_path.relative_to(repo_root).as_posix(),
        "feature_metadata_path": feature_metadata_path.relative_to(repo_root).as_posix(),
    }

    result = ClassicalCategoryTrainingResult(
        category=category,
        model_path=model_path,
        scaler_path=scaler_path,
        pca_path=pca_path,
        feature_metadata_path=feature_metadata_path,
        selected_feature_names=model_feature_names,
        train_patch_count=int(len(train_features)),
        val_patch_count=int(len(val_features)),
    )
    return result, train_score_df, val_score_df, summary


def train_classical_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict[str, Any],
    repo_root: str | Path = ".",
) -> ClassicalTrainingResult:
    """Train one classical One-Class SVM baseline per object category."""
    repo_root = Path(repo_root)
    seed = int(cfg.get("project", {}).get("seed", 42))
    paths_cfg = cfg["paths"]
    feature_cfg = cfg.get("features", {})
    feature_mode = _feature_mode(feature_cfg)
    fill_strategy = _fill_strategy(feature_cfg)
    requested_pca_components = _requested_pca_components(feature_cfg)
    input_feature_names = raw_patch_feature_names(cfg["patches"]["size"])

    categories = sorted(
        str(category)
        for category in train_df.loc[train_df["label"].astype(int) == 0, "category"].unique()
    )
    if not categories:
        raise ValueError("No normal training categories found.")

    category_results: list[ClassicalCategoryTrainingResult] = []
    category_summaries: list[dict[str, Any]] = []
    train_score_tables: list[pd.DataFrame] = []
    val_score_tables: list[pd.DataFrame] = []

    for category_index, category in enumerate(categories):
        train_category_df = _normal_rows_for_category(train_df, category)
        val_category_df = _normal_rows_for_category(val_df, category)
        if val_category_df.empty:
            raise ValueError(f"No normal validation rows found for category {category!r}.")

        result, train_scores_df, val_scores_df, summary = _train_one_category(
            category,
            train_category_df,
            val_category_df,
            cfg,
            repo_root,
            input_feature_names,
            seed + category_index * 1009,
        )
        category_results.append(result)
        category_summaries.append(summary)
        train_score_tables.append(train_scores_df)
        val_score_tables.append(val_scores_df)

    model_root = repo_root / paths_cfg["outputs_root"] / "models" / "classical"
    metrics_root = repo_root / paths_cfg["outputs_root"] / "metrics"
    logs_root = repo_root / paths_cfg["outputs_root"] / "logs"
    summary_path = logs_root / "classical_training_summary.json"
    category_summary_path = metrics_root / "classical_per_category.csv"
    train_score_path = metrics_root / "classical_train_patch_scores.csv"
    val_score_path = metrics_root / "classical_val_patch_scores.csv"

    train_scores_all = pd.concat(train_score_tables, ignore_index=True)
    val_scores_all = pd.concat(val_score_tables, ignore_index=True)
    metrics_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    train_scores_all.to_csv(train_score_path, index=False)
    val_scores_all.to_csv(val_score_path, index=False)
    pd.DataFrame(category_summaries).to_csv(category_summary_path, index=False)

    summary = {
        "model": "ocsvm",
        "training_scope": "per_category",
        "feature_mode": feature_mode,
        "fill_strategy": fill_strategy,
        "input_feature_count": len(input_feature_names),
        "pca_components_requested": int(requested_pca_components),
        "category_count": len(category_results),
        "train_patch_count": int(sum(result.train_patch_count for result in category_results)),
        "val_patch_count": int(sum(result.val_patch_count for result in category_results)),
        "model_feature_count": int(category_summaries[0]["model_feature_count"]),
        "selected_feature_names": list(category_results[0].selected_feature_names),
        "model_root": model_root.relative_to(repo_root).as_posix(),
        "category_summary_path": category_summary_path.relative_to(repo_root).as_posix(),
        "categories": category_summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return ClassicalTrainingResult(
        model_root=model_root,
        summary_path=summary_path,
        category_summary_path=category_summary_path,
        train_score_path=train_score_path,
        val_score_path=val_score_path,
        category_results=tuple(category_results),
        selected_feature_names=category_results[0].selected_feature_names,
        train_patch_count=summary["train_patch_count"],
        val_patch_count=summary["val_patch_count"],
    )
