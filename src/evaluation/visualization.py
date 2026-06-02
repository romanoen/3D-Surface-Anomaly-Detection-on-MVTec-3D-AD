"""Visualization helpers for figures used in debugging and reporting."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

_CACHE_ROOT = Path(tempfile.gettempdir()) / "mvtec_3d_ad_cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.data.preprocessing import load_binary_mask, transform_binary_mask_like_processed
from src.inference.anomaly_maps import normalize_map_for_display


def _draw_ground_truth_contour(axis: plt.Axes, ground_truth_mask: np.ndarray | None) -> None:
    """Draw a ground-truth contour when a processed annotation mask is available."""
    if ground_truth_mask is None:
        return

    ground_truth_mask = np.asarray(ground_truth_mask, dtype=bool)
    if not np.any(ground_truth_mask):
        return

    if np.all(ground_truth_mask):
        height, width = ground_truth_mask.shape
        axis.plot(
            [0, width - 1, width - 1, 0, 0],
            [0, 0, height - 1, height - 1, 0],
            color="cyan",
            linewidth=1.2,
        )
        return

    axis.contour(
        ground_truth_mask.astype(np.float32),
        levels=[0.5],
        colors=["cyan"],
        linewidths=1.2,
    )


def _draw_valid_mask_outline(axis: plt.Axes, valid_mask: np.ndarray | None) -> None:
    """Draw the processed object outline to anchor standalone heatmaps."""
    if valid_mask is None:
        return

    valid_mask = np.asarray(valid_mask, dtype=bool)
    if not np.any(valid_mask):
        return

    axis.contour(
        valid_mask.astype(np.float32),
        levels=[0.5],
        colors=["white"],
        linewidths=0.8,
        alpha=0.75,
    )


def save_method_comparison_figure(
    depth: np.ndarray,
    classical_heatmap: np.ndarray,
    autoencoder_heatmap: np.ndarray,
    output_path: str | Path,
    *,
    valid_mask: np.ndarray | None = None,
    ground_truth_mask: np.ndarray | None = None,
    title: str | None = None,
    classical_score: float | None = None,
    autoencoder_score: float | None = None,
) -> None:
    """Save a compact qualitative comparison without depth-texture overlays."""
    depth = np.asarray(depth, dtype=np.float32)
    classical_heatmap = np.asarray(classical_heatmap, dtype=np.float32)
    autoencoder_heatmap = np.asarray(autoencoder_heatmap, dtype=np.float32)
    if depth.shape != classical_heatmap.shape or depth.shape != autoencoder_heatmap.shape:
        raise ValueError(
            "Depth, classical heatmap, and autoencoder heatmap must share a shape, "
            f"got {depth.shape}, {classical_heatmap.shape}, and {autoencoder_heatmap.shape}"
        )

    depth_display = normalize_map_for_display(depth, valid_mask)
    classical_display = normalize_map_for_display(classical_heatmap, valid_mask)
    autoencoder_display = normalize_map_for_display(autoencoder_heatmap, valid_mask)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), layout="constrained")
    if title:
        fig.suptitle(title)

    panels = (
        ("depth + GT", depth_display, "gray"),
        ("classical heatmap", classical_display, "magma"),
        ("autoencoder heatmap", autoencoder_display, "magma"),
    )
    for axis, (panel_title, panel_image, cmap) in zip(axes, panels, strict=True):
        axis.imshow(panel_image, cmap=cmap, interpolation="nearest")
        axis.set_title(panel_title)
        axis.set_xticks([])
        axis.set_yticks([])
    _draw_ground_truth_contour(axes[0], ground_truth_mask)
    _draw_valid_mask_outline(axes[1], valid_mask)
    _draw_valid_mask_outline(axes[2], valid_mask)

    if classical_score is not None:
        axes[1].set_xlabel(f"score={classical_score:.4f}")
    if autoencoder_score is not None:
        axes[2].set_xlabel(f"score={autoencoder_score:.4f}")

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def select_representative_rows(
    per_image_df: pd.DataFrame,
    max_figures_per_category: int,
) -> pd.DataFrame:
    """Select a compact, deterministic set of representative rows per category."""
    if max_figures_per_category <= 0 or per_image_df.empty:
        return per_image_df.iloc[0:0].copy()

    selected_groups: list[pd.DataFrame] = []
    normal_budget = max(1, max_figures_per_category // 4)
    anomaly_budget = max_figures_per_category - normal_budget

    for _, category_df in per_image_df.groupby("category", sort=True, dropna=False):
        group = category_df.copy()
        group["classical_rank"] = group["classical_image_score"].rank(method="first", ascending=False, pct=True)
        group["autoencoder_rank"] = group["autoencoder_image_score"].rank(method="first", ascending=False, pct=True)
        group["combined_rank"] = group["classical_rank"] + group["autoencoder_rank"]

        normal_rows = (
            group.loc[group["label"].astype(int) == 0]
            .sort_values(["combined_rank", "sample_id"], ascending=[False, True])
            .head(normal_budget)
        )
        anomaly_rows = (
            group.loc[group["label"].astype(int) == 1]
            .sort_values(["combined_rank", "sample_id"], ascending=[False, True])
            .head(anomaly_budget)
        )
        selected_groups.append(pd.concat([normal_rows, anomaly_rows], ignore_index=False))

    selected = pd.concat(selected_groups).sort_values(
        ["category", "label", "combined_rank", "sample_id"],
        ascending=[True, False, False, True],
    )
    return selected.drop_duplicates(subset=["category", "sample_id"]).reset_index(drop=True)


def save_per_category_metric_figure(
    per_category_df: pd.DataFrame,
    metric_column: str,
    output_path: str | Path,
) -> None:
    """Save a per-category bar chart comparing the two benchmark methods."""
    if metric_column not in per_category_df:
        raise KeyError(f"Missing metric column {metric_column!r} in per_category_df.")

    plot_df = per_category_df.pivot(index="category", columns="method", values=metric_column)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    plot_df.plot(kind="bar", ax=ax, width=0.82)
    ax.set_title(f"{metric_column.upper()} by category")
    ax.set_xlabel("Category")
    ax.set_ylabel(metric_column.upper())
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="Method")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _method_color(method_name: str) -> str:
    """Return a consistent display color for a benchmark method."""
    palette = {
        "autoencoder": "#1f77b4",
        "classical": "#ff7f0e",
    }
    return palette.get(str(method_name).lower(), "#4c4c4c")


def _resolve_path(path_value: object, repo_root: Path) -> Path:
    """Resolve a potentially relative artifact path against the repository root."""
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return repo_root / path


def _load_processed_ground_truth_mask(
    row: pd.Series,
    cfg: Mapping[str, Any],
    repo_root: Path,
) -> np.ndarray | None:
    """Load a raw defect mask and transform it to processed-image coordinates."""
    mask_path_value = str(row.get("mask_path", "")).strip()
    if not mask_path_value:
        return None

    mask_path = _resolve_path(mask_path_value, repo_root)
    if not mask_path.exists():
        return None

    crop_keys = ("crop_top", "crop_bottom", "crop_left", "crop_right")
    if not all(key in row.index for key in crop_keys):
        return None

    crop_box = tuple(int(float(row[key])) for key in crop_keys)
    raw_mask = load_binary_mask(mask_path)
    return transform_binary_mask_like_processed(raw_mask, crop_box, dict(cfg))


def save_overall_roc_pr_figure(
    per_image_df: pd.DataFrame,
    score_columns: Mapping[str, str],
    output_path: str | Path,
    label_column: str = "label",
) -> None:
    """Save overall ROC and PR curves for all benchmark methods."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = per_image_df[label_column].astype(int).to_numpy()
    prevalence = float(labels.mean())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    roc_ax, pr_ax = axes

    roc_ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#999999", linewidth=1.2, label="random")
    pr_ax.axhline(prevalence, linestyle="--", color="#999999", linewidth=1.2, label=f"prevalence={prevalence:.3f}")

    for method_name, score_column in score_columns.items():
        scores = per_image_df[score_column].astype(np.float32).to_numpy()
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        color = _method_color(method_name)
        auroc = float(roc_auc_score(labels, scores))
        ap = float(average_precision_score(labels, scores))

        roc_ax.plot(
            fpr,
            tpr,
            linewidth=2.2,
            color=color,
            label=f"{method_name} (AUROC={auroc:.3f})",
        )
        pr_ax.plot(
            recall,
            precision,
            linewidth=2.2,
            color=color,
            label=f"{method_name} (AP={ap:.3f})",
        )

    roc_ax.set_title("Overall ROC Curve")
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.set_xlim(0.0, 1.0)
    roc_ax.set_ylim(0.0, 1.0)
    roc_ax.grid(linestyle="--", alpha=0.3)
    roc_ax.legend(title="Method")

    pr_ax.set_title("Overall Precision-Recall Curve")
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_xlim(0.0, 1.0)
    pr_ax.set_ylim(0.0, 1.0)
    pr_ax.grid(linestyle="--", alpha=0.3)
    pr_ax.legend(title="Method")

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_rank_histogram_figure(
    per_image_df: pd.DataFrame,
    score_columns: Mapping[str, str],
    output_path: str | Path,
    label_column: str = "label",
) -> None:
    """Save score-percentile histograms split by class for each method."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    method_names = list(score_columns.keys())
    fig, axes = plt.subplots(1, len(method_names), figsize=(6 * len(method_names), 4.5), layout="constrained")
    if len(method_names) == 1:
        axes = [axes]

    labels = per_image_df[label_column].astype(int).to_numpy()
    bins = np.linspace(0.0, 1.0, 21)

    for axis, method_name in zip(axes, method_names, strict=True):
        score_column = score_columns[method_name]
        rank_values = per_image_df[score_column].rank(method="average", pct=True).astype(np.float32).to_numpy()
        negatives = rank_values[labels == 0]
        positives = rank_values[labels == 1]

        axis.hist(
            negatives,
            bins=bins,
            density=True,
            alpha=0.6,
            color="#7f7f7f",
            label=f"normal (n={len(negatives)})",
        )
        axis.hist(
            positives,
            bins=bins,
            density=True,
            alpha=0.6,
            color=_method_color(method_name),
            label=f"anomaly (n={len(positives)})",
        )
        axis.set_title(f"{str(method_name).title()} Score Ranks")
        axis.set_xlabel("Score percentile rank")
        axis.set_ylabel("Density")
        axis.set_xlim(0.0, 1.0)
        axis.grid(linestyle="--", alpha=0.25)
        axis.legend()

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metric_delta_figure(
    per_category_df: pd.DataFrame,
    metric_column: str,
    output_path: str | Path,
    reference_method: str = "classical",
    comparison_method: str = "autoencoder",
) -> None:
    """Save a sorted per-category delta plot for one metric."""
    if metric_column not in per_category_df.columns:
        raise KeyError(f"Missing metric column {metric_column!r} in per_category_df.")

    pivot = per_category_df.pivot(index="category", columns="method", values=metric_column)
    if reference_method not in pivot.columns or comparison_method not in pivot.columns:
        raise KeyError(
            f"Expected both {reference_method!r} and {comparison_method!r} in per_category_df."
        )

    delta = (pivot[comparison_method] - pivot[reference_method]).sort_values(ascending=False)
    colors = [(_method_color(comparison_method) if value >= 0.0 else _method_color(reference_method)) for value in delta]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    ax.barh(delta.index.astype(str), delta.to_numpy(), color=colors)
    ax.axvline(0.0, color="#333333", linewidth=1.0)
    ax.set_title(f"{comparison_method.title()} - {reference_method.title()} {metric_column.upper()} Delta")
    ax.set_xlabel(f"Delta {metric_column.upper()}")
    ax.set_ylabel("Category")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.invert_yaxis()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def select_extreme_cases(
    per_image_df: pd.DataFrame,
    score_column: str,
    *,
    label_value: int,
    ascending: bool,
    top_k: int,
    diversify_categories: bool = True,
) -> pd.DataFrame:
    """Select the most extreme rows for one method and class label."""
    if top_k <= 0 or per_image_df.empty:
        return per_image_df.iloc[0:0].copy()

    filtered = per_image_df.loc[per_image_df["label"].astype(int) == int(label_value)].copy()
    if filtered.empty:
        return filtered

    filtered["_score_rank"] = filtered[score_column].rank(method="first", ascending=ascending)
    filtered = filtered.sort_values([score_column, "sample_position"], ascending=[ascending, True])

    if not diversify_categories:
        return filtered.head(top_k).drop(columns="_score_rank").reset_index(drop=True)

    selected_parts: list[pd.DataFrame] = []
    remaining = filtered.copy()
    selected_count = 0
    while selected_count < top_k and not remaining.empty:
        round_pick = remaining.groupby("category", sort=True, dropna=False).head(1)
        selected_parts.append(round_pick)
        selected_count += len(round_pick)
        remaining = remaining.drop(index=round_pick.index)

    selected = pd.concat(selected_parts, ignore_index=False).head(top_k)
    return selected.drop(columns="_score_rank").reset_index(drop=True)


def select_method_advantage_cases(
    per_image_df: pd.DataFrame,
    primary_score_column: str,
    secondary_score_column: str,
    *,
    label_value: int = 1,
    top_k: int,
) -> pd.DataFrame:
    """Select cases where one method ranks much more confidently than the other."""
    if top_k <= 0 or per_image_df.empty:
        return per_image_df.iloc[0:0].copy()

    filtered = per_image_df.loc[per_image_df["label"].astype(int) == int(label_value)].copy()
    if filtered.empty:
        return filtered

    filtered["_primary_rank"] = filtered[primary_score_column].rank(method="average", pct=True)
    filtered["_secondary_rank"] = filtered[secondary_score_column].rank(method="average", pct=True)
    filtered["_rank_gap"] = filtered["_primary_rank"] - filtered["_secondary_rank"]
    selected = filtered.sort_values(["_rank_gap", "sample_position"], ascending=[False, True])
    selected = selected.groupby("category", sort=True, dropna=False).head(1).sort_values(
        ["_rank_gap", "sample_position"],
        ascending=[False, True],
    )
    if len(selected) < top_k:
        fallback = filtered.drop(index=selected.index, errors="ignore").sort_values(
            ["_rank_gap", "sample_position"],
            ascending=[False, True],
        )
        selected = pd.concat([selected, fallback], ignore_index=False)
    return selected.head(top_k).drop(columns=["_primary_rank", "_secondary_rank", "_rank_gap"]).reset_index(drop=True)


def save_case_gallery(
    case_rows: pd.DataFrame,
    output_path: str | Path,
    *,
    repo_root: str | Path,
    cfg: Mapping[str, Any],
    title: str,
    classical_score_column: str = "classical_image_score",
    autoencoder_score_column: str = "autoencoder_image_score",
) -> None:
    """Save a compact gallery of depth plus standalone method heatmaps."""
    if case_rows.empty:
        return

    repo_root = Path(repo_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = len(case_rows)
    fig, axes = plt.subplots(row_count, 3, figsize=(11, 3.1 * row_count), layout="constrained")
    if row_count == 1:
        axes = np.asarray([axes])
    fig.suptitle(title)

    for row_index, (_, row) in enumerate(case_rows.iterrows()):
        depth_path = _resolve_path(row["processed_depth_path"], repo_root)
        valid_mask_path = _resolve_path(row["processed_mask_path"], repo_root)
        classical_heatmap_path = _resolve_path(row["classical_heatmap_path"], repo_root)
        autoencoder_heatmap_path = _resolve_path(row["autoencoder_heatmap_path"], repo_root)

        depth = np.load(depth_path).astype(np.float32, copy=False)
        valid_mask = np.load(valid_mask_path).astype(bool, copy=False)
        classical_heatmap = np.load(classical_heatmap_path).astype(np.float32, copy=False)
        autoencoder_heatmap = np.load(autoencoder_heatmap_path).astype(np.float32, copy=False)
        ground_truth_mask = _load_processed_ground_truth_mask(row, cfg, repo_root)

        depth_display = normalize_map_for_display(depth, valid_mask)
        classical_display = normalize_map_for_display(classical_heatmap, valid_mask)
        autoencoder_display = normalize_map_for_display(autoencoder_heatmap, valid_mask)

        panels = (
            ("depth + GT", depth_display, "gray", None, None, True, False),
            (
                "classical heatmap",
                classical_display,
                "magma",
                None,
                float(row[classical_score_column]),
                False,
                True,
            ),
            (
                "autoencoder heatmap",
                autoencoder_display,
                "magma",
                None,
                float(row[autoencoder_score_column]),
                False,
                True,
            ),
        )

        for column_index, (
            panel_title,
            panel_image,
            cmap,
            overlay_image,
            score_value,
            draw_ground_truth,
            draw_mask_outline,
        ) in enumerate(panels):
            axis = axes[row_index, column_index]
            axis.imshow(panel_image, cmap=cmap, interpolation="nearest")
            if overlay_image is not None:
                axis.imshow(overlay_image, cmap="magma", alpha=0.55, interpolation="nearest")
            if draw_ground_truth:
                _draw_ground_truth_contour(axis, ground_truth_mask)
            if draw_mask_outline:
                _draw_valid_mask_outline(axis, valid_mask)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(panel_title)
            if score_value is not None:
                axis.set_xlabel(f"score={score_value:.4f}")
            if column_index == 0:
                axis.set_ylabel(
                    f"{row['category']} / {row['defect_type']}\n"
                    f"id={row['sample_id']} label={int(row['label'])}",
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=40,
                )

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
