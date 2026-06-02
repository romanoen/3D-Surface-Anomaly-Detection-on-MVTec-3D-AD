"""Generate diagnostic result figures from the saved benchmark outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.visualization import (
    save_case_gallery,
    save_metric_delta_figure,
    save_overall_roc_pr_figure,
    save_rank_histogram_figure,
    select_extreme_cases,
    select_method_advantage_cases,
)
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate diagnostic figures from the shared benchmark outputs."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help="Path to the base config file. Defaults to configs/base.yaml.",
    )
    parser.add_argument(
        "--classical-config",
        type=Path,
        default=None,
        help="Path to the classical config file. Defaults to configs/classical.yaml.",
    )
    parser.add_argument(
        "--per-image",
        type=Path,
        default=None,
        help="Path to the merged benchmark per-image CSV. Defaults to outputs/metrics/per_image.csv.",
    )
    parser.add_argument(
        "--per-category",
        type=Path,
        default=None,
        help="Path to the benchmark per-category CSV. Defaults to outputs/metrics/per_category.csv.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory for the generated analysis figures. Defaults to fig/06_results/analysis.",
    )
    parser.add_argument(
        "--case-budget",
        type=int,
        default=6,
        help="How many examples to show in each qualitative gallery.",
    )
    return parser.parse_args()


def resolve_repo_root() -> Path:
    """Return the repository root based on this script location."""
    return Path(__file__).resolve().parents[1]


def main() -> None:
    """Generate analysis figures from an existing benchmark run."""
    args = parse_args()
    repo_root = resolve_repo_root()

    base_config_path = args.base_config or repo_root / "configs" / "base.yaml"
    classical_config_path = args.classical_config or repo_root / "configs" / "classical.yaml"
    cfg = load_config(base_config_path, classical_config_path)

    metrics_root = repo_root / cfg["paths"]["outputs_root"] / "metrics"
    per_image_path = args.per_image or metrics_root / "per_image.csv"
    per_category_path = args.per_category or metrics_root / "per_category.csv"
    output_root = args.output_root or (repo_root / cfg["paths"]["fig_root"] / "06_results" / "analysis")

    per_image_df = pd.read_csv(per_image_path, keep_default_na=False)
    per_category_df = pd.read_csv(per_category_path, keep_default_na=False)
    score_columns = {
        "classical": "classical_image_score",
        "autoencoder": "autoencoder_image_score",
    }

    output_root.mkdir(parents=True, exist_ok=True)

    save_overall_roc_pr_figure(
        per_image_df,
        score_columns,
        output_root / "overall_roc_pr.png",
    )
    save_rank_histogram_figure(
        per_image_df,
        score_columns,
        output_root / "score_rank_histograms.png",
    )
    save_metric_delta_figure(
        per_category_df,
        "auroc",
        output_root / "auroc_delta_by_category.png",
    )
    save_metric_delta_figure(
        per_category_df,
        "ap",
        output_root / "ap_delta_by_category.png",
    )

    case_budget = max(1, int(args.case_budget))
    case_specs = (
        (
            "classical_false_positives.png",
            "Classical False Positives",
            select_extreme_cases(
                per_image_df,
                "classical_image_score",
                label_value=0,
                ascending=False,
                top_k=case_budget,
            ),
        ),
        (
            "classical_false_negatives.png",
            "Classical False Negatives",
            select_extreme_cases(
                per_image_df,
                "classical_image_score",
                label_value=1,
                ascending=True,
                top_k=case_budget,
            ),
        ),
        (
            "autoencoder_false_positives.png",
            "Autoencoder False Positives",
            select_extreme_cases(
                per_image_df,
                "autoencoder_image_score",
                label_value=0,
                ascending=False,
                top_k=case_budget,
            ),
        ),
        (
            "autoencoder_false_negatives.png",
            "Autoencoder False Negatives",
            select_extreme_cases(
                per_image_df,
                "autoencoder_image_score",
                label_value=1,
                ascending=True,
                top_k=case_budget,
            ),
        ),
        (
            "autoencoder_advantage_cases.png",
            "Cases Where Autoencoder Helps Most",
            select_method_advantage_cases(
                per_image_df,
                "autoencoder_image_score",
                "classical_image_score",
                top_k=case_budget,
            ),
        ),
        (
            "classical_advantage_cases.png",
            "Cases Where Classical Helps Most",
            select_method_advantage_cases(
                per_image_df,
                "classical_image_score",
                "autoencoder_image_score",
                top_k=case_budget,
            ),
        ),
    )

    for filename, title, case_rows in case_specs:
        save_case_gallery(
            case_rows,
            output_root / filename,
            repo_root=repo_root,
            cfg=cfg,
            title=title,
        )

    print(f"Saved analysis figures under {output_root.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
