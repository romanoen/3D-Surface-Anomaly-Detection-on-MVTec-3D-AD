"""Training utilities for the autoencoder baseline."""

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

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loaders import PatchDataset, build_patch_dataset
from src.models.autoencoder import (
    build_autoencoder,
    load_autoencoder_checkpoint,
    save_autoencoder_checkpoint,
)
from src.utils.seed import set_global_seed


@dataclass(frozen=True)
class AutoencoderCategoryTrainingResult:
    """Paths and summary values produced by one category-specific autoencoder."""

    category: str
    checkpoint_path: Path
    history_path: Path
    figure_root: Path
    train_patch_count: int
    val_patch_count: int
    best_epoch: int
    best_val_loss: float
    epochs_completed: int


@dataclass(frozen=True)
class AutoencoderTrainingResult:
    """Paths and summary values produced by autoencoder training."""

    model_root: Path
    summary_path: Path
    category_summary_path: Path
    category_results: tuple[AutoencoderCategoryTrainingResult, ...]
    train_patch_count: int
    val_patch_count: int


class _TorchPatchDataset(Dataset[dict[str, Any]]):
    """Torch-friendly wrapper around the lazy patch dataset."""

    def __init__(self, patch_dataset: PatchDataset) -> None:
        self.patch_dataset = patch_dataset

    def __len__(self) -> int:
        """Return the number of retained patches."""
        return len(self.patch_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Convert one depth patch sample into tensors plus light metadata."""
        sample = self.patch_dataset[index]
        patch = torch.from_numpy(sample["patch"][None, ...].astype(np.float32, copy=False))
        mask = torch.from_numpy(sample["mask"][None, ...].astype(np.float32, copy=False))
        return {
            "patch": patch,
            "mask": mask,
            "sample_id": str(sample["sample_id"]),
            "category": str(sample["category"]),
            "patch_position": int(sample["patch_position"]),
        }


def _sample_patch_indices(
    patch_count: int,
    max_patches: int | None,
    seed: int,
) -> np.ndarray:
    """Return deterministic patch indices, optionally sampled without replacement."""
    if max_patches is None or int(max_patches) <= 0 or patch_count <= int(max_patches):
        return np.arange(patch_count, dtype=np.int64)

    rng = np.random.default_rng(seed)
    indices = rng.choice(patch_count, size=int(max_patches), replace=False)
    return np.sort(indices.astype(np.int64))


def _select_device(training_cfg: dict[str, Any]) -> torch.device:
    """Return the requested torch device."""
    requested = str(training_cfg.get("device", "auto")).lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


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


def masked_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_name: str = "mse",
) -> torch.Tensor:
    """Compute a mask-aware reconstruction loss over valid object pixels."""
    if reconstruction.shape != target.shape or target.shape != mask.shape:
        raise ValueError(
            "reconstruction, target, and mask must share the same shape, "
            f"got {tuple(reconstruction.shape)}, {tuple(target.shape)}, and {tuple(mask.shape)}"
        )

    loss_name = str(loss_name).lower()
    if loss_name == "mse":
        residual = (reconstruction - target) ** 2
    elif loss_name == "mae":
        residual = torch.abs(reconstruction - target)
    else:
        raise ValueError(f"Unsupported autoencoder loss: {loss_name}")

    weighted = residual * mask
    denominator = mask.sum().clamp_min(1.0)
    return weighted.sum() / denominator


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    loss_name: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Run one training or validation epoch and return the mean loss."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        patches = batch["patch"].to(device)
        masks = batch["mask"].to(device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            reconstruction = model(patches)
            loss = masked_reconstruction_loss(reconstruction, patches, masks, loss_name=loss_name)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        batch_size = int(patches.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def _save_loss_curve(history_df: pd.DataFrame, output_path: Path) -> None:
    """Save the train/validation reconstruction-loss curve."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="train")
    ax.plot(history_df["epoch"], history_df["val_loss"], marker="o", label="val")
    ax.set_title("Autoencoder reconstruction loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked loss")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_reconstruction_examples(
    model: nn.Module,
    dataset: _TorchPatchDataset,
    selected_indices: np.ndarray,
    device: torch.device,
    output_path: Path,
    max_examples: int,
) -> None:
    """Save compact input/reconstruction/residual examples from validation patches."""
    if max_examples <= 0 or selected_indices.size == 0:
        return

    chosen_indices = selected_indices[: max_examples]
    rows = len(chosen_indices)
    fig, axes = plt.subplots(rows, 3, figsize=(9, max(3, rows * 2.6)), layout="constrained")
    axes = np.atleast_2d(axes)

    model.eval()
    with torch.no_grad():
        for row_index, patch_index in enumerate(chosen_indices):
            sample = dataset[int(patch_index)]
            patch = sample["patch"].unsqueeze(0).to(device)
            mask = sample["mask"].unsqueeze(0).to(device)
            reconstruction = model(patch)
            residual = torch.abs(reconstruction - patch) * mask

            patch_np = patch.squeeze(0).squeeze(0).cpu().numpy()
            recon_np = reconstruction.squeeze(0).squeeze(0).cpu().numpy()
            residual_np = residual.squeeze(0).squeeze(0).cpu().numpy()
            mask_np = mask.squeeze(0).squeeze(0).cpu().numpy().astype(bool)

            axes[row_index, 0].imshow(np.ma.array(patch_np, mask=~mask_np), cmap="gray", vmin=0.0, vmax=1.0)
            axes[row_index, 0].set_title("input")
            axes[row_index, 1].imshow(np.ma.array(recon_np, mask=~mask_np), cmap="gray", vmin=0.0, vmax=1.0)
            axes[row_index, 1].set_title("reconstruction")
            axes[row_index, 2].imshow(np.ma.array(residual_np, mask=~mask_np), cmap="magma", interpolation="nearest")
            axes[row_index, 2].set_title("residual")
            axes[row_index, 0].set_ylabel(
                f"{sample['category']} / {sample['sample_id']}\npatch {sample['patch_position']}"
            )

            for axis in axes[row_index]:
                axis.set_xticks([])
                axis.set_yticks([])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _train_one_category(
    category: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict[str, Any],
    repo_root: Path,
    device: torch.device,
    seed: int,
) -> tuple[AutoencoderCategoryTrainingResult, dict[str, Any]]:
    """Train and persist one category-specific autoencoder."""
    set_global_seed(seed)
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    paths_cfg = cfg["paths"]
    category_path = _safe_path_part(category)

    train_patch_dataset = build_patch_dataset(train_df, cfg, repo_root=repo_root, normal_only=True)
    val_patch_dataset = build_patch_dataset(val_df, cfg, repo_root=repo_root, normal_only=True)
    if len(train_patch_dataset) == 0:
        raise ValueError(f"No normal training patches were available for category {category!r}.")
    if len(val_patch_dataset) == 0:
        raise ValueError(f"No normal validation patches were available for category {category!r}.")

    train_torch_dataset = _TorchPatchDataset(train_patch_dataset)
    val_torch_dataset = _TorchPatchDataset(val_patch_dataset)

    train_indices = _sample_patch_indices(
        len(train_torch_dataset),
        training_cfg.get("max_train_patches"),
        seed,
    )
    val_indices = _sample_patch_indices(
        len(val_torch_dataset),
        training_cfg.get("max_val_patches"),
        seed + 1,
    )

    train_dataset: Dataset[dict[str, Any]] = Subset(train_torch_dataset, train_indices.tolist())
    val_dataset: Dataset[dict[str, Any]] = Subset(val_torch_dataset, val_indices.tolist())

    batch_size = int(training_cfg.get("batch_size", 64))
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    num_workers = int(training_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_autoencoder(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg.get("learning_rate", 1e-3)))
    loss_name = str(training_cfg.get("loss", "mse")).lower()
    epochs = int(training_cfg.get("epochs", 50))
    patience = int(training_cfg.get("early_stopping_patience", 8))
    min_delta = float(training_cfg.get("early_stopping_min_delta", 0.0))
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if patience <= 0:
        raise ValueError(f"early_stopping_patience must be positive, got {patience}")

    checkpoint_name = str(training_cfg.get("checkpoint_name", "conv_autoencoder.pt")).strip() or "conv_autoencoder.pt"
    checkpoint_path = (
        repo_root
        / paths_cfg["outputs_root"]
        / "models"
        / "autoencoder"
        / category_path
        / checkpoint_name
    )
    history_path = (
        repo_root
        / paths_cfg["outputs_root"]
        / "metrics"
        / "autoencoder_histories"
        / f"{category_path}.csv"
    )
    figure_root = repo_root / paths_cfg["fig_root"] / "05_autoencoder" / "training" / category_path

    history_rows: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    progress = tqdm(range(1, epochs + 1), desc=f"Training autoencoder [{category}]")
    for epoch in progress:
        train_loss = _run_epoch(model, train_loader, device, loss_name, optimizer=optimizer)
        val_loss = _run_epoch(model, val_loader, device, loss_name, optimizer=None)
        history_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        progress.set_postfix(train_loss=f"{train_loss:.5f}", val_loss=f"{val_loss:.5f}")

        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_autoencoder_checkpoint(
                model,
                checkpoint_path,
                metadata={
                    "category": category,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "training_loss": train_loss,
                    "device": str(device),
                },
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    history_df = pd.DataFrame(history_rows)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(history_path, index=False)
    _save_loss_curve(history_df, figure_root / "loss_curve.png")
    best_model, _ = load_autoencoder_checkpoint(checkpoint_path, map_location=device)
    best_model = best_model.to(device)
    _save_reconstruction_examples(
        best_model,
        val_torch_dataset,
        val_indices,
        device,
        figure_root / "reconstruction_examples.png",
        max_examples=int(training_cfg.get("reconstruction_examples", 6)),
    )

    result = AutoencoderCategoryTrainingResult(
        category=category,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        figure_root=figure_root,
        train_patch_count=int(len(train_indices)),
        val_patch_count=int(len(val_indices)),
        best_epoch=int(best_epoch),
        best_val_loss=float(best_val_loss),
        epochs_completed=int(len(history_df)),
    )
    summary = {
        "category": category,
        "model": str(model_cfg.get("name", "conv_autoencoder")),
        "train_patch_count": result.train_patch_count,
        "val_patch_count": result.val_patch_count,
        "epochs_completed": result.epochs_completed,
        "best_epoch": result.best_epoch,
        "best_val_loss": result.best_val_loss,
        "device": str(device),
        "checkpoint_path": checkpoint_path.relative_to(repo_root).as_posix(),
        "history_path": history_path.relative_to(repo_root).as_posix(),
        "figure_root": figure_root.relative_to(repo_root).as_posix(),
        "model_config": model.config_dict(),
    }
    return result, summary


def train_autoencoder_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: dict[str, Any],
    repo_root: str | Path = ".",
) -> AutoencoderTrainingResult:
    """Train one autoencoder baseline per object category."""
    repo_root = Path(repo_root)
    seed = int(cfg.get("project", {}).get("seed", 42))
    training_cfg = cfg["training"]
    paths_cfg = cfg["paths"]
    device = _select_device(training_cfg)

    if not bool(training_cfg.get("per_category", True)):
        raise ValueError(
            "Autoencoder training must use per_category=True to match the shared implementation plan."
        )

    categories = sorted(
        str(category)
        for category in train_df.loc[train_df["label"].astype(int) == 0, "category"].unique()
    )
    if not categories:
        raise ValueError("No normal training categories found.")

    category_results: list[AutoencoderCategoryTrainingResult] = []
    category_summaries: list[dict[str, Any]] = []

    for category_index, category in enumerate(categories):
        train_category_df = _normal_rows_for_category(train_df, category)
        val_category_df = _normal_rows_for_category(val_df, category)
        if val_category_df.empty:
            raise ValueError(f"No normal validation rows found for category {category!r}.")

        result, summary = _train_one_category(
            category,
            train_category_df,
            val_category_df,
            cfg,
            repo_root,
            device,
            seed + category_index * 1009,
        )
        category_results.append(result)
        category_summaries.append(summary)

    model_root = repo_root / paths_cfg["outputs_root"] / "models" / "autoencoder"
    metrics_root = repo_root / paths_cfg["outputs_root"] / "metrics"
    logs_root = repo_root / paths_cfg["outputs_root"] / "logs"
    summary_path = logs_root / "autoencoder_training_summary.json"
    category_summary_path = metrics_root / "autoencoder_per_category.csv"

    metrics_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(category_summaries).to_csv(category_summary_path, index=False)

    summary = {
        "model": str(cfg["model"].get("name", "conv_autoencoder")),
        "training_scope": "per_category",
        "category_count": len(category_results),
        "train_patch_count": int(sum(result.train_patch_count for result in category_results)),
        "val_patch_count": int(sum(result.val_patch_count for result in category_results)),
        "device": str(device),
        "model_root": model_root.relative_to(repo_root).as_posix(),
        "category_summary_path": category_summary_path.relative_to(repo_root).as_posix(),
        "categories": category_summaries,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return AutoencoderTrainingResult(
        model_root=model_root,
        summary_path=summary_path,
        category_summary_path=category_summary_path,
        category_results=tuple(category_results),
        train_patch_count=summary["train_patch_count"],
        val_patch_count=summary["val_patch_count"],
    )
