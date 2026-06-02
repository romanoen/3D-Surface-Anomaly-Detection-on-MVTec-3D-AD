"""Compact convolutional autoencoder for reconstruction-based anomaly detection."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _normalize_encoder_channels(value: Sequence[int]) -> tuple[int, ...]:
    """Validate and normalize the encoder channel configuration."""
    channels = tuple(int(channel) for channel in value)
    if not channels:
        raise ValueError("encoder_channels must contain at least one block width.")
    if any(channel <= 0 for channel in channels):
        raise ValueError(f"encoder_channels must be positive, got {channels}")
    return channels


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Build a compact convolutional feature block."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class ConvAutoencoder(nn.Module):
    """Compact convolutional autoencoder for normalized depth patches."""

    def __init__(
        self,
        in_channels: int = 1,
        encoder_channels: Sequence[int] = (16, 32, 64),
        latent_channels: int = 64,
    ) -> None:
        super().__init__()
        if int(in_channels) <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if int(latent_channels) <= 0:
            raise ValueError(f"latent_channels must be positive, got {latent_channels}")

        self.in_channels = int(in_channels)
        self.encoder_channels = _normalize_encoder_channels(encoder_channels)
        self.latent_channels = int(latent_channels)

        encoder_blocks: list[nn.Module] = []
        current_channels = self.in_channels
        for output_channels in self.encoder_channels:
            encoder_blocks.append(_conv_block(current_channels, output_channels))
            current_channels = output_channels
        self.encoder_blocks = nn.ModuleList(encoder_blocks)
        self.pools = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2) for _ in self.encoder_channels])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(current_channels, self.latent_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.latent_channels, current_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.upsamplers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for output_channels in reversed(self.encoder_channels):
            self.upsamplers.append(
                nn.ConvTranspose2d(current_channels, output_channels, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(_conv_block(output_channels, output_channels))
            current_channels = output_channels

        self.output_layer = nn.Conv2d(current_channels, self.in_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Reconstruct a batch of normalized depth patches."""
        if inputs.ndim != 4:
            raise ValueError(f"Expected inputs with shape (N, C, H, W), got {tuple(inputs.shape)}")
        if int(inputs.shape[1]) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {int(inputs.shape[1])}"
            )

        spatial_shape = tuple(int(value) for value in inputs.shape[-2:])
        x = inputs
        for block, pool in zip(self.encoder_blocks, self.pools, strict=True):
            x = block(x)
            if x.shape[-2] < 2 or x.shape[-1] < 2:
                raise ValueError(
                    "Patch size became too small for the configured encoder depth, "
                    f"got intermediate shape {tuple(x.shape)}"
                )
            x = pool(x)

        x = self.bottleneck(x)
        for upsampler, block in zip(self.upsamplers, self.decoder_blocks, strict=True):
            x = upsampler(x)
            x = block(x)

        if tuple(x.shape[-2:]) != spatial_shape:
            x = F.interpolate(x, size=spatial_shape, mode="bilinear", align_corners=False)
        return self.output_layer(x)

    def config_dict(self) -> dict[str, Any]:
        """Return the architecture configuration needed to rebuild the module."""
        return {
            "name": "conv_autoencoder",
            "in_channels": self.in_channels,
            "encoder_channels": list(self.encoder_channels),
            "latent_channels": self.latent_channels,
        }


def build_autoencoder(model_cfg: dict[str, Any]) -> ConvAutoencoder:
    """Build the configured convolutional autoencoder."""
    return ConvAutoencoder(
        in_channels=int(model_cfg.get("in_channels", 1)),
        encoder_channels=model_cfg.get("encoder_channels", (16, 32, 64)),
        latent_channels=int(model_cfg.get("latent_channels", 64)),
    )


def save_autoencoder_checkpoint(
    model: ConvAutoencoder,
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a state-dict checkpoint plus the architecture configuration."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_config": model.config_dict(),
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, output_path)


def load_autoencoder_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device | None = None,
) -> tuple[ConvAutoencoder, dict[str, Any]]:
    """Load a saved autoencoder checkpoint and rebuild the model."""
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a checkpoint dictionary, got {type(payload).__name__}")
    if "model_config" not in payload or "state_dict" not in payload:
        raise ValueError(f"Checkpoint {checkpoint_path} is missing required keys.")

    model = build_autoencoder(payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    return model, payload
