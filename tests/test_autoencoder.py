"""Tests for the compact convolutional autoencoder model helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.models.autoencoder import ConvAutoencoder, load_autoencoder_checkpoint, save_autoencoder_checkpoint


class AutoencoderModelTestCase(unittest.TestCase):
    """Verify the autoencoder model shape and checkpoint behavior."""

    def test_forward_preserves_input_shape(self) -> None:
        """The autoencoder should reconstruct tensors with the same shape."""
        model = ConvAutoencoder(in_channels=1, encoder_channels=(4, 8), latent_channels=8)
        inputs = torch.rand(3, 1, 8, 8, dtype=torch.float32)

        outputs = model(inputs)

        self.assertEqual(tuple(outputs.shape), (3, 1, 8, 8))

    def test_checkpoint_roundtrip_restores_config_and_weights(self) -> None:
        """A saved checkpoint should rebuild the same architecture."""
        model = ConvAutoencoder(in_channels=1, encoder_channels=(4, 8), latent_channels=8)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model.pt"
            save_autoencoder_checkpoint(model, checkpoint_path, metadata={"note": "demo"})

            loaded_model, payload = load_autoencoder_checkpoint(checkpoint_path, map_location="cpu")

        self.assertIsInstance(loaded_model, ConvAutoencoder)
        self.assertEqual(loaded_model.config_dict()["encoder_channels"], [4, 8])
        self.assertEqual(payload["metadata"]["note"], "demo")


if __name__ == "__main__":
    unittest.main()
