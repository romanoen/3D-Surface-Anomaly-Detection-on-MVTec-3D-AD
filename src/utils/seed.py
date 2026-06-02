"""Random seed utilities for reproducible experiments."""

from __future__ import annotations

import random

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch is optional at import time
    torch = None


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and optionally PyTorch for repeatable runs."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
