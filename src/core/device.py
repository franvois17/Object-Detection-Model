"""Cross-platform device detection: CUDA / MPS / CPU."""

from __future__ import annotations

import os
import torch


def get_device() -> torch.device:
    """Return the best available torch device.

    Priority: CUDA > MPS > CPU.
    Sets MPS fallback env var automatically.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")

    return torch.device("cpu")


def device_label(device: torch.device) -> str:
    """Human-readable label for the device."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        return f"CUDA ({name})"
    if device.type == "mps":
        return "Apple MPS"
    return "CPU"


DEVICE = get_device()
DEVICE_LABEL = device_label(DEVICE)
