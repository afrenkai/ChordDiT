from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch


def select_device(requested: str | torch.device | None = None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_fallback_device(device: torch.device) -> torch.device:
    if device.type != "mps":
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
