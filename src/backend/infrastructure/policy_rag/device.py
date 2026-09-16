"""Device selection for local policy RAG models."""

from __future__ import annotations

import os

import torch


def resolve_policy_model_device(requested: str | None = None) -> str:
    """Prefer CUDA when available while keeping CPU fallback explicit."""
    if requested is None:
        requested = os.environ.get("MEDIGUARD_POLICY_RAG_DEVICE")
    if requested:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device {requested!r}, but CUDA is unavailable in the "
                f"current PyTorch environment ({torch.__version__})."
            )
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"
