"""Centralised device detection for ``src/models/`` (audit 5.1).

A single source of truth so CUDA / MPS / CPU selection cannot drift between
``EncoderFactory``, ``model_wrapper``, ``benchmark_runner``, ``utils``.
Apple Silicon contributors used to silently land on CPU because the
ad-hoc snippets in those four files only checked ``torch.cuda.is_available()``.
"""

from __future__ import annotations

from typing import Optional

import torch


def detect_device(prefer: Optional[str] = None) -> torch.device:
    """Return the best available device.

    Parameters
    ----------
    prefer
        If supplied, parsed via ``torch.device(prefer)`` and returned
        verbatim — caller-specified placement always wins. ``None`` (the
        default) triggers auto-detection: CUDA → MPS → CPU.
    """

    if prefer:
        return torch.device(prefer)

    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and getattr(mps_backend, "is_available", lambda: False)():
        return torch.device("mps")

    return torch.device("cpu")


__all__ = ["detect_device"]
