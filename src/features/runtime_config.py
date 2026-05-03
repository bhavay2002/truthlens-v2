"""Mutable runtime feature-flag registry.

Audit fix §6.6 + §9.1 — the previous code path set
``TRANSFORMER_AVAILABLE`` as a module-level boolean inside the
``emotion_intensity_features`` import-time ``try/except``. There was no
way to monkey-patch the flag from a test without reaching into the
module, and the config-side ``enable_emotion_transformer`` flag was
silently ignored at runtime.

This module owns the runtime flags so:

* tests can flip the flag via ``set_transformer_enabled(False)`` and
  exercise the lexicon-only path deterministically;
* the extractor reads the flag at *extract time* (not import time), so
  changing the config and re-running a request takes effect without a
  process restart.

Defaults are conservative (everything enabled) — the extractor's own
``try/except`` import guard still owns the question of whether the
underlying library is *physically* available.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

_lock = threading.RLock()

# Whether the HF emotion classifier should run. The extractor still
# checks the actual import outcome before attempting inference; this
# flag only gives operators a way to *force* the lexicon-only path
# without touching the model file.
_transformer_enabled: bool = (
    os.environ.get("TRUTHLENS_DISABLE_TRANSFORMER", "0") not in ("1", "true", "TRUE")
)

# Soft cap for the per-batch transformer forward pass. See audit §6.5.
_max_transformer_batch: int = int(
    os.environ.get("TRUTHLENS_TRANSFORMER_MAX_BATCH", "64") or 64
)

# Maximum effective sequence length per chunk and the chunk stride for
# overflow handling (audit §5.1). 256 + 64 keeps activation memory at
# half of the 512 default while still covering the document via
# overflow-window averaging.
_transformer_chunk_length: int = int(
    os.environ.get("TRUTHLENS_TRANSFORMER_CHUNK", "256") or 256
)
_transformer_chunk_stride: int = int(
    os.environ.get("TRUTHLENS_TRANSFORMER_STRIDE", "64") or 64
)

# Audit fix §8 — when ``True``, the analysis adapter raises on a
# silent ``importlib`` failure instead of swallowing it. Useful for
# CI where a renamed analyser module should fail the build, not
# silently disable the feature.
_analysis_adapters_strict: bool = (
    os.environ.get("TRUTHLENS_ANALYSIS_ADAPTERS_STRICT", "0") in ("1", "true", "TRUE")
)


def transformer_enabled() -> bool:
    with _lock:
        return _transformer_enabled


def set_transformer_enabled(value: bool) -> None:
    """Test hook — flip the runtime flag without touching the import."""
    global _transformer_enabled
    with _lock:
        _transformer_enabled = bool(value)


def max_transformer_batch() -> int:
    with _lock:
        return max(1, _max_transformer_batch)


def set_max_transformer_batch(value: int) -> None:
    global _max_transformer_batch
    with _lock:
        _max_transformer_batch = max(1, int(value))


def transformer_chunk_length() -> int:
    with _lock:
        return max(32, _transformer_chunk_length)


def transformer_chunk_stride() -> int:
    with _lock:
        return max(0, min(_transformer_chunk_stride, _transformer_chunk_length - 1))


def analysis_adapters_strict() -> bool:
    with _lock:
        return _analysis_adapters_strict


def set_analysis_adapters_strict(value: bool) -> None:
    """Test / operator hook for the §8 strict-load flag."""
    global _analysis_adapters_strict
    with _lock:
        _analysis_adapters_strict = bool(value)


def configure(
    *,
    transformer_enabled: Optional[bool] = None,
    max_batch: Optional[int] = None,
    chunk_length: Optional[int] = None,
    chunk_stride: Optional[int] = None,
    analysis_adapters_strict: Optional[bool] = None,
) -> None:
    """Bulk-update — convenience for the bootstrap path."""
    global _transformer_enabled, _max_transformer_batch
    global _transformer_chunk_length, _transformer_chunk_stride
    global _analysis_adapters_strict
    with _lock:
        if transformer_enabled is not None:
            _transformer_enabled = bool(transformer_enabled)
        if max_batch is not None:
            _max_transformer_batch = max(1, int(max_batch))
        if chunk_length is not None:
            _transformer_chunk_length = max(32, int(chunk_length))
        if chunk_stride is not None:
            _transformer_chunk_stride = max(0, int(chunk_stride))
        if analysis_adapters_strict is not None:
            _analysis_adapters_strict = bool(analysis_adapters_strict)
