"""Inference-layer constants — single source of truth for values that
were previously duplicated across modules (CFG-2, CFG-5, CFG-6).

Keeping these here lets every entry point (engine, cache, predict_api,
batch_inference, report_generator) agree on the same defaults without
re-declaring magic strings/numbers per module. Values that should be
configurable per deployment still live in ``config/config.yaml`` and
are surfaced via ``src.utils.settings``; the constants below are
defaults / contract versions that change only on a code release.
"""

from __future__ import annotations


INFERENCE_CACHE_VERSION: str = "v2"
"""Bump on any change to the cached prediction blob's schema. Read by
both ``InferenceCache`` (default) and ``predict_api`` so the two cannot
drift across entry points."""

DEFAULT_INFERENCE_BATCH_SIZE: int = 32
"""Default batch size used by ``InferenceEngine`` /
``BatchInferenceEngine`` / ``InferenceConfigLoader`` when neither the
caller nor ``settings.inference.batch_size`` provides one."""

DEFAULT_MAX_LENGTH: int = 512
"""Default tokenizer max-length when not supplied by settings."""

REPORT_VERSION: str = "v3"
"""Bump when the ``ReportGenerator`` output schema changes."""
