"""Single source of truth for runtime feature-pipeline configuration.

Audit fix §9 — settings that drove the feature pipeline used to live in
three places: per-module module-level constants
(``emotion_intensity_features.MODEL_NAME``, the spaCy model literal in
``spacy_doc.py`` / ``syntactic_features.py``), the mutable
``runtime_config`` flags (transformer enable / batch / chunk / stride),
and ad-hoc env vars consumed by ``CacheManager`` defaults. Drift
between those three sources caused stale tests, unexpected GPU loads,
and confusing operator behaviour.

This module gathers them into a single ``FeatureConfig`` dataclass.
All fields default from the same env vars the legacy code used, so
``FeatureConfig()`` reproduces today's behaviour byte-for-byte. New
code that wants to tweak the pipeline should:

    cfg = FeatureConfig(transformer_enabled=False)
    cfg.apply_to_runtime()
    bootstrap_feature_registry(config=cfg)

instead of mutating ``runtime_config`` directly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from src.features import runtime_config

logger = logging.getLogger(__name__)


# =========================================================
# DEFAULT HELPERS  (kept private so callers go through fields)
# =========================================================


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw not in ("0", "false", "False", "FALSE", "")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int env %s=%r — using default %d", name, raw, default)
        return default


# =========================================================
# CONFIG DATACLASS
# =========================================================


@dataclass
class FeatureConfig:
    """Runtime configuration for the feature pipeline.

    Every field has an env-var-driven default that matches the legacy
    behaviour. ``apply_to_runtime()`` pushes the values into the
    mutable :mod:`src.features.runtime_config` registry so existing
    consumers (the emotion extractor, the analysis adapters, the
    cache manager defaults, etc.) pick them up.
    """

    # -------------------------
    # Transformer
    # -------------------------
    emotion_model: str = (
        os.environ.get(
            "TRUTHLENS_EMOTION_MODEL",
            "j-hartmann/emotion-english-distilroberta-base",
        )
    )
    transformer_enabled: bool = (
        os.environ.get("TRUTHLENS_DISABLE_TRANSFORMER", "0")
        not in ("1", "true", "TRUE")
    )
    transformer_max_batch: int = _env_int("TRUTHLENS_TRANSFORMER_MAX_BATCH", 64)
    transformer_chunk_length: int = _env_int("TRUTHLENS_TRANSFORMER_CHUNK", 256)
    transformer_chunk_stride: int = _env_int("TRUTHLENS_TRANSFORMER_STRIDE", 64)

    # -------------------------
    # spaCy
    # -------------------------
    spacy_model: str = os.environ.get("TRUTHLENS_SPACY_MODEL", "en_core_web_sm")

    # -------------------------
    # Cache
    # -------------------------
    cache_max_memory_items: int = _env_int("TRUTHLENS_CACHE_MAX_ITEMS", 10_000)
    cache_max_memory_bytes: Optional[int] = _env_int(
        "TRUTHLENS_CACHE_MAX_BYTES", 512 * 1024 * 1024
    )
    feature_context_cache_size: int = _env_int(
        "TRUTHLENS_CONTEXT_CACHE_SIZE", 256
    )

    # -------------------------
    # Analysis adapters
    # -------------------------
    analysis_adapters_strict: bool = _env_bool(
        "TRUTHLENS_ANALYSIS_ADAPTERS_STRICT", False
    )

    # -------------------------
    # Torch CPU thread cap (audit fix §6.2)
    # -------------------------
    torch_thread_cap: int = _env_int("TRUTHLENS_TORCH_THREADS", 4)

    # =====================================================
    # APPLY
    # =====================================================

    def apply_to_runtime(self) -> None:
        """Push every field into :mod:`runtime_config`.

        Idempotent — safe to call repeatedly (e.g. once per bootstrap
        and again from a test fixture).
        """
        runtime_config.configure(
            transformer_enabled=self.transformer_enabled,
            max_batch=self.transformer_max_batch,
            chunk_length=self.transformer_chunk_length,
            chunk_stride=self.transformer_chunk_stride,
            analysis_adapters_strict=self.analysis_adapters_strict,
        )
        # Make the spaCy + emotion model names visible to the lazy
        # loaders that read os.environ at first call. Setting the
        # env var is intentional — the lazy loaders are scattered
        # across modules and reading from env keeps the diff small.
        os.environ["TRUTHLENS_EMOTION_MODEL"] = self.emotion_model
        os.environ["TRUTHLENS_SPACY_MODEL"] = self.spacy_model
        logger.info(
            "FeatureConfig applied | emotion=%s spacy=%s transformer=%s "
            "max_batch=%d chunk=%d stride=%d strict_adapters=%s "
            "ctx_cache=%d",
            self.emotion_model,
            self.spacy_model,
            self.transformer_enabled,
            self.transformer_max_batch,
            self.transformer_chunk_length,
            self.transformer_chunk_stride,
            self.analysis_adapters_strict,
            self.feature_context_cache_size,
        )
