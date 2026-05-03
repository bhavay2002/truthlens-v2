"""
File: predict_api.py 
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Union

from src.inference.constants import (
    DEFAULT_MAX_LENGTH,
    INFERENCE_CACHE_VERSION,
)
from src.inference.inference_engine import InferenceConfig, InferenceEngine
from src.inference.prediction_service import PredictionService
from src.inference.inference_logger import InferenceLogger
from src.inference.inference_cache import InferenceCache, InferenceCacheConfig
from src.inference.monitoring import InferenceMonitor
from src.inference.postprocessing import Postprocessor
from src.inference.result_formatter import ResultFormatter
from src.utils.input_validation import ensure_non_empty_text
from src.utils.settings import load_settings

# =========================================================
# GLOBAL SINGLETON
# =========================================================

_service: PredictionService | None = None
_lock = threading.Lock()


# =========================================================
# LOAD SERVICE
# =========================================================

def _get_service() -> PredictionService:
    global _service

    if _service is not None:
        return _service

    with _lock:
        if _service is None:

            # CRIT-5: pull the model path (and other inference defaults)
            # from settings rather than hardcoding ``"models"``. The
            # hardcoded literal pointed at a directory that does not
            # exist for any caller that respects ``config/config.yaml``.
            settings = load_settings()
            model_path = str(settings.model.path)
            device = str(getattr(settings.inference, "device", "auto"))
            max_length = int(
                getattr(settings.model, "max_length", DEFAULT_MAX_LENGTH)
            )
            # CFG-2: prefer the operator-supplied cache_version (lets a
            # rollout migrate caches without a code change) and fall back
            # to the package-level constant, NOT a divergent literal.
            cache_version = str(
                getattr(settings.inference, "cache_version",
                        INFERENCE_CACHE_VERSION)
            )

            # ---------------- ENGINE ----------------
            engine = InferenceEngine(
                InferenceConfig(
                    model_path=model_path,
                    device=device,
                    max_length=max_length,
                )
            )

            # ---------------- CACHE ----------------
            cache = InferenceCache(
                InferenceCacheConfig(
                    enable_memory_cache=True,
                    cache_version=cache_version,
                )
            )

            # ---------------- LOGGER ----------------
            logger = InferenceLogger()

            # ---------------- MONITOR ----------------
            monitor = InferenceMonitor()

            # ---------------- POSTPROCESSOR ----------------
            postprocessor = Postprocessor()

            # ---------------- FORMATTER ----------------
            formatter = ResultFormatter()

            # ---------------- SERVICE ----------------
            # UNUSED-FIX: ``InferenceMonitor`` was instantiated and
            # attached to ``_service.monitor`` but never updated. Pass it
            # via the ctor so PredictionService.predict* will call
            # ``monitor.update(...)`` on every request.
            _service = PredictionService(
                engine=engine,
                cache=cache,
                logger_=logger,
                formatter=formatter,
                monitor=monitor,
            )

            # attach optional components
            _service.postprocessor = postprocessor

    return _service


# =========================================================
# INPUT VALIDATION
# =========================================================

def _ensure_list(texts: Union[str, List[str]]) -> List[str]:

    if isinstance(texts, str):
        texts = [texts]

    if not isinstance(texts, list) or not texts:
        raise ValueError("texts must be non-empty list")

    for t in texts:
        ensure_non_empty_text(t)

    return texts


# =========================================================
# 🔥 MAIN API (HUMAN FRIENDLY)
# =========================================================

def predict(text: str) -> Dict[str, Any]:

    service = _get_service()
    return service.predict(text)


# =========================================================
# 🔥 BATCH API
# =========================================================

def predict_batch(texts: List[str]) -> List[Dict[str, Any]]:

    texts = _ensure_list(texts)
    service = _get_service()

    return service.predict_batch(texts)


# =========================================================
# 🔥 FULL PIPELINE (REPORT)
# =========================================================

def predict_full(text: str) -> Dict[str, Any]:

    service = _get_service()
    return service.predict_full(text)


# =========================================================
# 🔥 FORMATTED OUTPUT
# =========================================================

def predict_formatted(
    text: str,
    *,
    mode: str = "api",
) -> Dict[str, Any]:

    service = _get_service()
    return service.predict_formatted(text, mode=mode)


# =========================================================
# 🔥 EVALUATION ENTRYPOINT
# =========================================================

def predict_for_evaluation(texts: List[str]) -> Dict[str, Any]:

    texts = _ensure_list(texts)
    service = _get_service()

    return service.predict_for_evaluation(texts)


# =========================================================
# 🔥 UNCERTAINTY SUPPORT
# =========================================================

def predict_with_uncertainty(texts: List[str]) -> Dict[str, Any]:

    texts = _ensure_list(texts)
    service = _get_service()

    outputs = service.predict_for_evaluation(texts)

    results: Dict[str, Any] = {}

    import numpy as np

    for task, out in outputs.items():

        # CRIT-2: ``_meta`` (and any future scratch keys) sit alongside
        # the per-task entries; they are not classification heads and must
        # not be treated as such.
        if not isinstance(out, dict) or "probabilities" not in out:
            results[task] = out
            continue

        probs = out.get("probabilities")

        if probs is not None:
            probs_arr = np.asarray(probs)
            if probs_arr.ndim < 2:
                entropy = -(probs_arr * np.log(probs_arr + 1e-12)
                            + (1 - probs_arr) * np.log(1 - probs_arr + 1e-12))
            else:
                entropy = -np.sum(probs_arr * np.log(probs_arr + 1e-12), axis=-1)
        else:
            entropy = None

        results[task] = {
            **out,
            "entropy": entropy,
        }

    return results


# =========================================================
# 🔥 MONITORING ENDPOINT
# =========================================================

def get_metrics() -> Dict[str, Any]:

    service = _get_service()

    if hasattr(service, "monitor"):
        return service.monitor.snapshot()

    return {}


# =========================================================
# 🔥 RESET CACHE (OPTIONAL ADMIN)
# =========================================================

def clear_cache():

    service = _get_service()

    if service.cache:
        service.cache.clear()