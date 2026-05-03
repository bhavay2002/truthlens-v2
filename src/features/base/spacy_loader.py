"""Process-wide shared spaCy model loader (delegates to analysis cache).

Audit fix #1.7 — every ``spacy.load("en_core_web_sm")`` site under
``src/features/`` was holding its *own* ~50 MB pipeline copy because
this module had a private cache that was disjoint from
``src/analysis/spacy_loader``'s cache. Even with both sides asking for
the same model, the process ended up with two resident pipelines, each
warming its own NER/parser independently on first call.

This module now thinly wraps ``src.analysis.spacy_loader.get_nlp`` so
both code paths share **one** cache keyed on ``(model_name, ())`` (the
analysis side's "safe" mode also uses an empty disable tuple). The
public surface stays the same — callers still get a single shared
``Language`` (or ``None`` if the model isn't installed and they should
fall back to a regex/blank pipeline).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Lock + per-model-name "we already failed once, don't retry" set.
# The actual loaded ``Language`` instances live in
# ``src.analysis.spacy_loader._CACHE`` so memory is shared.
_lock = threading.Lock()
_failed: Dict[str, bool] = {}


def get_shared_nlp(model_name: str = "en_core_web_sm") -> Optional[Any]:
    """Return a shared spaCy ``Language`` for ``model_name``, or ``None``.

    Delegates to ``src.analysis.spacy_loader.get_nlp`` so the analysis
    and features layers share a single cache. The first failure for a
    given model name is remembered locally so we don't keep hammering
    the analysis loader (which raises ``RuntimeError`` on missing
    models — features callers want a soft ``None`` instead).
    """
    if _failed.get(model_name):
        return None

    try:
        from src.analysis.spacy_loader import get_nlp
    except Exception as exc:
        with _lock:
            if not _failed.get(model_name):
                logger.warning("Shared spaCy loader unavailable: %s", exc)
            _failed[model_name] = True
        return None

    try:
        # ``disable=()`` matches the analysis side's "safe" mode so the
        # cache key (model, disable_tuple) is unified across layers.
        return get_nlp(model_name, disable=())
    except Exception as exc:
        with _lock:
            if not _failed.get(model_name):
                logger.warning(
                    "spaCy model '%s' unavailable; using fallback. (%s)",
                    model_name, exc,
                )
            _failed[model_name] = True
        return None


def reset_shared_nlp() -> None:
    """Drop the local "already failed" set. Test-only.

    The actual model cache lives in ``src.analysis.spacy_loader``; call
    its ``clear_cache`` if you also need to drop the loaded pipelines.
    """
    with _lock:
        _failed.clear()
