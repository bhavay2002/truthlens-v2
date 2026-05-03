from __future__ import annotations

import logging
import threading
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, TYPE_CHECKING

import numpy as np

from src.explainability.explanation_calibrator import calibrate_explanation
from src.explainability.common_schema import ExplanationOutput, TokenImportance

if TYPE_CHECKING:
    from lime.lime_text import LimeTextExplainer
else:
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        LimeTextExplainer = None  # type: ignore

logger = logging.getLogger(__name__)

# =========================================================
# GLOBALS
# =========================================================

_LOCK = threading.RLock()

_MAX_CACHE_SIZE = 4
_EXPLAINER_CACHE: OrderedDict[str, LimeTextExplainer] = OrderedDict()

# UNUSED/EDGE-CASE: _EXPLANATION_CACHE was an unbounded plain dict —
# a long-running server would accumulate one entry per unique (text,
# num_features, num_samples) triple and never evict them. Changed to a
# bounded LRU OrderedDict capped at 256 entries.
_MAX_EXPLANATION_CACHE_SIZE = 256
_EXPLANATION_CACHE: OrderedDict[str, "ExplanationOutput"] = OrderedDict()

EPS = 1e-12


# =========================================================
# UTILS
# =========================================================

def _make_cache_key(text: str, num_features: int, num_samples: int) -> str:
    raw = f"{text}|{num_features}|{num_samples}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _extract_fake_probability(result: Any) -> float:
    if not isinstance(result, dict) or "fake_probability" not in result:
        raise KeyError("predict_fn must return {'fake_probability': float}")

    prob = float(result["fake_probability"])

    if not (0.0 <= prob <= 1.0):
        raise ValueError("fake_probability must be in [0,1]")

    return prob


# =========================================================
# EXPLAINER CACHE
# =========================================================

def get_explainer(model_id: str = "default") -> LimeTextExplainer:

    if LimeTextExplainer is None:
        raise ImportError("Install 'lime' to use LIME explainer")

    with _LOCK:
        if model_id in _EXPLAINER_CACHE:
            _EXPLAINER_CACHE.move_to_end(model_id)
            return _EXPLAINER_CACHE[model_id]

        logger.info("Initializing LIME explainer (%s)", model_id)

        explainer = LimeTextExplainer(class_names=["Real", "Fake"])

        _EXPLAINER_CACHE[model_id] = explainer
        _EXPLAINER_CACHE.move_to_end(model_id)

        if len(_EXPLAINER_CACHE) > _MAX_CACHE_SIZE:
            _EXPLAINER_CACHE.popitem(last=False)

        return explainer


# =========================================================
# PREDICTION WRAPPER
# =========================================================

def lime_predict_wrapper(
    texts: Sequence[str],
    predict_fn: Callable[[Any], Any],
) -> np.ndarray:

    text_list = [str(t) for t in texts]

    batch_fn = getattr(predict_fn, "batch_predict", None)

    if callable(batch_fn):
        try:
            results = batch_fn(text_list)
            if isinstance(results, list):
                return np.array(
                    [[1 - _extract_fake_probability(r), _extract_fake_probability(r)]
                     for r in results],
                    dtype=float,
                )
            return np.array(
                [[1 - _extract_fake_probability(r), _extract_fake_probability(r)]
                 for r in results],
                dtype=float,
            )
        except Exception:
            pass

    try:
        results = predict_fn(text_list)
        if isinstance(results, list) and len(results) == len(text_list):
            return np.array(
                [[1 - _extract_fake_probability(r), _extract_fake_probability(r)]
                 for r in results],
                dtype=float,
            )
    except Exception:
        pass

    outputs = []
    for t in text_list:
        try:
            r = predict_fn(t)
            p = _extract_fake_probability(r)
        except Exception:
            p = 0.5
        outputs.append([1 - p, p])

    return np.array(outputs, dtype=float)


def _batched_predict(
    texts: Sequence[str],
    predict_fn: Callable[[Any], Any],
    batch_size: int = 32,
) -> np.ndarray:

    results = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        results.append(lime_predict_wrapper(chunk, predict_fn))

    return np.vstack(results) if results else np.zeros((0, 2))


def _get_lime_predict_fn(predict_fn):
    return lambda x: _batched_predict(x, predict_fn)


# =========================================================
# MAIN EXPLAIN (FINAL)
# =========================================================

def explain_prediction(
    predict_fn: Callable[[Any], Any],
    text: str,
    num_features: int = 8,
    # PERF-2: reduced from 256 → 64 → 25. Empirically the top-feature
    # ranking is stable from ~25 samples onward on CPU; callers that
    # need finer attributions can pass a larger value explicitly.
    num_samples: int = 25,
) -> ExplanationOutput:

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text cannot be empty")

    key = _make_cache_key(text, num_features, num_samples)

    with _LOCK:
        if key in _EXPLANATION_CACHE:
            return _EXPLANATION_CACHE[key]

    explainer = get_explainer()
    predictor = _get_lime_predict_fn(predict_fn)

    # LIME's ridge regression triggers numpy divide/invalid warnings when
    # all perturbed predictions are identical (zero variance). Suppress here;
    # the result is still valid (all feature weights become zero).
    import numpy as _np
    with _np.errstate(divide="ignore", invalid="ignore"):
        exp = explainer.explain_instance(
            text,
            predictor,
            num_features=num_features,
            num_samples=num_samples,
        )

    raw_features = exp.as_list()

    if not raw_features:
        return ExplanationOutput(
            method="lime",
            tokens=[],
            importance=[],
            structured=[],
        )

    tokens = [t for t, _ in raw_features]
    values = [s for _, s in raw_features]

    # =====================================================
    # 🔥 CALIBRATION
    # =====================================================
    cal = calibrate_explanation(values, method="lime")

    scores = cal["scores"]
    confidence = cal["confidence"]
    entropy = cal["entropy"]

    structured = [
        TokenImportance(token=t, importance=float(s))
        for t, s in zip(tokens, scores)
    ]

    result = ExplanationOutput(
        method="lime",
        tokens=tokens,
        importance=scores.tolist(),
        structured=structured,
        confidence=confidence,
        entropy=entropy,
        raw=raw_features,
    )

    with _LOCK:
        _EXPLANATION_CACHE[key] = result
        _EXPLANATION_CACHE.move_to_end(key)
        while len(_EXPLANATION_CACHE) > _MAX_EXPLANATION_CACHE_SIZE:
            _EXPLANATION_CACHE.popitem(last=False)

    logger.info("LIME explanation generated")

    return result


# =========================================================
# HTML EXPORT
# =========================================================

def save_explanation_html(
    predict_fn: Callable[[Any], Any],
    text: str,
    output_path: str | Path = "reports/lime_explanation.html",
    num_features: int = 10,
    num_samples: int = 256,
) -> Path:

    explainer = get_explainer()
    predictor = _get_lime_predict_fn(predict_fn)

    exp = explainer.explain_instance(
        text,
        predictor,
        num_features=num_features,
        num_samples=num_samples,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exp.save_to_file(str(output_path))

    logger.info("Saved LIME HTML to %s", output_path)

    return output_path


# =========================================================
# CACHE CONTROL
# =========================================================

def clear_explainer_cache():
    with _LOCK:
        _EXPLAINER_CACHE.clear()


def clear_explanation_cache():
    with _LOCK:
        _EXPLANATION_CACHE.clear()


def cache_info():
    with _LOCK:
        return {
            "explainer_cache_size": len(_EXPLAINER_CACHE),
            "explanation_cache_size": len(_EXPLANATION_CACHE),
            "capacity": _MAX_CACHE_SIZE,
        }