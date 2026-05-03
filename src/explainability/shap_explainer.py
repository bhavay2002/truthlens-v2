from __future__ import annotations

from collections import OrderedDict
import hashlib
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np

from src.explainability.utils_validation import validate_tokens_scores
from src.explainability.explanation_calibrator import calibrate_explanation
from src.explainability.common_schema import ExplanationOutput, TokenImportance

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None  # type: ignore

logger = logging.getLogger(__name__)

# =========================================================
# CACHE CONFIG
# =========================================================

_MAX_EXPLAINER_CACHE_SIZE = 8
_MAX_VALUE_CACHE_SIZE = 64

_EXPLAINER_CACHE: "OrderedDict[Tuple[Any, ...], Any]" = OrderedDict()
_VALUE_CACHE: "OrderedDict[Tuple[Any, ...], Any]" = OrderedDict()

_LOCK = threading.RLock()

EPS = 1e-12

SPECIAL_TOKENS = {
    "[CLS]", "[SEP]", "<s>", "</s>",
    "[PAD]", "<pad>", "[UNK]", "<unk>",
}

# =========================================================
# UTIL
# =========================================================

def _process_shap_values(values):
    if isinstance(values, list):
        values = values[0]

    values = np.asarray(values, dtype=np.float32)

    if values.ndim == 3:
        values = values[:, :, -1]
    elif values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    elif values.ndim == 2:
        # EDGE CASE FIX: (seq_len, num_classes) from a multi-class SHAP
        # explainer — take the last column (positive/fake class).
        # Previously fell through, leaving a 2D array that broke every
        # downstream caller expecting a 1-D importance vector.
        values = values[:, -1]

    return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)


def _extract_fake_probability(result: Any) -> float:
    if not isinstance(result, dict) or "fake_probability" not in result:
        raise KeyError("predict_fn must return {'fake_probability': float}")

    p = float(result["fake_probability"])

    if not (0.0 <= p <= 1.0):
        raise ValueError("fake_probability must be in [0,1]")

    return p


# =========================================================
# PREDICT WRAPPER
# =========================================================

def shap_predict_wrapper(
    texts: Sequence[str],
    predict_fn: Callable[[str], Dict[str, Any]],
) -> np.ndarray:

    batch_fn = getattr(predict_fn, "batch_predict", None)

    if callable(batch_fn):
        try:
            results = batch_fn(list(texts))
            return np.array(
                [[1 - _extract_fake_probability(r), _extract_fake_probability(r)]
                 for r in results],
                dtype=float,
            )
        except Exception as e:
            logger.warning("Batch prediction failed: %s", e)

    outputs = []
    for t in texts:
        r = predict_fn(t)
        p = _extract_fake_probability(r)
        outputs.append([1 - p, p])

    return np.asarray(outputs, dtype=float)


# =========================================================
# CACHE KEY
# =========================================================

def _stable_predict_fn_key(predict_fn) -> Tuple[Any, ...]:

    stable_id = getattr(predict_fn, "__cache_key__", None)
    if isinstance(stable_id, str):
        return ("explicit", stable_id)

    bound = getattr(predict_fn, "__self__", None)
    if bound is not None:
        model_name = getattr(bound, "model_name", None)
        tokenizer_name = getattr(bound, "tokenizer_name", None)
        if model_name or tokenizer_name:
            return ("model", model_name, tokenizer_name)

    return (
        getattr(predict_fn, "__module__", "unknown"),
        getattr(predict_fn, "__qualname__", "unknown"),
    )


def _set_cache(cache, key, value, max_size):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_size:
        cache.popitem(last=False)


# =========================================================
# EXPLAINER CACHE
# =========================================================

def get_explainer(predict_fn):

    if shap is None:
        raise ImportError("Install shap to use SHAP explanations")

    key = _stable_predict_fn_key(predict_fn)

    with _LOCK:
        if key not in _EXPLAINER_CACHE:
            logger.info("Initializing SHAP explainer")

            masker = shap.maskers.Text()
            explainer = shap.Explainer(
                lambda x: shap_predict_wrapper(x, predict_fn),
                masker,
            )

            _set_cache(_EXPLAINER_CACHE, key, explainer, _MAX_EXPLAINER_CACHE_SIZE)

        return _EXPLAINER_CACHE[key]


# =========================================================
# VALUE CACHE
# =========================================================

def _get_shap_values(predict_fn, text):

    text_hash = hashlib.sha1(text.encode()).hexdigest()
    key = _stable_predict_fn_key(predict_fn) + (text_hash,)

    with _LOCK:
        if key in _VALUE_CACHE:
            _VALUE_CACHE.move_to_end(key)
            return _VALUE_CACHE[key]

    explainer = get_explainer(predict_fn)
    shap_values = explainer([text])

    with _LOCK:
        _set_cache(_VALUE_CACHE, key, shap_values, _MAX_VALUE_CACHE_SIZE)

    return shap_values


# =========================================================
# 🔥 MAIN EXPLAIN (FINAL)
# =========================================================

def explain_text(predict_fn, text: str) -> ExplanationOutput:

    if not text.strip():
        raise ValueError("text cannot be empty")

    shap_values = _get_shap_values(predict_fn, text)

    data = getattr(shap_values, "data", None)
    if data is None or len(data) == 0:
        return ExplanationOutput(
            method="shap",
            tokens=[],
            importance=[],
            structured=[],
        )

    tokens = list(data[0])
    values = _process_shap_values(shap_values.values[0])

    n = min(len(tokens), len(values))
    tokens = tokens[:n]
    values = values[:n]

    filtered = [(t, v) for t, v in zip(tokens, values) if t not in SPECIAL_TOKENS]

    if not filtered:
        return ExplanationOutput(
            method="shap",
            tokens=[],
            importance=[],
            structured=[],
        )

    tokens, values = zip(*filtered)
    tokens = list(tokens)
    values = list(values)

    validate_tokens_scores(tokens, values)

    # =====================================================
    # 🔥 CALIBRATION
    # =====================================================
    cal = calibrate_explanation(values, method="shap")

    scores = cal["scores"]              # np.ndarray
    confidence = cal["confidence"]
    entropy = cal["entropy"]

    structured = [
        TokenImportance(token=t, importance=float(s))
        for t, s in zip(tokens, scores)
    ]

    return ExplanationOutput(
        method="shap",
        tokens=tokens,
        importance=scores.tolist(),
        structured=structured,
        confidence=confidence,
        entropy=entropy,
    )


# =========================================================
# VISUALIZATION
# =========================================================

def plot_explanation(predict_fn, text):

    if shap is None:
        raise ImportError("SHAP not installed")

    shap_values = _get_shap_values(predict_fn, text)
    shap.plots.text(shap_values[0])


def save_explanation_html(predict_fn, text, output_path="reports/shap.html"):

    if shap is None:
        raise ImportError("SHAP not installed")

    shap_values = _get_shap_values(predict_fn, text)

    html = shap.plots.text(shap_values[0], display=False)
    html_str = str(html) if html else "<p>No SHAP output</p>"

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(html_str, encoding="utf-8")

    return path


# =========================================================
# CACHE UTIL
# =========================================================

def cache_stats():
    return {
        "explainer_cache": len(_EXPLAINER_CACHE),
        "value_cache": len(_VALUE_CACHE),
    }


def clear_cache():
    with _LOCK:
        _EXPLAINER_CACHE.clear()
        _VALUE_CACHE.clear()