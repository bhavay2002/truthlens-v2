from __future__ import annotations

import logging
from typing import Iterable, Dict, Any, Union, Optional

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


logger = logging.getLogger(__name__)

EPS = 1e-12
ArrayLike = Union[np.ndarray, "torch.Tensor"]


# =========================================================
# UTILS
# =========================================================

def _to_numpy(x: ArrayLike) -> np.ndarray:
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.nan_to_num(np.asarray(x), nan=0.0, posinf=1.0, neginf=0.0)


def _to_output(arr: np.ndarray, like: ArrayLike):
    # GPU-AG-4: preserve the *input* tensor's dtype and device so we
    # don't accidentally promote a fp16/bf16 tensor (e.g. when running
    # under autocast) to fp32 silently. The previous implementation
    # always returned float32, which mixed dtypes downstream and
    # produced autocast warnings or implicit casts.
    if TORCH_AVAILABLE and isinstance(like, torch.Tensor):
        return torch.from_numpy(arr.astype(np.float32, copy=False)).to(
            device=like.device,
            dtype=like.dtype,
        )
    return arr


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _clip_to_range(x, feature_range):
    a, b = feature_range
    return np.clip(x, a, b)


# =========================================================
# NORMALIZER
# =========================================================

class ScoreNormalizer:

    def __init__(
        self,
        method: str = "minmax",
        *,
        feature_range=(0.0, 1.0),
        strict: bool = False,
        clip: bool = True,
    ):
        self.method = method.lower()
        self.feature_range = feature_range
        self.strict = strict
        self.clip = clip

        self.stats: Dict[str, Any] = {}
        self.fitted = False

    # =====================================================
    # FIT
    # =====================================================

    def fit(self, values: ArrayLike):

        arr = _to_numpy(values)

        if arr.size == 0:
            raise ValueError("Empty input")

        if self.method == "minmax":
            self.stats["min"] = float(arr.min())
            self.stats["max"] = float(arr.max())

        elif self.method == "zscore":
            self.stats["mean"] = float(arr.mean())
            self.stats["std"] = float(arr.std())

        elif self.method == "robust":
            self.stats["median"] = float(np.median(arr))
            self.stats["iqr"] = float(
                np.percentile(arr, 75) - np.percentile(arr, 25)
            )

        elif self.method == "quantile":
            self.stats["sorted"] = np.sort(arr)

        else:
            raise ValueError(f"Unsupported method: {self.method}")

        self.fitted = True
        return self

    # =====================================================
    # TRANSFORM
    # =====================================================

    def transform(self, values: ArrayLike) -> ArrayLike:

        if not self.fitted:
            raise RuntimeError("Call fit() first")

        arr = _to_numpy(values)

        if self.method == "minmax":
            vmin = self.stats["min"]
            vmax = self.stats["max"]

            denom = max(vmax - vmin, EPS)
            norm = (arr - vmin) / denom

            a, b = self.feature_range
            result = norm * (b - a) + a

        elif self.method == "zscore":
            mean = self.stats["mean"]
            std = max(self.stats["std"], EPS)
            result = (arr - mean) / std

        elif self.method == "robust":
            median = self.stats["median"]
            iqr = max(self.stats["iqr"], EPS)
            result = (arr - median) / iqr

        elif self.method == "quantile":
            sorted_vals = self.stats["sorted"]
            ranks = np.searchsorted(sorted_vals, arr)
            result = ranks / len(sorted_vals)

        else:
            raise ValueError(f"Unsupported method: {self.method}")

        if self.clip:
            # NORM-AG-3: respect feature_range — previously clip() always
            # truncated to [0, 1] regardless of the configured range,
            # silently dropping the negative half of e.g. (-1, 1).
            result = _clip_to_range(result, self.feature_range)

        return _to_output(result.astype(np.float32), values)

    # =====================================================
    # FIT + TRANSFORM
    # =====================================================

    def fit_transform(self, values: ArrayLike):
        return self.fit(values).transform(values)

    # =====================================================
    # PROBABILITY-AWARE NORMALIZATION
    # =====================================================

    def normalize_probabilities(self, probs: ArrayLike) -> ArrayLike:

        arr = _to_numpy(probs)

        if arr.ndim == 2:
            arr = arr / (np.sum(arr, axis=1, keepdims=True) + EPS)

        return _to_output(arr.astype(np.float32), probs)

    # =====================================================
    # ENTROPY-AWARE NORMALIZATION
    # =====================================================

    def normalize_with_uncertainty(
        self,
        values: ArrayLike,
        entropy: Optional[np.ndarray],
    ) -> ArrayLike:

        arr = _to_numpy(values)

        if entropy is not None:
            entropy = _to_numpy(entropy)
            arr = arr * (1.0 - entropy)

        return self.transform(arr)

    # =====================================================
    # SERIALIZATION
    # =====================================================

    def state_dict(self):
        return {
            "method": self.method,
            "stats": self.stats,
            "feature_range": self.feature_range,
            "clip": self.clip,
        }

    def load_state_dict(self, state):
        self.method = state["method"]
        self.stats = state["stats"]
        self.feature_range = tuple(state["feature_range"])
        self.clip = state.get("clip", True)
        self.fitted = True


# =========================================================
# EXTRA UTILITIES
# =========================================================

def log_scale(values: ArrayLike):
    arr = _to_numpy(values)
    return np.log1p(arr)


def percentile_clip(values: ArrayLike, low=1, high=99):
    arr = _to_numpy(values)
    lo = np.percentile(arr, low)
    hi = np.percentile(arr, high)
    return np.clip(arr, lo, hi)


def sigmoid_calibration(values: ArrayLike):
    arr = _to_numpy(values)
    return 1 / (1 + np.exp(-arr))


# =========================================================
# CONVENIENCE WRAPPERS (stateless, single-call)
# =========================================================

def clip_scores(
    values: ArrayLike,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> np.ndarray:
    """Clip an array of scores to [min_value, max_value]."""
    arr = _to_numpy(values)
    return np.clip(arr, min_value, max_value)


def normalize_minmax(
    values: ArrayLike,
    feature_range=(0.0, 1.0),
) -> np.ndarray:
    """Min-max normalize values into feature_range in one call."""
    arr = _to_numpy(values)
    if arr.size == 0:
        raise ValueError("Input cannot be empty")
    norm = ScoreNormalizer(method="minmax", feature_range=feature_range)
    return norm.fit_transform(arr).astype(np.float32)


def normalize_zscore(values: ArrayLike) -> np.ndarray:
    """Z-score normalize values in one call."""
    arr = _to_numpy(values)
    if arr.size == 0:
        return arr
    norm = ScoreNormalizer(method="zscore", clip=False)
    return norm.fit_transform(arr).astype(np.float32)


def normalize_robust(values: ArrayLike) -> np.ndarray:
    """Robust (median/IQR) normalize values in one call."""
    arr = _to_numpy(values)
    if arr.size == 0:
        return arr
    norm = ScoreNormalizer(method="robust", clip=False)
    return norm.fit_transform(arr).astype(np.float32)


def normalize_pipeline(
    values: ArrayLike,
    method: str = "minmax",
    feature_range=(0.0, 1.0),
    clip: Optional[bool] = None,
) -> np.ndarray:
    """Generic single-call normalizer — dispatches to the ScoreNormalizer.

    ``clip`` defaults to True for minmax (output is already bounded) and
    False for zscore/robust (which produce unbounded outputs by design).
    """
    arr = _to_numpy(values)
    if arr.size == 0:
        return arr
    if clip is None:
        clip = method == "minmax"
    _SUPPORTED = {"minmax", "zscore", "robust", "quantile"}
    if method not in _SUPPORTED:
        raise ValueError(f"Unsupported normalization method: '{method}'. Choose from {sorted(_SUPPORTED)}")
    norm = ScoreNormalizer(method=method, feature_range=feature_range, clip=clip)
    return norm.fit_transform(arr).astype(np.float32)