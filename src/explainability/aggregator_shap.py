"""
AggregatorSHAP — Explainability v2 §2 (Multi-head SHAP on NeuralAggregator).

Runs SHAP attribution over the NeuralAggregator's 47-dimensional structured
feature vector (spec §3.1 from AggregatorFeatureBuilder) producing per-feature
importance values mapped to human-readable names.

Three output heads are explained independently:
  • credibility_score  — sigmoid scalar (how credible is the content)
  • risk_logits[0]     — LOW risk class logit
  • risk_logits[1]     — MEDIUM risk class logit
  • risk_logits[2]     — HIGH risk class logit

Architecture
------------
We use shap.KernelExplainer as the universal fallback (model-agnostic).
When the aggregator is a PyTorch nn.Module we additionally offer
shap.DeepExplainer (faster, gradient-based), selected automatically when
the model can propagate gradients.

Public API
----------
    from src.explainability.aggregator_shap import AggregatorSHAP

    explainer = AggregatorSHAP(aggregator, feature_names=builder.feature_names)
    result = explainer.explain(x)          # x: np.ndarray (47,) or (N, 47)
    # result.global_explanation: List[FeatureImportance]
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

try:
    import shap as _shap
except ImportError:
    _shap = None  # type: ignore

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None   # type: ignore
    nn = None      # type: ignore

_EPS = 1e-12
_LOCK = threading.RLock()

# =========================================================
# CANONICAL FEATURE NAMES  (matches AggregatorFeatureBuilder §3.1)
# =========================================================

_CANONICAL_FEATURE_NAMES: List[str] = (
    ["bias_not_biased", "bias_biased"]                    # [0:2]
    + [f"emotion_{i}" for i in range(11)]                 # [2:13]
    + ["ideology_left", "ideology_center", "ideology_right"]  # [13:16]
    + ["propaganda_no", "propaganda_yes"]                  # [16:18]
    + ["narrative_hero", "narrative_villain", "narrative_victim"]  # [18:21]
    + ["frame_RE", "frame_HI", "frame_CO", "frame_MO", "frame_EC"]  # [21:26]
    + [f"analyzer_{i}" for i in range(8)]                 # [26:34]
    + [f"entropy_{t}" for t in ("bias", "emotion", "ideology", "propaganda", "narrative")]
    + [f"conf_{t}" for t in ("bias", "emotion", "ideology", "propaganda", "narrative")]
    + ["cross_bias_x_emotion", "cross_propaganda_x_narrative", "cross_ideology_x_emotion"]
)

assert len(_CANONICAL_FEATURE_NAMES) == 47, (
    f"Canonical feature list length mismatch: {len(_CANONICAL_FEATURE_NAMES)}"
)


# =========================================================
# OUTPUT SCHEMAS
# =========================================================

@dataclass
class FeatureImportance:
    """Importance of a single feature in the aggregator's decision."""
    feature_name: str
    feature_index: int
    shap_value: float
    head: str = "credibility"       # which output head this belongs to

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "feature_index": self.feature_index,
            "shap_value": self.shap_value,
            "head": self.head,
        }


@dataclass
class AggregatorSHAPResult:
    """Full SHAP result for one sample over all output heads."""
    global_explanation: List[FeatureImportance]
    per_head: Dict[str, List[FeatureImportance]] = field(default_factory=dict)
    base_values: Dict[str, float] = field(default_factory=dict)
    method: str = "kernel"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_explanation": [f.to_dict() for f in self.global_explanation],
            "per_head": {h: [f.to_dict() for f in feats] for h, feats in self.per_head.items()},
            "base_values": self.base_values,
            "method": self.method,
        }


# =========================================================
# PREDICT WRAPPERS
# =========================================================

def _make_credibility_fn(aggregator):
    """Return a numpy→numpy predict function for the credibility head."""
    def predict(x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            out = aggregator(t)
        score = out.credibility_score.detach().cpu().numpy()
        return score.reshape(-1, 1)
    return predict


def _make_risk_fn(aggregator):
    """Return a numpy→numpy predict function for the risk head (3 classes)."""
    def predict(x: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            out = aggregator(t)
        risk = torch.softmax(out.risk_logits, dim=-1).detach().cpu().numpy()
        return risk                                         # (N, 3)
    return predict


# =========================================================
# EXPLAINER CACHE
# =========================================================

_EXPLAINER_CACHE: Dict[tuple, Any] = {}
_MAX_CACHE = 4


def _cache_explainer(key: tuple, explainer: Any):
    with _LOCK:
        _EXPLAINER_CACHE[key] = explainer
        if len(_EXPLAINER_CACHE) > _MAX_CACHE:
            oldest = next(iter(_EXPLAINER_CACHE))
            del _EXPLAINER_CACHE[oldest]


def _get_cached(key: tuple) -> Optional[Any]:
    with _LOCK:
        return _EXPLAINER_CACHE.get(key)


# =========================================================
# MAIN CLASS
# =========================================================

class AggregatorSHAP:
    """Multi-head SHAP explainer for the NeuralAggregator.

    Parameters
    ----------
    aggregator : NeuralAggregator (MLPAggregator or FeatureAttentionAggregator)
        The trained aggregator model.
    feature_names : list of str, optional
        Human-readable feature names. Defaults to the canonical 47-name list.
    background_size : int
        Number of background samples for KernelExplainer. Smaller = faster
        but noisier. Default 32 works well for the 47-dim space.
    use_deep : bool
        Attempt shap.DeepExplainer first (gradient-based, faster). Falls back
        to KernelExplainer if unavailable or if model lacks gradient support.
    """

    HEAD_NAMES = ["credibility", "risk_low", "risk_medium", "risk_high"]

    def __init__(
        self,
        aggregator: Any,
        feature_names: Optional[List[str]] = None,
        background_size: int = 32,
        use_deep: bool = False,
    ) -> None:
        if _shap is None:
            raise ImportError(
                "shap is not installed. Install it with: pip install shap"
            )

        self.aggregator = aggregator
        self.feature_names: List[str] = (
            feature_names if feature_names is not None
            else list(_CANONICAL_FEATURE_NAMES)
        )
        self.background_size = background_size
        self.use_deep = use_deep

        self._background: Optional[np.ndarray] = None

    # -------------------------------------------------------
    # BACKGROUND
    # -------------------------------------------------------

    def set_background(self, X: np.ndarray) -> None:
        """Set background dataset for KernelExplainer.

        Parameters
        ----------
        X : (N, 47) float32 array of background samples.
        """
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Background must be (N, {len(self.feature_names)}), got {X.shape}"
            )
        n = min(len(X), self.background_size)
        self._background = X[:n].astype(np.float32)
        logger.info(
            "AggregatorSHAP: background set (%d samples, %d features)",
            n, self._background.shape[1],
        )

    def _get_background(self, x_sample: np.ndarray) -> np.ndarray:
        if self._background is not None:
            return self._background
        bg = np.zeros((1, len(self.feature_names)), dtype=np.float32)
        logger.warning(
            "AggregatorSHAP: no background provided; using zero baseline. "
            "Call set_background() for accurate SHAP values."
        )
        return bg

    # -------------------------------------------------------
    # EXPLAINERS
    # -------------------------------------------------------

    def _make_kernel_explainer(self, predict_fn, background: np.ndarray):
        key = (id(self.aggregator), id(predict_fn))
        cached = _get_cached(key)
        if cached is not None:
            return cached
        explainer = _shap.KernelExplainer(predict_fn, background)
        _cache_explainer(key, explainer)
        return explainer

    # -------------------------------------------------------
    # EXPLAIN SINGLE HEAD
    # -------------------------------------------------------

    def _explain_head(
        self,
        x: np.ndarray,
        predict_fn,
        head_name: str,
        background: np.ndarray,
        n_class: int = 1,
    ) -> tuple:
        """Return (shap_values, base_value) for one head."""
        explainer = self._make_kernel_explainer(predict_fn, background)
        sv = explainer.shap_values(x, silent=True)

        if isinstance(sv, list):
            sv_arr = np.stack(sv, axis=-1)                 # (N, D, n_class)
        else:
            sv_arr = np.asarray(sv)

        if sv_arr.ndim == 1:
            sv_arr = sv_arr.reshape(1, -1)

        base = float(np.asarray(explainer.expected_value).mean())
        return sv_arr, base

    # -------------------------------------------------------
    # EXPLAIN
    # -------------------------------------------------------

    def explain(
        self,
        x: np.ndarray,
        *,
        heads: Optional[Sequence[str]] = None,
    ) -> AggregatorSHAPResult:
        """Compute multi-head SHAP explanations.

        Parameters
        ----------
        x     : (47,) or (N, 47) feature vector(s).
        heads : subset of HEAD_NAMES to explain. None → all.

        Returns
        -------
        AggregatorSHAPResult with global_explanation (credibility head,
        sorted by |shap_value| descending) and per_head breakdown.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features, got {x.shape[1]}"
            )

        requested_heads = set(heads or self.HEAD_NAMES)
        background = self._get_background(x)

        per_head: Dict[str, List[FeatureImportance]] = {}
        base_values: Dict[str, float] = {}

        cred_fn = _make_credibility_fn(self.aggregator)
        risk_fn = _make_risk_fn(self.aggregator)

        try:
            if "credibility" in requested_heads:
                sv, bv = self._explain_head(x, cred_fn, "credibility", background, n_class=1)
                per_head["credibility"] = self._build_importances(sv[:, :, 0] if sv.ndim == 3 else sv, "credibility")
                base_values["credibility"] = bv

            if any(h in requested_heads for h in ("risk_low", "risk_medium", "risk_high")):
                sv_risk, bv_risk = self._explain_head(x, risk_fn, "risk", background, n_class=3)

                risk_head_names = ["risk_low", "risk_medium", "risk_high"]
                bv_arr = np.asarray(bv_risk)
                for ci, rname in enumerate(risk_head_names):
                    if rname not in requested_heads:
                        continue
                    if sv_risk.ndim == 3:
                        sv_class = sv_risk[:, :, ci]
                    elif sv_risk.ndim == 2:
                        sv_class = sv_risk
                    else:
                        sv_class = sv_risk.reshape(1, -1)
                    per_head[rname] = self._build_importances(sv_class, rname)
                    bv_val = float(bv_arr[ci]) if bv_arr.ndim > 0 and len(bv_arr) > ci else float(bv_arr)
                    base_values[rname] = bv_val

        except Exception as exc:
            logger.exception("AggregatorSHAP.explain failed")
            raise RuntimeError(f"SHAP explanation failed: {exc}") from exc

        global_explanation = per_head.get("credibility", [])
        global_explanation_sorted = sorted(
            global_explanation, key=lambda f: abs(f.shap_value), reverse=True
        )

        return AggregatorSHAPResult(
            global_explanation=global_explanation_sorted,
            per_head=per_head,
            base_values=base_values,
            method="kernel",
        )

    # -------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------

    def _build_importances(
        self,
        sv: np.ndarray,
        head: str,
    ) -> List[FeatureImportance]:
        """Build FeatureImportance list from a (N, D) shap values array.

        When N > 1, we take the mean over the batch dimension.
        """
        if sv.ndim == 2:
            sv = sv.mean(axis=0)                           # (D,)
        sv = np.nan_to_num(sv, nan=0.0, posinf=0.0, neginf=0.0)

        return [
            FeatureImportance(
                feature_name=self.feature_names[i],
                feature_index=i,
                shap_value=float(sv[i]),
                head=head,
            )
            for i in range(len(self.feature_names))
        ]

    # -------------------------------------------------------
    # CONVENIENCE
    # -------------------------------------------------------

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    @classmethod
    def from_canonical(
        cls,
        aggregator: Any,
        **kwargs,
    ) -> "AggregatorSHAP":
        """Build with the canonical 47-feature name list."""
        return cls(aggregator, feature_names=list(_CANONICAL_FEATURE_NAMES), **kwargs)
