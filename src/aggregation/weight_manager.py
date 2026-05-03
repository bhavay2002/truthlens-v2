from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Iterable, List, Mapping
from threading import RLock

import numpy as np

# CFG-AG-1: WEIGHT_GROUPS / TASK_TO_GROUP / SCALAR_WEIGHT_KEYS now
# live in `aggregation_config.py`. Re-exported below so existing
# imports (`from .weight_manager import WEIGHT_GROUPS`) keep working.
from .aggregation_config import (
    WEIGHT_GROUPS,
    TASK_TO_GROUP,
    SCALAR_WEIGHT_KEYS as _SCALAR_KEYS,
)

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# DEFAULTS
# =========================================================

DEFAULT_WEIGHTS: Dict[str, float] = {
    "bias": 0.40,
    "emotion": 0.30,
    "narrative": 0.20,
    "analysis_influence_manipulation": 0.10,

    "discourse": 0.55,
    "graph": 0.35,
    "analysis_influence_credibility": 0.10,

    "credibility_bias_penalty": 0.20,

    "final_credibility": 0.5,
    "final_manipulation": 0.3,
    "final_ideology": 0.2,
}


__all__ = [
    "DEFAULT_WEIGHTS",
    "WEIGHT_GROUPS",
    "TASK_TO_GROUP",
    "WeightManager",
]


def _aggregate_group_signal(
    signal: Optional[Mapping[str, float]],
    default: float,
) -> Dict[str, float]:
    """Average a per-task signal into per-group values.

    EDGE-AG: every input value is filtered for NaN/Inf and a fallback
    default is returned when no usable values exist. The previous
    implementation relied on `np.clip(NaN, 0, 1)` which returns NaN
    and then poisoned the multiplicative scale chain.
    """
    if not signal:
        return {grp: default for grp in WEIGHT_GROUPS}

    out: Dict[str, float] = {}
    for grp in WEIGHT_GROUPS:
        vals: List[float] = []
        for task, group in TASK_TO_GROUP.items():
            if group != grp or task not in signal:
                continue
            v = signal[task]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if not np.isfinite(v):
                continue
            vals.append(float(np.clip(v, 0.0, 1.0)))
        out[grp] = float(np.mean(vals)) if vals else default
    return out


# =========================================================
# MANAGER
# =========================================================

class WeightManager:

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        *,
        version: str = "v2",
        frozen: bool = False,
        smoothing: float = 0.1,
        uncertainty_penalty: float = 0.2,
        scale_clip: tuple = (0.5, 2.0),
    ) -> None:

        self._lock = RLock()
        self.version = version
        self.frozen = frozen
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self.uncertainty_penalty = float(np.clip(uncertainty_penalty, 0.0, 1.0))
        # WGT-AG-4: symmetric clip in log-space (e.g. 0.5..2.0 → ±1
        # octave). The previous (0.1, 2.0) range allowed a 2x boost but
        # a 10x attenuation, biasing the system toward high-confidence
        # sections.
        self.scale_clip = (
            float(scale_clip[0]),
            float(scale_clip[1]),
        )

        # CRIT-AG-9: merge user-provided weights with the defaults so
        # callers may override only a subset (e.g. from
        # `WeightConfig.weights` populated by config.yaml).
        merged = dict(DEFAULT_WEIGHTS)
        if weights:
            merged.update(weights)

        self.weights = merged
        self._validate_weights(self.weights)
        self._clip_scalar_keys()
        self._normalize_weights()

        logger.info(
            "[WeightManager] Initialized | version=%s frozen=%s",
            self.version,
            self.frozen,
        )

    # =====================================================
    # LOAD
    # =====================================================

    def load_weights_from_config(self, config_path: str | Path) -> Dict[str, float]:

        with self._lock:

            if self.frozen:
                raise RuntimeError("Weights are frozen")

            config_path = Path(config_path)

            with config_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)

            if not isinstance(loaded, dict):
                raise ValueError("Weight config must be dict")

            merged = self.weights.copy()
            merged.update(loaded)

            self._validate_weights(merged)
            self.weights = merged
            self._clip_scalar_keys()
            self._normalize_weights()

            return self.get_weights()

    # =====================================================
    # VALIDATION
    # =====================================================

    def _validate_weights(self, weights: Dict[str, Any]) -> None:

        for k, v in weights.items():

            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"{k} must be numeric")

            if not np.isfinite(v) or v < 0:
                raise ValueError(f"{k} invalid: {v}")

        # WGT-AG-5: catch the all-zero-group case here with a clearer
        # message than `_normalize_group`'s generic ValueError.
        for grp, keys in WEIGHT_GROUPS.items():
            present = [k for k in keys if k in weights]
            if present and sum(float(weights[k]) for k in present) <= 0:
                raise ValueError(
                    f"Weight group {grp!r} sums to zero — at least one "
                    f"of {list(present)} must be > 0"
                )

    def _clip_scalar_keys(self) -> None:
        # WGT-AG-1: scalar (non-grouped) keys must be clipped to [0, 1]
        # because they are used directly as multipliers, not as
        # relative weights.
        for k in _SCALAR_KEYS:
            if k in self.weights:
                self.weights[k] = float(np.clip(self.weights[k], 0.0, 1.0))

    # =====================================================
    # NORMALIZATION
    # =====================================================

    def _normalize_group(self, keys: Iterable[str]) -> None:

        values = np.array([self.weights[k] for k in keys], dtype=np.float64)

        total = float(np.sum(values))

        if total <= 0:
            raise ValueError(
                f"Weight group {tuple(keys)!r} sums to zero — at least "
                "one weight must be > 0"
            )

        values = values / total

        for k, v in zip(keys, values):
            self.weights[k] = float(v)

    def _normalize_weights(self) -> None:
        for group in WEIGHT_GROUPS.values():
            self._normalize_group(group)

    # =====================================================
    # ADAPTIVE WEIGHTING (CRIT-AG-7, CRIT-AG-8, CRIT-AG-10, WGT-AG-4)
    # =====================================================

    def get_adaptive_weights(
        self,
        *,
        confidence: Optional[Dict[str, float]] = None,
        entropy: Optional[Dict[str, float]] = None,
        explanation_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:

        with self._lock:

            scaled = dict(self.weights)

            # CRIT-AG-8: derive a per-group scale factor from the
            # per-task confidence / entropy dicts. Every key inside a
            # group then receives the same multiplicative scale, so no
            # entry is left unmodulated to silently dominate the group
            # after renormalisation.
            conf_g = (
                _aggregate_group_signal(confidence, default=1.0)
                if confidence is not None else None
            )
            ent_g = (
                _aggregate_group_signal(entropy, default=0.0)
                if entropy is not None else None
            )

            for grp, keys in WEIGHT_GROUPS.items():

                scale = 1.0

                if conf_g is not None:
                    scale *= conf_g.get(grp, 1.0)

                if ent_g is not None:
                    # respect the configured uncertainty_penalty so that
                    # high-uncertainty groups are dampened by a known
                    # magnitude rather than zeroed out.
                    ent_val = float(np.clip(ent_g.get(grp, 0.0), 0.0, 1.0))
                    scale *= max(0.0, 1.0 - self.uncertainty_penalty * ent_val)

                if explanation_scores:
                    exp_vals = [
                        float(np.clip(explanation_scores[k], 0.0, 1.0))
                        for k in keys
                        if k in explanation_scores
                        and isinstance(explanation_scores[k], (int, float))
                        and np.isfinite(explanation_scores[k])
                    ]
                    if exp_vals:
                        scale *= 1.0 + float(np.mean(exp_vals))

                # WGT-AG-4: symmetric clip — equal headroom for boost
                # and attenuation in log-space.
                lo, hi = self.scale_clip
                scale = float(np.clip(scale, lo, hi))

                for k in keys:
                    scaled[k] = self.weights[k] * scale

            # CRIT-AG-10: renormalise the scaled vector BEFORE applying
            # the convex smoothing. Both inputs to the smoothing
            # combination then sum to 1 within each group, so the
            # smoothing factor `α` actually means "α toward adaptive".
            for keys in WEIGHT_GROUPS.values():
                total = sum(scaled[k] for k in keys) + EPS
                for k in keys:
                    scaled[k] /= total

            for k in self.weights:
                if k in _SCALAR_KEYS:
                    continue
                if k not in scaled:
                    scaled[k] = self.weights[k]
                if k in {kk for keys in WEIGHT_GROUPS.values() for kk in keys}:
                    scaled[k] = (
                        (1.0 - self.smoothing) * self.weights[k]
                        + self.smoothing * scaled[k]
                    )

            # Final renorm for floating-point cleanup.
            for keys in WEIGHT_GROUPS.values():
                total = sum(scaled[k] for k in keys) + EPS
                for k in keys:
                    scaled[k] /= total

            # Scalar (non-grouped) entries are passed through unchanged.
            for k in _SCALAR_KEYS:
                if k in self.weights:
                    scaled[k] = self.weights[k]

            return scaled

    # =====================================================
    # SIMPLE CONFIDENCE MODE (BACKWARD COMPAT)
    # =====================================================

    def get_weighted(
        self,
        confidence: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:

        return self.get_adaptive_weights(confidence=confidence)

    # =====================================================
    # UPDATE
    # =====================================================

    def adjust_weight(self, key: str, value: float) -> Dict[str, float]:

        with self._lock:

            if self.frozen:
                raise RuntimeError("Weights frozen")

            self.weights[key] = float(value)

            self._validate_weights(self.weights)
            self._clip_scalar_keys()
            self._normalize_weights()

            return self.get_weights()

    # =====================================================
    # ACCESS
    # =====================================================

    def get_weights(self) -> Dict[str, float]:
        with self._lock:
            return self.weights.copy()
