from __future__ import annotations

import logging
from typing import Dict, Optional, List, Any

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# CONFIG
# =========================================================

class RiskThresholds:
    def __init__(self, low: float = 0.3, medium: float = 0.6):
        if not (0.0 <= low < medium <= 1.0):
            raise ValueError("Invalid thresholds")

        self.low = low
        self.medium = medium


class RiskConfig:
    def __init__(
        self,
        default: RiskThresholds = RiskThresholds(),
        per_key: Optional[Dict[str, RiskThresholds]] = None,
        invert_keys: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        uncertainty_penalty: float = 0.2,
    ):
        self.default = default
        self.per_key = per_key or {}
        self.invert_keys = set(invert_keys or [])
        self.weights = weights or {}
        self.uncertainty_penalty = uncertainty_penalty


# =========================================================
# UTILS
# =========================================================

def _validate(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Invalid numeric value")

    if not np.isfinite(value):
        raise ValueError("Non-finite value")

    return float(np.clip(value, 0.0, 1.0))


def _entropy(probs):
    probs = np.asarray(probs)
    return -np.sum(probs * np.log(probs + EPS))


# =========================================================
# CORE RISK LOGIC
# =========================================================

def compute_risk_score(
    value: float,
    *,
    invert: bool,
    uncertainty: Optional[float],
    config: RiskConfig,
) -> float:

    value = _validate(value)

    if invert:
        value = 1.0 - value

    # uncertainty penalty
    if uncertainty is not None:
        value *= (1.0 - config.uncertainty_penalty * uncertainty)

    return float(np.clip(value, 0.0, 1.0))


def score_to_level(score: float, thresholds: RiskThresholds) -> str:

    if score < thresholds.low:
        return "LOW"
    elif score < thresholds.medium:
        return "MEDIUM"
    return "HIGH"


_DEFAULT_THRESHOLDS = RiskThresholds()


def score_to_risk_level(score: float, thresholds: Optional[RiskThresholds] = None) -> str:
    """Convenience wrapper around score_to_level using default thresholds."""
    return score_to_level(score, thresholds or _DEFAULT_THRESHOLDS)


# =========================================================
# MAIN API (UPGRADED)
# =========================================================

def assess_risk_levels(
    scores: Dict[str, float],
    *,
    probabilities: Optional[Dict[str, np.ndarray]] = None,
    config: Optional[RiskConfig] = None,
    return_scores: bool = False,
) -> Dict[str, Any]:

    if not isinstance(scores, dict):
        raise ValueError(f"scores must be a dict, got {type(scores).__name__}")

    config = config or RiskConfig()

    results = {}
    continuous_scores = {}

    for key, value in scores.items():
        # Skip non-numeric values gracefully
        try:
            float(value)
        except (TypeError, ValueError):
            continue

        thresholds = config.per_key.get(key, config.default)
        invert = key in config.invert_keys
        weight = config.weights.get(key, 1.0)

        # -------------------------
        # uncertainty
        # -------------------------
        uncertainty = None

        if probabilities and key in probabilities:
            uncertainty = _entropy(probabilities[key])

        # -------------------------
        # score computation
        # -------------------------
        risk_score = compute_risk_score(
            value,
            invert=invert,
            uncertainty=uncertainty,
            config=config,
        )

        risk_score *= weight

        level = score_to_level(risk_score, thresholds)

        results[key] = level
        continuous_scores[key] = risk_score

    if return_scores:
        return {
            "levels": results,
            "scores": continuous_scores,
        }

    return results


# =========================================================
# BATCH SUPPORT
# =========================================================

def assess_batch(
    batch_scores: List[Dict[str, float]],
    *,
    config: Optional[RiskConfig] = None,
) -> List[Dict[str, str]]:

    return [
        assess_risk_levels(scores, config=config)
        for scores in batch_scores
    ]


# =========================================================
# TRUTHLENS WRAPPER (UPGRADED)
# =========================================================

TRUTHLENS_RISK_KEY_MAP = {
    "truthlens_manipulation_risk": "manipulation_risk",
    "truthlens_credibility_score": "credibility_level",
    "truthlens_final_score": "overall_truthlens_rating",
}


# REC-AG-3: build the default `RiskConfig(invert_keys=...)` exactly
# once at import time instead of allocating a fresh instance on every
# article. Negligible per call but symptomatic of an avoidable hot-path
# allocation under batched inference.
_DEFAULT_TRUTHLENS_CONFIG = RiskConfig()


# CFG-AG-4: bridge for the Pydantic `aggregation_config.RiskConfig`
# (low_threshold, medium_threshold, uncertainty_penalty) into the
# runtime `RiskConfig` consumed here. Use this from the pipeline to
# avoid maintaining two divergent shapes.
def from_pydantic_config(
    pydantic_cfg: Any,
    *,
    invert_keys: Optional[List[str]] = None,
) -> "RiskConfig":
    return RiskConfig(
        default=RiskThresholds(
            low=float(getattr(pydantic_cfg, "low_threshold", 0.3)),
            medium=float(getattr(pydantic_cfg, "medium_threshold", 0.6)),
        ),
        uncertainty_penalty=float(
            getattr(pydantic_cfg, "uncertainty_penalty", 0.2)
        ),
        invert_keys=invert_keys,
    )


def assess_truthlens_risks(
    scores: Dict[str, float],
    *,
    probabilities: Optional[Dict[str, np.ndarray]] = None,
    config: Optional[RiskConfig] = None,
) -> Dict[str, Any]:

    if not isinstance(scores, dict):
        raise ValueError(f"scores must be a dict, got {type(scores).__name__}")

    # PERF-AG-4: previously this called assess_risk_levels three times
    # (once per key) and rebuilt RiskConfig validation each call. Issue
    # one batched call with all present scores instead.
    config = config or _DEFAULT_TRUTHLENS_CONFIG

    present = {
        k_in: scores[k_in]
        for k_in in TRUTHLENS_RISK_KEY_MAP
        if k_in in scores
    }

    if not present:
        return {}

    result = assess_risk_levels(
        present,
        probabilities=probabilities,
        config=config,
        return_scores=False,
    )

    return {
        k_out: result[k_in]
        for k_in, k_out in TRUTHLENS_RISK_KEY_MAP.items()
        if k_in in present and k_in in result
    }