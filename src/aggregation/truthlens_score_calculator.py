from __future__ import annotations

import logging
import math
from typing import Dict, Any, Optional, Mapping

import numpy as np

# CFG-AG-1: pull the canonical group definitions from
# `aggregation_config` so the calculator and the weight manager can
# never disagree on which keys belong to which group.
from .aggregation_config import WEIGHT_GROUPS

logger = logging.getLogger(__name__)
EPS = 1e-12


# =========================================================
# CONTRACT
# =========================================================

# CRIT-AG-3: the calculator's composite formulas reference these
# section names. They are now declared in one place and validated at
# `compute_scores` time so missing sections become a debug log line
# instead of a silent zero in the final score.
REQUIRED_SECTIONS = (
    "bias", "emotion", "narrative",
    "discourse", "graph", "ideology", "analysis",
)

_MANIPULATION_KEYS = WEIGHT_GROUPS["manipulation"]
_CREDIBILITY_KEYS  = WEIGHT_GROUPS["credibility"]
_FINAL_KEYS        = WEIGHT_GROUPS["final"]


# CRIT-AG-4: the previous code looked up `graph_centrality_mean` which
# is never produced by any upstream component. Accept the actual keys
# emitted by `src/graph/graph_analysis.py` and `feature_mapper`.
_GRAPH_SIGNAL_KEYS = (
    "centrality_mean",
    "graph_density",
    "avg_centrality",
    "consistency",
    "graph_consistency",
    "graph_centrality_mean",
)


# =========================================================
# SCORE VECTOR UTILITY
# =========================================================

# Canonical ordering for truthlens_score_vector — stable across versions.
_SCORE_VECTOR_KEYS: tuple = (
    "truthlens_bias_score",
    "truthlens_emotion_score",
    "truthlens_narrative_score",
    "truthlens_discourse_score",
    "truthlens_graph_score",
    "truthlens_ideology_score",
    "truthlens_manipulation_risk",
    "truthlens_credibility_score",
    "truthlens_final_score",
)


def truthlens_score_vector(scores: Dict[str, float]) -> "np.ndarray":
    """Return a fixed-order float32 numpy vector from a flat scores dict.

    Parameters
    ----------
    scores : dict
        Must contain every key listed in ``_SCORE_VECTOR_KEYS``.

    Returns
    -------
    np.ndarray, shape (9,), dtype float32

    Raises
    ------
    RuntimeError
        When any required key is absent from *scores*.
    """
    try:
        return np.array([scores[k] for k in _SCORE_VECTOR_KEYS], dtype=np.float32)
    except KeyError as exc:
        raise RuntimeError(f"Missing score key: {exc}") from exc


def _renorm_group(weights: Dict[str, float], keys) -> None:
    total = sum(weights[k] for k in keys if k in weights) + EPS
    for k in keys:
        if k in weights:
            weights[k] = float(weights[k] / total)


# =========================================================
# CORE
# =========================================================

class TruthLensScoreCalculator:

    def __init__(
        self,
        *,
        graph_influence_cap: float = 0.1,
        explanation_blend: float = 0.5,
    ):
        # WGT-AG-2: previously hardcoded inside `compute_scores` (0.1
        # for graph contribution, 0.5 for explanation blend). They are
        # now constructor-injected so they can be wired through
        # `AggregationConfig.fusion`.
        self.graph_influence_cap = float(np.clip(graph_influence_cap, 0.0, 1.0))
        self.explanation_blend = float(np.clip(explanation_blend, 0.0, 1.0))

    # =====================================================
    # MAIN
    # =====================================================

    def compute_scores(
        self,
        profile: Dict[str, Any],
        *,
        weights: Mapping[str, float],
        explanation_scores: Optional[Dict[str, float]] = None,
        # The following are accepted for backwards-compat with callers
        # that still pass them, but they are NOT applied here — see
        # CRIT-AG-7. The same modulation now happens exactly once,
        # inside `WeightManager.get_adaptive_weights`.
        confidence: Optional[Dict[str, float]] = None,
        entropy: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:

        # WGT-AG-3: weights are mandatory — there is no longer a second
        # source of truth (the deleted `ScoreWeights` dataclass) that
        # could silently disagree with `WeightManager.DEFAULT_WEIGHTS`.
        if not weights:
            raise ValueError(
                "TruthLensScoreCalculator.compute_scores requires "
                "explicit `weights` (typically WeightManager output)"
            )

        w = dict(weights)
        _renorm_group(w, _MANIPULATION_KEYS)
        _renorm_group(w, _CREDIBILITY_KEYS)
        _renorm_group(w, _FINAL_KEYS)

        # CRIT-AG-3: surface missing sections as a debug log rather
        # than failing — the calculator stays robust but the absence
        # is no longer invisible to the operator.
        missing = [s for s in REQUIRED_SECTIONS if s not in profile]
        if missing:
            logger.debug(
                "[TruthLensScoreCalculator] missing sections: %s",
                missing,
            )

        # CRIT-AG-4: aggregate any of the recognised graph keys instead
        # of demanding a single string that no upstream component
        # produces.
        graph_section = (
            profile.get("graph", {})
            if isinstance(profile.get("graph"), dict) else {}
        )
        graph_vals = []
        for k in _GRAPH_SIGNAL_KEYS:
            v = graph_section.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                graph_vals.append(float(v))
        graph_signal = float(
            np.clip(sum(graph_vals) / len(graph_vals), 0.0, 1.0)
        ) if graph_vals else 0.0

        section_scores: Dict[str, float] = {}
        section_debug: Dict[str, Any] = {}

        for section, data in profile.items():

            base_val = self._aggregate(data)

            # CRIT-AG-BLEND: the graph correction is a graph-network signal
            # (centrality, density, consistency). Applying it to every section
            # — bias, emotion, narrative, etc. — inflates unrelated scores and
            # confounds the composite formulas. Restrict it to the "graph"
            # section only, where it is semantically meaningful.
            if section == "graph":
                val = base_val + self.graph_influence_cap * graph_signal
            else:
                val = base_val

            debug_info: Dict[str, Any] = {
                "base": base_val,
                "graph_signal": graph_signal,
                "graph_influence": self.graph_influence_cap * graph_signal,
            }

            # CRIT-AG-7: confidence + entropy are NOT re-applied to the
            # value here. They are already baked into `weights` by
            # `WeightManager.get_adaptive_weights`. Doubling them
            # caused (e.g. conf=0.5, ent=0.5) sections to be attenuated
            # by ~0.11 instead of ~0.5, ranking-distorting the output.

            if explanation_scores and section in explanation_scores:
                exp_val = explanation_scores[section]
                if isinstance(exp_val, (int, float)) and math.isfinite(exp_val):
                    exp_score = float(np.clip(exp_val, 0.0, 1.0))
                    blend = self.explanation_blend
                    val = (1.0 - blend) * val + blend * exp_score
                    debug_info["explanation_score"] = exp_score
                    debug_info["explanation_blend"] = blend

            final_val = float(np.clip(val, 0.0, 1.0))
            section_scores[section] = final_val
            section_debug[section] = {**debug_info, "final": final_val}

        # =====================================================
        # COMPOSITE SCORES
        # =====================================================

        manipulation = self._manipulation(section_scores, w)
        credibility  = self._credibility(section_scores, w)
        final_score  = self._final(
            credibility, manipulation, section_scores.get("ideology", 0.0), w
        )

        return {
            "section_scores": section_scores,
            "manipulation_risk": manipulation,
            "credibility_score": credibility,
            "final_score": final_score,
            "debug": {
                "inputs": profile,
                "explanation_scores": explanation_scores,
                "graph_signal": graph_signal,
                "section_breakdown": section_debug,
                "weights_used": w,
            },
        }

    # =====================================================
    # AGGREGATION
    # =====================================================

    @staticmethod
    def _aggregate(section_data: Any) -> float:

        if not isinstance(section_data, dict):
            return 0.0

        # PERF-AG-3: avoid the np.array constructor overhead that
        # dominated this hot path for the typical 1-3 element vectors.
        # Using `math.fsum` keeps the running-sum precision on par
        # with numpy.mean for these sizes.
        #
        # CRIT-AG-SCALE: upstream section dicts can contain raw counts
        # (e.g. {"probability": 0.82, "intensity": 150.0}). Averaging
        # those with probabilities yields garbage (75.4 → clamped to 1.0,
        # masking the real signal). Clamp each value to [0, 1] before the
        # mean so the average stays in the probability space regardless of
        # what the feature extractor emits.
        vals = [
            min(1.0, max(0.0, float(v)))
            for v in section_data.values()
            if isinstance(v, (int, float))
            and not isinstance(v, bool)
            and math.isfinite(v)
        ]

        if not vals:
            return 0.0

        mean = math.fsum(vals) / len(vals)
        if mean < 0.0:
            return 0.0
        if mean > 1.0:
            return 1.0
        return mean

    # =====================================================
    # COMPONENT SCORES
    # =====================================================

    def _manipulation(self, s: Dict[str, float], w: Dict[str, float]) -> float:
        val = (
            w.get("bias", 0.0) * s.get("bias", 0.0) +
            w.get("emotion", 0.0) * s.get("emotion", 0.0) +
            w.get("narrative", 0.0) * s.get("narrative", 0.0) +
            w.get("analysis_influence_manipulation", 0.0) * s.get("analysis", 0.0)
        )
        return float(np.clip(val, 0.0, 1.0))

    def _credibility(self, s: Dict[str, float], w: Dict[str, float]) -> float:
        positive = (
            w.get("discourse", 0.0) * s.get("discourse", 0.0) +
            w.get("graph", 0.0) * s.get("graph", 0.0) +
            w.get("analysis_influence_credibility", 0.0) * s.get("analysis", 0.0)
        )
        # WGT-AG-1: credibility_bias_penalty is a scalar multiplier in
        # [0, 1] — clipping is enforced by WeightManager._clip_scalar_keys.
        penalty = float(np.clip(
            w.get("credibility_bias_penalty", 0.2) * s.get("bias", 0.0),
            0.0, 1.0,
        ))
        return float(np.clip(positive * (1.0 - penalty), 0.0, 1.0))

    def _final(self, c: float, m: float, i: float, w: Dict[str, float]) -> float:
        val = (
            w.get("final_credibility", 0.5) * c +
            w.get("final_manipulation", 0.3) * (1.0 - m) +
            w.get("final_ideology", 0.2) * (1.0 - i)
        )
        return float(np.clip(val, 0.0, 1.0))
