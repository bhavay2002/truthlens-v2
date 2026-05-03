"""
HybridScorer — blend neural and rule-based credibility scores.

Spec §7 (Aggregation Engine v2).

Formula
-------
    final_score = α * neural_score + (1 − α) * rule_score

where α is either:

Static (spec §7.1 — ``"static"`` mode)
    α = config.alpha  (default 0.7)

Dynamic / confidence-based (spec §7.2 — ``"dynamic"`` mode)
    α = clamp(base_alpha × mean_confidence, min_alpha, max_alpha)

    Intuition: when all task heads are confident, the neural score
    learned on those high-confidence examples is trustworthy →
    alpha stays close to ``base_alpha``. When the upstream models
    are uncertain (high entropy, low max-prob), fall back toward
    the interpretable rule-based score by shrinking alpha.

Fallback (spec §9 — availability guarantee)
    If ``neural_score`` is ``None`` (disabled or runtime error),
    ``HybridScorer.score`` returns a pure rule-based result with
    ``alpha=0.0``.

Usage
-----
    scorer = HybridScorer(alpha=0.7, dynamic=True)

    result = scorer.score(
        neural_score=0.82,
        rule_score=0.65,
        mean_confidence=0.78,
    )
    # {'final': 0.773, 'neural': 0.82, 'rule': 0.65, 'alpha': 0.546, 'mode': 'dynamic'}
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =========================================================
# SCORER
# =========================================================

class HybridScorer:
    """Blend neural and rule-based scores with static or dynamic alpha.

    Parameters
    ----------
    alpha : float
        Base neural weight in [0, 1]. Default 0.7 (spec §7.1).
    dynamic : bool
        If ``True``, alpha is scaled by ``mean_confidence`` (spec §7.2).
        If ``False``, always uses the fixed ``alpha``.
    min_alpha : float
        Minimum alpha used in dynamic mode (fallback floor). Default 0.2.
    max_alpha : float
        Maximum alpha used in dynamic mode (cap for very confident inputs).
        Default 0.9.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        dynamic: bool = True,
        min_alpha: float = 0.2,
        max_alpha: float = 0.9,
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1] (got {alpha})")
        if not (0.0 <= min_alpha <= max_alpha <= 1.0):
            raise ValueError("Require 0 ≤ min_alpha ≤ max_alpha ≤ 1")

        self.alpha     = float(alpha)
        self.dynamic   = dynamic
        self.min_alpha = float(min_alpha)
        self.max_alpha = float(max_alpha)

        logger.info(
            "HybridScorer | alpha=%.2f dynamic=%s min=%.2f max=%.2f",
            self.alpha, self.dynamic, self.min_alpha, self.max_alpha,
        )

    # -----------------------------------------------------------------------
    # MAIN
    # -----------------------------------------------------------------------

    def score(
        self,
        neural_score: Optional[float],
        rule_score: float,
        *,
        mean_confidence: Optional[float] = None,
        task_confidences: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Compute the hybrid final score.

        Parameters
        ----------
        neural_score : float or None
            Credibility score from the NeuralAggregator (sigmoid output).
            Pass ``None`` to indicate neural is unavailable → pure rule.
        rule_score : float
            Credibility score from ``TruthLensScoreCalculator``.
        mean_confidence : float, optional
            Mean task confidence (used in dynamic mode). If not provided
            and ``task_confidences`` is given, it is computed as the
            mean over that dict.
        task_confidences : dict, optional
            Per-task confidence scores. Used to compute ``mean_confidence``
            when it is not supplied directly.

        Returns
        -------
        dict with keys:
            ``final``   — blended credibility score in [0, 1]
            ``neural``  — neural_score (or None)
            ``rule``    — rule_score
            ``alpha``   — effective alpha used
            ``mode``    — ``"neural_disabled"`` / ``"static"`` / ``"dynamic"``
        """
        rule_score = float(np.clip(_safe_float(rule_score), 0.0, 1.0))

        # ── Fallback: neural unavailable ──────────────────────────────────
        if neural_score is None:
            return {
                "final":  rule_score,
                "neural": None,
                "rule":   rule_score,
                "alpha":  0.0,
                "mode":   "neural_disabled",
            }

        neural_score = float(np.clip(_safe_float(neural_score), 0.0, 1.0))

        # ── Compute effective alpha ───────────────────────────────────────
        if self.dynamic:
            conf = _resolve_confidence(mean_confidence, task_confidences)
            effective_alpha, mode = self._dynamic_alpha(conf), "dynamic"
        else:
            effective_alpha, mode = self.alpha, "static"

        # ── Blend ─────────────────────────────────────────────────────────
        final = effective_alpha * neural_score + (1.0 - effective_alpha) * rule_score
        final = float(np.clip(final, 0.0, 1.0))

        logger.debug(
            "HybridScorer | α=%.3f mode=%s neural=%.3f rule=%.3f → %.3f",
            effective_alpha, mode, neural_score, rule_score, final,
        )

        return {
            "final":  final,
            "neural": neural_score,
            "rule":   rule_score,
            "alpha":  effective_alpha,
            "mode":   mode,
        }

    def score_batch(
        self,
        neural_scores,
        rule_scores,
        *,
        mean_confidences=None,
    ) -> list:
        """Vectorised wrapper over :meth:`score`.

        Parameters
        ----------
        neural_scores : list / array of float or None
        rule_scores : list / array of float
        mean_confidences : list / array of float, optional

        Returns
        -------
        List of result dicts (same as :meth:`score`).
        """
        n = len(rule_scores)
        neural_scores = list(neural_scores) if neural_scores is not None else [None] * n
        mean_confidences = list(mean_confidences) if mean_confidences is not None else [None] * n

        # PERF-AG-BATCH: fast vectorised path for the common case where all
        # neural scores are available and alpha is static (no per-sample
        # confidence look-up needed). Collapses N Python call frames into a
        # single NumPy BLAS operation.
        if (
            not self.dynamic
            and all(ns is not None for ns in neural_scores)
        ):
            ns_arr = np.clip(
                np.array([_safe_float(ns) for ns in neural_scores], dtype=np.float64),
                0.0, 1.0,
            )
            rs_arr = np.clip(
                np.array([_safe_float(rs) for rs in rule_scores], dtype=np.float64),
                0.0, 1.0,
            )
            finals = np.clip(
                self.alpha * ns_arr + (1.0 - self.alpha) * rs_arr,
                0.0, 1.0,
            )
            return [
                {
                    "final":  float(f),
                    "neural": float(ns),
                    "rule":   float(rs),
                    "alpha":  self.alpha,
                    "mode":   "static",
                }
                for f, ns, rs in zip(finals, ns_arr, rs_arr)
            ]

        return [
            self.score(ns, rs, mean_confidence=mc)
            for ns, rs, mc in zip(neural_scores, rule_scores, mean_confidences)
        ]

    # -----------------------------------------------------------------------
    # ALPHA STRATEGIES
    # -----------------------------------------------------------------------

    def _dynamic_alpha(self, mean_confidence: float) -> float:
        """Scale alpha by mean confidence, clamped to [min_alpha, max_alpha].

        α = clamp(base_alpha × mean_confidence + min_alpha × (1 − mean_confidence),
                  min_alpha, max_alpha)

        This is a convex combination of base_alpha and min_alpha:
        • confidence = 1.0 → α = base_alpha
        • confidence = 0.0 → α = min_alpha  (fallback toward rule)
        """
        alpha_raw = (
            self.alpha     * mean_confidence
            + self.min_alpha * (1.0 - mean_confidence)
        )
        return float(np.clip(alpha_raw, self.min_alpha, self.max_alpha))

    # -----------------------------------------------------------------------
    # INSPECTION
    # -----------------------------------------------------------------------

    def expected_alpha(self, confidence: float) -> float:
        """Return the alpha that would be applied for a given confidence.

        Useful for logging / tuning the min/max_alpha bounds without a
        full forward pass.
        """
        if not self.dynamic:
            return self.alpha
        return self._dynamic_alpha(float(np.clip(confidence, 0.0, 1.0)))

    def __repr__(self) -> str:
        return (
            f"HybridScorer(alpha={self.alpha:.2f}, dynamic={self.dynamic}, "
            f"min={self.min_alpha:.2f}, max={self.max_alpha:.2f})"
        )


# =========================================================
# HELPERS
# =========================================================

def _safe_float(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def _resolve_confidence(
    mean_confidence: Optional[float],
    task_confidences: Optional[Dict[str, float]],
) -> float:
    """Return a scalar confidence in [0, 1]."""
    if mean_confidence is not None:
        return float(np.clip(_safe_float(mean_confidence), 0.0, 1.0))

    if task_confidences:
        vals = [
            float(v) for v in task_confidences.values()
            if isinstance(v, (int, float)) and math.isfinite(float(v))
        ]
        if vals:
            return float(np.clip(sum(vals) / len(vals), 0.0, 1.0))

    return 0.5   # neutral fallback when no confidence info available
