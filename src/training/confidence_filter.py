"""Confidence-based sample filter — Label Noise Amplification fix.

Problem
-------
When a dataset is weakly labelled, samples whose labels are essentially
random noise still flow through the shared encoder and corrupt its
representations.  In multi-task training the damage is amplified because
the shared encoder is updated by *every* task simultaneously — a noisy
emotion label therefore also degrades the bias and propaganda heads.

Solution
--------
Before each backward pass, measure the model's *own confidence* on each
sample in the batch (across all active task heads).  Samples on which
every head is near-uniform (high entropy → low confidence) are likely
mislabelled or out-of-domain.  Scale down their contribution to the loss
so they contribute weaker gradients to the shared encoder.

Design
------
Two gating modes are supported:

``hard``
    Samples with mean confidence < ``min_confidence`` are fully zeroed;
    the per-task loss is scaled by the fraction of kept samples.
    Gate factor ∈ {0, ..., 1} (proportion of batch kept).

``soft``
    Each sample's contribution is weighted by its mean confidence.
    Gate factor = mean(confidence_i) over the batch ∈ (0, 1].

The gate factor is a *scalar* that is applied to the already-computed
per-task scalar loss inside ``MultiTaskLoss.forward()``.  This avoids
touching TaskLossRouter or changing the reduction mode of any loss
function — the filter integrates as a post-hoc multiplicative rescaling.

Integration
-----------
Attach to ``MultiTaskLoss`` via::

    filter = ConfidenceFilter(ConfidenceFilterConfig(min_confidence=0.4))
    loss_module.attach_confidence_filter(filter)

When attached, ``MultiTaskLoss.forward()`` automatically calls
``filter.compute_gate_factor(logits, task_types)`` and multiplies
every per-task loss by the returned scalar.

Threading note
--------------
``ConfidenceFilter`` is stateless between calls (all state lives in the
returned tensor / scalar).  It is therefore safe to share across
DataLoader workers.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

class ConfidenceFilterConfig:
    """Tuning knobs for :class:`ConfidenceFilter`.

    Parameters
    ----------
    min_confidence:
        Threshold in [0, 1].  Samples with mean confidence below this
        value are down-weighted (hard mode) or contribute less via their
        raw confidence score (soft mode).  Typical values: 0.3–0.5.
        Setting to 0.0 effectively disables filtering.
    mode:
        ``"hard"`` — binary keep / zero gate (gate_factor = kept / B).
        ``"soft"`` — continuous weight = mean(confidence_i).
    min_gate_factor:
        Floor on the returned gate factor so the loss is never zeroed
        entirely (prevents the degenerate case where a fully noisy batch
        produces a zero-gradient step that looks like convergence).
        Default 0.05.
    log_every:
        Log the filter statistics every ``log_every`` calls.
        Set to 0 to disable periodic logging.
    task_types:
        Optional mapping of task name → ``"multi_class"`` |
        ``"multilabel"`` | ``"binary"`` | ``"regression"``.  Used to
        select the right entropy formula.  Defaults to ``"multi_class"``
        for unknown tasks.
    """

    def __init__(
        self,
        min_confidence: float = 0.35,
        mode: str = "soft",
        min_gate_factor: float = 0.05,
        log_every: int = 200,
        task_types: Optional[Dict[str, str]] = None,
    ) -> None:
        if not (0.0 <= min_confidence <= 1.0):
            raise ValueError(
                f"min_confidence must be in [0, 1] (got {min_confidence})"
            )
        if mode not in ("hard", "soft"):
            raise ValueError(f"mode must be 'hard' or 'soft' (got {mode!r})")
        if not (0.0 <= min_gate_factor <= 1.0):
            raise ValueError(
                f"min_gate_factor must be in [0, 1] (got {min_gate_factor})"
            )

        self.min_confidence = float(min_confidence)
        self.mode = mode
        self.min_gate_factor = float(min_gate_factor)
        self.log_every = int(log_every)
        self.task_types: Dict[str, str] = task_types or {}


# =========================================================
# FILTER
# =========================================================

class ConfidenceFilter:
    """Computes a scalar gate factor from per-sample model confidence.

    Parameters
    ----------
    config:
        Tuning parameters; :class:`ConfidenceFilterConfig` defaults
        applied when ``None``.
    """

    def __init__(
        self,
        config: Optional[ConfidenceFilterConfig] = None,
    ) -> None:
        self.cfg = config or ConfidenceFilterConfig()

        self._call_count: int = 0
        self._running_kept: float = 0.0
        self._running_total: float = 0.0

        logger.info(
            "ConfidenceFilter | mode=%s | min_confidence=%.3f | "
            "min_gate_factor=%.3f",
            self.cfg.mode,
            self.cfg.min_confidence,
            self.cfg.min_gate_factor,
        )

    # ----------------------------------------------------------------
    # LOW-LEVEL ENTROPY UTILITIES
    # ----------------------------------------------------------------

    @staticmethod
    def _multiclass_entropy(logits: torch.Tensor) -> torch.Tensor:
        """Per-sample entropy from multiclass logits (B, C) → (B,).

        Returns entropy normalised to [0, 1] by dividing by log(C).
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()
        raw_entropy = -(probs * log_probs).sum(dim=-1)           # (B,)
        max_entropy = math.log(max(logits.size(-1), 2))
        return (raw_entropy / max_entropy).clamp(0.0, 1.0)       # normalised

    @staticmethod
    def _multilabel_entropy(logits: torch.Tensor) -> torch.Tensor:
        """Per-sample average binary entropy from multilabel logits (B, L) → (B,).

        Returns mean binary entropy normalised to [0, 1].
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        p = torch.sigmoid(logits.float()).clamp(1e-7, 1.0 - 1e-7)
        h = -(p * p.log() + (1 - p) * (1 - p).log())            # (B, L)
        max_binary_entropy = math.log(2)
        return (h.mean(dim=-1) / max_binary_entropy).clamp(0.0, 1.0)  # (B,)

    # ----------------------------------------------------------------
    # PER-SAMPLE CONFIDENCE
    # ----------------------------------------------------------------

    def compute_sample_confidence(
        self,
        logits: Dict[str, torch.Tensor],
        task_types: Optional[Dict[str, str]] = None,
    ) -> torch.Tensor:
        """Compute per-sample mean confidence from all active task heads.

        Confidence is defined as ``1 - normalised_entropy``.  High
        confidence → model is certain about this sample (likely clean
        label).  Low confidence → model is near-uniform (likely noisy or
        out-of-domain).

        Parameters
        ----------
        logits:
            Per-task logit dict ``{task: (B, C)}`` or ``{task: (B,)}``.
        task_types:
            Optional override for per-task type inference.  Keys not
            present fall back to ``self.cfg.task_types`` then
            ``"multi_class"``.

        Returns
        -------
        (B,) confidence tensor on CPU in [0, 1].  Returns ones when
        ``logits`` is empty (no filtering applied).
        """
        if not logits:
            return torch.ones(1)

        merged_types: Dict[str, str] = {**self.cfg.task_types, **(task_types or {})}

        entropy_terms: list[torch.Tensor] = []
        B: Optional[int] = None

        for task, t_logits in logits.items():
            ttype = merged_types.get(task, "multi_class")
            canonical = ttype.replace("_", "").lower()

            with torch.no_grad():
                if canonical in ("multilabel", "binary"):
                    h = self._multilabel_entropy(t_logits)
                elif canonical == "regression":
                    # No meaningful entropy for regression — treat as fully
                    # confident (entropy 0) so regression tasks don't dilute
                    # the filter for classification heads.
                    if B is None:
                        B = t_logits.size(0) if t_logits.dim() >= 1 else 1
                    h = torch.zeros(B)
                else:
                    h = self._multiclass_entropy(t_logits)

            if B is None:
                B = h.size(0)

            entropy_terms.append(h.cpu())

        if not entropy_terms:
            return torch.ones(B or 1)

        mean_entropy = torch.stack(entropy_terms, dim=-1).mean(dim=-1)  # (B,)
        confidence = 1.0 - mean_entropy                                  # (B,)
        return confidence.clamp(0.0, 1.0)

    # ----------------------------------------------------------------
    # GATE FACTOR
    # ----------------------------------------------------------------

    def compute_gate_factor(
        self,
        logits: Dict[str, torch.Tensor],
        task_types: Optional[Dict[str, str]] = None,
    ) -> float:
        """Compute the scalar gate factor for this batch.

        In ``"hard"`` mode: gate = fraction of samples whose confidence
        exceeds ``min_confidence`` (if none qualify, ``min_gate_factor``
        is returned rather than zero).

        In ``"soft"`` mode: gate = mean(confidence_i) across all samples
        in the batch.  Samples that are confidently predicted pull the
        gate up; noisy samples pull it down.

        Parameters
        ----------
        logits:
            Per-task logit dict (same as ``MultiTaskLoss.forward``).
        task_types:
            Optional task-type mapping forwarded to
            ``compute_sample_confidence``.

        Returns
        -------
        float gate factor in [min_gate_factor, 1.0].
        """
        confidence = self.compute_sample_confidence(logits, task_types)
        B = float(confidence.size(0))

        if self.cfg.mode == "hard":
            kept = float((confidence >= self.cfg.min_confidence).sum())
            gate = kept / max(B, 1.0)
            self._running_kept += kept
        else:
            gate = float(confidence.mean())
            self._running_kept += float(
                (confidence >= self.cfg.min_confidence).sum()
            )

        self._running_total += B
        self._call_count += 1

        gate = max(gate, self.cfg.min_gate_factor)

        if (
            self.cfg.log_every > 0
            and self._call_count % self.cfg.log_every == 0
        ):
            kept_pct = 100.0 * self._running_kept / max(self._running_total, 1.0)
            logger.info(
                "ConfidenceFilter | step=%d | gate=%.4f | "
                "kept=%.1f%% (last %d calls)",
                self._call_count,
                gate,
                kept_pct,
                self.cfg.log_every,
            )
            self._running_kept = 0.0
            self._running_total = 0.0

        return gate

    # ----------------------------------------------------------------
    # STATS
    # ----------------------------------------------------------------

    def get_stats(self) -> Dict[str, float]:
        """Return running filter statistics for monitoring."""
        kept_pct = (
            100.0 * self._running_kept / max(self._running_total, 1.0)
        )
        return {
            "filter_call_count": float(self._call_count),
            "filter_kept_pct": kept_pct,
            "filter_mode": float({"hard": 0, "soft": 1}.get(self.cfg.mode, -1)),
            "filter_min_confidence": self.cfg.min_confidence,
        }

    def reset_stats(self) -> None:
        """Reset rolling statistics (call between epochs if desired)."""
        self._running_kept = 0.0
        self._running_total = 0.0

    def __repr__(self) -> str:
        return (
            f"ConfidenceFilter(mode={self.cfg.mode!r}, "
            f"min_confidence={self.cfg.min_confidence:.3f}, "
            f"calls={self._call_count})"
        )
