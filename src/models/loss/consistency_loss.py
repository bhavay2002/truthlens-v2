"""
ConsistencyLoss — cross-task representation consistency regularizer.

Spec §2.2 (Training Pipeline Upgrade / Phase 3 objective).

Formulation
-----------

    L_consistency = KL(p_bias || p_ideology)
                  + λ_cos * (1 − cos(H_emotion, H_bias))

where:
    p_bias, p_ideology  — softmax distributions over bias / ideology logits
    H_emotion, H_bias   — post-interaction task representations (B, D)
                          exposed via ``outputs["task_representations"]``
    λ_cos               — weight for the cosine alignment term (default 1.0)

Only terms for which the required keys are present in ``outputs`` are
computed; the function degrades gracefully when the model does not expose
task representations (i.e. returns only the KL term).

Usage (Phase 3 training loop)
------------------------------

    loss_fn = ConsistencyLoss(kl_tasks=("bias", "ideology"),
                              cosine_pairs=[("emotion", "bias")])

    for batch in loader:
        outputs = model(**batch)
        L_task  = loss_engine.compute(outputs, batch)[0]
        L_cons  = loss_fn(outputs)
        L_total = L_task + state.consistency_lambda * L_cons
        L_total.backward()

Design notes
------------
* KL divergence is computed as ``F.kl_div(log_p, q, reduction="batchmean")``
  which expects log-probabilities for the first argument.
* Cosine similarity is computed as ``F.cosine_similarity(H_a, H_b)``
  and converted to distance as ``1 - sim``, averaged over the batch.
* All operations are AMP-safe (no explicit float() casts needed inside
  the loss — inputs are cast to float32 before log/softmax for numerical
  stability, then converted back).
* Tasks absent from ``task_logits`` or ``task_representations`` are
  silently skipped so the loss never crashes during mixed-corpus training.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ConsistencyLoss(nn.Module):
    """Cross-task distribution and representation alignment loss.

    Parameters
    ----------
    kl_tasks:
        Pair of task names (p, q) for the KL term: KL(p_task_a || p_task_b).
        Only multi-class tasks are meaningful here (softmax distributions).
    cosine_pairs:
        List of (task_a, task_b) pairs whose post-interaction representations
        are aligned via cosine distance. Requires the model to emit
        ``outputs["task_representations"]``.
    kl_weight:
        Relative weight of the KL term (before the outer consistency_lambda
        applied in the training loop).
    cosine_weight:
        Relative weight of the cosine alignment term.
    temperature:
        Temperature applied to logits before softmax for the KL term.
        T > 1 softens distributions (recommended for cross-task alignment
        to avoid overly confident anchor distributions). Default 2.0.
    reduction:
        Reduction applied by ``F.kl_div``. ``"batchmean"`` (default)
        divides by batch size — correct for KL divergence.
    """

    def __init__(
        self,
        kl_tasks: Tuple[str, str] = ("bias", "ideology"),
        cosine_pairs: Optional[List[Tuple[str, str]]] = None,
        kl_weight: float = 1.0,
        cosine_weight: float = 1.0,
        temperature: float = 2.0,
        reduction: str = "batchmean",
    ) -> None:
        super().__init__()

        if len(kl_tasks) != 2:
            raise ValueError(
                f"kl_tasks must be a 2-tuple of task names (got {kl_tasks!r})"
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0 (got {temperature})")
        if reduction not in {"batchmean", "mean", "sum", "none"}:
            raise ValueError(f"Unsupported reduction: {reduction!r}")

        self.kl_tasks = tuple(kl_tasks)
        self.cosine_pairs: List[Tuple[str, str]] = list(cosine_pairs or [("emotion", "bias")])
        self.kl_weight = float(kl_weight)
        self.cosine_weight = float(cosine_weight)
        self.temperature = float(temperature)
        self.reduction = reduction

        logger.info(
            "ConsistencyLoss | kl_tasks=%s | cosine_pairs=%s | "
            "T=%.1f | kl_w=%.2f | cos_w=%.2f",
            self.kl_tasks, self.cosine_pairs,
            self.temperature, self.kl_weight, self.cosine_weight,
        )

    # -----------------------------------------------------------------------
    # FORWARD
    # -----------------------------------------------------------------------

    def forward(self, outputs: Dict[str, Any]) -> torch.Tensor:
        """Compute consistency loss from model forward outputs.

        Parameters
        ----------
        outputs:
            Model output dict. Reads:
              * ``outputs["task_logits"]``          — Dict[str, Tensor(B, C)]
              * ``outputs["task_representations"]`` — Dict[str, Tensor(B, D)]
                (optional; cosine term skipped if absent)

        Returns
        -------
        Scalar consistency loss tensor. Returns 0.0 (no-grad) when no
        computable terms are found (graceful degradation).
        """
        task_logits: Dict[str, torch.Tensor] = outputs.get("task_logits", {})
        task_reprs: Dict[str, torch.Tensor] = outputs.get(
            "task_representations", {}
        )

        terms: List[torch.Tensor] = []

        # ── KL term ──────────────────────────────────────────────────────
        kl = self._kl_term(task_logits)
        if kl is not None:
            terms.append(self.kl_weight * kl)

        # ── Cosine alignment terms ────────────────────────────────────────
        cos_total = self._cosine_term(task_reprs)
        if cos_total is not None:
            terms.append(self.cosine_weight * cos_total)

        if not terms:
            # No computable terms — return a detached zero
            device = _first_device(task_logits) or torch.device("cpu")
            return torch.tensor(0.0, device=device, requires_grad=False)

        return torch.stack(terms).sum()

    # -----------------------------------------------------------------------
    # TERM HELPERS
    # -----------------------------------------------------------------------

    def _kl_term(
        self,
        task_logits: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """KL(p_a || p_b) using softmax(logits / T) distributions.

        Reads ``self.kl_tasks = (task_a, task_b)``.
        Returns ``None`` if either task is missing.
        """
        task_a, task_b = self.kl_tasks

        if task_a not in task_logits or task_b not in task_logits:
            logger.debug(
                "ConsistencyLoss KL: task(s) %s or %s not in task_logits — "
                "skipping KL term.", task_a, task_b,
            )
            return None

        logits_a = task_logits[task_a].float()  # (B, C_a)
        logits_b = task_logits[task_b].float()  # (B, C_b)

        # Temperature-softened distributions
        p_a = F.softmax(logits_a / self.temperature, dim=-1)  # (B, C_a)
        p_b = F.softmax(logits_b / self.temperature, dim=-1)  # (B, C_b)

        # Align dimensions — take the minimum number of classes
        min_c = min(p_a.size(-1), p_b.size(-1))
        p_a = p_a[:, :min_c]
        p_b = p_b[:, :min_c]

        # Re-normalise after truncation so distributions sum to 1
        p_a = p_a / p_a.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        p_b = p_b / p_b.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        # KL(p_a || p_b): F.kl_div expects log(q) as first arg
        log_p_a = p_a.clamp_min(1e-9).log()  # (B, C)
        kl = F.kl_div(log_p_a, p_b, reduction=self.reduction)

        return kl

    def _cosine_term(
        self,
        task_reprs: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Mean cosine distance over ``self.cosine_pairs``.

        ``cosine_distance = 1 - cosine_similarity``, averaged across
        the batch and across all pairs. Returns ``None`` if no pairs
        have both tasks present.
        """
        pair_distances: List[torch.Tensor] = []

        for task_a, task_b in self.cosine_pairs:
            if task_a not in task_reprs or task_b not in task_reprs:
                logger.debug(
                    "ConsistencyLoss cosine: task(s) %s or %s not in "
                    "task_representations — skipping pair.",
                    task_a, task_b,
                )
                continue

            H_a = task_reprs[task_a].float()  # (B, D)
            H_b = task_reprs[task_b].float()  # (B, D)

            # cosine_similarity: (B,) in [-1, 1]
            sim = F.cosine_similarity(H_a, H_b, dim=-1)
            # cosine distance: (B,) in [0, 2] → normalise to [0, 1]
            dist = (1.0 - sim) / 2.0
            pair_distances.append(dist.mean())

        if not pair_distances:
            return None

        return torch.stack(pair_distances).mean()

    # -----------------------------------------------------------------------
    # UTILITIES
    # -----------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"kl_tasks={self.kl_tasks}, "
            f"cosine_pairs={self.cosine_pairs}, "
            f"T={self.temperature:.1f}, "
            f"kl_w={self.kl_weight:.2f}, "
            f"cos_w={self.cosine_weight:.2f}"
        )


# =========================================================
# HELPERS
# =========================================================

def _first_device(
    tensor_dict: Dict[str, torch.Tensor],
) -> Optional[torch.device]:
    """Return the device of the first tensor in the dict, or None."""
    for v in tensor_dict.values():
        if isinstance(v, torch.Tensor):
            return v.device
    return None
