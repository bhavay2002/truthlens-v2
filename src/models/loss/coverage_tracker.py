from __future__ import annotations

import logging
from typing import Dict

import torch

logger = logging.getLogger(__name__)


class EMACoverageTracker:
    """
    EMA-based task coverage tracker for multi-task imbalance correction.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        floor: float = 0.05,
        cap: float = 10.0,
        enabled: bool = True,
        warmup_steps: int = 0,  # ✅ optional (no behavior change by default)
        temper: float = 0.5,
    ) -> None:

        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0,1], got {alpha}")

        if floor <= 0.0:
            raise ValueError(f"floor must be positive, got {floor}")

        if cap < 1.0:
            raise ValueError(f"cap must be >= 1.0, got {cap}")

        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

        if not 0.0 < temper <= 1.0:
            raise ValueError(
                f"temper must be in (0, 1], got {temper}"
            )

        self.alpha = float(alpha)
        self.floor = float(floor)
        self.cap = float(cap)
        self.enabled = enabled
        self.warmup_steps = int(warmup_steps)
        # N3: ``temper`` is the exponent applied to the
        # ``target_cov / cov`` ratio when deriving the per-task
        # multiplier. ``temper=1.0`` recovers the old behaviour
        # (multiplier = target / cov, capped); ``temper=0.5`` (the
        # default) is sqrt-tempering, which is what the audit asks
        # for: it dampens early-step swings when one task dominates
        # the EMA, while still removing imbalance asymptotically.
        self.temper = float(temper)

        # per-task EMA coverage
        self._coverage: Dict[str, float] = {}

        # diagnostics
        self._steps: Dict[str, int] = {}

        logger.info(
            "EMACoverageTracker initialized | alpha=%.3f floor=%.3f cap=%.2f enabled=%s warmup=%d",
            self.alpha,
            self.floor,
            self.cap,
            self.enabled,
            self.warmup_steps,
        )

    # =========================================================
    # UPDATE COVERAGE
    # =========================================================

    def update(
        self,
        task: str,
        labels: torch.Tensor,
        ignore_index: float | int = -100,
        task_type: str = "multiclass",
    ) -> None:

        if not self.enabled:
            return

        # Normalize task_type to the canonical form used elsewhere in the
        # loss stack (see ``TaskLossConfig.__post_init__`` in
        # ``src/models/loss/multitask_loss.py`` — it folds
        # ``multi_class``/``multi-class`` → ``multiclass`` and
        # ``multi_label``/``multi-label`` → ``multilabel``). Accept both
        # spellings here so legacy callers don't crash.
        canonical_task_type = str(task_type).replace("_", "").replace("-", "").lower()

        if not torch.is_tensor(labels) or labels.numel() == 0:
            has_label = False

        else:
            if canonical_task_type == "multiclass":
                has_label = bool(labels.ne(ignore_index).any())

            elif canonical_task_type in {"binary", "multilabel"}:
                has_label = bool(labels.ne(float(ignore_index)).any())

            elif canonical_task_type == "regression":
                has_label = True

            else:
                raise ValueError(f"Unknown task_type: {task_type}")

        prev = self._coverage.get(task, 0.0)

        new_cov = (
            self.alpha * (1.0 if has_label else 0.0)
            + (1.0 - self.alpha) * prev
        )

        self._coverage[task] = new_cov
        self._steps[task] = self._steps.get(task, 0) + 1

    # =========================================================
    # APPLY WEIGHTING
    # =========================================================

    def _target_coverage(self) -> float:
        """Mean of the per-task EMA coverage values, floored.

        N3: anchoring multipliers to the *mean* coverage rather than
        ``1.0`` keeps them centred around 1 (so the total loss scale
        is preserved) and gives an interpretable target — "the typical
        task". Falls back to ``floor`` when no task has been seen yet.
        """
        if not self._coverage:
            return self.floor
        return max(
            sum(self._coverage.values()) / len(self._coverage),
            self.floor,
        )

    def _multiplier_for(self, cov: float, target_cov: float) -> float:
        """Sqrt-tempered (default) multiplier with a hard cap.

        Implements ``min((target / cov) ** temper, cap)``. The temper
        exponent < 1 dampens swings when ``cov`` is far below
        ``target`` while still removing imbalance asymptotically.
        """
        cov_safe = max(cov, self.floor)
        ratio = target_cov / cov_safe
        return min(ratio ** self.temper, self.cap)

    def weight(
        self,
        task: str,
        loss: torch.Tensor,
    ) -> torch.Tensor:

        if not self.enabled:
            return loss

        if not torch.is_tensor(loss):
            raise TypeError("loss must be tensor")

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected for task '{task}'")

        step = self._steps.get(task, 0)

        # ---- warmup (no aggressive boosting early) ----
        if step < self.warmup_steps:
            return loss

        target_cov = self._target_coverage()
        multiplier = self._multiplier_for(
            self._coverage.get(task, 0.0),
            target_cov,
        )

        # ---- device-safe tensor ----
        multiplier_tensor = loss.new_tensor(multiplier)

        weighted = loss * multiplier_tensor

        return weighted

    # =========================================================
    # UTILITIES
    # =========================================================

    def get_coverage(self) -> Dict[str, float]:
        return dict(self._coverage)

    def get_multipliers(self) -> Dict[str, float]:
        target_cov = self._target_coverage()
        return {
            t: self._multiplier_for(cov, target_cov)
            for t, cov in self._coverage.items()
        }

    def reset(self) -> None:
        self._coverage.clear()
        self._steps.clear()

        logger.info("EMACoverageTracker state reset")