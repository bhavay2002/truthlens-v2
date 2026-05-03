"""DynamicTaskWeightBalancer — Dominant Task overfitting fix.

Problem
-------
In multi-task training one task (most commonly ``emotion`` which has
large, well-labelled datasets) tends to dominate the shared encoder:
its loss is consistently higher in absolute terms, so its gradients are
larger and its head effectively captures most of the encoder's capacity.
The other heads then underfit.

Standard EMA normalisation (``EMALossNormalizer``) dampens this by
dividing each task loss by a running mean, but it operates *before*
the static task weights and does not adapt the weights themselves.  When
one task is structurally larger (e.g. 5-class vs 2-class, or larger
dataset → higher absolute CE) the normaliser alone is insufficient.

Solution
--------
``DynamicTaskWeightBalancer`` is a :class:`~src.models.loss.base_balancer.BaseBalancer`
that:

1. Maintains an EMA of the *raw* per-task loss magnitude.
2. After each step computes a *dominance ratio* for each task:
       dominance_t = ema_loss_t / mean(ema_losses)
3. Computes adaptive inverse-dominance weights:
       raw_weight_t  = 1 / dominance_t^temperature
       norm_weight_t = raw_weight_t * N / sum(raw_weights)
   where N = number of tasks, so the weights always average to 1.0 and
   the total gradient magnitude is not suppressed (unlike simply
   dividing by a large per-task weight).
4. In ``combine(task_losses)`` returns the weighted sum of raw task
   losses using the PREVIOUS step's weights (so the current batch's loss
   isn't used to compute its own weight — avoids circular dependency).
5. Enforces a ``max_weight`` cap (default 3.0) so no single task can
   dominate the combination in the other direction.

Interaction with static task weights
--------------------------------------
``DynamicTaskWeightBalancer`` operates on ``raw_losses`` (before
``TaskLossConfig.weight`` is applied) because it is called from
``MultiTaskLoss.balancer.combine(raw_losses)`` — see the MultiTaskLoss
forward pass.  The static per-task weights in ``LossEngineConfig``
therefore act as a *prior* that is then refined at each step by the
dynamic balancer.  To lean on only the dynamic balancer, set all static
task weights to 1.0.

Integration
-----------
::

    from src.training.dynamic_task_balancer import (
        DynamicTaskBalancerConfig,
        DynamicTaskWeightBalancer,
    )

    balancer = DynamicTaskWeightBalancer(
        tasks=["bias", "emotion", "propaganda", "ideology",
               "clickbait", "narrative_frame"],
        config=DynamicTaskBalancerConfig(ema_alpha=0.05, temperature=0.5),
    )
    loss_engine.attach_balancer(balancer)

The balancer is wired automatically by ``create_multitask_trainer_fn``
when ``settings.loss.use_dynamic_balancer = true`` in ``config.yaml``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch

from src.models.loss.base_balancer import BaseBalancer

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

class DynamicTaskBalancerConfig:
    """Tuning knobs for :class:`DynamicTaskWeightBalancer`.

    Parameters
    ----------
    ema_alpha:
        EMA decay for tracking per-task raw loss magnitude.  Smaller
        values → slower adaptation (smoother but lags).  Larger values
        → faster adaptation (more noise).  Recommended: 0.05–0.15.
    temperature:
        Exponent on the dominance ratio before inverting.  T = 1.0 gives
        full inverse proportionality; T < 1 dampens the correction
        (less aggressive rebalancing); T = 0 disables dynamic
        rebalancing (all weights → 1.0).  Recommended: 0.5.
    max_weight:
        Upper cap on the dynamic weight for any single task.  Prevents
        a task with a near-zero loss from receiving an astronomically
        high weight.  Recommended: 3.0–5.0.
    min_weight:
        Lower floor on the dynamic weight.  Prevents a dominating task
        from being fully suppressed.  Recommended: 0.1–0.3.
    warmup_steps:
        Number of update steps before dynamic weights are applied.
        During warmup the balancer returns uniform weights (all 1.0) so
        the EMA has time to stabilise.  Recommended: 50–200.
    log_every:
        Log current dynamic weights every ``log_every`` steps.
    """

    def __init__(
        self,
        ema_alpha: float = 0.1,
        temperature: float = 0.5,
        max_weight: float = 3.0,
        min_weight: float = 0.1,
        warmup_steps: int = 100,
        log_every: int = 100,
    ) -> None:
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1] (got {ema_alpha})")
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0 (got {temperature})")
        if max_weight < 1.0:
            raise ValueError(f"max_weight must be >= 1.0 (got {max_weight})")
        if not (0.0 <= min_weight <= 1.0):
            raise ValueError(f"min_weight must be in [0, 1] (got {min_weight})")

        self.ema_alpha = float(ema_alpha)
        self.temperature = float(temperature)
        self.max_weight = float(max_weight)
        self.min_weight = float(min_weight)
        self.warmup_steps = int(warmup_steps)
        self.log_every = int(log_every)


# =========================================================
# BALANCER
# =========================================================

class DynamicTaskWeightBalancer(BaseBalancer):
    """EMA-based adaptive task weight balancer.

    Extends :class:`~src.models.loss.base_balancer.BaseBalancer` and
    plugs into ``MultiTaskLoss`` via ``attach_task_balancer()``.

    Parameters
    ----------
    tasks:
        Ordered list of task names.  Must match the keys in the
        ``task_losses`` dict that ``combine()`` receives.
    config:
        Tuning parameters.  Defaults applied if ``None``.
    """

    def __init__(
        self,
        tasks: List[str],
        config: Optional[DynamicTaskBalancerConfig] = None,
    ) -> None:
        super().__init__()

        if not tasks:
            raise ValueError("tasks cannot be empty")

        self.tasks = list(tasks)
        self.cfg = config or DynamicTaskBalancerConfig()

        # EMA loss estimates: initialised to 1.0 (neutral)
        self._ema: Dict[str, float] = {t: 1.0 for t in tasks}

        # Current dynamic weights (updated at step end)
        self._weights: Dict[str, float] = {t: 1.0 for t in tasks}

        # Step counter for warmup
        self._step: int = 0

        # Buffer for raw losses received in on_before_backward
        self._last_raw_losses: Dict[str, float] = {}

        self._mark_initialized()

        logger.info(
            "DynamicTaskWeightBalancer | tasks=%s | ema_alpha=%.3f | "
            "temperature=%.2f | warmup=%d",
            tasks,
            self.cfg.ema_alpha,
            self.cfg.temperature,
            self.cfg.warmup_steps,
        )

    # ----------------------------------------------------------------
    # BASEBALANCER CONTRACT
    # ----------------------------------------------------------------

    def combine(
        self,
        task_losses: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Combine per-task raw losses using current dynamic weights.

        Returns a weighted scalar total loss.  Uses the weights computed
        at the END of the *previous* step (buffered in ``_weights``) so
        the current batch's loss is not used to weight itself.
        """
        total: Optional[torch.Tensor] = None

        for task, loss in task_losses.items():
            w = float(self._weights.get(task, 1.0))
            weighted = loss * w
            total = weighted if total is None else total + weighted

        if total is None:
            ref = next(iter(task_losses.values()))
            return ref * 0.0

        return total

    def on_before_backward(
        self,
        task_losses: Dict[str, torch.Tensor],
        shared_parameters=None,
    ) -> None:
        """Buffer raw loss scalars for EMA update after backward.

        We buffer here (not in ``combine``) because ``combine`` is
        called AFTER the normaliser/coverage chain — the ``task_losses``
        passed to ``combine`` are the *raw* losses, which is exactly
        what we want for dominance tracking.
        """
        self._last_raw_losses = {
            t: float(v.detach().cpu()) for t, v in task_losses.items()
        }

    def on_step_end(self) -> None:
        """Update EMA estimates and recompute adaptive weights."""
        if not self._last_raw_losses:
            return

        alpha = self.cfg.ema_alpha

        # ── EMA update ────────────────────────────────────────────────
        for task in self.tasks:
            if task in self._last_raw_losses:
                loss_val = self._last_raw_losses[task]
                self._ema[task] = (
                    (1.0 - alpha) * self._ema[task] + alpha * loss_val
                )

        self._step += 1

        # ── Dynamic weights (skip during warmup) ─────────────────────
        if self._step <= self.cfg.warmup_steps:
            self._weights = {t: 1.0 for t in self.tasks}
            return

        ema_values = [max(self._ema[t], 1e-8) for t in self.tasks]
        mean_ema = sum(ema_values) / max(len(ema_values), 1)

        raw_weights: Dict[str, float] = {}
        for task, ema_val in zip(self.tasks, ema_values):
            dominance = ema_val / mean_ema            # > 1 → dominating
            if self.cfg.temperature == 0.0:
                raw_weights[task] = 1.0
            else:
                # Inverse dominance with temperature damping
                raw_weights[task] = 1.0 / (dominance ** self.cfg.temperature)

        # Normalise so weights average to 1.0
        weight_sum = sum(raw_weights.values())
        n = len(self.tasks)
        new_weights: Dict[str, float] = {}
        for task, rw in raw_weights.items():
            w = rw * n / max(weight_sum, 1e-8)
            w = min(w, self.cfg.max_weight)
            w = max(w, self.cfg.min_weight)
            new_weights[task] = w

        self._weights = new_weights
        self._last_raw_losses = {}

        if self.cfg.log_every > 0 and self._step % self.cfg.log_every == 0:
            ema_fmt = {t: f"{self._ema[t]:.4f}" for t in self.tasks}
            w_fmt = {t: f"{self._weights[t]:.3f}" for t in self.tasks}
            logger.info(
                "DynamicTaskWeightBalancer | step=%d | ema=%s | weights=%s",
                self._step,
                ema_fmt,
                w_fmt,
            )

    # ----------------------------------------------------------------
    # MONITORING
    # ----------------------------------------------------------------

    def get_weights(self) -> Dict[str, float]:
        """Return current dynamic task weights for monitoring."""
        return dict(self._weights)

    def get_ema_losses(self) -> Dict[str, float]:
        """Return current EMA loss estimates per task."""
        return dict(self._ema)

    def get_dominance(self) -> Dict[str, float]:
        """Return dominance ratio (ema_t / mean_ema) per task."""
        ema_values = [max(self._ema[t], 1e-8) for t in self.tasks]
        mean_ema = sum(ema_values) / max(len(ema_values), 1)
        return {t: self._ema[t] / mean_ema for t in self.tasks}

    def debug_state(self) -> Dict[str, float]:
        state = {}
        for t in self.tasks:
            state[f"ema_loss_{t}"] = self._ema[t]
            state[f"dynamic_weight_{t}"] = self._weights.get(t, 1.0)
        state["step"] = float(self._step)
        return state

    def __repr__(self) -> str:
        return (
            f"DynamicTaskWeightBalancer("
            f"tasks={self.tasks}, "
            f"step={self._step}, "
            f"weights={{{', '.join(f'{t}: {self._weights.get(t, 1.0):.3f}' for t in self.tasks)}}})"
        )
