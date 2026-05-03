"""
CurriculumScheduler — phased multi-task training schedule.

Spec §2 (Training Pipeline Upgrade).

Three phases
------------

Phase 1 — Task-Specific Warmup  (epochs 1 .. phase1_epochs)
    * Cross-task interaction layer is frozen.
    * Only ``warmup_tasks`` contribute to the loss.
    * Hard-mining is OFF.
    * Goal: build strong task-specific priors before interaction weights
      receive any gradient.

Phase 2 — Joint Multi-Task Training  (phase1_epochs+1 .. phase1+phase2)
    * All components unfrozen.
    * All enabled tasks active.
    * Task-balanced sampling + masked loss.
    * Hard-mining is ON (partial: pool fills, sampling starts at epoch 2).
    * Interaction strength is ramped from 0→1 via ``interaction_alpha``.

Phase 3 — Cross-Task Consistency Refinement  (phase1+phase2+1 .. end)
    * Same as phase 2, PLUS:
    * ConsistencyLoss is added to total loss weighted by ``consistency_lambda``.
    * Hard-mining is FULL.

Usage
-----

    sched = CurriculumScheduler(model, CurriculumConfig())
    for epoch in range(1, total_epochs + 1):
        state = sched.on_epoch_start(epoch)
        sched.apply_to_model(model, state)
        for batch in loader:
            step_state = sched.on_step(global_step)
            # state.apply_consistency_loss → add ConsistencyLoss to total
            # state.interaction_alpha     → pass to interaction layer
            ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


# =========================================================
# STATE — returned by on_epoch_start / on_step
# =========================================================

@dataclass
class CurriculumState:
    """Snapshot of the curriculum position; consumed by the training loop.

    The training loop should read this state and:
      * call ``scheduler.apply_to_model(model, state)`` once per epoch to
        freeze / unfreeze the right components.
      * gate the ConsistencyLoss on ``state.apply_consistency_loss``.
      * pass ``state.interaction_alpha`` to any alpha-gated component.
      * set the hard-miner mode from ``state.hard_mining_mode``.
    """

    phase: int
    active_tasks: List[str]
    freeze_interaction: bool
    apply_consistency_loss: bool
    hard_mining_mode: str       # "off" | "partial" | "full"
    interaction_alpha: float    # 0.0 → 1.0 ramp (phase 2+)
    consistency_lambda: float   # weight for L_consistency


# =========================================================
# CONFIG
# =========================================================

@dataclass
class CurriculumConfig:
    """Hyper-parameters for the three-phase curriculum.

    All epoch indices are *inclusive* and 1-based (matching the loop
    ``for epoch in range(1, total_epochs + 1)``).

    Parameters
    ----------
    phase1_epochs:
        Number of task-specific warmup epochs (interaction frozen).
    phase2_epochs:
        Number of joint multi-task training epochs.
    phase3_epochs:
        Number of consistency-refinement epochs. Training continues
        until ``phase1 + phase2 + phase3`` epochs are complete.
    warmup_tasks:
        Tasks that are active in Phase 1. Defaults to all tasks if
        left empty (discovered from the model at ``on_epoch_start``).
    consistency_lambda:
        Loss weight for the consistency term added in Phase 3.
    alpha_ramp_steps:
        Global training steps over which ``interaction_alpha`` is
        linearly ramped from 0.0 to 1.0, starting at the first Phase 2
        step. Set to 0 to jump directly to alpha=1.0.
    interaction_module_name:
        Attribute name of the cross-task interaction layer on the model
        (used by ``apply_to_model``). Defaults to ``"interaction"``.
    """

    phase1_epochs: int = 2
    phase2_epochs: int = 4
    phase3_epochs: int = 4
    warmup_tasks: List[str] = field(default_factory=list)
    consistency_lambda: float = 0.1
    alpha_ramp_steps: int = 1000
    interaction_module_name: str = "interaction"

    def __post_init__(self) -> None:
        if self.phase1_epochs < 0:
            raise ValueError("phase1_epochs must be >= 0")
        if self.phase2_epochs < 1:
            raise ValueError("phase2_epochs must be >= 1")
        if self.phase3_epochs < 0:
            raise ValueError("phase3_epochs must be >= 0")
        if not (0.0 <= self.consistency_lambda <= 10.0):
            raise ValueError("consistency_lambda must be in [0, 10]")
        if self.alpha_ramp_steps < 0:
            raise ValueError("alpha_ramp_steps must be >= 0")


# =========================================================
# SCHEDULER
# =========================================================

class CurriculumScheduler:
    """Three-phase curriculum controller for ``InteractingMultiTaskModel``.

    The scheduler is stateless with respect to the model — it returns
    ``CurriculumState`` objects describing WHAT should happen; the caller
    decides HOW to act on them (freeze, add loss, etc.). The only mutable
    state held internally is the global-step counter used for the alpha
    ramp.

    Parameters
    ----------
    config:
        Curriculum hyper-parameters.
    all_tasks:
        Full list of tasks enabled in the model. If empty, the scheduler
        falls back to ``config.warmup_tasks`` for phase 1 and emits a
        warning for phase 2+.
    """

    def __init__(
        self,
        config: Optional[CurriculumConfig] = None,
        all_tasks: Optional[List[str]] = None,
    ) -> None:
        self.config = config or CurriculumConfig()
        self._all_tasks: List[str] = list(all_tasks or [])
        # Global step counter — incremented by on_step(); used for alpha ramp.
        self._global_step: int = 0
        # Step at which Phase 2 started (set on first Phase 2 epoch).
        self._phase2_start_step: Optional[int] = None

        logger.info(
            "CurriculumScheduler | phases=[%d, %d, %d] | tasks=%s | "
            "warmup_tasks=%s | lambda=%.3f | alpha_ramp=%d",
            self.config.phase1_epochs,
            self.config.phase2_epochs,
            self.config.phase3_epochs,
            self._all_tasks,
            self.config.warmup_tasks,
            self.config.consistency_lambda,
            self.config.alpha_ramp_steps,
        )

    # -----------------------------------------------------------------------
    # PHASE HELPERS
    # -----------------------------------------------------------------------

    def get_phase(self, epoch: int) -> int:
        """Map epoch number (1-based) to phase (1/2/3)."""
        p1 = self.config.phase1_epochs
        p2 = self.config.phase2_epochs
        if epoch <= p1:
            return 1
        if epoch <= p1 + p2:
            return 2
        return 3

    def _active_tasks(self, phase: int) -> List[str]:
        if phase == 1:
            warmup = self.config.warmup_tasks
            if warmup:
                return list(warmup)
            # Fall back to all tasks if warmup_tasks not specified
            logger.debug(
                "CurriculumScheduler: warmup_tasks not set — "
                "using all tasks in Phase 1."
            )
            return list(self._all_tasks)
        return list(self._all_tasks)

    def _interaction_alpha(self) -> float:
        """Linear ramp of interaction alpha from 0 → 1 over alpha_ramp_steps."""
        ramp = self.config.alpha_ramp_steps
        if ramp <= 0 or self._phase2_start_step is None:
            return 1.0
        elapsed = self._global_step - self._phase2_start_step
        return float(min(1.0, elapsed / ramp))

    # -----------------------------------------------------------------------
    # MAIN API
    # -----------------------------------------------------------------------

    def on_epoch_start(self, epoch: int) -> CurriculumState:
        """Return the curriculum state for the given epoch.

        Call once at the top of each training epoch, before any batch
        processing. The returned state describes the full configuration
        for the epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number (1-based).

        Returns
        -------
        CurriculumState
        """
        phase = self.get_phase(epoch)

        # Record Phase 2 start step (once)
        if phase >= 2 and self._phase2_start_step is None:
            self._phase2_start_step = self._global_step
            logger.info(
                "CurriculumScheduler: entering Phase 2 at "
                "epoch=%d global_step=%d",
                epoch,
                self._global_step,
            )

        active = self._active_tasks(phase)
        freeze_interaction = phase == 1
        apply_consistency = phase == 3
        hard_mode = {1: "off", 2: "partial", 3: "full"}[phase]
        alpha = self._interaction_alpha()

        state = CurriculumState(
            phase=phase,
            active_tasks=active,
            freeze_interaction=freeze_interaction,
            apply_consistency_loss=apply_consistency,
            hard_mining_mode=hard_mode,
            interaction_alpha=alpha,
            consistency_lambda=self.config.consistency_lambda,
        )

        logger.info(
            "CurriculumScheduler | epoch=%d phase=%d "
            "active_tasks=%s freeze_interaction=%s "
            "hard_mining=%s alpha=%.3f",
            epoch, phase, active, freeze_interaction, hard_mode, alpha,
        )

        return state

    def on_step(self) -> None:
        """Increment the internal global step counter.

        Call once per optimiser step (NOT per gradient-accumulation
        micro-step) so the alpha ramp tracks actual parameter updates.
        """
        self._global_step += 1

    def get_alpha(self) -> float:
        """Current interaction alpha (for manual queries between epochs)."""
        return self._interaction_alpha()

    # -----------------------------------------------------------------------
    # MODEL MUTATOR
    # -----------------------------------------------------------------------

    def apply_to_model(self, model: nn.Module, state: CurriculumState) -> None:
        """Freeze / unfreeze model components according to ``state``.

        This is the ONLY method that mutates the model. Keeping the
        freeze/unfreeze logic here (rather than in the training loop)
        ensures the curriculum is the single source of truth for which
        components are trainable.

        Behaviour by phase
        ------------------
        Phase 1:
            Freeze the cross-task interaction layer
            (``config.interaction_module_name``). All other components
            are left at their current requires_grad state (the caller is
            responsible for ensuring non-interaction components are
            unfrozen before entering Phase 1).

        Phase 2+:
            Unfreeze the cross-task interaction layer.
        """
        interaction = getattr(
            model, self.config.interaction_module_name, None
        )
        if interaction is None:
            logger.debug(
                "CurriculumScheduler.apply_to_model: model has no "
                "attribute %r — skipping freeze/unfreeze.",
                self.config.interaction_module_name,
            )
            return

        requires_grad = not state.freeze_interaction
        frozen_count = 0
        for p in interaction.parameters():
            if p.requires_grad != requires_grad:
                p.requires_grad = requires_grad
                frozen_count += 1

        if frozen_count:
            action = "Unfroze" if requires_grad else "Froze"
            logger.info(
                "CurriculumScheduler: %s %d params in '%s' (phase %d)",
                action,
                frozen_count,
                self.config.interaction_module_name,
                state.phase,
            )

    # -----------------------------------------------------------------------
    # UTILITIES
    # -----------------------------------------------------------------------

    def total_epochs(self) -> int:
        """Sum of all three phase lengths."""
        cfg = self.config
        return cfg.phase1_epochs + cfg.phase2_epochs + cfg.phase3_epochs

    def phase_boundaries(self) -> Dict[str, int]:
        """Return the last epoch of each phase."""
        p1 = self.config.phase1_epochs
        p2 = self.config.phase2_epochs
        p3 = self.config.phase3_epochs
        return {
            "phase1_end": p1,
            "phase2_end": p1 + p2,
            "phase3_end": p1 + p2 + p3,
        }

    def set_all_tasks(self, tasks: List[str]) -> None:
        """Update the full task list (e.g. after model construction)."""
        self._all_tasks = list(tasks)

    def reset(self) -> None:
        """Reset internal step counters (useful between training runs)."""
        self._global_step = 0
        self._phase2_start_step = None
