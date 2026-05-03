from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

import torch
import torch.nn as nn

from src.models.loss.task_loss_router import TaskLossRouter
from src.models.loss.loss_normalizer import EMALossNormalizer
from src.models.loss.coverage_tracker import EMACoverageTracker
from src.models.loss.base_balancer import BaseBalancer
from src.training.loss_functions import FocalLoss

logger = logging.getLogger(__name__)


# =========================================================
# SHARED-PARAMETER HELPER (A4.5)
# =========================================================

def gather_shared_parameters(
    model: torch.nn.Module,
) -> Optional[Iterable[torch.nn.Parameter]]:
    """Return the model's *shared* parameters for GradNorm-style balancers.

    Multi-task gradient balancers (``GradNorm``, ``PCGrad``, …) need the
    parameters that all tasks share — typically the encoder trunk —
    in order to compute per-task gradient norms against a *common*
    surface. Models can publish this via
    ``get_optimization_parameters()``. Plain ``nn.Module``s without
    that contract get ``None``, which the balancer interprets as
    "skip the gradient-shaping step" rather than crashing.

    A4.5: documents the previously implicit ``shared_parameters``
    contract on :meth:`MultiTaskLoss.forward` and gives callers a
    one-liner so the boilerplate isn't duplicated at every training
    step.
    """

    fn = getattr(model, "get_optimization_parameters", None)
    if not callable(fn):
        return None

    return fn()


# =========================================================
# CONFIG
# =========================================================

@dataclass
class TaskLossConfig:
    task_type: str
    weight: float = 1.0
    ignore_index: int = -100
    pos_weight: Optional[torch.Tensor] = None

    # ── Loss-level balancing (LOSS-LVL-3) ────────────────────────────
    # ``class_weights`` is the multiclass analogue of ``pos_weight``:
    # a per-class scalar tensor that scales each class's contribution
    # to ``CrossEntropyLoss``. Set it from
    # ``training.loss_balancer.plan_for_labels`` so the rare classes
    # contribute a gradient signal that is not drowned out by the
    # majority class. ``None`` keeps the unweighted CE baseline.
    class_weights: Optional[torch.Tensor] = None
    # When the dominant class exceeds the focal threshold (~0.9), even
    # class weights are not enough — flip ``use_focal=True`` to switch
    # the multiclass head to ``FocalLoss(weight=class_weights)``. The
    # ``(1 - p_t)^gamma`` factor down-weights the easy majority samples
    # so gradients keep flowing from the hard, rare ones.
    use_focal: bool = False
    focal_gamma: float = 2.0

    # ── Multilabel column filtering ──────────────────────────────────
    # When the dataset has dropped degenerate columns (see
    # ``utils.label_cleaning.remove_single_class_columns``), the
    # labels tensor has shape ``(B, K)`` while the model head still
    # outputs ``(B, C)`` with ``C >= K``. ``TaskLossRouter`` slices
    # the logits down to ``valid_label_indices`` before computing BCE
    # so the unused output neurons receive zero gradient and don't
    # corrupt the encoder. Leave as ``None`` to keep the original
    # full-width behaviour.
    valid_label_indices: Optional[List[int]] = None

    def __post_init__(self) -> None:
        # Canonical form: drop underscores, lowercase.
        # Accepts "multi_class"/"multiclass", "multi_label"/"multilabel",
        # "binary", "regression" — written as a single source of truth so
        # the rest of the loss layer can branch on a stable enum.
        canonical = str(self.task_type).replace("_", "").lower()
        if canonical not in {"multiclass", "multilabel", "binary", "regression"}:
            raise ValueError(f"invalid task_type={self.task_type!r}")
        object.__setattr__(self, "task_type", canonical)


# =========================================================
# MULTI-TASK LOSS (ORCHESTRATOR)
# =========================================================

class MultiTaskLoss(nn.Module):
    """
    Fully modular multi-task loss system.

    Pipeline:
        logits
          → TaskLossRouter
          → EMALossNormalizer
          → EMACoverageTracker
          → static weighting
          → BaseBalancer (GradNorm / Uncertainty)
          → final normalization

    Designed for:
    - multi-task imbalance
    - sparse supervision
    - research experimentation
    - production training
    """

    VALID_NORMALIZATION = {"active", "fixed", "sum"}

    def __init__(
        self,
        task_configs: Dict[str, TaskLossConfig],
        *,
        normalization: str = "active",
        use_normalizer: bool = True,
        use_coverage: bool = True,
        normalizer_alpha: Optional[float] = None,
    ) -> None:
        super().__init__()

        if not task_configs:
            raise ValueError("task_configs cannot be empty")

        if normalization not in self.VALID_NORMALIZATION:
            raise ValueError(f"Invalid normalization: {normalization}")

        self.task_configs = task_configs
        self.task_names = list(task_configs.keys())
        self.normalization = normalization

        # =====================================================
        # LOSS FUNCTIONS
        # =====================================================

        loss_functions = nn.ModuleDict()

        for name, cfg in task_configs.items():

            if cfg.task_type == "multiclass":
                # LOSS-LVL-3: pick the right loss for the observed
                # class distribution. ``use_focal`` is set upstream by
                # ``training.loss_balancer`` when the dominant class
                # exceeds the focal threshold; ``class_weights`` is set
                # whenever the distribution is imbalanced enough to
                # warrant inverse-frequency reweighting. Both fall back
                # to vanilla CE when neither is set, matching the prior
                # behaviour exactly on balanced data.
                cw = cfg.class_weights
                if cfg.use_focal:
                    loss_functions[name] = FocalLoss(
                        gamma=cfg.focal_gamma,
                        weight=cw,
                        ignore_index=cfg.ignore_index,
                        reduction="mean",
                    )
                else:
                    loss_functions[name] = nn.CrossEntropyLoss(
                        weight=cw,
                        ignore_index=cfg.ignore_index,
                    )

            elif cfg.task_type in {"binary", "multilabel"}:
                loss_functions[name] = nn.BCEWithLogitsLoss(
                    reduction="none",
                    pos_weight=cfg.pos_weight
                )

            elif cfg.task_type == "regression":
                loss_functions[name] = nn.MSELoss()

            else:
                raise ValueError(f"{name}: invalid task_type={cfg.task_type}")

        # =====================================================
        # MODULES
        # =====================================================

        # GPU-3 (loss buffers): ``loss_functions`` is an ``nn.ModuleDict`` that
        # holds ``nn.BCEWithLogitsLoss(pos_weight=…)`` and
        # ``nn.CrossEntropyLoss(weight=…)``. PyTorch registers ``pos_weight`` /
        # ``weight`` as *buffers* on those loss modules — so they only move
        # device when their parent ``nn.Module`` walks them via ``.to(device)``.
        # ``TaskLossRouter`` is a plain Python class (deliberately, to keep the
        # routing logic stateless), which means ``self.router.loss_functions``
        # is INVISIBLE to ``MultiTaskLoss.children()`` / ``.modules()`` — and
        # therefore invisible to ``MultiTaskLoss.to(device)``. The result is
        # the classic "Expected all tensors to be on the same device, but
        # found cuda:0 and cpu" crash on the very first BCE forward pass when
        # ``pos_weight`` / ``class_weights`` are populated by the loss
        # balancer (e.g. emotion multilabel).
        #
        # Fix: assign the ``ModuleDict`` to ``self.loss_functions`` BEFORE
        # passing it into the router. PyTorch's ``__setattr__`` then registers
        # it as a submodule of ``MultiTaskLoss``, so a single call to
        # ``MultiTaskLoss.to(device)`` (already done in
        # ``TrainingStep.__init__``) propagates to every per-task loss buffer.
        # The router still gets the same ``ModuleDict`` reference so its
        # routing logic is unchanged.
        self.loss_functions = loss_functions
        self.router = TaskLossRouter(loss_functions, task_configs)

        # NORMALIZER-ALPHA-DAMP: pass through the YAML-tunable EMA alpha
        # when the caller supplied one; otherwise fall back to the
        # ``EMALossNormalizer`` default (0.1) so legacy callers are
        # unaffected.
        if use_normalizer:
            if normalizer_alpha is not None:
                self.normalizer = EMALossNormalizer(alpha=float(normalizer_alpha))
            else:
                self.normalizer = EMALossNormalizer()
        else:
            self.normalizer = None
        self.coverage = EMACoverageTracker() if use_coverage else None

        self.balancer: Optional[BaseBalancer] = None

        # diagnostics
        self.last_active_heads: int = 0

        logger.info(
            "MultiTaskLoss initialized | tasks=%s | norm=%s",
            self.task_names,
            self.normalization,
        )

    # =========================================================
    # BALANCER
    # =========================================================

    def attach_task_balancer(self, balancer: BaseBalancer) -> None:
        if not isinstance(balancer, BaseBalancer):
            raise TypeError("balancer must inherit BaseBalancer")

        self.balancer = balancer
        logger.info("Balancer attached: %s", balancer.__class__.__name__)

    # =========================================================
    # MAIN FORWARD
    # =========================================================

    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        *,
        shared_parameters: Optional[Iterable[torch.nn.Parameter]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute weighted multi-task loss.

        ``shared_parameters`` (A4.5) is the iterable of trunk / shared
        parameters that GradNorm-style balancers will compute per-task
        gradient norms against. Most callers should obtain it via
        :func:`gather_shared_parameters(model)`; non-balancer setups can
        leave it ``None``.
        """

        if not isinstance(logits, dict) or not isinstance(labels, dict):
            raise TypeError("logits and labels must be dict")

        task_losses: Dict[str, torch.Tensor] = {}
        raw_losses: Dict[str, torch.Tensor] = {}

        total_loss: Optional[torch.Tensor] = None
        active_heads = 0

        # =====================================================
        # PER-TASK LOOP
        # =====================================================

        for task in self.task_names:

            if task not in logits or task not in labels:
                continue

            cfg = self.task_configs[task]

            # -------------------------
            # 1. RAW LOSS (IMPORTANT)
            # -------------------------

            raw_loss = self.router.compute(
                task,
                logits[task],
                labels[task],
            )

            # REC-1: per-task ``torch.isfinite`` was previously checked HERE,
            # again in ``LossEngine.compute`` (one per task + one for total),
            # and again in ``TrainingStep.run`` for the total — three full
            # device-host syncs per step, two of them N×. Keep ONLY the
            # cheapest single ``isnan().any()`` reduce on the aggregated
            # ``total_loss`` at the TrainingStep boundary; NaN propagates
            # through the sum so any per-task NaN is still caught there.

            loss = raw_loss

            # -------------------------
            # 2. COVERAGE UPDATE
            # -------------------------

            if self.coverage is not None:
                self.coverage.update(
                    task,
                    labels[task],
                    ignore_index=cfg.ignore_index,
                    task_type=cfg.task_type,
                )

            # -------------------------
            # 3. EMA NORMALIZATION
            # -------------------------

            if self.normalizer is not None:
                loss = self.normalizer.normalize(task, loss)

            # -------------------------
            # 4. COVERAGE WEIGHTING
            # -------------------------

            if self.coverage is not None:
                loss = self.coverage.weight(task, loss)

            # -------------------------
            # 5. STATIC TASK WEIGHT
            # -------------------------

            weighted_loss = loss * float(cfg.weight)

            # MT-4: the second element of the return tuple is the per-task
            # loss view that downstream consumers (TaskScheduler EMA,
            # AutoDebugEngine.LossTracker EMA, instrumentation) treat as a
            # raw loss magnitude. Returning the *weighted+normalized+
            # coverage-multiplied* value would (a) corrupt the adaptive
            # scheduler's softmax of EMA losses (it sees post-normalized
            # ratios and reverse-amplifies the weighting), (b) make
            # spike/anomaly detection thresholds non-comparable across
            # tasks with different static weights, and (c) hide regressions
            # in raw model performance behind balancing changes. Track both
            # — ``total_loss`` accumulates the WEIGHTED value (correct for
            # backward) while the returned per-task dict holds RAW values
            # (correct for diagnostics).
            task_losses[task] = weighted_loss  # internal, used for total_loss
            raw_losses[task] = raw_loss

            total_loss = (
                weighted_loss
                if total_loss is None
                else total_loss + weighted_loss
            )

            active_heads += 1

        # =====================================================
        # EMPTY BATCH SAFE (AMP SAFE)
        # =====================================================

        if active_heads == 0:
            for t in logits.values():
                if torch.is_tensor(t) and t.requires_grad:
                    return t.sum() * 0.0, {}
            return torch.zeros((), requires_grad=False), {}

        # =====================================================
        # BALANCER HOOK (BEFORE BACKWARD)
        # =====================================================

        if self.balancer is not None:
            self.balancer.on_before_backward(
                raw_losses,
                shared_parameters=shared_parameters,
            )

        # =====================================================
        # BALANCER COMBINATION
        # =====================================================

        if self.balancer is not None:
            total_loss = self.balancer(raw_losses)

        # =====================================================
        # FINAL NORMALIZATION
        # =====================================================

        if self.normalization == "active":
            total_loss = total_loss / float(active_heads)

        elif self.normalization == "fixed":
            total_loss = total_loss / float(len(self.task_names))

        # "sum" → no change

        self.last_active_heads = active_heads

        # MT-4: return RAW per-task losses (see MT-4 comment above).
        return total_loss, raw_losses

    # =========================================================
    # TRAINING HOOKS (CALL FROM TRAINER)
    # =========================================================

    def on_after_backward(self) -> None:
        if self.balancer is not None:
            self.balancer.on_after_backward()

    def on_step_end(self) -> None:
        if self.balancer is not None:
            self.balancer.on_step_end()

    # =========================================================
    # DEBUG / MONITORING
    # =========================================================

    def get_stats(self) -> Dict[str, Dict]:

        stats: Dict[str, Dict] = {}

        if self.normalizer:
            stats["loss_mean"] = self.normalizer.get_running_means()

        if self.coverage:
            stats["coverage"] = self.coverage.get_coverage()
            stats["coverage_weights"] = self.coverage.get_multipliers()

        if self.balancer and hasattr(self.balancer, "get_weights"):
            stats["balancer_weights"] = self.balancer.get_weights()

        stats["active_heads"] = self.last_active_heads

        return stats