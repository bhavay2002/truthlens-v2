from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Iterable

import torch
import torch.nn as nn

from src.models.loss.task_loss_router import TaskLossRouter
from src.models.loss.loss_normalizer import EMALossNormalizer
from src.models.loss.coverage_tracker import EMACoverageTracker
from src.models.loss.base_balancer import BaseBalancer

if TYPE_CHECKING:
    from src.training.confidence_filter import ConfidenceFilter

# CIRCULAR-IMPORT FIX: ``src.training.loss_functions`` is part of the
# ``src.training`` package whose ``__init__`` re-exports ``LossEngine``
# from ``loss_engine.py``, which in turn imports from this module at load
# time.  A top-level import here therefore creates a circular dependency
# that causes an ``ImportError`` when ``multitask_loss`` is the first
# module in the cycle to be resolved.
#
# Fix: defer the import to the first ``MultiTaskLoss.__init__`` call.
# ``FocalLoss`` is only used inside that method, so the lazy import is
# semantically identical and avoids the cycle entirely.
_FocalLoss = None


def _get_focal_loss_cls():
    global _FocalLoss
    if _FocalLoss is None:
        from src.training.loss_functions import FocalLoss as _FL
        _FocalLoss = _FL
    return _FocalLoss

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
    class_weights: Optional[torch.Tensor] = None
    use_focal: bool = False
    focal_gamma: float = 2.0

    # ── Multilabel column filtering ──────────────────────────────────
    valid_label_indices: Optional[List[int]] = None

    # ── Semantic alignment upgrades ───────────────────────────────────
    # Per-task temperature scaling applied to logits before loss.
    # T > 1 softens the distribution (reduces overconfidence, e.g.
    # emotion T=1.5); T < 1 sharpens it (e.g. propaganda T=0.8).
    # T = 1.0 is a no-op and the default.
    temperature: float = 1.0

    # Label smoothing epsilon.  For multiclass tasks PyTorch's built-in
    # F.cross_entropy label_smoothing is used.  For multilabel tasks the
    # binary targets are softened: y_smooth = y*(1-ε) + ε*0.5.
    # ε = 0.05 is recommended for multiclass; ε = 0.01 for multilabel.
    # 0.0 disables smoothing (default, preserves original behaviour).
    label_smoothing: float = 0.0

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
          → TaskLossRouter  (temperature scaling + label smoothing applied here)
          → EMALossNormalizer
          → EMACoverageTracker
          → static weighting
          → BaseBalancer (GradNorm / Uncertainty)
          → final normalization

    Designed for:
    - multi-task imbalance
    - sparse supervision (task_mask gating)
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

        FocalLoss = _get_focal_loss_cls()

        for name, cfg in task_configs.items():

            if cfg.task_type == "multiclass":
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

        # GPU-3: assign the ModuleDict to self BEFORE the router so
        # .to(device) propagates loss-function buffers (pos_weight, etc.).
        self.loss_functions = loss_functions
        self.router = TaskLossRouter(loss_functions, task_configs)

        if use_normalizer:
            if normalizer_alpha is not None:
                self.normalizer = EMALossNormalizer(alpha=float(normalizer_alpha))
            else:
                self.normalizer = EMALossNormalizer()
        else:
            self.normalizer = None
        self.coverage = EMACoverageTracker() if use_coverage else None

        self.balancer: Optional[BaseBalancer] = None

        # CONFIDENCE-FILTER: attached via attach_confidence_filter().
        # Stored as Any to avoid a circular import at module level;
        # the TYPE_CHECKING guard at the top provides IDE type info.
        self._confidence_filter: Optional[Any] = None

        # diagnostics
        self.last_active_heads: int = 0
        self.last_confidence_gate: float = 1.0

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

    def attach_confidence_filter(self, confidence_filter: Any) -> None:
        """Attach a :class:`~src.training.confidence_filter.ConfidenceFilter`.

        When attached, every forward pass computes a per-batch gate factor
        from the model's own prediction confidence and multiplies each
        per-task loss by that factor.  Samples the model is highly
        uncertain about (likely mislabelled) therefore contribute weaker
        gradients to the shared encoder.

        Parameters
        ----------
        confidence_filter:
            A ``ConfidenceFilter`` instance.  Accepted as ``Any`` to
            avoid a circular import; runtime duck-typing is used.
        """
        if not hasattr(confidence_filter, "compute_gate_factor"):
            raise TypeError(
                "confidence_filter must expose compute_gate_factor(logits, task_types)"
            )
        self._confidence_filter = confidence_filter
        logger.info(
            "ConfidenceFilter attached: %s",
            confidence_filter.__class__.__name__,
        )

    # =========================================================
    # MAIN FORWARD
    # =========================================================

    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        *,
        shared_parameters: Optional[Iterable[torch.nn.Parameter]] = None,
        task_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute weighted multi-task loss.

        Parameters
        ----------
        logits:
            Per-task logit tensors {task: (B, C)}.
        labels:
            Per-task label tensors {task: (B,) or (B, C)}.
        shared_parameters:
            (A4.5) Shared trunk parameters for GradNorm-style balancers.
        task_mask:
            Optional (B, num_tasks) long tensor where entry [i, t] = 1
            indicates sample i carries a valid label for task t.  When
            provided it gates the per-task loss so that rows without a
            label for a given task contribute zero gradient. This enables
            partial supervision in mixed-task batches.  When None the
            existing per-task ignore_index masking in TaskLossRouter
            is the sole guard (single-task batch behaviour unchanged).
        """

        if not isinstance(logits, dict) or not isinstance(labels, dict):
            raise TypeError("logits and labels must be dict")

        # Build a name→column-index look-up for task_mask slicing.
        # We match by position in self.task_names so the mask ordering
        # is stable regardless of the dict iteration order.
        task_to_col: Dict[str, int] = {t: i for i, t in enumerate(self.task_names)}

        # ── CONFIDENCE GATE (Label Noise Amplification fix) ───────────────
        # Compute a scalar gate factor ∈ (0, 1] from model confidence before
        # the per-task loss loop.  Batches in which the model is uniformly
        # uncertain across all heads (likely mislabelled) receive a smaller
        # gate factor and therefore contribute weaker gradients to the shared
        # encoder.  gate_factor = 1.0 when no filter is attached (no-op).
        gate_factor: float = 1.0
        if self._confidence_filter is not None:
            task_types_for_filter: Dict[str, str] = {
                t: cfg.task_type for t, cfg in self.task_configs.items()
            }
            gate_factor = self._confidence_filter.compute_gate_factor(
                logits, task_types_for_filter
            )
        self.last_confidence_gate = gate_factor

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
            # 1. RAW LOSS
            # -------------------------

            raw_loss = self.router.compute(
                task,
                logits[task],
                labels[task],
            )

            # -------------------------
            # 1b. TASK-MASK GATING
            #
            # When a per-row task_mask is available, scale the scalar
            # loss by the fraction of active rows for this task.
            # This is equivalent to computing the loss only on rows
            # where the mask is 1 and normalising by that count, which
            # prevents batches with very few labelled rows for a task
            # from producing disproportionately large gradient updates.
            # -------------------------
            if task_mask is not None and task in task_to_col:
                col_idx = task_to_col[task]
                if col_idx < task_mask.shape[1]:
                    col = task_mask[:, col_idx].float().to(raw_loss.device)
                    active_frac = col.mean().clamp_min(1e-6)
                    raw_loss = raw_loss * active_frac

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

            # -------------------------
            # 5b. CONFIDENCE GATE
            #
            # Apply the batch-level confidence gate factor computed
            # before the per-task loop.  gate_factor ∈ (0, 1] so
            # low-confidence batches contribute smaller gradients to
            # the shared encoder without zeroing them entirely
            # (the min_gate_factor floor in ConfidenceFilter prevents
            # complete gradient suppression).
            #
            # When no filter is attached gate_factor = 1.0 (no-op).
            # -------------------------
            if gate_factor != 1.0:
                weighted_loss = weighted_loss * gate_factor

            task_losses[task] = weighted_loss
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
