from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import torch

from src.models.loss.multitask_loss import MultiTaskLoss, TaskLossConfig
from src.models.loss.base_balancer import BaseBalancer

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class LossEngineConfig:
    task_types: Dict[str, str]
    task_weights: Optional[Dict[str, float]] = None
    ignore_index: int = -100

    # LOSS-LVL-3: per-task loss-level balancing inputs, threaded straight
    # through to ``TaskLossConfig``. ``create_trainer_fn`` builds these
    # from the training-set label distribution via
    # ``training.loss_balancer.plan_for_dataframe`` before instantiating
    # the engine. Leaving them empty preserves the original unweighted
    # behaviour exactly.
    class_weights: Optional[Dict[str, "torch.Tensor"]] = None
    pos_weights: Optional[Dict[str, "torch.Tensor"]] = None
    use_focal: Optional[Dict[str, bool]] = None
    focal_gamma: Optional[Dict[str, float]] = None

    # Per-task surviving multilabel column indices (computed once on the
    # train split). Threaded through to ``TaskLossConfig`` so the router
    # slices the model's full-width logits down to match the reduced
    # label tensor produced by ``MultiLabelDataset(valid_label_indices=…)``.
    valid_label_indices: Optional[Dict[str, list]] = None

    # CFG-5: ``normalization`` selects the reduction strategy used when
    # combining per-task losses inside ``MultiTaskLoss``:
    #
    #   * ``"active"`` (default, recommended for multi-task training) —
    #     divide the summed weighted losses by the number of tasks that
    #     produced a non-zero loss this step. Keeps the aggregate scale
    #     stable when some heads receive no labels in a given batch.
    #   * ``"sum"`` — keep the raw weighted sum. **Required** for true
    #     single-task runs (one entry in ``task_types``); otherwise the
    #     "/ 1" division is a no-op and exporting "sum" makes the metric
    #     comparable to the loss reported in single-head literature.
    #     ``LossEngine.__init__`` auto-overrides ``"active"`` -> ``"sum"``
    #     for single-task configs and emits a warning.
    #   * ``"mean"`` — same as ``"active"`` but divides by ``len(task_types)``
    #     regardless of which heads contributed; appropriate when every
    #     task is expected to be active in every batch.
    normalization: str = "active"
    use_normalizer: bool = True
    use_coverage: bool = True

    # NORMALIZER-ALPHA-DAMP: EMA decay forwarded to ``EMALossNormalizer``
    # inside ``MultiTaskLoss``. Lower → smoother running mean / longer
    # effective averaging window → quieter per-task normalisation factors
    # → fewer downstream gradient spikes. ``None`` keeps the
    # ``EMALossNormalizer`` default (0.1) for back-compat with callers
    # that don't set it.
    normalizer_alpha: Optional[float] = None

    # LOSS-3: TrainingStep divides the loss by ``gradient_accumulation_steps``
    # so the gradient magnitude matches a single big batch. MultiTaskLoss
    # ALSO divides by ``active_heads``. The two are correct in isolation but
    # compose multiplicatively, so the effective per-task weight is
    # ``cfg.weight / (active_heads × grad_accum)``. To preserve the configured
    # static task weights when grad_accum > 1, pre-scale ``cfg.weight`` by
    # ``gradient_accumulation_steps`` here.
    gradient_accumulation_steps: int = 1


# =========================================================
# LOSS ENGINE
# =========================================================

class LossEngine:

    def __init__(self, config: LossEngineConfig):

        self.config = config

        # -------------------------------------------------
        # BUILD TASK CONFIGS
        # -------------------------------------------------

        task_configs: Dict[str, TaskLossConfig] = {}

        # LOSS-3: pre-scale static task weights by grad_accum so the
        # downstream loss/grad_accum division in TrainingStep doesn't
        # silently shrink the configured per-task weight.
        ga = max(1, int(getattr(config, "gradient_accumulation_steps", 1)))

        cw_map = config.class_weights or {}
        pw_map = config.pos_weights or {}
        focal_map = config.use_focal or {}
        gamma_map = config.focal_gamma or {}
        valid_idx_map = config.valid_label_indices or {}

        for task, task_type in config.task_types.items():
            weight = (config.task_weights or {}).get(task, 1.0)
            valid_idx = valid_idx_map.get(task)

            task_configs[task] = TaskLossConfig(
                task_type=task_type,
                weight=float(weight) * float(ga),
                ignore_index=config.ignore_index,
                class_weights=cw_map.get(task),
                pos_weight=pw_map.get(task),
                use_focal=bool(focal_map.get(task, False)),
                focal_gamma=float(gamma_map.get(task, 2.0)),
                valid_label_indices=(
                    [int(i) for i in valid_idx] if valid_idx else None
                ),
            )

        # -------------------------------------------------
        # CORE LOSS MODULE
        #
        # MT-1: ``create_trainer_fn`` builds one Trainer per task, so this
        # engine is overwhelmingly invoked with a SINGLE task. The full
        # multi-task plumbing (EMA normalizer, coverage tracker, GradNorm /
        # uncertainty balancer, ``normalization="active"`` head-count
        # division) is not just no-op overhead in that regime — it is
        # actively misleading. Specifically:
        #   * ``EMALossNormalizer`` rescales the loss by an EMA of itself;
        #     with one task this is a divide-by-self that flattens the
        #     gradient signal of the early steps and inflates it later.
        #   * ``EMACoverageTracker`` multiplies by a running coverage
        #     ratio that is always 1.0 for one task — pure GPU work.
        #   * ``normalization="active"`` divides by ``active_heads`` which
        #     is always 1; harmless but advertises a behaviour that
        #     doesn't apply.
        #   * ``attach_balancer`` would silently no-op since balancers
        #     need >= 2 tasks to balance between.
        # Force-disable them in the single-task path, log a clear warning,
        # and reject ``attach_balancer`` so callers can't be fooled into
        # thinking multi-task balancing is active when it isn't. The full
        # MultiTaskLoss wiring stays available unchanged for any future
        # caller that genuinely passes >1 tasks.
        # -------------------------------------------------

        single_task = len(task_configs) <= 1

        if single_task:
            if config.use_normalizer or config.use_coverage or config.normalization != "sum":
                logger.warning(
                    "MT-1: LossEngine instantiated with %d task(s); "
                    "disabling EMA normalizer, coverage tracker, and "
                    "switching normalization='sum'. Multi-task balancing "
                    "is a no-op in this configuration.",
                    len(task_configs),
                )
            effective_use_normalizer = False
            effective_use_coverage = False
            effective_normalization = "sum"
        else:
            effective_use_normalizer = config.use_normalizer
            effective_use_coverage = config.use_coverage
            effective_normalization = config.normalization

        self._single_task = single_task

        self.loss_module = MultiTaskLoss(
            task_configs=task_configs,
            normalization=effective_normalization,
            use_normalizer=effective_use_normalizer,
            use_coverage=effective_use_coverage,
            # NORMALIZER-ALPHA-DAMP: forward the YAML-tunable EMA alpha
            # into the normalizer (see LossEngineConfig docstring).
            normalizer_alpha=config.normalizer_alpha,
        )

        self._balancer: Optional[BaseBalancer] = None

        logger.info(
            "LossEngine initialized | tasks=%s | norm=%s",
            list(task_configs.keys()),
            config.normalization,
        )

    # =====================================================
    # BALANCER
    # =====================================================

    def attach_balancer(self, balancer: BaseBalancer) -> None:

        # MT-1: balancers (GradNorm / Uncertainty) need >= 2 tasks to
        # balance between. With one task they're a no-op that still pays
        # the per-step ``on_before_backward`` autograd-grad cost — and
        # worse, the caller is led to believe multi-task balancing is
        # active. Reject loudly so the bug is impossible to ship silently.
        # This runs BEFORE the BaseBalancer type check so the user gets
        # the most informative error first (the type check is a generic
        # contract guard; the single-task check is a specific config bug).
        if self._single_task:
            raise RuntimeError(
                "MT-1: cannot attach a balancer to a single-task LossEngine; "
                "balancers require >= 2 tasks. Build a multi-task LossEngine "
                "(pass >1 entries in LossEngineConfig.task_types) before "
                "calling attach_balancer."
            )

        if not isinstance(balancer, BaseBalancer):
            raise TypeError("balancer must inherit from BaseBalancer")

        self._balancer = balancer
        self.loss_module.attach_task_balancer(balancer)

        logger.info("Balancer attached: %s", balancer.__class__.__name__)

    # =====================================================
    # MAIN
    # =====================================================

    def compute(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        *,
        shared_parameters=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        if "labels" not in batch:
            raise RuntimeError("Missing 'labels'")

        # Single-task model classes (BiasClassifier, IdeologyClassifier,
        # …) emit ``outputs["logits"]`` (a single tensor) rather than the
        # multi-head ``outputs["task_logits"]`` dict that
        # MultiTaskTruthLensModel produces. When the LossEngine is
        # configured with exactly one task, treat the bare ``logits``
        # tensor as that task's logits and synthesise the ``task_logits``
        # dict the rest of the engine expects. Multi-task callers MUST
        # still supply ``task_logits`` — there is no way to disambiguate
        # a single tensor across multiple heads.
        if "task_logits" not in outputs:
            if "logits" in outputs and len(self.config.task_types) == 1:
                only_task = next(iter(self.config.task_types.keys()))
                outputs["task_logits"] = {only_task: outputs["logits"]}
            else:
                raise RuntimeError("Missing 'task_logits'")

        logits = outputs["task_logits"]
        labels = batch["labels"]

        # Single-task collate emits ``labels`` as a bare tensor, but
        # MultiTaskLoss requires a {task: tensor} dict. Wrap it up using
        # the only configured task name so the per-task loop matches.
        if not isinstance(labels, dict) and len(self.config.task_types) == 1:
            only_task = next(iter(self.config.task_types.keys()))
            labels = {only_task: labels}

        # -------------------------------------------------
        # CORE LOSS  (BUG-10: forward shared_parameters into the
        # loss module so GradNorm-style balancers can compute task
        # gradients. The balancer's on_before_backward hook is
        # already fired inside MultiTaskLoss.forward — we MUST NOT
        # invoke it again here, otherwise stateful balancers double-
        # advance their internal step counters every iteration.)
        # -------------------------------------------------

        total_loss, task_losses = self.loss_module(
            logits,
            labels,
            shared_parameters=shared_parameters,
        )

        # -------------------------------------------------
        # NUMERICAL SAFETY
        #
        # REC-1: the original layer issued THREE separate ``torch.isfinite``
        # reductions per step:
        #   * MultiTaskLoss.forward — once per task (N device-host syncs)
        #   * LossEngine.compute    — once per task + once aggregate (N+1)
        #   * TrainingStep.run      — once on the aggregate
        # That's 2N+2 forced syncs per step for an event that fires at most
        # a handful of times in an entire run. We now keep ONLY the
        # ``TrainingStep.run`` check (cheapest boundary — exactly the one
        # ``skip_nan_loss`` semantics need to honour). NaN propagates
        # through the aggregation, so any per-task NaN still surfaces there.
        # -------------------------------------------------

        # -------------------------------------------------
        # DEBUG ATTACHMENTS (SAFE)
        # -------------------------------------------------

        outputs["task_losses"] = {
            k: v.detach() for k, v in task_losses.items()
        }

        outputs["total_loss"] = total_loss.detach()

        # REC-2: ``mean_loss`` was a host-side ``.item()`` sync computed every
        # single step (forces the GPU to drain) and was attached to
        # ``outputs["loss_stats"]`` — but no consumer ever reads it. Trainer
        # logs ``raw_loss`` (the total) directly, the tracker logs per-task
        # losses, and instrumentation has its own EMA. Dropping the
        # computation removes one full host-device sync per step.
        # ``num_tasks`` is kept (it's a Python int; no GPU work).
        outputs["loss_stats"] = {
            "num_tasks": len(task_losses),
        }

        return total_loss, task_losses

    # =====================================================
    # TRAINING HOOKS
    # =====================================================

    def on_after_backward(self) -> None:

        if self._balancer is not None:
            try:
                self._balancer.on_after_backward()
            except Exception as e:
                logger.warning("Balancer post-backward failed: %s", e)

    def on_step_end(self) -> None:

        if self._balancer is not None:
            try:
                self._balancer.on_step_end()
            except Exception as e:
                logger.warning("Balancer step-end failed: %s", e)

    # =====================================================
    # STATS
    # =====================================================

    def get_stats(self) -> Dict[str, Any]:

        stats = {}

        if hasattr(self.loss_module, "get_stats"):
            stats.update(self.loss_module.get_stats())

        if self._balancer is not None:
            if hasattr(self._balancer, "get_weights"):
                stats["balancer_weights"] = self._balancer.get_weights()

        return stats

    # =====================================================
    # RESET
    # =====================================================

    def reset(self) -> None:

        if hasattr(self.loss_module, "reset"):
            self.loss_module.reset()

        logger.info("LossEngine reset complete")