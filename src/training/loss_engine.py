from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
    # combining per-task losses inside ``MultiTaskLoss``.
    normalization: str = "active"
    use_normalizer: bool = True
    use_coverage: bool = True

    # NORMALIZER-ALPHA-DAMP: EMA decay forwarded to ``EMALossNormalizer``.
    normalizer_alpha: Optional[float] = None

    # LOSS-3: gradient accumulation step pre-scaling.
    gradient_accumulation_steps: int = 1

    # ── Semantic alignment upgrades ───────────────────────────────────
    # Per-task temperature scaling. Dict maps task name → temperature T.
    # T > 1 softens predictions (e.g. emotion: 1.5, ideology: 1.2).
    # T < 1 sharpens predictions (e.g. propaganda: 0.8).
    # Tasks absent from this dict default to T = 1.0 (no-op).
    task_temperatures: Optional[Dict[str, float]] = None

    # Per-task label smoothing epsilon.
    # Multiclass tasks: ε = 0.05 recommended.
    # Multilabel tasks: ε = 0.01 recommended.
    # Tasks absent from this dict default to ε = 0.0 (disabled).
    label_smoothing: Optional[Dict[str, float]] = None


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
        temp_map = config.task_temperatures or {}
        smooth_map = config.label_smoothing or {}

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
                temperature=float(temp_map.get(task, 1.0)),
                label_smoothing=float(smooth_map.get(task, 0.0)),
            )

        # -------------------------------------------------
        # CORE LOSS MODULE
        #
        # MT-1: single-task path disables multi-task plumbing that would
        # be misleading / actively harmful in that regime.
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
            normalizer_alpha=config.normalizer_alpha,
        )

        self._balancer: Optional[BaseBalancer] = None

        logger.info(
            "LossEngine initialized | tasks=%s | norm=%s | temps=%s | smoothing=%s",
            list(task_configs.keys()),
            config.normalization,
            temp_map or "none",
            smooth_map or "none",
        )

    # =====================================================
    # BALANCER
    # =====================================================

    def attach_balancer(self, balancer: BaseBalancer) -> None:

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

        # Single-task model classes emit ``outputs["logits"]`` rather than
        # the multi-head ``outputs["task_logits"]`` dict. Synthesise the dict.
        if "task_logits" not in outputs:
            if "logits" in outputs and len(self.config.task_types) == 1:
                only_task = next(iter(self.config.task_types.keys()))
                outputs["task_logits"] = {only_task: outputs["logits"]}
            else:
                raise RuntimeError("Missing 'task_logits'")

        logits = outputs["task_logits"]
        labels = batch["labels"]

        if not isinstance(labels, dict) and len(self.config.task_types) == 1:
            only_task = next(iter(self.config.task_types.keys()))
            labels = {only_task: labels}

        # ── Task-presence mask (partial supervision) ──────────────────
        # Extracted from the batch when present (collate now propagates it).
        # Shape: (B, num_tasks) — 1 where the row has a valid label for task t.
        task_mask: Optional[torch.Tensor] = batch.get("task_mask", None)

        # -------------------------------------------------
        # CORE LOSS
        # -------------------------------------------------

        total_loss, task_losses = self.loss_module(
            logits,
            labels,
            shared_parameters=shared_parameters,
            task_mask=task_mask,
        )

        # -------------------------------------------------
        # DEBUG ATTACHMENTS
        # -------------------------------------------------

        outputs["task_losses"] = {
            k: v.detach() for k, v in task_losses.items()
        }

        outputs["total_loss"] = total_loss.detach()

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
