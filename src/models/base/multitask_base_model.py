from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class MultiTaskBaseModel(BaseModel):

    def __init__(self, task_configs: Dict[str, Dict[str, Any]]) -> None:
        super().__init__()

        if not isinstance(task_configs, dict) or not task_configs:
            raise ValueError("task_configs must be non-empty dict")

        self.task_configs = task_configs
        self.task_heads: nn.ModuleDict = nn.ModuleDict()
        self.loss_functions: Dict[str, nn.Module] = {}

    # =====================================================
    # ENCODE
    # =====================================================

    @abstractmethod
    def encode(self, *inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError

    # =====================================================
    # REGISTER
    # =====================================================

    def register_task_head(
        self,
        task_name: str,
        head: nn.Module,
        loss_fn: nn.Module,
    ) -> None:

        if task_name in self.task_heads:
            raise ValueError(f"Task already exists: {task_name}")

        self.task_heads[task_name] = head
        self.loss_functions[task_name] = loss_fn

        logger.info("Registered head: %s", task_name)

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(
        self,
        *inputs: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        task: Optional[str] = None,
        return_features: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:

        shared = self.encode(*inputs, **kwargs)

        if shared.dim() != 2:
            raise ValueError(f"Expected 2D features, got {shared.shape}")

        active_task = task
        if active_task is None and labels and len(labels) == 1:
            active_task = next(iter(labels))

        task_list = [active_task] if active_task else list(self.task_heads.keys())

        outputs: Dict[str, Any] = {"tasks": {}}
        # A4.1: collect per-task losses in a list and reduce ONCE via
        # ``torch.stack(...).sum()`` instead of the previous ternary
        # ``loss if total_loss is None else total_loss + loss``. The
        # ternary allocated a fresh tensor and pushed an autograd node
        # per task, deepening the backward graph by N levels for an
        # N-task model. The single-shot reduce is also clearer about
        # the active-task vs multi-task split below.
        per_task_losses: list[torch.Tensor] = []

        for name in task_list:

            head = self.task_heads.get(name)
            if head is None:
                raise ValueError(f"No head: {name}")

            logits = head(shared)

            cfg = self.task_configs.get(name, {})
            task_type = cfg.get("type", "classification")

            task_out: Dict[str, torch.Tensor] = {"logits": logits}

            # Derived per-task statistics are inference-only; computing
            # them in training mode is wasted compute that also inflates
            # the autograd graph for tensors the loss never touches (P1).
            if not self.training:
                # N1: stable entropy in log-space — never take ``log``
                # of an EPS-shifted probability. For multiclass we use
                # ``log_softmax``; for multilabel we use ``logsigmoid``
                # of ``±logits`` for the per-label binary entropy.
                if task_type == "multilabel":
                    log_p = F.logsigmoid(logits)
                    log_1mp = F.logsigmoid(-logits)
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).long()
                    confidence = probs.max(dim=-1).values
                    entropy = -(
                        probs * log_p + (1.0 - probs) * log_1mp
                    ).mean(dim=-1)
                else:
                    # A6.2: derive ``preds`` / ``confidence`` from
                    # ``logits`` / ``log_probs`` directly — softmax is
                    # monotone so ``argmax(logits) == argmax(probs)``,
                    # and ``log_probs.max().exp()`` equals
                    # ``probs.max()`` without recomputing softmax.
                    log_probs = F.log_softmax(logits, dim=-1)
                    probs = log_probs.exp()
                    preds = torch.argmax(logits, dim=-1)
                    confidence = log_probs.max(dim=-1).values.exp()
                    entropy = -(probs * log_probs).sum(dim=-1)

                task_out["probabilities"] = probs
                task_out["predictions"] = preds
                task_out["confidence"] = confidence
                task_out["entropy"] = entropy

            if labels and name in labels:

                loss_fn = self.loss_functions.get(name)
                if loss_fn is None:
                    raise RuntimeError(f"No loss_fn for {name}")

                target = labels[name]

                if task_type == "multilabel":
                    target = target.float()

                loss = loss_fn(logits, target)
                task_out["loss"] = loss

                # A4.1: append; final reduce happens once, outside the loop.
                per_task_losses.append(loss)

            outputs["tasks"][name] = task_out

        if active_task:
            outputs.update(outputs["tasks"][active_task])

        # A4.1: explicit active-task vs multi-task split. In active-task
        # mode there is at most ONE entry in ``per_task_losses``, so we
        # promote it directly. In multi-task mode we ``stack().sum()``
        # to collapse all per-task losses with a single allocation and a
        # single autograd node.
        if per_task_losses:
            if active_task:
                outputs["loss"] = per_task_losses[0]
            else:
                outputs["loss"] = torch.stack(per_task_losses).sum()

        if return_features:
            outputs["shared_features"] = shared

        return outputs

    # =====================================================
    # PREDICT
    # =====================================================

    @torch.inference_mode()
    def predict(
        self,
        *inputs: torch.Tensor,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, torch.Tensor]]:

        was_training = self.training
        self.eval()

        try:
            outputs = self.forward(*inputs, task=task, **kwargs)
        finally:
            if was_training:
                self.train()

        return {
            name: {
                "predictions": out["predictions"],
                "probabilities": out["probabilities"],
                "confidence": out["confidence"],
            }
            for name, out in outputs["tasks"].items()
        }

    # =====================================================
    # TASKS
    # =====================================================

    def get_tasks(self) -> Dict[str, Dict[str, Any]]:
        return self.task_configs