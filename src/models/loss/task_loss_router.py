from __future__ import annotations

import logging
from typing import Dict, TYPE_CHECKING

import torch
import torch.nn as nn

# CIRCULAR-IMPORT FIX: ``multitask_loss`` imports ``TaskLossRouter`` from this
# module at module-load time. Importing ``TaskLossConfig`` from ``multitask_loss``
# at module level here creates a cycle that fails on the second module's load
# ("cannot import name 'TaskLossConfig' from partially initialized module ...").
# ``TaskLossConfig`` is only used as a *type annotation* in this file, so we
# import it under ``TYPE_CHECKING``. Combined with ``from __future__ import
# annotations`` at the top of the file, every annotation is already a string
# and never resolved at runtime — zero behavioural change, no runtime cost.
if TYPE_CHECKING:
    from src.models.loss.multitask_loss import TaskLossConfig

logger = logging.getLogger(__name__)


class TaskLossRouter:
    """
    Responsible ONLY for computing per-task raw losses.

    This module:
    - routes task → correct loss function
    - handles dtype/device alignment
    - handles masking (ignore_index)
    - returns raw (unweighted, unnormalized) loss

    It does NOT:
    - normalize loss
    - apply task weights
    - apply balancing (GradNorm / uncertainty)
    """

    def __init__(
        self,
        loss_functions: nn.ModuleDict,
        task_configs: Dict[str, TaskLossConfig],
    ) -> None:

        if not isinstance(loss_functions, nn.ModuleDict):
            raise TypeError("loss_functions must be nn.ModuleDict")

        self.loss_functions = loss_functions
        self.task_configs = task_configs

        logger.info(
            "TaskLossRouter initialized | tasks=%s",
            list(task_configs.keys()),
        )

    # =========================================================
    # MAIN ROUTER
    # =========================================================

    def compute(
        self,
        task_name: str,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute raw loss for a single task.

        Returns
        -------
        torch.Tensor
            Scalar loss (mean reduced)
        """

        if task_name not in self.task_configs:
            raise ValueError(f"Unknown task: {task_name}")

        cfg = self.task_configs[task_name]
        loss_fn = self.loss_functions[task_name]

        if not torch.is_tensor(logits):
            raise TypeError(f"{task_name}: logits must be tensor")

        if not torch.is_tensor(labels):
            raise TypeError(f"{task_name}: labels must be tensor")

        if logits.numel() == 0:
            raise RuntimeError(f"{task_name}: empty logits")

        # device alignment
        if labels.device != logits.device:
            labels = labels.to(logits.device)

        # AMP safety
        logits = logits.float()

        # route (TaskLossConfig normalises to canonical form: no underscores)
        if cfg.task_type == "multiclass":
            return self._multiclass_loss(task_name, logits, labels, cfg, loss_fn)

        elif cfg.task_type == "binary":
            return self._binary_loss(task_name, logits, labels, cfg, loss_fn)

        elif cfg.task_type == "multilabel":
            return self._multilabel_loss(task_name, logits, labels, cfg, loss_fn)

        elif cfg.task_type == "regression":
            return self._regression_loss(task_name, logits, labels, cfg, loss_fn)

        else:
            raise ValueError(f"{task_name}: invalid task_type '{cfg.task_type}'")

    # =========================================================
    # TASK TYPES
    # =========================================================

    def _multiclass_loss(
        self,
        task: str,
        logits: torch.Tensor,
        labels: torch.Tensor,
        cfg: TaskLossConfig,
        loss_fn: nn.Module,
    ) -> torch.Tensor:

        labels = labels.long()

        # one-hot → index
        if labels.dim() == 2:
            labels = labels.argmax(dim=1)

        # ignore masked
        valid = labels.ne(cfg.ignore_index)
        if not bool(valid.any()):
            return self._zero_loss(logits)

        loss = loss_fn(logits, labels)

        return loss.mean()

    def _binary_loss(
        self,
        task: str,
        logits: torch.Tensor,
        labels: torch.Tensor,
        cfg: TaskLossConfig,
        loss_fn: nn.Module,
    ) -> torch.Tensor:

        labels = labels.float()

        if logits.dim() == 1:
            logits = logits.unsqueeze(1)

        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        loss = loss_fn(logits, labels)

        return loss.mean()

    def _multilabel_loss(
        self,
        task: str,
        logits: torch.Tensor,
        labels: torch.Tensor,
        cfg: TaskLossConfig,
        loss_fn: nn.Module,
    ) -> torch.Tensor:

        labels = labels.float()

        # When the dataset has dropped degenerate columns (all-0 or all-1
        # in the train split), the model head still emits the full-width
        # logits — slice them down to the surviving columns so they match
        # the reduced labels and the (already-reduced) ``pos_weight``
        # tensor inside ``loss_fn``. The dropped logit columns receive
        # zero gradient and therefore stop poisoning the shared encoder.
        valid_idx = cfg.valid_label_indices
        if valid_idx is not None and len(valid_idx) != logits.shape[-1]:
            if logits.shape[-1] < max(valid_idx) + 1:
                raise ValueError(
                    f"{task}: valid_label_indices reference column "
                    f"{max(valid_idx)} but logits have width "
                    f"{logits.shape[-1]}"
                )
            idx_t = torch.as_tensor(
                valid_idx, dtype=torch.long, device=logits.device
            )
            logits = logits.index_select(-1, idx_t)

        if logits.shape != labels.shape:
            raise ValueError(
                f"{task}: shape mismatch {logits.shape} vs {labels.shape}"
            )

        ignore_index = float(cfg.ignore_index)

        valid_mask = labels.ne(ignore_index)

        if not bool(valid_mask.any()):
            return self._zero_loss(logits)

        safe_labels = torch.where(
            valid_mask,
            labels,
            torch.zeros_like(labels),
        )

        raw_loss = loss_fn(logits, safe_labels)

        masked_loss = raw_loss * valid_mask.to(raw_loss.dtype)

        return masked_loss.sum() / valid_mask.sum().clamp_min(1)

    def _regression_loss(
        self,
        task: str,
        logits: torch.Tensor,
        labels: torch.Tensor,
        cfg: TaskLossConfig,
        loss_fn: nn.Module,
    ) -> torch.Tensor:

        labels = labels.float().view_as(logits)

        loss = loss_fn(logits, labels)

        return loss.mean()

    # =========================================================
    # UTILS
    # =========================================================

    def _zero_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Return gradient-safe zero loss.
        """
        if logits.requires_grad:
            return logits.sum() * 0.0
        return torch.zeros((), requires_grad=False)