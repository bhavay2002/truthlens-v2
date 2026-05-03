from __future__ import annotations

import logging
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseBalancer(nn.Module):
    """
    Base interface for multi-task loss balancing strategies.
    """

    def __init__(self) -> None:
        super().__init__()
        self._initialized = False

    # =========================================================
    # MAIN API
    # =========================================================

    def forward(
        self,
        task_losses: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Safe entrypoint: validates inputs before combining.
        """
        self._validate_inputs(task_losses)
        return self.combine(task_losses)

    def combine(
        self,
        task_losses: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Subclasses MUST implement this instead of forward().
        """
        raise NotImplementedError

    # =========================================================
    # OPTIONAL HOOKS
    # =========================================================

    def on_before_backward(
        self,
        task_losses: Dict[str, torch.Tensor],
        shared_parameters=None,
    ) -> None:
        return None

    def on_after_backward(self) -> None:
        return None

    def on_step_end(self) -> None:
        return None

    # =========================================================
    # INTERNAL UTILITIES
    # =========================================================

    def _validate_inputs(
        self,
        task_losses: Dict[str, torch.Tensor],
    ) -> None:

        if not isinstance(task_losses, dict) or not task_losses:
            raise ValueError("task_losses must be a non-empty dict")

        devices = set()

        for task, loss in task_losses.items():

            if not torch.is_tensor(loss):
                raise TypeError(f"{task}: loss must be tensor")

            if loss.numel() != 1:
                raise ValueError(f"{task}: loss must be scalar")

            if not torch.isfinite(loss):
                raise RuntimeError(f"{task}: non-finite loss detected")

            devices.add(loss.device)

        if len(devices) > 1:
            raise RuntimeError("All task losses must be on same device")

    def _mark_initialized(self) -> None:
        self._initialized = True

    # =========================================================
    # DEBUGGING
    # =========================================================

    def debug_state(self) -> Dict[str, float]:
        return {}

    def extra_repr(self) -> str:
        return f"{self.__class__.__name__}(initialized={self._initialized})"