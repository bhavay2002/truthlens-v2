from __future__ import annotations

import logging
from typing import Dict, List

import torch
import torch.nn as nn

from .base_balancer import BaseBalancer

logger = logging.getLogger(__name__)


class UncertaintyBalancer(BaseBalancer):

    def __init__(
        self,
        task_names: List[str],
        init_log_var: float = 0.0,
        clamp_min: float = -10.0,
        clamp_max: float = 10.0,
    ) -> None:
        super().__init__()

        if not task_names:
            raise ValueError("task_names must be non-empty")

        self.task_names = task_names

        self.log_vars = nn.ParameterDict({
            t: nn.Parameter(torch.tensor([init_log_var], dtype=torch.float32))
            for t in task_names
        })

        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        logger.info(
            "UncertaintyBalancer initialized | tasks=%s",
            task_names,
        )

    # =========================================================
    # COMBINE (FIXED)
    # =========================================================

    def combine(
        self,
        task_losses: Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        total_loss = None

        for task in self.task_names:

            if task not in task_losses:
                continue

            loss = task_losses[task]
            log_var = self.log_vars[task]

            # safer clamp
            log_var_clamped = log_var.clamp(
                self.clamp_min,
                self.clamp_max,
            )

            precision = torch.exp(-log_var_clamped)

            weighted = precision * loss + log_var_clamped

            total_loss = weighted if total_loss is None else total_loss + weighted

        if total_loss is None:
            return torch.zeros((), requires_grad=True)

        return total_loss

    # =========================================================
    # UTILITIES
    # =========================================================

    def get_weights(self) -> Dict[str, float]:
        weights = {}

        for t, log_var in self.log_vars.items():
            precision = torch.exp(-log_var.detach())
            weights[t] = float(precision.item())

        return weights

    def get_log_vars(self) -> Dict[str, float]:
        return {
            t: float(v.detach().item())
            for t, v in self.log_vars.items()
        }

    def extra_repr(self) -> str:
        return f"tasks={self.task_names}"