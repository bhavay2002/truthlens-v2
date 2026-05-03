from __future__ import annotations

import logging
from typing import Dict, List

import torch
import torch.nn as nn

from .base_balancer import BaseBalancer

logger = logging.getLogger(__name__)


class GradNormBalancer(BaseBalancer):

    def __init__(
        self,
        task_names: List[str],
        alpha: float = 1.5,
    ) -> None:
        super().__init__()

        if not task_names:
            raise ValueError("task_names must be non-empty")

        self.task_names = task_names
        self.alpha = float(alpha)

        self.log_weights = nn.Parameter(torch.zeros(len(task_names)))

        self.initial_losses: Dict[str, float] = {}
        self._initialized = False

        self._last_grad_norms: Dict[str, torch.Tensor] = {}

    # =========================================================
    # COMBINE (FIXED)
    # =========================================================

    def combine(
        self,
        task_losses: Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        # N6: only consider tasks that actually contributed a loss to
        # this batch. The previous ``task_losses[t]`` indexing raised a
        # KeyError when a task was missing (real after we added the
        # ignore-index path through ``_compute_task_loss``) AND it
        # silently included tasks whose loss was ``None`` in the wrong
        # rows of ``self.log_weights``. Iterate by *position* so the
        # parameter and weight dimensions stay aligned; skip rows whose
        # loss is missing or ``None``.
        active_indices: list[int] = []
        active_losses: list[torch.Tensor] = []

        for i, t in enumerate(self.task_names):
            loss = task_losses.get(t)
            if loss is None:
                continue
            active_indices.append(i)
            active_losses.append(loss)

        if not active_losses:
            raise RuntimeError(
                "GradNormBalancer.combine() received no active task "
                "losses; nothing to combine."
            )

        # Select the *active* rows of log_weights (preserving autograd)
        # and renormalise so the mean remains ~1 across active tasks.
        active_idx_tensor = torch.tensor(
            active_indices,
            device=self.log_weights.device,
        )
        active_log_weights = self.log_weights.index_select(
            0, active_idx_tensor
        )
        weights = torch.exp(active_log_weights)
        weights = weights * (len(weights) / weights.sum().detach())

        losses = torch.stack(active_losses)

        if not self._initialized:
            # Record initial losses only for the tasks that showed up
            # this batch; tasks that arrive later will be initialised
            # on their first appearance via the same code path.
            self.initial_losses = {
                t: float(task_losses[t].detach().item())
                for t in self.task_names
                if task_losses.get(t) is not None
            }
            self._initialized = True
        else:
            # Late-arriving task — record its initial loss now so the
            # subsequent ``current / initial`` ratio in
            # ``on_before_backward`` is well-defined.
            for t in self.task_names:
                loss_t = task_losses.get(t)
                if loss_t is None or t in self.initial_losses:
                    continue
                self.initial_losses[t] = float(loss_t.detach().item())

        weighted_losses = weights * losses

        return weighted_losses.sum()

    # =========================================================
    # GRADNORM CORE
    # =========================================================

    def on_before_backward(
        self,
        task_losses: Dict[str, torch.Tensor],
        shared_parameters,
    ) -> None:

        if not self._initialized:
            return

        if shared_parameters is None:
            raise RuntimeError("GradNorm requires shared_parameters")

        # N6: same active-task filter as ``combine`` — operate only on
        # the rows of ``log_weights`` whose task showed up in this
        # batch and whose initial loss has been recorded. Iterating by
        # position keeps parameter / weight / grad-norm dimensions in
        # lock-step.
        active_indices: list[int] = []
        active_tasks: list[str] = []
        active_losses: list[torch.Tensor] = []

        for i, task in enumerate(self.task_names):
            loss = task_losses.get(task)
            if loss is None or task not in self.initial_losses:
                continue
            active_indices.append(i)
            active_tasks.append(task)
            active_losses.append(loss)

        if not active_losses:
            return

        weights = torch.exp(self.log_weights)
        weights = weights * (len(weights) / weights.sum().detach())

        grad_norms = []

        for idx, loss in zip(active_indices, active_losses):

            grads = torch.autograd.grad(
                weights[idx] * loss,
                shared_parameters,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )

            valid_grads = [g.norm() for g in grads if g is not None]

            if not valid_grads:
                grad_norm = torch.tensor(0.0, device=weights.device)
            else:
                grad_norm = torch.stack(valid_grads).mean()

            grad_norms.append(grad_norm)

        grad_norms = torch.stack(grad_norms)

        self._last_grad_norms = {
            t: g.detach() for t, g in zip(active_tasks, grad_norms)
        }

        current_losses = torch.tensor(
            [loss.detach().item() for loss in active_losses],
            device=grad_norms.device,
        )

        initial_losses = torch.tensor(
            [self.initial_losses[t] for t in active_tasks],
            device=grad_norms.device,
        )

        loss_ratios = current_losses / initial_losses.clamp_min(1e-8)

        avg_ratio = loss_ratios.mean()
        relative_rates = loss_ratios / avg_ratio

        avg_grad_norm = grad_norms.mean().detach()

        target_grad_norms = avg_grad_norm * (relative_rates ** self.alpha)

        grad_loss = torch.nn.functional.l1_loss(
            grad_norms,
            target_grad_norms.detach(),
        )

        grad_loss.backward(retain_graph=True)

    # =========================================================
    # DEBUG
    # =========================================================

    def get_last_grad_norms(self) -> Dict[str, float]:
        return {
            k: float(v.item()) for k, v in self._last_grad_norms.items()
        }

    def get_weights(self) -> Dict[str, float]:
        weights = torch.exp(self.log_weights).detach()
        weights = weights * (len(weights) / weights.sum())
        return {
            t: float(w.item())
            for t, w in zip(self.task_names, weights)
        }