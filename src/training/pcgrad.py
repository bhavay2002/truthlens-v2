"""
PCGradOptimizer — Explainability v2 §6 (PCGrad + task gating).

Implements PCGrad (Gradient Surgery for Multi-Task Learning, Yu et al. 2020)
as a drop-in wrapper around any PyTorch optimizer.

Algorithm
---------
For each task i, compute gradient g_i over the shared parameters.  For each
pair (i, j) where g_i · g_j < 0 (conflict), replace g_i with:

    g_i ← g_i − (g_i · g_j / ||g_j||²) · g_j

The final gradient applied is the average of the projected per-task vectors.

Task gating
-----------
When a ConfidenceFilter is attached, tasks whose gate_factor falls below
``gate_threshold`` are excluded from the gradient aggregation for that step.
This implements §6 "task gating" without requiring a separate code path.

Usage
-----
    from src.training.pcgrad import PCGradOptimizer

    opt = PCGradOptimizer(torch.optim.AdamW(model.parameters(), lr=2e-5))

    # In training loop:
    opt.zero_grad()
    losses = {task: compute_loss(task) for task in tasks}
    opt.pc_backward(losses)
    opt.step()

    # With task gating:
    opt = PCGradOptimizer(base_opt, gate_threshold=0.3)
    opt.set_gate_factors({"bias": 0.5, "propaganda": 0.1})  # 0.1 < 0.3 → gated out
    opt.pc_backward(losses)  # propaganda excluded
"""

from __future__ import annotations

import copy
import logging
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_EPS = 1e-12


# =========================================================
# PCGrad core
# =========================================================

def _project_conflicting(
    g_i: torch.Tensor,
    g_j: torch.Tensor,
) -> torch.Tensor:
    """Project g_i onto the normal plane of g_j when they conflict.

    If g_i · g_j >= 0 returns g_i unchanged (no conflict).
    Otherwise: g_i ← g_i − (g_i · g_j / ||g_j||²) · g_j
    """
    dot = torch.dot(g_i, g_j)
    if dot >= 0:
        return g_i
    norm_sq = g_j.dot(g_j).clamp_min(_EPS)
    return g_i - (dot / norm_sq) * g_j


def _pcgrad_project(
    task_grads: List[torch.Tensor],
) -> torch.Tensor:
    """Apply PCGrad projection over a list of per-task gradient vectors.

    Parameters
    ----------
    task_grads : List of 1-D gradient tensors (one per task), all same length.

    Returns
    -------
    Aggregated (mean) projected gradient, same shape as each input.
    """
    if not task_grads:
        raise ValueError("task_grads cannot be empty")

    projected: List[torch.Tensor] = []

    for i, g_i in enumerate(task_grads):
        g = g_i.clone()
        for j, g_j in enumerate(task_grads):
            if i == j:
                continue
            g = _project_conflicting(g, g_j)
        projected.append(g)

    return torch.stack(projected, dim=0).mean(dim=0)


# =========================================================
# PARAM FLATTENING HELPERS
# =========================================================

def _flatten_params(params: Iterable[nn.Parameter]) -> Tuple[torch.Tensor, List[torch.Size], List[int]]:
    """Flatten all parameter gradients into a single 1-D vector."""
    grads: List[torch.Tensor] = []
    shapes: List[torch.Size] = []
    numel: List[int] = []

    for p in params:
        if p.grad is None:
            g = torch.zeros_like(p).view(-1)
        else:
            g = p.grad.view(-1)
        grads.append(g)
        shapes.append(p.shape)
        numel.append(p.numel())

    return torch.cat(grads), shapes, numel


def _unflatten_into_params(
    flat: torch.Tensor,
    params: List[nn.Parameter],
    shapes: List[torch.Size],
    numel: List[int],
) -> None:
    """Write a flattened gradient vector back into parameter .grad fields."""
    offset = 0
    for p, shape, n in zip(params, shapes, numel):
        chunk = flat[offset: offset + n].view(shape)
        if p.grad is None:
            p.grad = chunk.clone()
        else:
            p.grad.copy_(chunk)
        offset += n


# =========================================================
# OPTIMIZER WRAPPER
# =========================================================

class PCGradOptimizer:
    """PCGrad wrapper for any PyTorch optimizer.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The underlying optimizer (AdamW, SGD, …).
    gate_threshold : float
        Gate factor below which a task's gradient is excluded.
        Default 0.0 means all tasks always participate.
    reduction : "mean" | "sum"
        How to aggregate projected per-task gradients.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        gate_threshold: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        self.optimizer = optimizer
        self.gate_threshold = gate_threshold
        self.reduction = reduction

        self._gate_factors: Dict[str, float] = {}

    # -------------------------------------------------------
    # GATE FACTORS
    # -------------------------------------------------------

    def set_gate_factors(self, gate_factors: Dict[str, float]) -> None:
        """Update per-task gate factors (from ConfidenceFilter or similar).

        Tasks with gate_factor < gate_threshold are excluded from the PCGrad
        aggregation step for the current batch.
        """
        self._gate_factors = dict(gate_factors)

    def _is_gated_in(self, task: str) -> bool:
        gf = self._gate_factors.get(task, 1.0)
        return gf >= self.gate_threshold

    # -------------------------------------------------------
    # PARAMETER GROUPS
    # -------------------------------------------------------

    def _all_params(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        seen: set = set()
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p not in seen and p.requires_grad:
                    params.append(p)
                    seen.add(p)
        return params

    # -------------------------------------------------------
    # PC_BACKWARD  (main entry point)
    # -------------------------------------------------------

    def pc_backward(
        self,
        task_losses: Dict[str, torch.Tensor],
        retain_graph: bool = False,
    ) -> None:
        """Compute per-task gradients, apply PCGrad projection, write back.

        Parameters
        ----------
        task_losses : {task_name: scalar_loss_tensor}
        retain_graph : whether to retain autograd graph after each backward
            (set True if you need it for other purposes).
        """
        params = self._all_params()

        # Filter to active (gated-in) tasks
        active_tasks = {
            task: loss
            for task, loss in task_losses.items()
            if self._is_gated_in(task)
        }

        if not active_tasks:
            logger.warning(
                "PCGradOptimizer.pc_backward: all tasks gated out "
                "(gate_threshold=%.3f). No gradient applied.",
                self.gate_threshold,
            )
            return

        gated_out = set(task_losses.keys()) - set(active_tasks.keys())
        if gated_out:
            logger.debug("PCGrad: gated-out tasks: %s", sorted(gated_out))

        n = len(active_tasks)
        task_grad_vecs: List[torch.Tensor] = []

        # ── Compute per-task gradient vectors ─────────────────────────
        for idx, (task, loss) in enumerate(active_tasks.items()):
            # Zero existing grads to isolate this task's contribution
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()

            keep_graph = retain_graph or (idx < n - 1)
            loss.backward(retain_graph=keep_graph)

            flat, shapes, numel = _flatten_params(params)
            task_grad_vecs.append(flat.clone())

        # ── PCGrad projection ─────────────────────────────────────────
        projected = _pcgrad_project(task_grad_vecs)

        if self.reduction == "sum":
            projected = projected * n

        # ── Write projected gradient back into .grad ──────────────────
        _unflatten_into_params(projected, params, shapes, numel)

    # -------------------------------------------------------
    # DELEGATE STANDARD OPTIMIZER METHODS
    # -------------------------------------------------------

    def step(self, closure: Optional[Callable] = None):
        return self.optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict) -> None:
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    def __repr__(self) -> str:
        return (
            f"PCGradOptimizer(optimizer={self.optimizer.__class__.__name__}, "
            f"gate_threshold={self.gate_threshold}, reduction={self.reduction!r})"
        )
