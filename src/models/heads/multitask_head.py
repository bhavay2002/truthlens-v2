from __future__ import annotations

import logging
from typing import Dict, Optional, Any

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class MultiTaskHead(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.task_heads: nn.ModuleDict = nn.ModuleDict()
        self.loss_fns: Dict[str, nn.Module] = {}
        # A4.2: task weights live in a ``BufferDict``-shaped registry so
        # they survive ``state_dict()`` / ``load_state_dict()`` round
        # trips. Previously we kept ``Dict[str, float]`` which was
        # invisible to checkpointing — a re-loaded model trained under
        # the default weight 1.0 instead of the tuned per-task value.
        # We register one zero-dim buffer per task name and look them up
        # via ``_task_weight_attr``; this also gives DDP / mixed-precision
        # contexts a real tensor to hook on if a future scheduler wants
        # to anneal task weights at runtime.
        self._task_weight_names: list[str] = []

    # =====================================================
    # REGISTER
    # =====================================================

    def register_task(
        self,
        task_name: str,
        head: nn.Module,
        loss_fn: Optional[nn.Module] = None,
        weight: float = 1.0,
    ) -> None:

        if not isinstance(task_name, str) or not task_name.strip():
            raise ValueError("task_name must be a valid string")

        if task_name in self.task_heads:
            raise ValueError(f"Task '{task_name}' already registered")

        if not isinstance(head, nn.Module):
            raise TypeError("head must be nn.Module")

        self.task_heads[task_name] = head

        if loss_fn is not None:
            self.loss_fns[task_name] = loss_fn

        # A4.2: register the per-task weight as a real buffer so it
        # round-trips through ``state_dict()`` and is visible to DDP.
        attr = self._task_weight_attr(task_name)
        self.register_buffer(attr, torch.tensor(float(weight)))
        self._task_weight_names.append(task_name)

        logger.info("Registered multitask head: %s", task_name)

    @staticmethod
    def _task_weight_attr(task_name: str) -> str:
        """Internal attribute name for a task's weight buffer.

        We can't use the bare task name because ``register_buffer`` and
        ``state_dict()`` share the module's attribute namespace with
        ``self.task_heads`` etc., and dotted task names would collide
        with the module-tree separator. The ``_task_weight__`` prefix
        makes the buffer easy to filter and impossible to confuse with
        a sub-module.
        """
        safe = task_name.replace(".", "__").replace("/", "__")
        return f"_task_weight__{safe}"

    def _get_task_weight(self, task_name: str) -> torch.Tensor:
        return getattr(self, self._task_weight_attr(task_name))

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:

        if features is None:
            raise ValueError("features cannot be None")

        if features.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {features.shape}")

        if not features.is_contiguous():
            features = features.contiguous()

        outputs: Dict[str, Any] = {
            "tasks": {},
        }

        # A4.1: accumulate WEIGHTED per-task losses in a list and reduce
        # ONCE via ``torch.stack(...).sum()``. The previous ``total_loss
        # + loss`` ternary allocated a fresh tensor and pushed an
        # autograd node per task, making the backward graph N levels
        # deeper than necessary for an N-task model.
        weighted_losses: list[torch.Tensor] = []

        for task_name, head in self.task_heads.items():

            head_output = head(features)

            # A3.4: dict contract is mandatory. Heads must return a dict
            # containing at least ``logits`` (see :class:`BaseHead`).
            # The previous tensor-fallback path silently passed broken
            # heads through training and only crashed at calibration.
            if not isinstance(head_output, dict):
                raise TypeError(
                    f"Task '{task_name}' head must return a dict (got "
                    f"{type(head_output).__name__}); see "
                    f"src.models.heads.base_head.BaseHead."
                )
            if "logits" not in head_output:
                raise RuntimeError(
                    f"Task '{task_name}' head dict missing 'logits' key "
                    f"(keys={list(head_output)})"
                )

            logits = head_output["logits"]

            # P2.3: only materialise the *rich* per-task dict
            # (``probabilities``, ``predictions``, ``confidence``,
            # ``entropy`` …) at evaluation time. During training those
            # derived tensors are never read — every loss function
            # consumes ``logits`` directly — but eagerly building them
            # keeps the corresponding ``softmax`` / ``argmax`` tensors
            # alive on the autograd graph for the entire backward pass.
            if self.training:
                task_output: Dict[str, Any] = {"logits": logits}
            else:
                # Defensively COPY the sub-head's dict before mutating
                # ``task_output["loss"]`` below — aliasing the sub-head's
                # own dict would leak the previous batch's loss key into
                # the next call.
                task_output = dict(head_output)

            outputs["tasks"][task_name] = task_output

            # -------------------------
            # LOSS
            # -------------------------
            if labels is not None and task_name in labels:

                if task_name not in self.loss_fns:
                    raise RuntimeError(
                        f"No loss function for task '{task_name}'"
                    )

                loss_fn = self.loss_fns[task_name]
                task_labels = labels[task_name]

                loss = loss_fn(logits, task_labels)

                # A4.4: publish exactly ONE loss key per task — the RAW
                # loss. The previous code stored both ``loss`` (raw) and
                # ``weighted_loss`` in the same dict; downstream callers
                # that summed ``loss`` AND added ``weighted_loss`` to a
                # parent objective would double-count. With a single key
                # the contract is unambiguous: the engine that computed
                # the loss owns weighting.
                task_output["loss"] = loss

                weight = self._get_task_weight(task_name)
                weighted_losses.append(weight * loss)

        if weighted_losses:
            # Single-allocation reduce + single autograd node (A4.1).
            outputs["total_loss"] = torch.stack(weighted_losses).sum()

        return outputs

    # =====================================================
    # PREDICT
    # =====================================================

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> Dict[str, Any]:
        # A6.3: ``predict`` is a *single* forward call, not a re-forward.
        # Earlier audit drafts flagged this as duplicated work because
        # ``MultiTaskHead.forward`` itself materialises the rich per-task
        # dict in eval mode (predictions / probabilities / confidence /
        # entropy). We then just re-shape that dict into a slimmer
        # prediction-only view below. The forward pass is run exactly
        # once; do not add caching here.

        was_training = self.training
        self.eval()

        try:
            outputs = self.forward(features)
        finally:
            if was_training:
                self.train()

        predictions: Dict[str, Any] = {}

        for task_name, task_output in outputs["tasks"].items():

            predictions[task_name] = {
                "predictions": task_output.get("predictions"),
                "probabilities": task_output.get("probabilities"),
                "confidence": task_output.get("confidence"),
            }

        return predictions

    # =====================================================
    # UTILS
    # =====================================================

    def get_tasks(self) -> Dict[str, nn.Module]:
        return dict(self.task_heads)

    def set_task_weight(self, task_name: str, weight: float) -> None:
        if task_name not in self.task_heads:
            raise ValueError(f"Task '{task_name}' not found")
        # A4.2: write into the registered buffer in-place so the new
        # value (a) survives ``state_dict()`` and (b) is visible to any
        # downstream module that already captured the buffer reference.
        buf = self._get_task_weight(task_name)
        with torch.no_grad():
            buf.fill_(float(weight))

    def get_task_weights(self) -> Dict[str, float]:
        return {
            name: float(self._get_task_weight(name).item())
            for name in self._task_weight_names
        }