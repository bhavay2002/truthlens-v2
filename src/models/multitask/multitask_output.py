from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import torch
 

# =========================================================
# TASK OUTPUT
# =========================================================

@dataclass
class TaskOutput:

    logits: torch.Tensor
    probabilities: Optional[torch.Tensor] = None
    predictions: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None

    # -------------------------
    # UTILITIES
    # -------------------------

    def detach(self) -> "TaskOutput":
        return TaskOutput(
            logits=self.logits.detach(),
            probabilities=self._safe_detach(self.probabilities),
            predictions=self._safe_detach(self.predictions),
            loss=self._safe_detach(self.loss),
        )

    def detach_(self) -> "TaskOutput":
        # P2.4: in-place detach. The fluent ``detach()`` returns a new
        # ``TaskOutput`` and a new tensor for every field, which doubles
        # transient memory while the autograd graph is still being torn
        # down. The vast majority of callers only need to break the
        # graph in place before serialising or storing outputs across
        # batches; for them ``detach_()`` avoids the extra allocations.
        self.logits = self.logits.detach()
        if isinstance(self.probabilities, torch.Tensor):
            self.probabilities = self.probabilities.detach()
        if isinstance(self.predictions, torch.Tensor):
            self.predictions = self.predictions.detach()
        if isinstance(self.loss, torch.Tensor):
            self.loss = self.loss.detach()
        return self

    def to(self, device: torch.device) -> "TaskOutput":
        return TaskOutput(
            logits=self.logits.to(device),
            probabilities=self._safe_to(self.probabilities, device),
            predictions=self._safe_to(self.predictions, device),
            loss=self._safe_to(self.loss, device),
        )

    def _safe_detach(self, x):
        return x.detach() if isinstance(x, torch.Tensor) else x

    def _safe_to(self, x, device):
        return x.to(device) if isinstance(x, torch.Tensor) else x


# =========================================================
# MULTI TASK OUTPUT
# =========================================================

@dataclass
class MultiTaskOutput:

    tasks: Dict[str, TaskOutput] = field(default_factory=dict)

    loss: Optional[torch.Tensor] = None
    task_losses: Optional[Dict[str, torch.Tensor]] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    # =====================================================
    # FACTORY
    # =====================================================

    @classmethod
    def from_model_outputs(
        cls,
        outputs: Dict[str, Any],
        *,
        task_names: Optional[Iterable[str]] = None,
    ) -> "MultiTaskOutput":
        """Adapt a model's raw forward dict into a :class:`MultiTaskOutput`.

        A3.7 — the previous "any dict-with-'logits' key counts as a task"
        fallback was a footgun: a stray entry like ``outputs["debug"] =
        {"logits": ...}`` would be interpreted as a task and pollute the
        loss / metrics surface. We now require either a ``task_logits``
        dict (the canonical fast path) or an explicit ``task_names``
        whitelist supplied by the caller (typically
        ``model.get_task_names()``). Legacy auto-discovery is gone.
        """

        if isinstance(outputs.get("multitask_output"), MultiTaskOutput):
            return outputs["multitask_output"]

        multitask = cls()

        # FAST PATH — single source of truth.
        if "task_logits" in outputs:
            task_logits = outputs["task_logits"]
            for task, logits in task_logits.items():
                payload = outputs.get(task) if isinstance(outputs.get(task), dict) else {}
                multitask.tasks[task] = TaskOutput(
                    logits=logits,
                    probabilities=payload.get("probabilities"),
                    predictions=payload.get("predictions"),
                    loss=payload.get("loss"),
                )

        # WHITELIST PATH — caller knows the task names.
        elif task_names is not None:
            for task_name in task_names:
                payload = outputs.get(task_name)
                if not isinstance(payload, dict):
                    raise KeyError(
                        f"Task {task_name!r} not present in model outputs "
                        f"(have: {sorted(k for k in outputs if isinstance(k, str))})"
                    )
                logits = payload.get("logits")
                if not isinstance(logits, torch.Tensor):
                    raise TypeError(
                        f"Task {task_name!r}: 'logits' missing or not a Tensor"
                    )
                multitask.tasks[task_name] = TaskOutput(
                    logits=logits,
                    probabilities=payload.get("probabilities"),
                    predictions=payload.get("predictions"),
                    loss=payload.get("loss"),
                )
        else:
            raise RuntimeError(
                "from_model_outputs requires either 'task_logits' in the "
                "model output dict or an explicit task_names= whitelist "
                "(typically model.get_task_names()). Auto-discovery has "
                "been removed (A3.7)."
            )

        multitask.loss = outputs.get("loss")
        multitask.task_losses = outputs.get("task_losses")

        return multitask

    # =====================================================
    # LOSS ENGINE INTERFACE ( IMPORTANT)
    # =====================================================

    def to_loss_inputs(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Convert to LossEngine-compatible format.

        Returns:
            {
                "task_logits": {...}
            }
        """
        return {
            "task_logits": {
                task: out.logits for task, out in self.tasks.items()
            }
        }

    # =====================================================
    # ACCESSORS
    # =====================================================

    def get_logits(self, task_name: str) -> torch.Tensor:
        return self.tasks[task_name].logits

    def get_predictions(self, task_name: str):
        return self.tasks[task_name].predictions

    def get_probabilities(self, task_name: str):
        return self.tasks[task_name].probabilities

    def get_task_loss(self, task_name: str):
        return self.tasks[task_name].loss

    # =====================================================
    # DEVICE OPS
    # =====================================================

    def to(self, device: torch.device) -> "MultiTaskOutput":
        return MultiTaskOutput(
            tasks={k: v.to(device) for k, v in self.tasks.items()},
            loss=self.loss.to(device) if isinstance(self.loss, torch.Tensor) else self.loss,
            task_losses={
                k: v.to(device) for k, v in (self.task_losses or {}).items()
            } if self.task_losses else None,
            metadata=self.metadata,
        )

    def detach(self) -> "MultiTaskOutput":
        return MultiTaskOutput(
            tasks={k: v.detach() for k, v in self.tasks.items()},
            loss=self.loss.detach() if isinstance(self.loss, torch.Tensor) else self.loss,
            task_losses={
                k: v.detach() for k, v in (self.task_losses or {}).items()
            } if self.task_losses else None,
            metadata=self.metadata,
        )

    def detach_(self) -> "MultiTaskOutput":
        # P2.4: in-place detach across the whole multi-task output.
        # See ``TaskOutput.detach_`` — same rationale, applied
        # recursively. Returns ``self`` for fluent chaining.
        for task_output in self.tasks.values():
            task_output.detach_()
        if isinstance(self.loss, torch.Tensor):
            self.loss = self.loss.detach()
        if self.task_losses:
            for k, v in list(self.task_losses.items()):
                if isinstance(v, torch.Tensor):
                    self.task_losses[k] = v.detach()
        return self

    # =====================================================
    # SERIALIZATION
    # =====================================================

    def to_dict(self, detach: bool = True) -> Dict[str, Any]:

        result = {}

        for task_name, task_output in self.tasks.items():

            if detach:
                task_output = task_output.detach()

            result[task_name] = {
                "logits": task_output.logits,
                "probabilities": task_output.probabilities,
                "predictions": task_output.predictions,
                "loss": task_output.loss,
            }

        result["loss"] = self.loss.detach() if detach and isinstance(self.loss, torch.Tensor) else self.loss

        if self.task_losses:
            result["task_losses"] = {
                k: v.detach() if detach else v
                for k, v in self.task_losses.items()
            }

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    def to_flat_prediction_dict(self) -> Dict[str, Any]:

        flat = {}

        for task_name, task_output in self.tasks.items():

            flat[f"{task_name}_logits"] = task_output.logits

            if task_output.probabilities is not None:
                flat[f"{task_name}_probabilities"] = task_output.probabilities

            if task_output.predictions is not None:
                flat[f"{task_name}_predictions"] = task_output.predictions

        if self.loss is not None:
            flat["loss"] = self.loss

        if self.task_losses:
            flat["task_losses"] = self.task_losses

        if self.metadata:
            flat["metadata"] = self.metadata

        return flat