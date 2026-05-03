"""Multi-task DataLoader iterator.

`MultiTaskLoader` mixes a set of per-task ``torch.utils.data.DataLoader``
objects into a single batch stream that downstream multi-task training
code (``MultiTaskTruthLensModel`` + ``LossEngine``/``MultiTaskLoss``)
can consume directly.

Contract enforced on every emitted batch
----------------------------------------
1. The batch is the *entire* batch produced by exactly ONE underlying
   per-task loader. We do NOT mix samples from different tasks inside
   a single forward pass — the per-task heads + per-task loss routing
   require single-task batches.

2. ``batch["task"]`` (set by ``src.data_processing.collate``) is left
   untouched so the training step / scheduler / instrumentation layer
   can route on it.

3. ``batch["labels"]`` is rewrapped from ``Tensor`` into
   ``{task_name: Tensor}``.  ``MultiTaskLoss.forward`` requires labels
   as a per-task ``dict`` (it iterates ``self.task_names`` and skips
   tasks that are absent from the dict). Without this rewrap the
   single-task batch contract from ``ClassificationDataset`` /
   ``MultiLabelDataset`` is incompatible with ``MultiTaskLoss`` and
   the loss step crashes with ``TypeError("logits and labels must be
   dict")``. We perform the wrap here — at the multi-task batch
   boundary — so neither the per-task datasets nor the engine layer
   need to know about it. The rewrap is idempotent: dict labels pass
   through unchanged.

Sampling strategies
-------------------
- ``"weighted"``: at each step, sample one task with probability
  proportional to ``task_weights`` (with replacement). Per-task
  iterators wrap when exhausted, so a single epoch of the multi-task
  loader corresponds to ``sum(len(loader) for loader in dataloaders.values())``
  steps — every per-task loader gets at least one full pass on
  expectation under uniform weights, and longer loaders get
  proportionally more updates. This is the recommended path for
  joint multi-task training over a shared encoder.

- ``"round_robin"``: cycles tasks deterministically in registration
  order. Iteration stops when the SHORTEST per-task loader is
  exhausted (so every task contributes the same number of updates per
  epoch). ``__len__`` reflects this: ``num_tasks * min(per-task lens)``.
  Used for validation, where every task should be seen the same number
  of times and reproducibility matters.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, Iterator, Mapping, Optional

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTS
# =========================================================

VALID_STRATEGIES = ("weighted", "round_robin")


# =========================================================
# MULTI-TASK LOADER
# =========================================================

class MultiTaskLoader:
    """Iterator over a dict of per-task DataLoaders.

    See module docstring for the contract details.
    """

    def __init__(
        self,
        dataloaders: Mapping[str, DataLoader],
        task_weights: Optional[Mapping[str, float]] = None,
        strategy: str = "weighted",
        *,
        seed: int = 42,
    ) -> None:

        if not dataloaders:
            raise ValueError("dataloaders cannot be empty")

        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy: {strategy!r}. "
                f"Must be one of {VALID_STRATEGIES}."
            )

        # MT-FACTORY: lock task ordering at construction time so the
        # round-robin path is deterministic and the weighted path's
        # rng draws are reproducible across runs with the same seed.
        self.tasks = list(dataloaders.keys())
        self.dataloaders: Dict[str, DataLoader] = dict(dataloaders)
        self.strategy = strategy

        # MT-FACTORY: normalize weights once. Missing tasks default to
        # 1.0 so callers don't have to enumerate every key. Zero / None
        # collapses to uniform — failing here would be more annoying
        # than helpful at config-load time.
        self.task_weights = self._normalize_weights(task_weights)

        self._rng = random.Random(seed)
        self._rr_index = 0

        # Lazy iterators — built on first __iter__ call.
        self._iters: Dict[str, Iterator] = {}

        logger.info(
            "MultiTaskLoader initialized | strategy=%s | tasks=%s | weights=%s",
            self.strategy,
            self.tasks,
            self.task_weights,
        )

    # =====================================================
    # PUBLIC API
    # =====================================================

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # MT-FACTORY: re-arm a fresh iterator for every per-task loader
        # at the start of every epoch. This is the only place we call
        # ``iter(loader)`` so the underlying DataLoader workers (when
        # ``persistent_workers=True``) get a clean restart per epoch
        # without us holding stale iterator handles.
        self._iters = {
            task: iter(loader)
            for task, loader in self.dataloaders.items()
        }
        self._rr_index = 0

        for _ in range(len(self)):
            task = self._next_task()

            try:
                batch = next(self._iters[task])
            except StopIteration:
                if self.strategy == "weighted":
                    # Weighted sampling treats a task's epoch as
                    # "wrap-around": rebuild the iterator and try once.
                    # If it fails twice in a row the loader is empty.
                    self._iters[task] = iter(self.dataloaders[task])
                    try:
                        batch = next(self._iters[task])
                    except StopIteration:
                        raise RuntimeError(
                            f"DataLoader for task={task!r} is empty"
                        )
                else:
                    # Round-robin: shortest loader determines epoch end.
                    return

            yield self._prepare_batch(batch, task)

    def __len__(self) -> int:
        if self.strategy == "weighted":
            return sum(len(loader) for loader in self.dataloaders.values())

        # round_robin: shortest loader caps the epoch
        shortest = min(len(loader) for loader in self.dataloaders.values())
        return shortest * len(self.tasks)

    # =====================================================
    # INTERNALS
    # =====================================================

    def _next_task(self) -> str:
        if self.strategy == "round_robin":
            task = self.tasks[self._rr_index]
            self._rr_index = (self._rr_index + 1) % len(self.tasks)
            return task

        # weighted
        weights = [self.task_weights[t] for t in self.tasks]
        return self._rng.choices(self.tasks, weights=weights, k=1)[0]

    def _prepare_batch(
        self,
        batch: Dict[str, Any],
        task: str,
    ) -> Dict[str, Any]:

        if not isinstance(batch, dict):
            raise TypeError(
                "MultiTaskLoader requires dataset batches to be dicts "
                f"(task={task!r} produced {type(batch).__name__}). "
                "Fix the dataset / collate_fn rather than reshaping here."
            )

        # MT-FACTORY: enforce single-task batches. ``collate.py`` already
        # raises on mixed-task batches at the per-task loader level; this
        # is a defensive double-check at the multi-task boundary so a
        # custom dataset that bypasses our collate can't silently feed
        # mixed-task batches to the loss router.
        observed_task = batch.get("task")
        if observed_task is not None and observed_task != task:
            raise RuntimeError(
                f"MultiTaskLoader sampled task={task!r} but the underlying "
                f"loader yielded a batch with task={observed_task!r}. "
                "Per-task loaders must produce single-task batches."
            )
        batch["task"] = task

        # MT-FACTORY: rewrap labels into the dict-of-task contract that
        # MultiTaskLoss requires. Idempotent — already-dict labels pass
        # through.
        labels = batch.get("labels")
        if labels is None:
            raise KeyError(
                f"Batch for task={task!r} is missing 'labels'. "
                "Datasets must emit a 'labels' tensor in every batch."
            )

        if isinstance(labels, torch.Tensor):
            batch["labels"] = {task: labels}
        elif isinstance(labels, dict):
            # Pass through, but enforce the key is present — this catches
            # the case where a caller pre-wrapped labels under the wrong
            # task name (which would silently no-op the loss for this
            # task because MultiTaskLoss skips missing keys).
            if task not in labels:
                raise KeyError(
                    f"Pre-wrapped labels dict for task={task!r} does not "
                    f"contain key {task!r} (keys={list(labels)})."
                )
        else:
            raise TypeError(
                f"Batch labels for task={task!r} must be Tensor or dict "
                f"(got {type(labels).__name__})."
            )

        return batch

    def _normalize_weights(
        self,
        weights: Optional[Mapping[str, float]],
    ) -> Dict[str, float]:

        if not weights:
            return {t: 1.0 for t in self.tasks}

        out: Dict[str, float] = {}
        for t in self.tasks:
            w = float(weights.get(t, 1.0))
            if w < 0:
                raise ValueError(
                    f"task_weights[{t!r}] = {w} must be non-negative"
                )
            out[t] = w

        total = sum(out.values())
        if total <= 0:
            logger.warning(
                "MultiTaskLoader: all task_weights are zero; falling back "
                "to uniform sampling."
            )
            return {t: 1.0 for t in self.tasks}

        return out


__all__ = ["MultiTaskLoader", "VALID_STRATEGIES"]
