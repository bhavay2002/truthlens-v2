#src\models\training\evaluation_engine.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn

from src.training.training_utils import move_batch_to_device

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class EvaluationConfig:
    task_types: Dict[str, str]
    device: Optional[str] = None
    ignore_index: int = -100
    threshold: float = 0.5

    # EVAL-MULTILABEL-SLICE: Per-task surviving multilabel column indices,
    # identical contract to ``LossEngineConfig.valid_label_indices``. When
    # the loss-balancer drops degenerate columns from the train split (e.g.
    # emotion: 11 kept of 20), the dataset emits labels of shape
    # ``[B, K_kept]`` while the model head still emits full-width logits of
    # shape ``[B, C_full]``. The training loss path handles this by slicing
    # logits via ``TaskLossRouter._multilabel_loss`` (lines 179-190 of
    # ``src/models/loss/task_loss_router.py``); without the same slicing
    # here, the multilabel evaluator hits
    #   ``IndexError: The shape of the mask [B, K_kept]
    #     does not match tensor [B, C_full]``
    # at ``preds = preds[mask]`` on the very first val batch. ``None`` (the
    # default) preserves the original full-width behaviour for callers /
    # tasks that don't drop columns.
    valid_label_indices: Optional[Dict[str, List[int]]] = None


# =========================================================
# STREAMING METRICS  (PERF-1: keep accumulators on-device)
# =========================================================
#
# Original implementation called ``.sum().item()`` on every batch which
# forces a host-device sync per metric per batch (3-6× val-loop slowdown
# on GPU). The new metrics:
#   * lazily allocate accumulator tensors on the FIRST batch's device
#   * never sync inside ``update`` — only on ``compute``
#   * expose ``sync_distributed`` so DDP can SUM-reduce raw counters
#     (PERF-2: correct DDP merging requires (sum, count) reductions, not
#     pre-divided averages)
# =========================================================


def _all_reduce_sum(*tensors: torch.Tensor) -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        for t in tensors:
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)


class StreamingAccuracy:
    def __init__(self):
        self.correct: Optional[torch.Tensor] = None  # device tensor, lazy
        self.total: int = 0                          # host int (numel is metadata, no sync)

    def update(self, preds, labels):
        if self.correct is None:
            self.correct = torch.zeros((), device=preds.device, dtype=torch.float64)
        self.correct = self.correct + (preds == labels).sum().to(self.correct.dtype)
        self.total += int(labels.numel())

    def sync_distributed(self):
        if self.correct is None:
            return
        total_t = torch.tensor(
            float(self.total),
            device=self.correct.device,
            dtype=self.correct.dtype,
        )
        _all_reduce_sum(self.correct, total_t)
        self.total = int(total_t.item())

    def compute(self):
        if self.correct is None:
            return 0.0
        return float(self.correct.item()) / max(self.total, 1)

    def reset(self):
        self.correct = None
        self.total = 0


class StreamingF1:
    def __init__(self):
        self.tp: Optional[torch.Tensor] = None
        self.fp: Optional[torch.Tensor] = None
        self.fn: Optional[torch.Tensor] = None

    def _ensure(self, device):
        if self.tp is None:
            self.tp = torch.zeros((), device=device, dtype=torch.float64)
            self.fp = torch.zeros((), device=device, dtype=torch.float64)
            self.fn = torch.zeros((), device=device, dtype=torch.float64)

    def update(self, preds, labels):
        self._ensure(preds.device)
        preds = preds.view(-1).to(torch.int64)
        labels = labels.view(-1).to(torch.int64)
        self.tp += torch.logical_and(preds == 1, labels == 1).sum().to(self.tp.dtype)
        self.fp += torch.logical_and(preds == 1, labels == 0).sum().to(self.fp.dtype)
        self.fn += torch.logical_and(preds == 0, labels == 1).sum().to(self.fn.dtype)

    def sync_distributed(self):
        if self.tp is None:
            return
        _all_reduce_sum(self.tp, self.fp, self.fn)

    def compute(self):
        if self.tp is None:
            return 0.0
        tp = float(self.tp.item())
        fp = float(self.fp.item())
        fn = float(self.fn.item())
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        return 2 * precision * recall / (precision + recall + 1e-12)

    def reset(self):
        self.tp = self.fp = self.fn = None


class StreamingMSE:
    def __init__(self):
        self.sum_sq: Optional[torch.Tensor] = None
        self.count: int = 0

    def update(self, preds, targets):
        if self.sum_sq is None:
            self.sum_sq = torch.zeros((), device=preds.device, dtype=torch.float64)
        diff = preds.float() - targets.float()
        self.sum_sq = self.sum_sq + (diff ** 2).sum().to(self.sum_sq.dtype)
        self.count += int(targets.numel())

    def sync_distributed(self):
        if self.sum_sq is None:
            return
        count_t = torch.tensor(
            float(self.count),
            device=self.sum_sq.device,
            dtype=self.sum_sq.dtype,
        )
        _all_reduce_sum(self.sum_sq, count_t)
        self.count = int(count_t.item())

    def compute(self):
        if self.sum_sq is None:
            return 0.0
        return float(self.sum_sq.item()) / max(self.count, 1)

    def reset(self):
        self.sum_sq = None
        self.count = 0


# =========================================================
# EVALUATION ENGINE
# =========================================================

class EvaluationEngine:

    def __init__(self, config: EvaluationConfig):

        self.config = config

        self.device = torch.device(
            config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info("EvaluationEngine initialized | device=%s", self.device)

    # =====================================================
    # MAIN
    # =====================================================

    @torch.inference_mode()
    def evaluate(self, model: nn.Module, dataloader) -> Dict[str, Any]:

        # GPU-4: ``model.to(device)`` was called unconditionally on every
        # ``evaluate`` invocation. When the model is already on the right
        # device this is a no-op cost; when the model has been wrapped
        # (DDP / torch.compile) re-moving it can break those wrappers
        # (DDP holds its device_ids at wrap time and an in-place move
        # corrupts the bucket assignment). Probe the first parameter and
        # only move when the device actually differs.
        try:
            current_device = next(model.parameters()).device
        except StopIteration:
            current_device = self.device

        if current_device != self.device:
            model = model.to(self.device)

        model.eval()

        metrics = self._init_metrics()

        for i, batch in enumerate(dataloader):

            batch = self._move_batch(batch)

            # Strip non-tensor metadata (``task``) injected by collate
            # so the strict single-task model signatures don't choke.
            model_batch = {
                k: v for k, v in batch.items()
                if k not in ("task",)
            }
            outputs = model(**model_batch)

            self._update_metrics(metrics, outputs, batch)

        return self._compute_metrics(metrics)

    # =====================================================
    # METRICS INIT
    # =====================================================

    def _init_metrics(self):

        metrics = {}

        for task, ttype in self.config.task_types.items():

            if ttype == "multiclass":
                metrics[task] = StreamingAccuracy()

            elif ttype == "multilabel":
                metrics[task] = StreamingF1()

            # MT-2: ``binary`` was previously dropped on the floor — no
            # metric was constructed for tasks like ``propaganda`` (binary
            # in the registry), so ``_compute_metrics`` returned an empty
            # ``{task}_score`` and the Trainer's early-stopping monitor
            # never saw a value → training silently consumed the full
            # epoch budget. Treat binary as the 1-D analogue of multilabel:
            # ``StreamingF1`` already operates on {0, 1} predictions and
            # labels, so it works directly once we threshold the sigmoid
            # of a single logit per sample (handled in ``_update_metrics``).
            elif ttype == "binary":
                metrics[task] = StreamingF1()

            elif ttype == "regression":
                metrics[task] = StreamingMSE()

        return metrics

    # =====================================================
    # UPDATE
    # =====================================================

    def _update_metrics(self, metrics, outputs, batch):

        task_logits = outputs.get("task_logits")
        # Single-task model classes emit ``outputs["logits"]`` (a single
        # tensor) instead of the multi-head ``task_logits`` dict the
        # MultiTaskTruthLensModel produces. Mirror the LossEngine
        # synthesis so single-task evaluation works without forcing
        # every model class to emit both shapes.
        if task_logits is None and "logits" in outputs and len(self.config.task_types) == 1:
            only_task = next(iter(self.config.task_types.keys()))
            task_logits = {only_task: outputs["logits"]}
        if task_logits is None:
            return

        # EDGE-2: ``batch["labels"]`` may be a single tensor (single-task
        # collate) instead of a per-task dict (multi-task collate). The
        # previous ``task not in batch["labels"]`` did ``in`` on a tensor
        # and raised ``TypeError`` mid-eval.  Normalize to a dict here so
        # both collate styles work transparently — single-tensor batches
        # are interpreted as labels for the *only* task in ``task_logits``.
        raw_labels = batch.get("labels")
        if raw_labels is None:
            return

        if isinstance(raw_labels, dict):
            labels_by_task = raw_labels
        elif isinstance(raw_labels, torch.Tensor):
            if len(task_logits) != 1:
                logger.warning(
                    "EDGE-2: batch['labels'] is a tensor but the model "
                    "produced %d task heads — cannot disambiguate. "
                    "Pass labels as a {task: tensor} dict for multi-task.",
                    len(task_logits),
                )
                return
            only_task = next(iter(task_logits.keys()))
            labels_by_task = {only_task: raw_labels}
        else:
            logger.warning(
                "EDGE-2: unexpected batch['labels'] type %s — skipping "
                "metric update.",
                type(raw_labels).__name__,
            )
            return

        for task, logits in task_logits.items():

            if task not in labels_by_task:
                continue

            labels = labels_by_task[task].to(logits.device)
            ttype = self.config.task_types.get(task)

            if ttype == "multiclass":

                preds = torch.argmax(logits, dim=-1)

                if labels.dim() == 2:
                    labels = labels.argmax(dim=-1)

                mask = labels != self.config.ignore_index
                preds = preds[mask]
                labels = labels[mask]

            elif ttype == "multilabel":

                # EVAL-MULTILABEL-SLICE: when the loss-balancer dropped
                # degenerate columns from the train split, ``labels`` here
                # has shape ``[B, K_kept]`` (from
                # ``MultiLabelDataset(valid_label_indices=…)``) while
                # ``logits`` is still the head's full ``[B, C_full]``.
                # Slice logits down to the same surviving columns so
                # ``preds[mask]`` doesn't crash with
                #   ``IndexError: shape of the mask [B, K_kept] does not
                #     match tensor [B, C_full]``.
                # This mirrors ``TaskLossRouter._multilabel_loss``
                # (src/models/loss/task_loss_router.py:179-190) so the
                # evaluation slicing can never silently disagree with the
                # training loss slicing — both consult the same per-task
                # ``valid_label_indices`` map produced once on the train
                # split by ``training.loss_balancer.plan_for_dataframe``.
                valid_idx = (
                    (self.config.valid_label_indices or {}).get(task)
                )
                if (
                    valid_idx is not None
                    and len(valid_idx) != logits.shape[-1]
                ):
                    if logits.shape[-1] < max(valid_idx) + 1:
                        raise ValueError(
                            f"{task}: valid_label_indices reference column "
                            f"{max(valid_idx)} but logits have width "
                            f"{logits.shape[-1]}"
                        )
                    idx_t = torch.as_tensor(
                        valid_idx, dtype=torch.long, device=logits.device,
                    )
                    logits = logits.index_select(-1, idx_t)

                preds = (torch.sigmoid(logits) > self.config.threshold).int()

                # Surface a clear error if shapes still disagree (e.g. a
                # caller forgot to pass ``valid_label_indices`` to the
                # EvaluationConfig but the dataset is dropping columns).
                # Without this guard the next line crashes with the
                # opaque PyTorch ``shape of the mask … does not match
                # tensor …`` message that buries the real cause.
                if preds.shape != labels.shape:
                    raise ValueError(
                        f"{task}: multilabel eval shape mismatch — "
                        f"preds {tuple(preds.shape)} vs labels "
                        f"{tuple(labels.shape)}. If the loss-balancer "
                        f"dropped degenerate columns, pass "
                        f"valid_label_indices={{'{task}': […]}} into "
                        f"EvaluationConfig (mirrors LossEngineConfig)."
                    )

                mask = labels != self.config.ignore_index
                preds = preds[mask]
                labels = labels[mask]

            # MT-2: binary head emits a single logit per sample (shape
            # ``[B]`` or ``[B, 1]``). Flatten to ``[B]`` so the predictions
            # and labels both reduce to {0, 1} 1-D tensors that match
            # StreamingF1's update contract.
            elif ttype == "binary":

                if logits.dim() > 1 and logits.size(-1) == 1:
                    logits = logits.squeeze(-1)

                preds = (torch.sigmoid(logits) > self.config.threshold).int()

                if labels.dim() > 1 and labels.size(-1) == 1:
                    labels = labels.squeeze(-1)

                mask = labels != self.config.ignore_index
                preds = preds[mask]
                labels = labels[mask]

            elif ttype == "regression":

                preds = logits

                # N-LOW-2: regression targets can legitimately contain
                # NaN sentinels (e.g. missing-value rows that survived a
                # join, or quarantined samples flagged by a label-quality
                # filter). Without a finite mask those NaNs propagate into
                # MSE / MAE accumulators and silently poison the metric
                # for the whole epoch. Mask non-finite AND ignore_index.
                finite_mask = torch.isfinite(labels)
                if labels.dtype.is_floating_point:
                    keep = finite_mask & (labels != self.config.ignore_index)
                else:
                    keep = labels != self.config.ignore_index
                preds = preds[keep]
                labels = labels[keep]

            else:
                continue

            # PERF-1: keep tensors on device — Streaming* metrics now hold
            # accumulators on the GPU and only sync at compute() time.
            metrics[task].update(preds.detach(), labels.detach())

    # =====================================================
    # COMPUTE
    # =====================================================

    def _compute_metrics(self, metrics):

        results = {}

        for task, metric in metrics.items():

            # PERF-2: Reduce raw (numerator, denominator) accumulators across
            # ranks BEFORE dividing — averaging post-divided rank-local scores
            # is mathematically wrong when shards have different sample
            # counts (drop_last=False). Each Streaming* metric implements
            # `sync_distributed` which all_reduces only its raw counters.
            if hasattr(metric, "sync_distributed"):
                metric.sync_distributed()

            results[f"{task}_score"] = metric.compute()

        return results

    # =====================================================
    # DEVICE
    # =====================================================

    def _move_batch(self, batch):
        # GPU-2: ``non_blocking=True`` is silently ignored unless the
        # source tensor is in pinned memory AND the destination is CUDA.
        # The previous inline lambda set ``non_blocking=True``
        # unconditionally — which gave a false impression of an async
        # H2D copy on CPU runs and on un-pinned tensors. Delegate to the
        # shared utility that gates ``non_blocking`` on
        # ``tensor.is_pinned()`` so the flag carries its real meaning.
        return move_batch_to_device(batch, self.device, non_blocking=True)