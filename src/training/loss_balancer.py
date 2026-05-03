"""
Loss-level balancing planner.

Given the labels of a single task's training set, decide:

* whether class-weighted CE / pos-weight is needed,
* whether the task is so imbalanced that focal loss should kick in,
* which multilabel columns are degenerate (only one class observed in
  the training fold) and must be dropped to avoid training "garbage
  heads" — these heads cannot learn anything useful and their gradients
  pollute the shared encoder.

This is the third layer of the imbalance strategy:

    Layer 1: TaskScheduler            — between-task imbalance
    Layer 2: data-level samplers      — within-task exposure
    Layer 3: loss-level (this module) — within-task gradient signal
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from src.training.loss_functions import compute_class_weights, compute_pos_weight

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class LossBalancerConfig:
    """Thresholds that decide when each loss-level remedy fires.

    Defaults are deliberately conservative: a balanced dataset gets no
    weighting at all so the loss matches the unweighted baseline reported
    in the literature.
    """
    # Multiclass: max class proportion above which we apply class weights.
    weight_threshold: float = 0.7
    # Multiclass: max class proportion above which we switch to focal loss.
    focal_threshold: float = 0.9
    focal_gamma: float = 2.0

    # Multilabel: minimum positive ratio in a column for it to be kept.
    # 0.0 means "drop only columns that are entirely 0 or entirely 1".
    multilabel_min_pos_ratio: float = 0.0


# =========================================================
# REPORT
# =========================================================

@dataclass
class LossBalancingPlan:
    task_type: str
    distribution: Dict[str, float] = field(default_factory=dict)
    max_ratio: float = 0.0
    class_weights: Optional[torch.Tensor] = None
    pos_weight: Optional[torch.Tensor] = None
    use_focal: bool = False
    focal_gamma: float = 2.0
    valid_label_indices: Optional[List[int]] = None
    dropped_label_indices: List[int] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# =========================================================
# ANALYSIS
# =========================================================

def _to_array(labels) -> np.ndarray:
    if isinstance(labels, pd.Series):
        return labels.to_numpy()
    if isinstance(labels, pd.DataFrame):
        return labels.to_numpy()
    if isinstance(labels, torch.Tensor):
        return labels.detach().cpu().numpy()
    return np.asarray(labels)


def _multiclass_plan(
    labels: np.ndarray,
    num_classes: int,
    config: LossBalancerConfig,
) -> LossBalancingPlan:
    arr = labels.astype(np.int64).ravel()
    counts = np.bincount(arr, minlength=num_classes)
    total = int(counts.sum())
    if total == 0:
        return LossBalancingPlan(task_type="multiclass", notes=["no_labels"])

    ratios = counts / total
    max_ratio = float(ratios.max())
    dist = {str(i): float(r) for i, r in enumerate(ratios)}

    plan = LossBalancingPlan(
        task_type="multiclass",
        distribution=dist,
        max_ratio=max_ratio,
        focal_gamma=config.focal_gamma,
    )

    if max_ratio >= config.weight_threshold:
        plan.class_weights = compute_class_weights(arr, num_classes)
        plan.notes.append(
            f"class_weights enabled (max_ratio={max_ratio:.3f} >= {config.weight_threshold})"
        )

    if max_ratio >= config.focal_threshold:
        plan.use_focal = True
        plan.notes.append(
            f"focal loss enabled gamma={config.focal_gamma} "
            f"(max_ratio={max_ratio:.3f} >= {config.focal_threshold})"
        )

    return plan


def _multilabel_plan(
    labels: np.ndarray,
    config: LossBalancerConfig,
) -> LossBalancingPlan:
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)

    n_rows, n_cols = labels.shape
    if n_rows == 0:
        return LossBalancingPlan(task_type="multilabel", notes=["no_labels"])

    binary = (labels > 0).astype(np.float32)
    pos = binary.sum(axis=0)
    pos_ratio = pos / max(n_rows, 1)

    # Single source of truth for "is this column degenerate?". The
    # ``label_cleaning`` helper is the same one the dataset factory
    # uses to drop columns at materialisation time, so the planner's
    # idea of "valid" and the dataset's idea of "valid" can never
    # silently diverge — a misalignment that would produce a
    # ``logits[..., K] vs labels[..., K']`` shape error at the first
    # batch.
    from src.utils.label_cleaning import (
        remove_single_class_columns,
        valid_indices_from_mask,
    )

    _, base_mask = remove_single_class_columns(binary, min_pos=1, min_neg=1)
    min_ratio = float(config.multilabel_min_pos_ratio)
    valid = base_mask & (pos_ratio >= min_ratio)
    valid_idx = valid_indices_from_mask(valid)
    dropped_idx = [i for i in range(n_cols) if i not in set(valid_idx)]

    if not valid_idx:
        return LossBalancingPlan(
            task_type="multilabel",
            distribution={str(i): float(r) for i, r in enumerate(pos_ratio)},
            valid_label_indices=[],
            dropped_label_indices=dropped_idx,
            notes=["all_columns_degenerate"],
        )

    kept = binary[:, valid_idx]
    pos_weight = compute_pos_weight(torch.from_numpy(kept))

    dist = {str(i): float(r) for i, r in enumerate(pos_ratio)}

    # MULTILABEL-FOCAL-FIX: previously the multilabel branch only
    # produced ``pos_weight`` and never set ``use_focal``, so on a
    # task like emotion with max positive ratio ≈ 0.95 (one column
    # collapsed at the negative class) the loss kept matching
    # majority predictions and the head plateaued at F1 ~0.5. The
    # multiclass and binary branches both gate focal loss on
    # ``max_ratio >= focal_threshold``; mirror that here so heavy
    # multilabel skew triggers the same remedy. Use the *minority*
    # ratio per column (min of pos/neg) to detect skew — a column
    # with pos_ratio=0.95 and one with pos_ratio=0.05 are both
    # equally pathological for BCE.
    if pos_ratio.size:
        skew_per_col = np.maximum(pos_ratio, 1.0 - pos_ratio)
        max_skew = float(skew_per_col.max())
    else:
        max_skew = 0.0

    plan = LossBalancingPlan(
        task_type="multilabel",
        distribution=dist,
        max_ratio=max_skew,
        pos_weight=pos_weight,
        focal_gamma=config.focal_gamma,
        valid_label_indices=valid_idx,
        dropped_label_indices=dropped_idx,
    )

    if max_skew >= config.focal_threshold:
        plan.use_focal = True
        plan.notes.append(
            f"focal loss enabled gamma={config.focal_gamma} "
            f"(max_skew={max_skew:.3f} >= {config.focal_threshold})"
        )

    if dropped_idx:
        plan.notes.append(
            f"dropped {len(dropped_idx)} degenerate label column(s): {dropped_idx}"
        )

    return plan


def _binary_plan(
    labels: np.ndarray,
    config: LossBalancerConfig,
) -> LossBalancingPlan:
    arr = labels.astype(np.float32).ravel()
    n = arr.size
    if n == 0:
        return LossBalancingPlan(task_type="binary", notes=["no_labels"])

    pos = float((arr > 0).sum())
    neg = float(n - pos)
    pos_ratio = pos / max(n, 1)
    dist = {"0": float(neg / n), "1": float(pos_ratio)}

    plan = LossBalancingPlan(
        task_type="binary",
        distribution=dist,
        max_ratio=float(max(pos_ratio, 1 - pos_ratio)),
        focal_gamma=config.focal_gamma,
    )

    if pos > 0 and neg > 0:
        plan.pos_weight = compute_pos_weight(arr.reshape(-1, 1))

    if plan.max_ratio >= config.focal_threshold:
        plan.use_focal = True
        plan.notes.append(
            f"focal loss enabled gamma={config.focal_gamma} "
            f"(max_ratio={plan.max_ratio:.3f} >= {config.focal_threshold})"
        )

    return plan


# =========================================================
# PUBLIC ENTRY POINTS
# =========================================================

def plan_for_labels(
    labels,
    *,
    task_type: str,
    num_classes: Optional[int] = None,
    config: Optional[LossBalancerConfig] = None,
) -> LossBalancingPlan:
    """Build a :class:`LossBalancingPlan` from raw labels.

    ``task_type`` accepts the canonical forms used by ``TaskLossConfig``:
    ``"multiclass"``, ``"multilabel"``, ``"binary"``.
    """
    config = config or LossBalancerConfig()
    canonical = task_type.replace("_", "").lower()
    arr = _to_array(labels)

    if canonical == "multiclass":
        if num_classes is None:
            num_classes = int(arr.max()) + 1 if arr.size else 0
        plan = _multiclass_plan(arr, int(num_classes), config)

    elif canonical == "multilabel":
        plan = _multilabel_plan(arr, config)

    elif canonical == "binary":
        plan = _binary_plan(arr, config)

    else:
        raise ValueError(f"unsupported task_type={task_type!r}")

    if plan.notes:
        for note in plan.notes:
            logger.info("LossBalancer | %s", note)
    return plan


def plan_for_dataframe(
    df: pd.DataFrame,
    *,
    label_columns: Sequence[str],
    task_type: str,
    num_classes: Optional[int] = None,
    config: Optional[LossBalancerConfig] = None,
) -> LossBalancingPlan:
    """Convenience wrapper that pulls labels out of a training DataFrame."""
    cols = [c for c in label_columns if c in df.columns]
    if not cols:
        return LossBalancingPlan(
            task_type=task_type.replace("_", "").lower(),
            notes=["label_columns_missing"],
        )

    if len(cols) == 1:
        labels = df[cols[0]].to_numpy()
    else:
        labels = df[list(cols)].to_numpy()

    return plan_for_labels(
        labels,
        task_type=task_type,
        num_classes=num_classes,
        config=config,
    )
