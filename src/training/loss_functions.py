#src\models\training\loss_functions.py
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# CLASS-WEIGHT / POS-WEIGHT HELPERS  (loss-level balancing)
# =========================================================

def compute_class_weights(
    labels: Union[Sequence[int], torch.Tensor],
    num_classes: int,
    *,
    normalize: bool = True,
    smoothing: float = 1.0,
) -> torch.Tensor:
    """Inverse-frequency class weights for ``CrossEntropyLoss(weight=…)``.

    weight_c = total / (num_classes * count_c)  (sklearn 'balanced' style)

    A small ``smoothing`` constant is added to every count so unseen
    classes do not produce ``inf`` weights, which would explode the loss
    on the first batch that happens to contain them.
    """
    if isinstance(labels, torch.Tensor):
        arr = labels.detach().cpu().long().tolist()
    else:
        arr = [int(x) for x in labels]

    counts = [0] * int(num_classes)
    for x in arr:
        if 0 <= x < num_classes:
            counts[x] += 1

    counts_t = torch.tensor(counts, dtype=torch.float64) + float(smoothing)
    total = counts_t.sum()
    weights = total / (float(num_classes) * counts_t)

    if normalize:
        weights = weights * (float(num_classes) / weights.sum())

    return weights.float()


def compute_pos_weight(
    labels: Union[Sequence[Sequence[float]], torch.Tensor],
    *,
    smoothing: float = 1.0,
    clip_max: Optional[float] = 100.0,
) -> torch.Tensor:
    """Per-column ``pos_weight`` for ``BCEWithLogitsLoss``.

    pos_weight_c = neg_c / pos_c   (after smoothing).

    NaN / negative entries are treated as missing and excluded from the
    counts, matching the multilabel ignore semantics used elsewhere in
    the loss layer.
    """
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)

    labels = labels.float()

    if labels.dim() == 1:
        labels = labels.unsqueeze(1)

    valid = torch.isfinite(labels) & (labels >= 0)
    safe = torch.where(valid, labels, torch.zeros_like(labels))

    pos = (safe * valid.float()).sum(dim=0)
    total = valid.float().sum(dim=0)
    neg = total - pos

    pos_w = (neg + float(smoothing)) / (pos + float(smoothing))

    if clip_max is not None:
        pos_w = pos_w.clamp(max=float(clip_max))

    return pos_w.float()


# =========================================================
# FOCAL LOSS
# =========================================================

class FocalLoss(nn.Module):
    """Multiclass Focal Loss (Lin et al., 2017).

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Use this for severe class imbalance where simple class weights are
    not enough to keep gradients meaningful — the ``(1 - p_t)^gamma``
    term down-weights well-classified examples so the network keeps
    focusing on the hard, rare ones.

    Parameters
    ----------
    gamma:
        Focusing parameter. ``gamma=0`` recovers weighted cross-entropy.
    weight:
        Per-class weights (same role as ``CrossEntropyLoss(weight=…)``).
    ignore_index:
        Targets equal to this value are skipped.
    reduction:
        ``"mean"`` (default), ``"sum"`` or ``"none"``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"invalid reduction={reduction!r}")

        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)
        self.reduction = reduction
        # Register as buffer so .to(device) follows the parent module.
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None  # type: ignore[assignment]

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        # one-hot → index, in keeping with the rest of the loss layer
        if targets.dim() == 2 and targets.shape == logits.shape:
            targets = targets.argmax(dim=1)

        targets = targets.long()

        weight = self.weight.to(logits.device) if self.weight is not None else None

        ce = F.cross_entropy(
            logits.float(),
            targets,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        # ``F.cross_entropy`` already returns -log p_t per sample, so
        # ``pt = exp(-ce)`` gives the correct (alpha-free) p_t. The
        # alpha_t factor is folded into ``ce`` via ``weight=…`` already.
        pt = torch.exp(-ce.detach()).clamp(min=1e-8, max=1.0)
        focal = ((1.0 - pt) ** self.gamma) * ce

        if self.reduction == "mean":
            # Match ``nn.CrossEntropyLoss``'s convention exactly so that
            # ``FocalLoss(gamma=0, weight=w)`` is bit-equivalent to
            # ``CrossEntropyLoss(weight=w)``: when ``weight`` is set,
            # PyTorch divides by ``sum(weight[targets])`` rather than by
            # the sample count. Without that, focal-with-weights and
            # CE-with-weights would silently disagree on the loss
            # *scale*, breaking gradient comparisons across epochs.
            valid = targets.ne(self.ignore_index)
            if weight is not None:
                w_per_sample = weight.index_select(
                    0, torch.where(valid, targets, torch.zeros_like(targets))
                )
                w_per_sample = w_per_sample * valid.to(w_per_sample.dtype)
                denom = w_per_sample.sum().clamp_min(1e-12).to(focal.dtype)
            else:
                denom = valid.sum().clamp_min(1).to(focal.dtype)
            return focal.sum() / denom

        if self.reduction == "sum":
            return focal.sum()

        return focal


# =========================================================
# PURE LOSS FUNCTIONS (NO TASK LOGIC)
# =========================================================

def binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Binary cross-entropy with optional positive-class re-weighting.

    EDGE-CASE (section 9, imbalanced binary): on heavily imbalanced data
    (e.g. ``99% / 1%`` like minority-class hate-speech detection), plain
    BCE collapses to "always predict the majority class" because the
    gradient from the rare positives is dwarfed by the negatives. The
    standard remedy is ``pos_weight`` — a per-class scalar (or tensor of
    shape ``[num_classes]``) that scales the positive term so the
    effective gradient is balanced. Exposing it here keeps callers from
    re-implementing the loss just to pass that single argument.
    """
    targets = targets.float()

    if targets.dim() == 1:
        targets = targets.unsqueeze(1)

    if logits.shape != targets.shape:
        raise RuntimeError("Shape mismatch in binary_loss")

    return F.binary_cross_entropy_with_logits(
        logits.float(),
        targets,
        pos_weight=pos_weight,
    )


def multiclass_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:

    if targets.dim() == 2:
        targets = targets.argmax(dim=1)

    mask = targets != ignore_index

    if not mask.any():
        return logits.sum() * 0.0

    return F.cross_entropy(
        logits[mask].float(),
        targets[mask].long(),
    )


def multilabel_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: float = -100.0,
) -> torch.Tensor:
    """
    Multi-label binary cross-entropy with element-wise ignore semantics.

    LOSS-4: Ignore-mask contract (now explicit).

    Multi-label targets are conventionally float in {0.0, 1.0}, so the
    integer sentinel used by ``multiclass_loss`` (``-100``) does not
    naturally appear in normal data — meaning the original mask
    ``targets != -100.0`` was effectively always all-True and the
    ``ignore_index`` argument was a no-op for the common case.

    The supported sentinels are now documented and consistently masked:
      * ``ignore_index`` (default ``-100.0``): explicit float sentinel.
        Pass any value (e.g. ``float('nan')``) to override.
      * ``NaN`` targets: ALWAYS treated as ignored. NaN labels would
        otherwise propagate through ``BCEWithLogits`` and corrupt the
        loss silently with no traceback.

    Both rules combine — an element is included iff it is finite AND not
    equal to ``ignore_index``.
    """

    targets = targets.float()

    if logits.shape != targets.shape:
        raise RuntimeError("Shape mismatch in multilabel_loss")

    # Finite-and-not-sentinel mask. ``isfinite`` masks NaN AND ±inf.
    mask = torch.isfinite(targets) & (targets != ignore_index)

    if not mask.any():
        return logits.sum() * 0.0

    safe_targets = torch.where(mask, targets, torch.zeros_like(targets))

    loss = F.binary_cross_entropy_with_logits(
        logits.float(),
        safe_targets,
        reduction="none",
    )

    loss = loss * mask.float()

    return loss.sum() / mask.sum().clamp_min(1)


def regression_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:

    if preds.shape != targets.shape:
        raise RuntimeError("Shape mismatch in regression_loss")

    return F.mse_loss(preds.float(), targets.float())


def masked_task_loss(losses: Dict[str, torch.Tensor], task_mask: Dict[str, torch.Tensor]) -> torch.Tensor:
    total = None
    active = 0.0
    for task, loss in losses.items():
        mask = task_mask.get(task)
        if mask is None:
            continue
        value = loss * mask.float()
        total = value if total is None else total + value
        active += float(mask.float().sum().item())
    if total is None or active <= 0:
        return torch.tensor(0.0, device=next(iter(losses.values())).device if losses else "cpu")
    return total / active

