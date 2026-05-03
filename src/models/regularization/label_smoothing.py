from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross Entropy Loss.

    Works for multiclass classification.

    Args:
        num_classes: number of classes
        smoothing: smoothing factor (0.0 = no smoothing)
        reduction: "mean" | "sum" | "none"
        ignore_index: optional index to ignore
    """

    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.1,
        reduction: str = "mean",
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()

        if not (0.0 <= smoothing < 1.0):
            raise ValueError("smoothing must be in [0, 1)")

        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index

        self.confidence = 1.0 - smoothing

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C)
            target: (B,)
        """

        log_probs = F.log_softmax(logits, dim=-1)

        if self.ignore_index is not None:
            mask = target != self.ignore_index
            logits = logits[mask]
            target = target[mask]
            log_probs = log_probs[mask]

            if target.numel() == 0:
                return torch.tensor(0.0, device=logits.device)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        loss = torch.sum(-true_dist * log_probs, dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# =========================================================
# MULTILABEL LABEL SMOOTHING
# =========================================================

class MultiLabelSmoothingLoss(nn.Module):
    """
    Label smoothing for multilabel classification using BCE.

    Args:
        smoothing: smoothing factor
        reduction: "mean" | "sum" | "none"
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if not (0.0 <= smoothing < 1.0):
            raise ValueError("smoothing must be in [0, 1)")

        self.smoothing = smoothing
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C)
            targets: (B, C) binary
        """

        smoothed_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing

        loss = F.binary_cross_entropy_with_logits(
            logits,
            smoothed_targets,
            reduction=self.reduction,
        )

        return loss