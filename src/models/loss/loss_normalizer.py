from __future__ import annotations

import logging
from typing import Dict

import torch

logger = logging.getLogger(__name__)


class EMALossNormalizer:
    """
    Exponential Moving Average (EMA) loss normalization.

    Purpose
    -------
    Normalize per-task losses so different magnitude scales
    do not dominate multi-task training.

    Formula
    -------
        running_mean_t = α * loss_t + (1-α) * running_mean_t
        normalized_loss = loss / max(running_mean_t, floor)

    Features
    --------
    - Per-task tracking
    - Numerically stable (floor)
    - AMP safe (detach for stats)
    - Lazy initialization
    """

    def __init__(
        self,
        alpha: float = 0.1,
        floor: float = 1e-3,
        enabled: bool = True,
    ) -> None:

        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0,1], got {alpha}")

        if floor <= 0.0:
            raise ValueError(f"floor must be positive, got {floor}")

        self.alpha = float(alpha)
        self.floor = float(floor)
        self.enabled = enabled

        # per-task running mean
        self._running_mean: Dict[str, float] = {}

        logger.info(
            "EMALossNormalizer initialized | alpha=%.3f floor=%.6f enabled=%s",
            self.alpha,
            self.floor,
            self.enabled,
        )

    # =========================================================
    # MAIN
    # =========================================================

    def normalize(
        self,
        task: str,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalize loss for a given task.

        Parameters
        ----------
        task : str
        loss : torch.Tensor (scalar)

        Returns
        -------
        torch.Tensor
            Normalized loss
        """

        if not self.enabled:
            return loss

        if not torch.is_tensor(loss):
            raise TypeError("loss must be a torch.Tensor")

        if loss.numel() != 1:
            raise ValueError("loss must be a scalar tensor")

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss detected for task '{task}'")

        # --- extract scalar safely ---
        loss_val = float(loss.detach().item())

        # --- lazy init ---
        if task not in self._running_mean:
            self._running_mean[task] = loss_val
            return loss  # first step: don't distort

        # --- EMA update ---
        prev = self._running_mean[task]
        new_mean = self.alpha * loss_val + (1.0 - self.alpha) * prev
        self._running_mean[task] = new_mean

        # --- normalize ---
        denom = max(new_mean, self.floor)

        normalized = loss / denom

        return normalized

    # =========================================================
    # UTILITIES
    # =========================================================

    def get_running_means(self) -> Dict[str, float]:
        """
        Return current EMA statistics.
        """
        return dict(self._running_mean)

    def reset(self) -> None:
        """
        Reset all EMA statistics (call per epoch if needed).
        """
        self._running_mean.clear()
        logger.info("EMALossNormalizer state reset")