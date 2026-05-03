from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class CalibrationMetricConfig:
    n_bins: int = 15

    def __post_init__(self):
        if self.n_bins <= 1:
            raise ValueError("n_bins must be > 1")


# =========================================================
# METRICS
# =========================================================

class CalibrationMetrics:

    def __init__(self, config: CalibrationMetricConfig | None = None):
        self.config = config or CalibrationMetricConfig()

    # -----------------------------------------------------

    @staticmethod
    def _validate(probs: torch.Tensor, labels: torch.Tensor):
        if probs.ndim != 2:
            raise ValueError("probs must be [N, C]")
        if labels.ndim != 1:
            raise ValueError("labels must be [N]")
        if probs.shape[0] != labels.shape[0]:
            raise ValueError("size mismatch")

    # -----------------------------------------------------

    @staticmethod
    def _to_probs(
        x: torch.Tensor,
        *,
        is_logits: Optional[bool] = None,
    ) -> torch.Tensor:
        """Coerce ``x`` to a row-stochastic probability matrix.

        N4: callers are expected to pass ``is_logits`` explicitly. The
        old auto-detect heuristic ("if anything is outside [0, 1], assume
        logits") silently misclassifies pre-softmax probabilities that
        happen to lie in that range and any probability tensor with a
        floating-point round-off below zero. We only fall back to the
        heuristic when ``is_logits is None`` and emit a warning so the
        ambiguity shows up in logs instead of being absorbed.
        """
        if is_logits is True:
            return torch.softmax(x, dim=1)
        if is_logits is False:
            return x

        x_min = float(x.min())
        x_max = float(x.max())
        if x_max > 1.0 or x_min < 0.0:
            logger.warning(
                "CalibrationMetrics._to_probs: is_logits not specified "
                "and tensor falls outside [0, 1] (min=%.4f, max=%.4f); "
                "applying softmax. Pass is_logits=True/False explicitly "
                "to silence this warning.",
                x_min,
                x_max,
            )
            return torch.softmax(x, dim=1)

        logger.warning(
            "CalibrationMetrics._to_probs: is_logits not specified and "
            "tensor lies in [0, 1]; assuming probabilities. Pass "
            "is_logits=True/False explicitly to silence this warning."
        )
        return x

    # =====================================================
    # ECE
    # =====================================================

    def expected_calibration_error(
        self,
        logits_or_probs: torch.Tensor,
        labels: torch.Tensor,
        *,
        is_logits: Optional[bool] = None,
    ) -> float:

        probs = self._to_probs(logits_or_probs, is_logits=is_logits)
        self._validate(probs, labels)

        conf, preds = torch.max(probs, dim=1)
        acc = preds.eq(labels)

        bins = torch.linspace(0, 1, self.config.n_bins + 1)
        ece = torch.zeros(1, device=probs.device)

        for i in range(self.config.n_bins):
            mask = (conf > bins[i]) & (conf <= bins[i + 1])

            if mask.sum() > 0:
                bin_acc = acc[mask].float().mean()
                bin_conf = conf[mask].mean()
                weight = mask.float().mean()
                ece += torch.abs(bin_conf - bin_acc) * weight

        return float(ece.item())

    # =====================================================
    # MCE
    # =====================================================

    def maximum_calibration_error(
        self,
        logits_or_probs: torch.Tensor,
        labels: torch.Tensor,
        *,
        is_logits: Optional[bool] = None,
    ) -> float:

        probs = self._to_probs(logits_or_probs, is_logits=is_logits)
        self._validate(probs, labels)

        conf, preds = torch.max(probs, dim=1)
        acc = preds.eq(labels)

        bins = torch.linspace(0, 1, self.config.n_bins + 1)
        mce = torch.zeros(1, device=probs.device)

        for i in range(self.config.n_bins):
            mask = (conf > bins[i]) & (conf <= bins[i + 1])

            if mask.sum() > 0:
                error = torch.abs(
                    conf[mask].mean() - acc[mask].float().mean()
                )
                mce = torch.maximum(mce, error)

        return float(mce.item())

    # =====================================================
    # BRIER
    # =====================================================

    def brier_score(
        self,
        logits_or_probs: torch.Tensor,
        labels: torch.Tensor,
        *,
        is_logits: Optional[bool] = None,
    ) -> float:

        probs = self._to_probs(logits_or_probs, is_logits=is_logits)
        self._validate(probs, labels)

        one_hot = F.one_hot(labels, num_classes=probs.shape[1]).float()
        score = torch.mean(torch.sum((probs - one_hot) ** 2, dim=1))

        return float(score.item())

    # =====================================================
    # NLL
    # =====================================================

    def negative_log_likelihood(
        self,
        logits_or_probs: torch.Tensor,
        labels: torch.Tensor,
        *,
        is_logits: Optional[bool] = None,
    ) -> float:
        """Mean negative-log-likelihood under the predictive distribution.

        N5: when given probabilities we route through ``log`` of a
        ``clamp_min``-floored copy and use ``F.nll_loss`` (the
        old ``log(probs + 1e-12)`` formulation drifted by EPS for any
        confident prediction). When given logits we route through
        ``log_softmax`` + ``F.nll_loss`` rather than ``F.cross_entropy``,
        which is mathematically equivalent but makes the call site
        explicit and easy to audit.
        """

        if is_logits is True:
            log_probs = F.log_softmax(logits_or_probs, dim=1)
            return float(F.nll_loss(log_probs, labels).item())

        if is_logits is False:
            log_probs = logits_or_probs.clamp_min(1e-12).log()
            return float(F.nll_loss(log_probs, labels).item())

        # Auto-detect (warned in ``_to_probs``); mirror the original
        # heuristic here so callers that haven't been updated yet still
        # work, but emit a warning so the ambiguity is visible.
        x_min = float(logits_or_probs.min())
        x_max = float(logits_or_probs.max())
        if x_max <= 1.0 and x_min >= 0.0:
            logger.warning(
                "negative_log_likelihood: is_logits not specified; "
                "tensor lies in [0, 1] so assuming probabilities."
            )
            log_probs = logits_or_probs.clamp_min(1e-12).log()
        else:
            logger.warning(
                "negative_log_likelihood: is_logits not specified; "
                "tensor outside [0, 1] (min=%.4f, max=%.4f) so "
                "assuming logits.",
                x_min,
                x_max,
            )
            log_probs = F.log_softmax(logits_or_probs, dim=1)

        return float(F.nll_loss(log_probs, labels).item())

    # =====================================================
    # RELIABILITY
    # =====================================================

    def reliability_statistics(
        self,
        logits_or_probs: torch.Tensor,
        labels: torch.Tensor,
        *,
        is_logits: Optional[bool] = None,
    ) -> Dict[str, np.ndarray]:

        probs = self._to_probs(logits_or_probs, is_logits=is_logits)
        self._validate(probs, labels)

        conf, preds = torch.max(probs, dim=1)
        acc = preds.eq(labels).float()

        conf_np = conf.cpu().numpy()
        acc_np = acc.cpu().numpy()

        bins = np.linspace(0.0, 1.0, self.config.n_bins + 1)

        bin_acc = np.zeros(self.config.n_bins)
        bin_conf = np.zeros(self.config.n_bins)
        bin_counts = np.zeros(self.config.n_bins)

        for i in range(self.config.n_bins):
            mask = (conf_np > bins[i]) & (conf_np <= bins[i + 1])

            if mask.sum() > 0:
                bin_acc[i] = acc_np[mask].mean()
                bin_conf[i] = conf_np[mask].mean()
                bin_counts[i] = mask.sum()

        return {
            "bin_accuracy": bin_acc,
            "bin_confidence": bin_conf,
            "bin_counts": bin_counts,
            "bin_boundaries": bins,
        }

    # =====================================================
    # ALL
    # =====================================================

    def compute_all_metrics(
        self,
        logits_or_probs: torch.Tensor,
        labels: torch.Tensor,
        *,
        is_logits: Optional[bool] = None,
    ) -> Dict[str, float]:

        metrics = {
            "ece": self.expected_calibration_error(
                logits_or_probs, labels, is_logits=is_logits
            ),
            "mce": self.maximum_calibration_error(
                logits_or_probs, labels, is_logits=is_logits
            ),
            "brier_score": self.brier_score(
                logits_or_probs, labels, is_logits=is_logits
            ),
            "nll": self.negative_log_likelihood(
                logits_or_probs, labels, is_logits=is_logits
            ),
        }

        logger.info("Calibration metrics: %s", metrics)

        return metrics