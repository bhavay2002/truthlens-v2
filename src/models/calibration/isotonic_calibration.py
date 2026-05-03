"""Isotonic-regression post-hoc calibration.

This module owns ``IsotonicCalibrator`` and ``IsotonicCalibrationConfig``.
The classes used to live inside ``temperature_scaling.py`` because of an
historical file-name swap; they have been moved here so each file in
``src.models.calibration`` once again contains exactly what its name
says. ``CalibrationMetrics`` lives in ``calibration_metrics.py`` —
nothing in this module re-exports it any more.

Public API:
  • :class:`IsotonicCalibrator`
  • :class:`IsotonicCalibrationConfig`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class IsotonicCalibrationConfig:
    out_of_bounds: str = "clip"
    increasing: bool = True
    normalize: bool = True


# =========================================================
# CALIBRATOR
# =========================================================

class IsotonicCalibrator:

    def __init__(
        self,
        config: Optional[IsotonicCalibrationConfig] = None,
    ) -> None:
        self.config = config or IsotonicCalibrationConfig()

        self._calibrators: List[IsotonicRegression] = []
        self._num_classes: int = 0
        self._is_fitted: bool = False

    # =====================================================
    # INTERNAL
    # =====================================================

    def _prepare(self, probs: np.ndarray) -> np.ndarray:

        probs = np.asarray(probs, dtype=np.float64)

        # binary vector -> convert to 2-class
        if probs.ndim == 1:
            probs = np.stack([1.0 - probs, probs], axis=1)

        if probs.ndim != 2:
            raise ValueError("probs must be (N, C)")

        if np.any((probs < 0) | (probs > 1)):
            raise ValueError("probs must be in [0,1]")

        return probs

    # =====================================================
    # FIT
    # =====================================================

    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> "IsotonicCalibrator":

        probs = self._prepare(probabilities)
        labels = np.asarray(labels, dtype=np.int64)

        if probs.shape[0] != labels.shape[0]:
            raise ValueError("size mismatch")

        n_classes = probs.shape[1]

        if labels.min() < 0 or labels.max() >= n_classes:
            raise ValueError("invalid labels")

        self._num_classes = n_classes
        self._calibrators = []

        for c in range(n_classes):
            y = (labels == c).astype(np.float64)
            x = probs[:, c]

            model = IsotonicRegression(
                out_of_bounds=self.config.out_of_bounds,
                increasing=self.config.increasing,
            )
            model.fit(x, y)

            self._calibrators.append(model)

        self._is_fitted = True

        logger.info("Isotonic fitted | classes=%d", n_classes)

        return self

    # =====================================================
    # PREDICT
    # =====================================================

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:

        if not self._is_fitted:
            raise RuntimeError("Calibrator not fitted")

        probs = self._prepare(probabilities)

        if probs.shape[1] != self._num_classes:
            raise ValueError("class mismatch")

        calibrated = np.zeros_like(probs)

        for i, model in enumerate(self._calibrators):
            calibrated[:, i] = model.predict(probs[:, i])

        if self.config.normalize:
            sums = calibrated.sum(axis=1, keepdims=True)
            sums[sums == 0] = 1.0
            calibrated = calibrated / sums

        return calibrated

    # =====================================================
    # FIT + TRANSFORM
    # =====================================================

    def calibrate(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:

        self.fit(probabilities, labels)
        return self.predict_proba(probabilities)


__all__ = ["IsotonicCalibrator", "IsotonicCalibrationConfig"]
