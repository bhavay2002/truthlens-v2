"""
File: explanation_monitor.py
Module: Explainability Monitoring
"""

from __future__ import annotations

import logging
from typing import List, Dict

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-12


class ExplanationMonitor:
    """
    Minimal production monitor for explainability scores.

    Core API:
        - update(scores)
        - summary()

    Tracks:
        - running statistics
        - optional drift
    """

    def __init__(self, max_history: int = 500) -> None:
        if max_history <= 0:
            raise ValueError("max_history must be > 0")

        self.max_history = max_history
        self.history: List[np.ndarray] = []

    # =====================================================
    # UPDATE
    # =====================================================

    def update(self, scores: List[float]) -> None:
        """
        Add new scores to history.
        """

        arr = np.asarray(scores, dtype=float)

        if arr.size == 0:
            return

        # sanitize
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # normalize (important for comparability)
        arr = np.abs(arr)
        total = float(np.sum(arr))
        if total > 0:
            arr = arr / (total + EPS)

        self.history.append(arr)

        if len(self.history) > self.max_history:
            self.history.pop(0)

    # =====================================================
    # RUNNING STATS
    # =====================================================

    def _stack(self) -> np.ndarray:
        if not self.history:
            return np.empty((0,))
        return np.concatenate([arr.ravel() for arr in self.history])

    def mean(self) -> float:
        if not self.history:
            return 0.0
        return float(np.mean(self._stack()))

    def std(self) -> float:
        if not self.history:
            return 0.0
        return float(np.std(self._stack()))

    def min(self) -> float:
        if not self.history:
            return 0.0
        return float(np.min(self._stack()))

    def max(self) -> float:
        if not self.history:
            return 0.0
        return float(np.max(self._stack()))

    # =====================================================
    # DRIFT (OPTIONAL)
    # =====================================================

    def drift(self) -> float:
        """
        Simple L1 drift between last two entries.
        """
        if len(self.history) < 2:
            return 0.0

        a = self.history[-2]
        b = self.history[-1]

        n = min(len(a), len(b))
        return float(np.mean(np.abs(a[:n] - b[:n])))

    # =====================================================
    # SUMMARY
    # =====================================================

    def summary(self) -> Dict[str, float]:
        """
        Core monitoring output.
        """

        return {
            "mean": self.mean(),
            "std": self.std(),
            "min": self.min(),
            "max": self.max(),
            "drift": self.drift(),
            "history_size": len(self.history),
        }

    # =====================================================
    # RESET
    # =====================================================

    def reset(self) -> None:
        self.history.clear()