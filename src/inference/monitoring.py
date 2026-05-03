"""
File: monitoring.py
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class MonitoringConfig:
    window_size: int = 500

    # thresholds
    latency_ms_threshold: float = 500
    confidence_threshold: float = 0.4
    entropy_threshold: float = 1.5


# =========================================================
# METRIC WINDOW
# =========================================================

class MetricWindow:
    """
    Rolling window for statistics.
    """

    def __init__(self, size: int):
        self.values = deque(maxlen=size)

    def add(self, value: float):
        self.values.append(value)

    def mean(self):
        return float(np.mean(self.values)) if self.values else 0.0

    def p95(self):
        return float(np.percentile(self.values, 95)) if self.values else 0.0

    def max(self):
        return max(self.values) if self.values else 0.0

    def size(self):
        return len(self.values)


# =========================================================
# MONITOR
# =========================================================

class InferenceMonitor:

    def __init__(self, config: Optional[MonitoringConfig] = None):

        self.config = config or MonitoringConfig()

        self.latency = MetricWindow(self.config.window_size)
        self.confidence = MetricWindow(self.config.window_size)
        self.entropy = MetricWindow(self.config.window_size)

        self.error_count = 0
        self.total_requests = 0

        self._lock = Lock()

        logger.info("InferenceMonitor initialized")

    # =====================================================
    # UPDATE
    # =====================================================

    def update(
        self,
        *,
        latency_ms: float,
        confidence: Optional[float] = None,
        probabilities: Optional[np.ndarray] = None,
        error: bool = False,
    ):

        with self._lock:

            self.total_requests += 1

            if error:
                self.error_count += 1

            self.latency.add(latency_ms)

            if confidence is not None:
                self.confidence.add(confidence)

            if probabilities is not None:
                entropy = self._compute_entropy(probabilities)
                self.entropy.add(entropy)

            self._check_alerts()

    # =====================================================
    # ENTROPY
    # =====================================================

    def _compute_entropy(self, probs):

        probs = np.asarray(probs)
        return float(-np.sum(probs * np.log(probs + 1e-12)))

    # =====================================================
    # ALERTS
    # =====================================================

    def _check_alerts(self):

        # latency
        if self.latency.mean() > self.config.latency_ms_threshold:
            logger.warning("High latency detected")

        # low confidence
        if self.confidence.mean() < self.config.confidence_threshold:
            logger.warning("Confidence drop detected")

        # high uncertainty
        if self.entropy.mean() > self.config.entropy_threshold:
            logger.warning("Uncertainty spike detected")

    # =====================================================
    # METRICS SNAPSHOT
    # =====================================================

    def snapshot(self) -> Dict[str, Any]:

        with self._lock:

            error_rate = (
                self.error_count / self.total_requests
                if self.total_requests > 0
                else 0.0
            )

            return {
                "latency_mean_ms": self.latency.mean(),
                "latency_p95_ms": self.latency.p95(),
                "latency_max_ms": self.latency.max(),

                "confidence_mean": self.confidence.mean(),
                "entropy_mean": self.entropy.mean(),
                "entropy_p95": self.entropy.p95(),

                "error_rate": error_rate,
                "total_requests": self.total_requests,
            }

    # =====================================================
    # RESET
    # =====================================================

    def reset(self):

        with self._lock:
            self.latency = MetricWindow(self.config.window_size)
            self.confidence = MetricWindow(self.config.window_size)
            self.entropy = MetricWindow(self.config.window_size)

            self.error_count = 0
            self.total_requests = 0

        logger.info("Monitoring reset")