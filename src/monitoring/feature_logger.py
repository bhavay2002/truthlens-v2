from __future__ import annotations

import time
import logging
from contextlib import contextmanager
from typing import Dict, Any, Optional

logger = logging.getLogger("truthlens.monitoring")


# =========================================================
# FEATURE ANOMALY LOGGING
# =========================================================

def log_feature_stats(
    features: Dict[str, float],
    *,
    task: str,
    step: Optional[int] = None,
    threshold: float = 0.9,
) -> None:
    """
    Detect and log anomalous feature values.

    Args:
        features: dict of feature_name -> value
        task: task name (e.g. toxicity, sentiment)
        step: training step (optional)
        threshold: anomaly threshold
    """
    anomalies = {}

    for k, v in features.items():
        if isinstance(v, (int, float)) and abs(v) > threshold:
            anomalies[k] = round(v, 6)

    if anomalies:
        logger.warning(
            "feature_anomaly_detected",
            extra={
                "task": task,
                "step": step,
                "num_anomalies": len(anomalies),
                "anomalies": anomalies,
            },
        )


# =========================================================
# FEATURE DISTRIBUTION SUMMARY
# =========================================================

def summarize_features(features: Dict[str, float]) -> Dict[str, float]:
    """
    Compute simple statistics for feature monitoring.
    """
    values = [v for v in features.values() if isinstance(v, (int, float))]

    if not values:
        return {}

    return {
        "mean": round(sum(values) / len(values), 6),
        "max": round(max(values), 6),
        "min": round(min(values), 6),
    }


def log_feature_summary(
    features: Dict[str, float],
    *,
    task: str,
    step: Optional[int] = None,
) -> None:
    summary = summarize_features(features)

    if summary:
        logger.info(
            "feature_summary",
            extra={
                "task": task,
                "step": step,
                **summary,
            },
        )


# =========================================================
# PERFORMANCE / TIMING
# =========================================================

@contextmanager
def time_block(name: str, *, task: Optional[str] = None):
    """
    Context manager to measure execution time.

    Example:
        with time_block("feature_extraction", task="toxicity"):
            features = extractor(text)
    """
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        logger.info(
            "timing",
            extra={
                "component": name,
                "task": task,
                "latency_sec": round(duration, 4),
            },
        )


def log_request_latency(
    duration: float,
    *,
    task: str,
) -> None:
    logger.info(
        "request_latency",
        extra={
            "task": task,
            "latency_sec": round(duration, 4),
        },
    )


# =========================================================
# FAILURE LOGGING
# =========================================================

def log_failure(
    error: Exception,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    logger.error(
        "pipeline_failure",
        extra={
            "error_type": type(error).__name__,
            "error_msg": str(error),
            "context": context or {},
        },
    )