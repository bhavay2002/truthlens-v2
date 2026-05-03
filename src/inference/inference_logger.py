from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any

import numpy as np

from src.utils import current_datetime

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

MAX_VECTOR_LOG = 10  # truncate arrays


# =========================================================
# SAFE SERIALIZATION
# =========================================================

def _truncate_vector(x):
    if x is None:
        return None
    x = np.asarray(x)
    return x[:MAX_VECTOR_LOG].tolist()


# =========================================================
# DATA CLASS
# =========================================================

@dataclass
class InferenceLogEntry:

    article_id: str
    trace_id: str
    processing_time_ms: float

    model_versions: Dict[str, str]
    feature_count: int

    # 🔥 prediction
    predicted_label: Optional[Any]
    prediction_confidence: Optional[float]

    # 🔥 probabilities / logits
    probabilities: Optional[Any]
    logits: Optional[Any]

    # 🔥 uncertainty
    entropy: Optional[float]
    p95_entropy: Optional[float]

    timestamp: float


# =========================================================
# LOGGER
# =========================================================

class InferenceLogger:

    def __init__(
        self,
        service_name: str = "truthlens-inference",
        enable_json_logs: bool = True,
        log_vectors: bool = False,
    ):
        self.service_name = service_name
        self.enable_json_logs = enable_json_logs
        self.log_vectors = log_vectors

        logger.info("InferenceLogger initialized")

    # =====================================================
    # UTILS
    # =====================================================

    def generate_article_id(self) -> str:
        return str(uuid.uuid4())

    def generate_trace_id(self) -> str:
        return str(uuid.uuid4())

    def start_timer(self) -> float:
        return time.perf_counter()

    def stop_timer(self, start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000

    # =====================================================
    # ENTRY CREATION (UPGRADED 🔥)
    # =====================================================

    def create_log_entry(
        self,
        *,
        article_id: Optional[str],
        start_time: float,
        model_versions: Dict[str, str],
        feature_count: int,

        predicted_label: Optional[Any] = None,
        prediction_confidence: Optional[float] = None,

        probabilities: Optional[Any] = None,
        logits: Optional[Any] = None,

        entropy: Optional[float] = None,
        p95_entropy: Optional[float] = None,
    ) -> InferenceLogEntry:

        if article_id is None:
            article_id = self.generate_article_id()

        trace_id = self.generate_trace_id()
        processing_time_ms = self.stop_timer(start_time)

        # 🔥 SAFE VECTOR LOGGING
        if not self.log_vectors:
            probabilities = None
            logits = None
        else:
            probabilities = _truncate_vector(probabilities)
            logits = _truncate_vector(logits)

        return InferenceLogEntry(
            article_id=article_id,
            trace_id=trace_id,
            processing_time_ms=processing_time_ms,
            model_versions=model_versions,
            feature_count=feature_count,
            predicted_label=predicted_label,
            prediction_confidence=prediction_confidence,
            probabilities=probabilities,
            logits=logits,
            entropy=entropy,
            p95_entropy=p95_entropy,
            timestamp=float(current_datetime().timestamp()),
        )

    # =====================================================
    # EMIT
    # =====================================================

    def log(self, entry: InferenceLogEntry, level=logging.INFO):

        record = {
            "service": self.service_name,
            "event": "inference",
            **asdict(entry),
        }

        try:
            msg = json.dumps(record) if self.enable_json_logs else str(record)
            logger.log(level, msg)

        except Exception as e:
            logger.exception("Logging failed: %s", e)

    # =====================================================
    # HIGH-LEVEL API
    # =====================================================

    def log_prediction(
        self,
        *,
        start_time: float,
        model_versions: Dict[str, str],
        feature_count: int,

        article_id: Optional[str] = None,
        predicted_label: Optional[Any] = None,
        prediction_confidence: Optional[float] = None,

        probabilities: Optional[Any] = None,
        logits: Optional[Any] = None,

        entropy: Optional[float] = None,
        p95_entropy: Optional[float] = None,
    ):

        entry = self.create_log_entry(
            article_id=article_id,
            start_time=start_time,
            model_versions=model_versions,
            feature_count=feature_count,
            predicted_label=predicted_label,
            prediction_confidence=prediction_confidence,
            probabilities=probabilities,
            logits=logits,
            entropy=entropy,
            p95_entropy=p95_entropy,
        )

        # 🔥 AUTO ALERTING (OPTIONAL)
        if entropy is not None and p95_entropy is not None:
            if entropy > p95_entropy:
                logger.warning("High uncertainty detected")

        self.log(entry)