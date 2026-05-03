"""
File: schema.py
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional

import numpy as np


# =========================================================
# BASE VALIDATION
# =========================================================

def _ensure_array(x, name: str):
    if x is None:
        return None
    arr = np.asarray(x)
    if arr.ndim == 0:
        raise ValueError(f"{name} must not be scalar")
    return arr


# =========================================================
# PREDICTION
# =========================================================

@dataclass
class TaskPrediction:

    logits: Optional[Any] = None
    probabilities: Optional[Any] = None
    predictions: Optional[Any] = None

    def validate(self):

        self.logits = _ensure_array(self.logits, "logits")
        self.probabilities = _ensure_array(self.probabilities, "probabilities")
        self.predictions = _ensure_array(self.predictions, "predictions")

        if self.probabilities is not None and self.predictions is not None:
            if len(self.probabilities) != len(self.predictions):
                raise ValueError("Mismatch between probabilities and predictions")

        return self


@dataclass
class PredictionOutput:

    tasks: Dict[str, TaskPrediction]

    def validate(self):

        if not self.tasks:
            raise ValueError("No tasks in prediction output")

        for task, pred in self.tasks.items():
            if not isinstance(pred, TaskPrediction):
                raise TypeError(f"{task} must be TaskPrediction")
            pred.validate()

        return self

    def to_dict(self):
        return {
            k: asdict(v) for k, v in self.tasks.items()
        }


# =========================================================
# EVALUATION
# =========================================================

@dataclass
class EvaluationMetrics:

    metrics: Dict[str, float]

    def validate(self):
        for k, v in self.metrics.items():
            if not isinstance(v, (int, float)):
                raise TypeError(f"{k} must be numeric")
        return self


@dataclass
class CalibrationOutput:

    ece: float
    classwise_ece: Optional[Dict[str, float]] = None
    temperature: Optional[float] = None

    def validate(self):
        if self.ece < 0:
            raise ValueError("ECE must be >= 0")
        return self


@dataclass
class UncertaintyOutput:

    mean_entropy: float
    p95_entropy: Optional[float] = None
    p99_entropy: Optional[float] = None

    def validate(self):
        if self.mean_entropy < 0:
            raise ValueError("Entropy must be >= 0")
        return self


# =========================================================
# CORRELATION
# =========================================================

@dataclass
class CorrelationOutput:

    matrix: Dict[str, Dict[str, float]]

    def validate(self):
        if not self.matrix:
            raise ValueError("Empty correlation matrix")
        return self


# =========================================================
# REPORT
# =========================================================

@dataclass
class ReportSchema:

    article_summary: Dict[str, Any]

    predictions: Dict[str, Any]

    evaluation: Optional[Dict[str, Any]] = None
    calibration: Optional[Dict[str, Any]] = None
    uncertainty: Optional[Dict[str, Any]] = None
    task_correlation: Optional[Dict[str, Any]] = None

    aggregation: Optional[Dict[str, Any]] = None
    explainability: Optional[Dict[str, Any]] = None

    metadata: Optional[Dict[str, Any]] = None

    def validate(self):

        if not isinstance(self.article_summary, dict):
            raise TypeError("article_summary must be dict")

        if not isinstance(self.predictions, dict):
            raise TypeError("predictions must be dict")

        return self

    def to_dict(self):
        return asdict(self)


# =========================================================
# API RESPONSE
# =========================================================

@dataclass
class APIResponse:

    predictions: Dict[str, Any]
    confidence: Dict[str, float]
    uncertainty: Optional[Dict[str, Any]]
    timestamp: str

    def validate(self):

        if not isinstance(self.predictions, dict):
            raise TypeError("predictions must be dict")

        if not isinstance(self.confidence, dict):
            raise TypeError("confidence must be dict")

        return self


# =========================================================
# UTILS
# =========================================================

def build_prediction_output(raw_outputs: Dict[str, Any]) -> PredictionOutput:

    tasks = {}

    for task, out in raw_outputs.items():

        tasks[task] = TaskPrediction(
            logits=out.get("logits"),
            probabilities=out.get("probabilities"),
            predictions=out.get("predictions"),
        )

    return PredictionOutput(tasks=tasks).validate()