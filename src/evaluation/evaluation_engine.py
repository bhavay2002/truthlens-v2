"""Lightweight evaluation orchestrator used by the trainer for in-loop eval."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from torch.utils.data import DataLoader

from src.evaluation.metrics_engine import MetricsEngine
from src.evaluation.prediction_collector import collect_all_tasks_from_loader

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """Run a DataLoader through a model and reduce to multi-task metrics."""

    def __init__(
        self,
        metrics_engine: Optional[MetricsEngine] = None,
        task_types: Optional[Dict[str, str]] = None,
    ):
        self.metrics_engine = metrics_engine or MetricsEngine()
        self.task_types = task_types or {}
        logger.info("EvaluationEngine initialized")

    def evaluate(
        self,
        model,
        dataloader: DataLoader,
        *,
        device=None,
    ) -> Dict[str, Any]:
        logger.info("Running evaluation...")

        predictions = self._collect_predictions(model=model, dataloader=dataloader, device=device)

        if not self.task_types:
            self.task_types = {
                task: data.get("task_type") for task, data in predictions.items()
                if isinstance(data, dict) and data.get("task_type")
            }

        metrics = self.metrics_engine.compute_multitask(
            predictions=predictions,
            task_types=self.task_types,
        )

        return {"metrics": metrics, "val_loss": self._extract_val_loss(metrics)}

    def _collect_predictions(self, model, dataloader: DataLoader, device=None):
        return collect_all_tasks_from_loader(
            model=model,
            dataloader=dataloader,
            device=device,
        )

    @staticmethod
    def _extract_val_loss(metrics: Dict[str, Any]) -> float:
        agg = metrics.get("__aggregate__") or {}

        if "log_loss" in agg:
            return float(agg["log_loss"])

        log_losses = [
            float(m["log_loss"])
            for k, m in metrics.items()
            if k != "__aggregate__" and isinstance(m, dict) and isinstance(m.get("log_loss"), (int, float))
        ]
        if log_losses:
            return float(np.mean(log_losses))

        # CRIT E4: prefer 1 - balanced_accuracy. Raw accuracy collapses on
        # imbalanced classes (a 95/5 binary task scores ~0.95 from a class-prior
        # baseline) and silently locks early stopping onto a degenerate model.
        if isinstance(agg.get("balanced_accuracy"), (int, float)):
            return float(1.0 - agg["balanced_accuracy"])

        balanced = [
            float(m["balanced_accuracy"])
            for k, m in metrics.items()
            if k != "__aggregate__"
            and isinstance(m, dict)
            and isinstance(m.get("balanced_accuracy"), (int, float))
        ]
        if balanced:
            return float(1.0 - np.mean(balanced))

        # Last-resort fallback when balanced_accuracy is missing entirely
        # (e.g. multilabel-only runs); kept for backwards compatibility.
        if isinstance(agg.get("accuracy"), (int, float)):
            return float(1.0 - agg["accuracy"])

        accuracies = [
            float(m["accuracy"])
            for k, m in metrics.items()
            if k != "__aggregate__" and isinstance(m, dict) and isinstance(m.get("accuracy"), (int, float))
        ]
        if accuracies:
            return float(1.0 - np.mean(accuracies))

        return 0.0
