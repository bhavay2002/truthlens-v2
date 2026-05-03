"""
File: postprocessing.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# CONFIG
# =========================================================

@dataclass
class PostprocessingConfig:

    # threshold for binary / multilabel
    threshold: float = 0.5

    # optional per-task thresholds
    task_thresholds: Optional[Dict[str, float]] = None

    # label mapping
    label_maps: Optional[Dict[str, Dict[int, str]]] = None

    # calibration
    apply_calibration: bool = False


# =========================================================
# CORE CLASS
# =========================================================

class Postprocessor:

    def __init__(self, config: Optional[PostprocessingConfig] = None):
        self.config = config or PostprocessingConfig()
        logger.info("Postprocessor initialized")

    # =====================================================
    # MAIN ENTRYPOINT
    # =====================================================

    def process(
        self,
        outputs: Dict[str, Any],
        *,
        task_types: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Process raw model outputs into final predictions.

        outputs:
        {
            task: {
                logits: np.ndarray
                probabilities: np.ndarray
            }
        }

        PP-3: ``task_types`` is REQUIRED and must contain an entry for
        every task in ``outputs``. The previous silent ``"multiclass"``
        default was responsible for emotion (multilabel, 20 heads) being
        softmax-collapsed into a single argmax label at inference time.
        Misconfiguration must surface, not degrade.
        """

        if task_types is None:
            raise ValueError(
                "Postprocessor.process() requires explicit task_types; "
                "silent default to 'multiclass' is unsafe for multilabel "
                "heads (e.g. emotion)."
            )

        results = {}

        for task, out in outputs.items():

            logits = out.get("logits")
            probs = out.get("probabilities")

            if task not in task_types:
                raise KeyError(
                    f"Postprocessor.process(): task_types missing entry for "
                    f"'{task}'. Known: {list(task_types)}"
                )
            task_type = task_types[task]

            if probs is None and logits is not None:
                probs = self._compute_probs(logits, task_type)

            preds = self._predict(probs, task, task_type)

            labels = self._map_labels(task, preds)

            confidence = self._confidence(probs, preds, task_type)

            results[task] = {
                "predictions": preds,
                "labels": labels,
                "confidence": confidence,
                "probabilities": probs,
                "logits": logits,
            }

        return results

    # =====================================================
    # PROBABILITIES
    # =====================================================

    def _compute_probs(self, logits, task_type):

        logits = np.asarray(logits)

        if task_type == "multiclass":
            e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            return e / (np.sum(e, axis=1, keepdims=True) + EPS)

        elif task_type in ("binary", "multilabel"):
            return 1 / (1 + np.exp(-logits))

        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    # =====================================================
    # PREDICTIONS
    # =====================================================

    def _predict(self, probs, task, task_type):

        probs = np.asarray(probs)

        threshold = self._get_threshold(task)

        if task_type == "multiclass":
            return np.argmax(probs, axis=1)

        elif task_type == "binary":
            return (probs > threshold).astype(int)

        elif task_type == "multilabel":
            return (probs > threshold).astype(int)

        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    # =====================================================
    # THRESHOLDS
    # =====================================================

    def _get_threshold(self, task):

        if self.config.task_thresholds and task in self.config.task_thresholds:
            return self.config.task_thresholds[task]

        return self.config.threshold

    # =====================================================
    # LABEL MAPPING
    # =====================================================

    def _map_labels(self, task, preds):

        if not self.config.label_maps:
            return preds

        mapping = self.config.label_maps.get(task)

        if not mapping:
            return preds

        return [mapping.get(int(p), str(p)) for p in preds]

    # =====================================================
    # CONFIDENCE
    # =====================================================

    def _confidence(self, probs, preds, task_type):

        probs = np.asarray(probs)

        if task_type == "multiclass":
            return np.max(probs, axis=1)

        elif task_type == "binary":
            return probs

        elif task_type == "multilabel":
            return np.max(probs, axis=1)

        return None

    # =====================================================
    # CALIBRATION HOOK (OPTIONAL)
    # =====================================================

    def apply_calibration(
        self,
        probs: np.ndarray,
        calibrator,
    ) -> np.ndarray:

        if not self.config.apply_calibration:
            return probs

        try:
            return calibrator.predict_proba(probs)
        except Exception as e:
            logger.warning("Calibration failed: %s", e)
            return probs

    # =====================================================
    # PP-2: PER-TASK THRESHOLD LOADING
    # =====================================================
    #
    # Training emits per-label F1-optimal thresholds via
    # ``src/evaluation/threshold_optimizer.py``. The previous inference
    # path hardcoded ``0.5`` for every multilabel head, throwing away
    # those tuned values. Operators now persist the optimizer's output
    # to ``<model_dir>/thresholds.json`` (or any path) and we plumb it
    # through here.

    def load_task_thresholds(self, path: str | Path) -> bool:
        """Populate ``config.task_thresholds`` from a JSON file.

        Returns True if thresholds were loaded, False if the file is
        absent. Raises on malformed content (silent fall-through to
        0.5 was the original bug).
        """
        path = Path(path)
        if not path.exists():
            return False

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, dict):
            raise ValueError(
                f"Threshold file {path} must be a JSON object "
                f"{{task: float}}; got {type(payload).__name__}"
            )

        thresholds: Dict[str, float] = {}
        for task, value in payload.items():
            try:
                thresholds[str(task)] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Threshold for task '{task}' is not a float: {value!r}"
                ) from exc

        existing = dict(self.config.task_thresholds or {})
        existing.update(thresholds)
        self.config.task_thresholds = existing
        logger.info(
            "Postprocessor: loaded %d task thresholds from %s",
            len(thresholds), path,
        )
        return True