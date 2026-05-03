"""
File: prediction_service.py
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional

import numpy as np

from src.inference.inference_engine import InferenceEngine
from src.inference.inference_logger import InferenceLogger
from src.inference.inference_cache import InferenceCache, InferenceCacheConfig
from src.inference.monitoring import InferenceMonitor
from src.inference.report_generator import ReportGenerator
from src.inference.result_formatter import ResultFormatter

logger = logging.getLogger(__name__)


# =========================================================
# SERVICE
# =========================================================

class PredictionService:

    def __init__(
        self,
        engine: InferenceEngine,
        *,
        cache: Optional[InferenceCache] = None,
        logger_: Optional[InferenceLogger] = None,
        report_generator: Optional[ReportGenerator] = None,
        formatter: Optional[ResultFormatter] = None,
        monitor: Optional[InferenceMonitor] = None,
    ):

        self.engine = engine

        self.cache = cache or InferenceCache(
            InferenceCacheConfig(enable_memory_cache=True)
        )

        self.logger = logger_ or InferenceLogger()
        self.report_generator = report_generator or ReportGenerator()
        self.formatter = formatter or ResultFormatter()
        # UNUSED-FIX: ``predict_api`` was building an ``InferenceMonitor``
        # and attaching it to ``service.monitor`` but no code path ever
        # called ``monitor.update(...)``. Hold it here so ``predict``,
        # ``predict_full_batch`` and ``predict_full`` can record latency
        # / outcome metrics every request.
        self.monitor = monitor

        logger.info("PredictionService initialized")

    # =====================================================
    # MONITOR HELPERS
    # =====================================================

    def _record_monitor(
        self,
        *,
        start_time: float,
        confidence: Optional[float] = None,
        probabilities: Optional[Any] = None,
        error: bool = False,
    ) -> None:
        """Best-effort metric emission. Monitor failures must never
        propagate into the request path — log and move on."""
        if self.monitor is None:
            return
        try:
            latency_ms = self.logger.stop_timer(start_time)
            self.monitor.update(
                latency_ms=latency_ms,
                confidence=confidence,
                probabilities=probabilities,
                error=error,
            )
        except Exception as exc:  # pragma: no cover - never break inference
            logger.debug("Monitor update failed: %s", exc)

    # =====================================================
    # CORE PREDICT
    # =====================================================

    def predict(
        self,
        text: str,
        *,
        use_cache: bool = True,
    ) -> Dict[str, Any]:

        def _compute() -> Dict[str, Any]:
            start = self.logger.start_timer()
            try:
                outputs = self.engine.predict_for_evaluation([text])
                preds = self._postprocess(outputs)
            except Exception:
                # UNUSED-FIX: emit a failure metric before re-raising so
                # operators see error-rate spikes in the monitor instead
                # of only in the application logs.
                self._record_monitor(start_time=start, error=True)
                raise
            self.logger.log_prediction(
                start_time=start,
                model_versions={},
                feature_count=0,
                predicted_label=preds.get("label"),
                prediction_confidence=preds.get("confidence"),
            )
            self._record_monitor(
                start_time=start,
                confidence=preds.get("confidence"),
            )
            return preds

        # LAT-5: route through the cache's single-flight helper so
        # concurrent requests for the same text only run one forward
        # pass. When caching is disabled, fall back to direct compute.
        if use_cache and self.cache is not None:
            return self.cache.get_or_compute(text, _compute)

        return _compute()

    # =====================================================
    # BATCH
    # =====================================================

    def predict_batch(
        self,
        texts: List[str],
    ) -> List[Dict[str, Any]]:

        outputs = self.engine.predict_for_evaluation(texts)
        return self._build_records_from_engine_output(outputs)

    # =====================================================
    # 🔥 LAT-1: BATCHED FULL PREDICTION
    # =====================================================

    def predict_full_batch(
        self,
        texts: List[str],
        *,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Batched equivalent of ``predict``.

        LAT-1: previously every batched caller (e.g. ``advanced_analysis``)
        looped ``self.predict(t)`` per text, which re-tokenised and ran a
        forward pass for every single sample. We now run a single forward
        pass over the entire batch and split the results per-sample.
        """

        if not texts:
            return []

        # ---------------- CACHE LOOKUP ----------------
        results: List[Optional[Dict[str, Any]]] = [None] * len(texts)
        pending_idx: List[int] = []
        pending_texts: List[str] = []

        if use_cache and self.cache is not None:
            for i, t in enumerate(texts):
                cached = self.cache.get(t)
                if cached is not None:
                    results[i] = cached
                else:
                    pending_idx.append(i)
                    pending_texts.append(t)
        else:
            pending_idx = list(range(len(texts)))
            pending_texts = list(texts)

        # ---------------- BATCHED INFERENCE ----------------
        if pending_texts:
            start = self.logger.start_timer()
            try:
                outputs = self.engine.predict_for_evaluation(pending_texts)
                records = self._build_records_from_engine_output(outputs)
            except Exception:
                self._record_monitor(start_time=start, error=True)
                raise

            for slot, record in zip(pending_idx, records):
                results[slot] = record
                if use_cache and self.cache is not None:
                    self.cache.set(texts[slot], record)

            self.logger.log_prediction(
                start_time=start,
                model_versions={},
                feature_count=len(pending_texts),
                predicted_label=None,
                prediction_confidence=None,
            )
            # Record one monitor sample per item with the per-item
            # confidence so latency averages are not skewed by batching.
            for record in records:
                self._record_monitor(
                    start_time=start,
                    confidence=record.get("confidence"),
                )

        return [r if r is not None else {} for r in results]

    def _build_records_from_engine_output(
        self,
        outputs: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """CRIT-2: read the nested ``{task: {...}}`` contract returned by
        ``InferenceEngine.predict_for_evaluation``."""

        task_name = getattr(self.engine, "DEFAULT_TASK_NAME", "main")
        task_out = outputs.get(task_name) or {}

        probs_arr = task_out.get("calibrated_probabilities")
        if probs_arr is None:
            probs_arr = task_out.get("probabilities")
        preds_arr = task_out.get("predictions")
        is_legacy_binary = bool(
            getattr(self.engine, "_is_legacy_binary_label_map", lambda _n: False)(
                probs_arr.shape[-1] if probs_arr is not None else 0
            )
        )

        n = (
            len(preds_arr) if preds_arr is not None
            else (len(probs_arr) if probs_arr is not None else 0)
        )

        results = []
        for i in range(n):

            conf = float(np.max(probs_arr[i])) if probs_arr is not None else None
            # CRIT-4: do not invent a fake_probability when the head is not
            # the legacy binary classifier.
            if is_legacy_binary and probs_arr is not None:
                fake_prob = float(probs_arr[i][1])
            else:
                fake_prob = None

            results.append({
                "label": int(preds_arr[i]) if preds_arr is not None else None,
                "confidence": conf,
                "fake_probability": fake_prob,
            })

        return results

    # =====================================================
    # FULL PIPELINE
    # =====================================================

    def predict_full(
        self,
        text: str,
        *,
        use_cache: bool = True,
    ) -> Dict[str, Any]:

        # REC-2: ``predict()`` consults the cache, but ``predict_full``
        # used to skip it entirely — every call re-ran the forward pass
        # plus the (much heavier) report generation. Honour the same
        # cache contract by namespacing the full-report blob under a
        # distinct key so it does not collide with the basic-prediction
        # entry written by ``predict()``.
        cache_key = f"__full__::{text}" if text is not None else None
        if use_cache and self.cache is not None and cache_key is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        start = self.logger.start_timer()
        try:
            outputs = self.engine.predict_for_evaluation([text])

            # ---------------- UNCERTAINTY ----------------
            uncertainty = self._compute_uncertainty(outputs)

            # ---------------- REPORT ----------------
            report = self.report_generator.generate_report(
                article_text=text,
                predictions=outputs,
                uncertainty=uncertainty,
            )
        except Exception:
            self._record_monitor(start_time=start, error=True)
            raise

        self._record_monitor(start_time=start)

        if use_cache and self.cache is not None and cache_key is not None:
            try:
                self.cache.set(cache_key, report)
            except Exception as exc:  # pragma: no cover
                logger.debug("Full-report cache set failed: %s", exc)

        return report

    # =====================================================
    # FORMATTED OUTPUT
    # =====================================================

    def predict_formatted(
        self,
        text: str,
        *,
        mode: str = "api",
    ) -> Dict[str, Any]:

        report = self.predict_full(text)

        if mode == "api":
            return self.formatter.format_api_response(report)

        elif mode == "dashboard":
            return self.formatter.format_dashboard_report(report)

        elif mode == "research":
            return self.formatter.format_research_export(report)

        else:
            raise ValueError("Invalid mode")

    # =====================================================
    # EVALUATION MODE
    # =====================================================

    def predict_for_evaluation(
        self,
        texts: List[str],
    ) -> Dict[str, Any]:

        return self.engine.predict_for_evaluation(texts)

    # =====================================================
    # POSTPROCESS
    # =====================================================

    def _postprocess(self, outputs):
        # CRIT-2: ``predict_for_evaluation`` now returns the nested
        # ``{task: {...}}`` contract; the single-text result is the head
        # of the per-task arrays.
        records = self._build_records_from_engine_output(outputs)
        if not records:
            return {"label": None, "confidence": None, "fake_probability": None}
        return records[0]

    # =====================================================
    # UNCERTAINTY
    # =====================================================

    def _compute_uncertainty(self, outputs):

        results = {}

        for task, out in outputs.items():

            # CRIT-2: ``_meta`` (and any other non-task scratch keys) sit
            # alongside per-task entries; skip anything that does not match
            # the nested ``{logits/probabilities/...}`` shape.
            if not isinstance(out, dict) or "probabilities" not in out:
                continue

            probs = out["probabilities"]

            if probs is None:
                continue

            probs = np.asarray(probs)
            eps = 1e-12

            # PP-4: pick the correct entropy formula by task type.
            # Categorical entropy assumes rows sum to 1 (multiclass softmax).
            # Multilabel heads are independent Bernoullis and need
            # ``-Σ_k [p_k log p_k + (1-p_k) log(1-p_k)]``; using the
            # categorical formula on multilabel probabilities gave a
            # systematically wrong uncertainty signal.
            task_type = out.get("task_type")
            if task_type is None:
                # Fallback inference: rows that don't sum to ~1 across the
                # last axis are almost certainly multilabel sigmoids.
                task_type = "multiclass"
                if probs.ndim >= 2:
                    row_sums = probs.sum(axis=-1)
                    if not np.allclose(row_sums, 1.0, atol=1e-3):
                        task_type = "multilabel"
                else:
                    task_type = "binary"

            if task_type == "multilabel":
                entropy = -np.sum(
                    probs * np.log(probs + eps)
                    + (1 - probs) * np.log(1 - probs + eps),
                    axis=-1,
                )
            elif task_type == "binary":
                # Per-sample Bernoulli entropy.
                entropy = -(probs * np.log(probs + eps)
                            + (1 - probs) * np.log(1 - probs + eps))
            else:
                # multiclass / categorical
                entropy = -np.sum(probs * np.log(probs + eps), axis=-1)

            entropy = np.atleast_1d(entropy)

            results[task] = {
                "mean_entropy": float(np.mean(entropy)),
                "p95_entropy": float(np.percentile(entropy, 95)),
            }

        return results