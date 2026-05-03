from __future__ import annotations

import logging
import time
from typing import Iterable, List, Dict, Any, Generator, Optional, Tuple

from src.analysis.analysis_pipeline import AnalysisPipeline
from src.analysis.feature_context import FeatureContext

logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTS
# =========================================================

DEFAULT_TIMEOUT = None


# =========================================================
# BATCH PROCESSOR
# =========================================================

class BatchProcessor:

    def __init__(
        self,
        pipeline: AnalysisPipeline,
        batch_size: int = 32,
        max_length: int = 100_000,
        drop_empty: bool = True,
        *,
        enable_profiling: bool = False,
    ):
        self.pipeline = pipeline
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.drop_empty = drop_empty
        self.enable_profiling = enable_profiling

        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        logger.info(
            "BatchProcessor initialized | batch_size=%d | max_length=%d",
            self.batch_size,
            self.max_length,
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def process(
        self,
        texts: Iterable[str],
        *,
        return_generator: bool = False,
    ) -> List[Dict[str, Any]] | Generator[Dict[str, Any], None, None]:

        generator = self._process_generator(texts)

        return generator if return_generator else list(generator)

    # =========================================================
    # CORE GENERATOR
    # =========================================================

    def _process_generator(
        self,
        texts: Iterable[str],
    ) -> Generator[Dict[str, Any], None, None]:

        batch: List[Tuple[int, str]] = []
        idx = 0

        for text in texts:

            processed = self._prepare_text(text)

            if processed is None:
                idx += 1
                continue

            batch.append((idx, processed))
            idx += 1

            if len(batch) >= self.batch_size:
                yield from self._run_batch(batch)
                batch.clear()

        if batch:
            yield from self._run_batch(batch)

    # =========================================================
    # TEXT PREPARATION
    # =========================================================

    def _prepare_text(self, text: Any) -> Optional[str]:

        if not isinstance(text, str):
            logger.warning("Skipping non-string input")
            return None

        text = text.strip()

        if self.drop_empty and not text:
            return None

        if len(text) > self.max_length:
            logger.warning("Text too long, truncating")
            text = text[: self.max_length]

        return text

    # =========================================================
    # BATCH EXECUTION (UPGRADED)
    # =========================================================

    def _run_batch(
        self,
        batch: List[Tuple[int, str]],
    ) -> Generator[Dict[str, Any], None, None]:

        indices, texts = zip(*batch)

        # IMPORTANT: AnalysisPipeline.run_batch expects List[str]. We pass the
        # raw strings so spaCy's `nlp.pipe` runs once across the batch and
        # `from_doc` materializes a fresh FeatureContext per item with its
        # own (per-doc) shared cache. Sharing a single dict across items
        # caused cross-contamination of the cached spaCy doc (CRIT-A2).
        text_list = list(texts)

        start_time = time.perf_counter()

        try:
            results = self.pipeline.run_batch(text_list)

            latency = time.perf_counter() - start_time

            if self.enable_profiling:
                logger.debug(
                    "Batch processed | size=%d | latency=%.4f sec | throughput=%.2f/s",
                    len(text_list),
                    latency,
                    len(text_list) / max(latency, 1e-8),
                )

            # preserve order
            for idx, result in zip(indices, results):
                yield self._attach_meta(result, idx)

        except Exception:
            logger.exception("Batch failed → fallback")

            yield from self._fallback_batch(batch)

    # =========================================================
    # FALLBACK (ROBUST)
    # =========================================================

    def _fallback_batch(
        self,
        batch: List[Tuple[int, str]],
    ) -> Generator[Dict[str, Any], None, None]:

        for idx, text in batch:

            # AnalysisPipeline.run accepts a raw text string and builds its
            # own per-call FeatureContext. We deliberately do NOT reuse a
            # cross-item shared cache here: each item must get a fresh
            # spaCy doc to avoid the contamination described in CRIT-A8.

            try:
                result = self.pipeline.run(text)
                yield self._attach_meta(result, idx)

            except Exception:
                logger.exception("Single item failed")

                yield self._empty_result(idx)

    # =========================================================
    # METADATA
    # =========================================================

    def _attach_meta(
        self,
        result: Dict[str, Any],
        idx: int,
    ) -> Dict[str, Any]:

        meta = result.setdefault("meta", {})
        meta["index"] = idx
        meta["success"] = True

        return result

    # =========================================================
    # EMPTY RESULT
    # =========================================================

    def _empty_result(self, idx: int) -> Dict[str, Any]:

        return {
            "features": {},
            "profile": {},
            "propaganda": {},
            "meta": {
                "index": idx,
                "success": False,
                "error": True,
            },
        }