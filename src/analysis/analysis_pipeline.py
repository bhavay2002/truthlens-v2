from __future__ import annotations

import logging 
import time
from typing import Dict, List, Any, Optional

from spacy.tokens import Doc

from src.analysis.spacy_loader import (
    get_shared_nlp,
    _configure_torch_threads_for_multiprocess,
)
from src.analysis.feature_context import FeatureContext
from src.analysis.feature_merger import FeatureMerger
from src.analysis.analysis_config import (
    ANALYSIS_CONFIG,
    AnalysisConfig,
    build_default_config,
    validate_config_against_registry,
)
from src.analysis.analysis_registry import AnalyzerRegistry, AnalyzerExecution

logger = logging.getLogger(__name__)

from src.monitoring.feature_logger import (
    log_feature_stats,
    log_feature_summary,
    time_block,
    log_failure,
)


# =========================================================
# PIPELINE (UPGRADED)
# =========================================================

class AnalysisPipeline:
    """
    Production-grade analysis pipeline.

    Features:
    - registry-driven execution
    - structured outputs
    - latency tracking
    - batch optimization
    - fail-safe execution
    """

    def __init__(
        self,
        registry: AnalyzerRegistry,
        *,
        config: Optional[AnalysisConfig] = None,
        nlp_mode: str = "safe",
    ):
        self.config = config or build_default_config()
        self.registry = registry
        self.nlp_mode = nlp_mode

        # CRIT-A3: fail fast if AnalysisConfig.analyzers references analyzer
        # names that don't exist in the registry. The previous behavior was
        # a silent no-op for ablation flags / per-analyzer ordering.
        validate_config_against_registry(self.config, self.registry.list())

        self.nlp = get_shared_nlp(mode=nlp_mode)
        self.merger = FeatureMerger()

        logger.info(
            "AnalysisPipeline initialized | analyzers=%d | mode=%s",
            len(self.registry.list()),
            nlp_mode,
        )

    # =====================================================
    # SINGLE RUN
    # =====================================================



    def run(self, text: str) -> Dict[str, Any]:
    
        start = time.time()
    
        try:
            # -------------------------------
            # VALIDATION
            # -------------------------------
            text = self._validate(text)
    
            # -------------------------------
            # NLP + CONTEXT (TIMED)
            # -------------------------------
            with time_block("nlp_processing"):
                doc = self.nlp(text)
                ctx = FeatureContext.from_doc(doc, mode=self.nlp_mode)
    
            # -------------------------------
            # ANALYZER EXECUTION (TIMED)
            # -------------------------------
            with time_block("analyzer_execution"):
                results = self._execute(ctx)
    
            # -------------------------------
            # FEATURE MERGING (TIMED)
            # -------------------------------
            with time_block("feature_merging"):
                merged, vector, keys = self._post_process(results)
    
            # -------------------------------
            # 🔍 FEATURE OBSERVABILITY
            # -------------------------------
            log_feature_stats(merged, task="analysis")
            log_feature_summary(merged, task="analysis")
    
            return {
                "sections": {k: v.output for k, v in results.items()},
                "features": merged,
                "vector": vector,
                "feature_keys": keys,
                "meta": self._build_meta(results, start),
            }
    
        except Exception as e:
            log_failure(
                e,
                context={
                    "stage": "analysis_pipeline.run",
                    "input_text": text[:200] if isinstance(text, str) else None,
                },
            )
            raise
    
    # =====================================================
    # BATCH RUN (OPTIMIZED)
    # =====================================================
    
    def run_batch(self, texts: List[str]) -> List[Dict[str, Any]]:

        if not texts:
            return []

        start = time.time()

        texts = [self._validate(t) for t in texts]

        # PERF-A5: honor the configured spaCy worker count instead of
        # hard-coding `n_process=1`. The previous default discarded
        # spaCy's tuned multi-process pickle path on every batch and
        # produced a 4-8x regression on CPU-bound batches > ~50 items.
        # Falls back to 1 if the config value is missing or invalid so
        # behavior stays deterministic in tests.
        try:
            n_process = int(getattr(ANALYSIS_CONFIG.spacy, "n_process", 1) or 1)
        except (TypeError, ValueError):
            n_process = 1
        if n_process < 1:
            n_process = 1

        # Section 6: prevent torch thread oversubscription when spaCy
        # forks worker processes. No-op when n_process == 1.
        if n_process > 1:
            _configure_torch_threads_for_multiprocess()

        docs = list(
            self.nlp.pipe(
                texts,
                batch_size=self.config.pipeline.batch_size,
                n_process=n_process,
            )
        )

        results = []

        for doc in docs:
            ctx = FeatureContext.from_doc(doc, mode=self.nlp_mode)

            exec_results = self._execute(ctx)
            merged, vector, keys = self._post_process(exec_results)

            results.append({
                "sections": {k: v.output for k, v in exec_results.items()},
                "features": merged,
                "vector": vector,
                "feature_keys": keys,
                "meta": self._build_meta(exec_results, start),
            })

        return results

    # =====================================================
    # EXECUTION ENGINE (CORE)
    # =====================================================

    def _execute(
        self,
        ctx: FeatureContext,
    ) -> Dict[str, AnalyzerExecution]:

        return self.registry.run_all(
            ctx,
            extra_inputs=self._extra_inputs(ctx),
        )

    # =====================================================
    # POST PROCESSING
    # =====================================================

    def _post_process(
        self,
        results: Dict[str, AnalyzerExecution],
    ):

        sections = {k: v.output for k, v in results.items()}

        merged = self.merger.merge(sections)
        vector, keys = self.merger.to_vector(sections)

        return merged, vector, keys

    # =====================================================
    # META / OBSERVABILITY
    # =====================================================

    def _build_meta(
        self,
        results: Dict[str, AnalyzerExecution],
        start_time: float,
    ) -> Dict[str, Any]:

        total_time = time.time() - start_time

        failures = [
            k for k, v in results.items() if not v.success
        ]

        latencies = {
            k: v.latency for k, v in results.items()
        }

        return {
            "total_latency": total_time,
            "analyzer_latency": latencies,
            "failed_analyzers": failures,
            "num_analyzers": len(results),
        }

    # =====================================================
    # EXTRA INPUTS (EXTENSIBILITY)
    # =====================================================

    def _extra_inputs(self, ctx: FeatureContext) -> Dict[str, Any]:
        """
        Hook for passing extra shared inputs to analyzers.
        """
        return {}

    # =====================================================
    # VALIDATION
    # =====================================================

    def _validate(self, text: Any) -> str:

        if not isinstance(text, str):
            raise ValueError("Input must be string")

        text = text.strip() 

        if not text:
            raise ValueError("Empty text")

        # Section 9: cap the configured max_text_length at the loaded
        # spaCy model's `nlp.max_length`. The default config limit
        # (100K) is well under the loader's 2M ceiling, but a custom
        # AnalysisConfig (or a future loader change) could otherwise
        # let an oversized string reach `nlp(...)`, which raises an
        # opaque `E088` from spaCy. Truncating here gives a single,
        # consistent failure mode.
        cfg_limit = self.config.global_config.max_text_length
        spacy_limit = getattr(self.nlp, "max_length", cfg_limit) or cfg_limit
        effective_limit = min(cfg_limit, spacy_limit)

        if len(text) > effective_limit:
            if self.config.global_config.truncate_text:
                text = text[:effective_limit]
            else:
                raise ValueError("Text too long")

        return text