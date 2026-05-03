#src\analysis\orchestrator.py

from __future__ import annotations

import logging
import time 
from typing import Dict, Any, Optional, List

from src.analysis.analysis_pipeline import AnalysisPipeline
from src.analysis.bias_profile_builder import BiasProfileBuilder
from src.analysis.propaganda_pattern_detector import PropagandaPatternDetector

logger = logging.getLogger(__name__)


# =========================================================
# ORCHESTRATOR
# =========================================================

class AnalysisOrchestrator:

    def __init__(
        self,
        pipeline: AnalysisPipeline,
        builder: Optional[BiasProfileBuilder] = None,
        propaganda_detector: Optional[PropagandaPatternDetector] = None,
        enable_timing: bool = True,
    ):
        self.pipeline = pipeline
        self.builder = builder or BiasProfileBuilder()
        self.propaganda = propaganda_detector or PropagandaPatternDetector()
        self.enable_timing = enable_timing

        logger.info("AnalysisOrchestrator initialized (final)")

    # =====================================================
    # SINGLE
    # =====================================================

    def run(self, text: str) -> Dict[str, Any]:

        start_total = time.perf_counter()

        try:
            text = self._validate_input(text)

            # -------------------------
            # PIPELINE
            # -------------------------
            t0 = time.perf_counter()
            raw = self.pipeline.run(text)
            t_pipeline = time.perf_counter() - t0

            if hasattr(raw, "model_dump"):
                raw = raw.model_dump()

            # -------------------------
            # POST PROCESS
            # -------------------------
            t1 = time.perf_counter()
            result = self._post_process(raw, text)
            t_post = time.perf_counter() - t1

            # -------------------------
            # META (TIMING)
            # -------------------------
            if self.enable_timing:
                result.setdefault("meta", {})
                result["meta"]["timing"] = {
                    "pipeline_ms": round(t_pipeline * 1000, 2),
                    "postprocess_ms": round(t_post * 1000, 2),
                    "total_ms": round((time.perf_counter() - start_total) * 1000, 2),
                }

            return result

        except Exception:
            logger.exception("Orchestrator run failed")
            return self._error_response()

    # =====================================================
    # BATCH
    # =====================================================

    def run_batch(self, texts: List[str]) -> List[Dict[str, Any]]:

        if not texts:
            return []

        start_total = time.perf_counter()

        try:
            indexed = list(enumerate(texts))

            t0 = time.perf_counter()
            raw_batch = self.pipeline.run_batch([t for _, t in indexed])
            t_pipeline = time.perf_counter() - t0

            results: List[Dict[str, Any]] = []

            for (idx, text), raw in zip(indexed, raw_batch):
                try:
                    if hasattr(raw, "model_dump"):
                        raw = raw.model_dump()

                    result = self._post_process(raw, text)

                    result.setdefault("meta", {})
                    result["meta"]["index"] = idx

                    results.append(result)

                except Exception:
                    logger.exception("Post-process failed")
                    results.append(self._error_response())

            # -------------------------
            # GLOBAL TIMING
            # -------------------------
            if self.enable_timing:
                total_time = (time.perf_counter() - start_total) * 1000

                for r in results:
                    r.setdefault("meta", {})
                    r["meta"]["batch_timing"] = {
                        "pipeline_total_ms": round(t_pipeline * 1000, 2),
                        "batch_total_ms": round(total_time, 2),
                    }

            return results

        except Exception:
            logger.exception("Batch failed → fallback to sequential")
            return [self.run(t) for t in texts]

    # =====================================================
    # POST PROCESS
    # =====================================================

    # Sections that AnalysisPipeline reports back from `run_all`.
    # Used both for safe extraction and for confidence aggregation.
    _ALL_SECTIONS = (
        "rhetorical",
        "argument",
        "context",
        "discourse",
        "emotion",
        "framing",
        "information",
        "information_omission",
        "ideology",
        "narrative_role",
        "narrative_conflict",
        "narrative_propagation",
        "narrative_temporal",
        "source",
    )

    # Sections aggregated to form the "narrative" view used by the
    # bias profile and propaganda detector.
    _NARRATIVE_SECTIONS = (
        "narrative_role",
        "narrative_conflict",
        "narrative_propagation",
        "narrative_temporal",
    )

    def _post_process(self, raw: Dict[str, Any], text: str) -> Dict[str, Any]:

        # -------------------------
        # SAFE EXTRACTION
        # -------------------------
        sections: Dict[str, Dict[str, float]] = {
            name: self._safe_section(raw.get(name, {}))
            for name in self._ALL_SECTIONS
        }

        narrative_section = self._merge_sections(
            sections, self._NARRATIVE_SECTIONS
        )

        # -------------------------
        # PROFILE
        # -------------------------
        # `bias` reflects framing signals (the primary bias surface);
        # `narrative` is the merged narrative view.
        profile = self.builder.build_profile(
            bias=sections["framing"],
            emotion=sections["emotion"],
            narrative=narrative_section,
            discourse=sections["discourse"],
            argument=sections.get("argument", {}),
            ideology=sections["ideology"],
        )

        # -------------------------
        # PROPAGANDA
        # -------------------------
        # The propaganda detector reads narrative-conflict signals
        # (polarization_ratio, conflict_intensity) — pass the conflict
        # section, not framing.
        propaganda = self.propaganda.analyze(
            emotion_features=sections["emotion"],
            narrative_features=sections["narrative_conflict"],
            rhetorical_features=sections["rhetorical"],
            argument_features=sections["argument"],
            information_features=sections["information"],
        )

        # -------------------------
        # CONFIDENCE
        # -------------------------
        confidence = self._confidence(sections)

        # -------------------------
        # META
        # -------------------------
        meta = {
            "input_length": len(text),
            "num_features": sum(len(v) for v in sections.values()),
            "confidence": confidence,
        }

        return {
            "features": raw,
            "profile": profile,
            "propaganda": propaganda,
            "meta": meta,
        }

    # =====================================================
    # SECTION HELPERS
    # =====================================================

    @staticmethod
    def _safe_section(section: Any) -> Dict[str, float]:
        if not isinstance(section, dict):
            return {}
        cleaned: Dict[str, float] = {}
        for k, v in section.items():
            if not isinstance(k, str):
                continue
            if isinstance(v, (int, float)):
                cleaned[k] = float(v)
        return cleaned

    @staticmethod
    def _merge_sections(
        sections: Dict[str, Dict[str, float]],
        names: tuple,
    ) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        for name in names:
            section = sections.get(name) or {}
            for k, v in section.items():
                # If two sub-sections share a key (e.g. `polarization_ratio`),
                # average rather than silently overwriting.
                if k in merged:
                    merged[k] = (merged[k] + float(v)) / 2.0
                else:
                    merged[k] = float(v)
        return merged

    # =====================================================
    # CONFIDENCE
    # =====================================================

    def _confidence(self, sections: Dict[str, Dict[str, float]]) -> float:
        """
        Confidence is the bounded mean over per-section means of finite,
        in-range feature values. Aggregating per-section first prevents
        any single large section (e.g. narrative_propagation with 15
        keys) from dominating the score.
        """

        section_means = []

        for section in sections.values():
            vals = [
                float(v)
                for v in section.values()
                if isinstance(v, (int, float))
                and v == v  # NaN check
                and v not in (float("inf"), float("-inf"))
            ]
            if not vals:
                continue
            section_means.append(sum(vals) / len(vals))

        if not section_means:
            return 0.0

        mean_val = sum(section_means) / len(section_means)

        return float(min(max(mean_val, 0.0), 1.0))

    # =====================================================
    # VALIDATION
    # =====================================================

    def _validate_input(self, text: Any) -> str:

        if not isinstance(text, str):
            raise ValueError("Input must be string")

        text = text.strip()

        if not text:
            raise ValueError("Empty text")

        return text

    # =====================================================
    # ERROR
    # =====================================================

    def _error_response(self) -> Dict[str, Any]:

        return {
            "features": {},
            "profile": {},
            "propaganda": {},
            "meta": {
                "error": True,
                "confidence": 0.0,
            },
        }