"""
File Name: integration_runner.py
Module: Analysis - Integration Runner

Description:
    Backward-compatibility shim that aggregates the set of single-text
    analyzers in :mod:`src.analysis` into a single object exposing
    ``analyze_text(text)``. Returns a dict keyed by analyzer name (e.g.
    ``"narrative_propagation"``, ``"framing"``) where each value is the
    analyzer's ``analyze(text)`` output.

    This module exists to satisfy callers that imported
    ``src.analysis.integration_runner.AnalysisIntegrationRunner`` after the
    pipeline refactor. Per-analyzer failures are caught and logged so a single
    broken analyzer does not bring down the whole batch.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _default_analyzers() -> List[Tuple[str, Any]]:
    """Lazily construct the default analyzer set.

    Imports are local so an optional analyzer that fails to import does not
    prevent the rest of the runner from working.
    """
    pairs: List[Tuple[str, Any]] = []

    def _try(name: str, factory):
        try:
            pairs.append((name, factory()))
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to construct analyzer %s", name)

    from src.analysis.argument_mining import ArgumentMiningAnalyzer
    from src.analysis.framing_analysis import FramingAnalyzer
    from src.analysis.ideological_language_detector import (
        IdeologicalLanguageDetector,
    )
    from src.analysis.information_density_analyzer import (
        InformationDensityAnalyzer,
    )
    from src.analysis.information_omission_detector import (
        InformationOmissionDetector,
    )
    from src.analysis.context_omission_detector import ContextOmissionDetector
    from src.analysis.discourse_coherence_analyzer import (
        DiscourseCoherenceAnalyzer,
    )
    from src.analysis.emotion_target_analysis import EmotionTargetAnalyzer
    from src.analysis.narrative_conflict import NarrativeConflictAnalyzer
    from src.analysis.narrative_propagation import (
        NarrativePropagationAnalyzer,
    )
    from src.analysis.narrative_role_extractor import NarrativeRoleExtractor
    from src.analysis.narrative_temporal_analyzer import (
        NarrativeTemporalAnalyzer,
    )
    from src.analysis.propaganda_pattern_detector import (
        PropagandaPatternDetector,
    )
    from src.analysis.rhetorical_device_detector import (
        RhetoricalDeviceDetector,
    )
    from src.analysis.source_attribution_analyzer import (
        SourceAttributionAnalyzer,
    )

    _try("argument_mining", ArgumentMiningAnalyzer)
    _try("framing", FramingAnalyzer)
    _try("ideological_language", IdeologicalLanguageDetector)
    _try("information_density", InformationDensityAnalyzer)
    _try("information_omission", InformationOmissionDetector)
    _try("context_omission", ContextOmissionDetector)
    _try("discourse_coherence", DiscourseCoherenceAnalyzer)
    _try("emotion_target", EmotionTargetAnalyzer)
    _try("narrative_conflict", NarrativeConflictAnalyzer)
    _try("narrative_propagation", NarrativePropagationAnalyzer)
    _try("narrative_role", NarrativeRoleExtractor)
    _try("narrative_temporal", NarrativeTemporalAnalyzer)
    _try("propaganda_patterns", PropagandaPatternDetector)
    _try("rhetorical_devices", RhetoricalDeviceDetector)
    _try("source_attribution", SourceAttributionAnalyzer)

    return pairs


class AnalysisIntegrationRunner:
    """Run every registered single-text analyzer over a piece of text.

    Parameters
    ----------
    analyzers:
        Optional iterable of ``(name, analyzer_instance)`` pairs. Each
        analyzer must expose an ``analyze(text: str) -> dict`` method. When
        ``None``, the default set defined in :func:`_default_analyzers` is
        used.
    fail_fast:
        When ``True``, the first analyzer error is re-raised. When ``False``
        (default), errors are logged and the analyzer's slot is recorded as
        ``{"error": "<message>"}`` so downstream consumers see structured
        feedback instead of a missing key.
    """

    def __init__(
        self,
        analyzers: Optional[Iterable[Tuple[str, Any]]] = None,
        *,
        fail_fast: bool = False,
    ) -> None:
        self._analyzers: List[Tuple[str, Any]] = (
            list(analyzers) if analyzers is not None else _default_analyzers()
        )
        self._fail_fast = fail_fast
        logger.info(
            "AnalysisIntegrationRunner initialized | analyzers=%d",
            len(self._analyzers),
        )

    @property
    def analyzers(self) -> List[Tuple[str, Any]]:
        return list(self._analyzers)

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Run every analyzer on ``text`` and return a dict of results.

        Per-analyzer exceptions are caught and recorded under
        ``{"error": str(exc)}`` unless ``fail_fast`` is set.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise ValueError("text must be a non-empty string")

        # Build a single FeatureContext so every analyzer shares the
        # same spaCy doc cache and lazy-computed token state.
        # Use from_doc(mode="safe") so the "ner" and "syntax" cache
        # slots are pre-seeded with the full-pipeline doc — the same
        # Vocab that EmotionTargetAnalyzer's PhraseMatcher was built
        # with. A plain FeatureContext(text=text) leaves those slots
        # empty, forcing get_doc(..., task="ner") to load the NER-task
        # nlp (different Vocab object) and triggering the
        # "doc.vocab does not match PhraseMatcher vocab" warning on
        # every article.
        from src.analysis.feature_context import FeatureContext
        from src.analysis.spacy_loader import get_shared_nlp
        try:
            nlp = get_shared_nlp("safe")
            doc = nlp(text)
            ctx = FeatureContext.from_doc(doc, mode="safe")
        except Exception:
            logger.exception("FeatureContext build failed; falling back to raw text")
            ctx = None

        results: Dict[str, Any] = {}
        for name, analyzer in self._analyzers:
            try:
                if ctx is not None and hasattr(analyzer, "analyze"):
                    # BUG-A-PROP-RUNNER guard: some analyzers (currently
                    # PropagandaPatternDetector) consume *upstream feature
                    # dicts* rather than a FeatureContext.  Calling
                    # analyze(ctx) on them ignores the context, falls
                    # back to all-empty upstream dicts, and silently
                    # returns an all-zero feature dict — corrupting
                    # downstream consumers.  Record a structured gap
                    # instead so the caller sees an explicit signal.
                    if getattr(analyzer, "requires_upstream_features", False):
                        results[name] = {
                            "skipped": (
                                "requires_upstream_features — "
                                "invoke via the orchestrator with "
                                "upstream feature dicts"
                            )
                        }
                        continue
                    # Modern analyzers expect a FeatureContext.
                    results[name] = analyzer.analyze(ctx)
                else:
                    results[name] = analyzer.analyze(text)
            except TypeError:
                # Legacy analyzers may still expect raw text.
                try:
                    results[name] = analyzer.analyze(text)
                except Exception as exc:
                    if self._fail_fast:
                        raise
                    logger.exception("Analyzer %s failed", name)
                    results[name] = {"error": str(exc)}
            except Exception as exc:
                if self._fail_fast:
                    raise
                logger.exception("Analyzer %s failed", name)
                results[name] = {"error": str(exc)}
        return results

    def analyze_batch(self, texts: Iterable[str]) -> List[Dict[str, Any]]:
        """Run the runner over an iterable of texts."""
        return [self.analyze_text(t) for t in texts]
