from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext

logger = logging.getLogger(__name__)


# =========================================================
# SPEC
# =========================================================

@dataclass
class AnalyzerSpec:
    name: str
    analyzer: BaseAnalyzer

    enabled: bool = True
    requires: List[str] = field(default_factory=list)
    provides: List[str] = field(default_factory=list)

    order: int = 0
    critical: bool = False  # 🔥 new (fail pipeline if breaks)

    # CRIT-A5: cached set of keyword argument names accepted by
    # `analyzer.analyze`. Used to filter `extra_inputs` so we never
    # forward kwargs an analyzer doesn't declare (which would raise
    # TypeError). `None` means "no filtering computed yet".
    accepted_kwargs: Optional[Set[str]] = None
    accepts_var_kwargs: bool = False


# =========================================================
# EXECUTION RESULT (NEW)
# =========================================================

@dataclass
class AnalyzerExecution:
    output: Dict[str, float]
    latency: float
    success: bool
    error: Optional[str] = None


# =========================================================
# REGISTRY
# =========================================================

class AnalyzerRegistry:

    def __init__(self):
        self._registry: Dict[str, AnalyzerSpec] = {}

    # -----------------------------------------------------

    def register(
        self,
        name: str,
        analyzer: BaseAnalyzer,
        *,
        enabled: bool = True,
        requires: Optional[List[str]] = None,
        provides: Optional[List[str]] = None,
        order: int = 0,
        critical: bool = False,
    ) -> None:

        if name in self._registry:
            raise ValueError(f"Analyzer '{name}' already registered")

        accepted, accepts_var = _inspect_analyzer_kwargs(analyzer)

        self._registry[name] = AnalyzerSpec(
            name=name,
            analyzer=analyzer,
            enabled=enabled,
            requires=requires or [],
            provides=provides or [],
            order=order,
            critical=critical,
            accepted_kwargs=accepted,
            accepts_var_kwargs=accepts_var,
        )

    # -----------------------------------------------------

    def get_active(self) -> List[AnalyzerSpec]:
        return [s for s in self._registry.values() if s.enabled]

    # -----------------------------------------------------

    def get_ordered(self) -> List[AnalyzerSpec]:
        ordered = sorted(self.get_active(), key=lambda x: x.order)
        self._validate_dependencies(ordered)
        return ordered

    # -----------------------------------------------------
    # 🔥 DEPENDENCY VALIDATION (NEW)
    # -----------------------------------------------------

    def _validate_dependencies(self, specs: List[AnalyzerSpec]):

        names = {s.name for s in specs}

        for spec in specs:
            for dep in spec.requires:
                if dep not in names:
                    raise RuntimeError(
                        f"Analyzer '{spec.name}' requires missing '{dep}'"
                    )

        # cycle detection (simple DFS)
        visited = set()
        stack = set()

        def dfs(node: str):
            if node in stack:
                raise RuntimeError(f"Cyclic dependency detected at '{node}'")
            if node in visited:
                return

            stack.add(node)
            visited.add(node)

            for dep in self._registry[node].requires:
                dfs(dep)

            stack.remove(node)

        for s in specs:
            dfs(s.name)

    # -----------------------------------------------------
    # 🔥 MAIN EXECUTION (UPGRADED)
    # -----------------------------------------------------

    def run_all(
        self,
        ctx: FeatureContext,
        *,
        extra_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, AnalyzerExecution]:

        extra_inputs = extra_inputs or {}
        results: Dict[str, AnalyzerExecution] = {}

        for spec in self.get_ordered():

            start = time.time()

            try:
                # CRIT-A5: per-analyzer keyword arguments. We forward
                # ONLY the kwargs the analyzer's `analyze()` actually
                # declares (or all of them if it accepts **kwargs).
                # Previously every analyzer received the full
                # `extra_inputs` dict, which raised TypeError for any
                # analyzer that didn't list every key explicitly.
                if spec.accepts_var_kwargs or spec.accepted_kwargs is None:
                    kwargs = dict(extra_inputs)
                else:
                    kwargs = {
                        k: v
                        for k, v in extra_inputs.items()
                        if k in spec.accepted_kwargs
                    }

                # Invoke through the BaseAnalyzer __call__ wrapper when
                # available so caching, validation, and fallbacks run.
                runner = (
                    spec.analyzer
                    if callable(spec.analyzer)
                    else spec.analyzer.analyze
                )

                output = runner(ctx, **kwargs) if kwargs else runner(ctx)

                if not isinstance(output, dict):
                    raise TypeError("Analyzer output must be dict")

                latency = time.time() - start

                results[spec.name] = AnalyzerExecution(
                    output=output,
                    latency=latency,
                    success=True,
                )

            except Exception as e:

                latency = time.time() - start

                logger.exception("Analyzer failed: %s", spec.name)

                if spec.critical:
                    raise RuntimeError(
                        f"Critical analyzer failed: {spec.name}"
                    ) from e

                results[spec.name] = AnalyzerExecution(
                    output={},
                    latency=latency,
                    success=False,
                    error=str(e),
                )

        return results

    # -----------------------------------------------------

    def list(self) -> List[str]:
        return list(self._registry.keys())


# =========================================================
# KWARG INTROSPECTION (CRIT-A5)
# =========================================================

def _inspect_analyzer_kwargs(analyzer: Any) -> tuple[Optional[Set[str]], bool]:
    """Return (accepted_keyword_names, accepts_var_keyword) for an analyzer.

    Inspects ``analyzer.analyze`` (preferred) and falls back to ``__call__``.
    Returns ``(None, True)`` if introspection fails so the caller forwards
    everything (matching legacy behavior).
    """
    target = getattr(analyzer, "analyze", None) or analyzer
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return None, True

    accepted: Set[str] = set()
    accepts_var = False
    for pname, param in sig.parameters.items():
        if pname in ("self", "ctx"):
            continue
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            accepted.add(pname)
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_var = True

    return accepted, accepts_var


# =========================================================
# DEFAULT REGISTRY (PRODUCTION SET)
# =========================================================

def build_default_registry() -> "AnalyzerRegistry":
    """
    Construct the default analyzer registry used by
    :class:`AnalysisPipeline`.

    Imports are local to keep the module lightweight at import time and
    to avoid cyclic imports between the registry and individual
    analyzers.
    """
    from src.analysis.rhetorical_device_detector import RhetoricalDeviceDetector
    from src.analysis.argument_mining import ArgumentMiningAnalyzer
    from src.analysis.context_omission_detector import ContextOmissionDetector
    from src.analysis.discourse_coherence_analyzer import (
        DiscourseCoherenceAnalyzer,
    )
    from src.analysis.emotion_target_analysis import EmotionTargetAnalyzer
    from src.analysis.framing_analysis import FramingAnalyzer
    from src.analysis.information_density_analyzer import (
        InformationDensityAnalyzer,
    )
    from src.analysis.information_omission_detector import (
        InformationOmissionDetector,
    )
    from src.analysis.ideological_language_detector import (
        IdeologicalLanguageDetector,
    )
    from src.analysis.narrative_role_extractor import NarrativeRoleExtractor
    from src.analysis.narrative_conflict import NarrativeConflictAnalyzer
    from src.analysis.narrative_propagation import (
        NarrativePropagationAnalyzer,
    )
    from src.analysis.narrative_temporal_analyzer import (
        NarrativeTemporalAnalyzer,
    )
    from src.analysis.source_attribution_analyzer import (
        SourceAttributionAnalyzer,
    )

    reg = AnalyzerRegistry()

    reg.register("rhetorical", RhetoricalDeviceDetector(), order=1)
    reg.register("argument", ArgumentMiningAnalyzer(), order=2)
    reg.register("context", ContextOmissionDetector(), order=3)
    reg.register("discourse", DiscourseCoherenceAnalyzer(), order=4)
    reg.register("emotion", EmotionTargetAnalyzer(), order=5)
    reg.register("framing", FramingAnalyzer(), order=6)
    reg.register("information", InformationDensityAnalyzer(), order=7)
    reg.register(
        "information_omission", InformationOmissionDetector(), order=8
    )
    reg.register("ideology", IdeologicalLanguageDetector(), order=9)
    reg.register("narrative_role", NarrativeRoleExtractor(), order=10)
    reg.register("narrative_conflict", NarrativeConflictAnalyzer(), order=11)
    reg.register(
        "narrative_propagation", NarrativePropagationAnalyzer(), order=12
    )
    reg.register("narrative_temporal", NarrativeTemporalAnalyzer(), order=13)
    reg.register("source", SourceAttributionAnalyzer(), order=14)

    return reg


# =========================================================
# DEFAULT REGISTRY SINGLETON  (GPU-5, v13/v14 audit)
# =========================================================
# Each call to ``build_default_registry()`` constructs ~14 analyzer
# objects (some of which load lexicon files / spaCy components /
# Pydantic configs). Previously every ``TruthLensPipeline.__init__``
# rebuilt the registry from scratch, which dominated cold-start time
# for short-lived processes (e.g., per-request workers, the API's
# fallback path). Mirror the ``get_default_pipeline()`` pattern from
# ``src/graph/graph_pipeline.py`` (G-R1) and cache a single
# process-wide instance.

_DEFAULT_REGISTRY: Optional["AnalyzerRegistry"] = None


def get_default_registry() -> "AnalyzerRegistry":
    """Return a process-wide ``AnalyzerRegistry`` singleton.

    Lazy-built on first call, then memoised. Callers that need a
    fresh, isolated registry (e.g., tests that mutate it) should
    keep calling :func:`build_default_registry` directly.
    """
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = build_default_registry()
    return _DEFAULT_REGISTRY