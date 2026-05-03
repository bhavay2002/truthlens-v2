from __future__ import annotations

import importlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features import runtime_config

logger = logging.getLogger(__name__)


# =========================================================
# HELPERS
# =========================================================

def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _sanitize_key(key: str) -> str:
    """Ensure consistent feature naming."""
    key = key.lower()
    key = re.sub(r"[^a-z0-9_]", "_", key)
    key = re.sub(r"_+", "_", key)
    return key.strip("_")


# =========================================================
# Adapter flattening budgets — audit §3.5 + §5.4
# =========================================================
# The previous ``_numeric_output`` recursively flattened *any* dict
# without bounds. If an analyser ever returned a deeply nested debug
# blob, the adapter expanded it into hundreds of new feature columns
# (which the schema then silently dropped). Those bound-less
# expansions were also wasted JSON / dict allocation work.
#
# Three caps:
#   * MAX_DEPTH (§3.5)        — refuse to descend past 3 nesting levels.
#   * MAX_FANOUT (§3.5)       — at any single level, flatten at most 32
#                               keys from the same dict.
#   * MAX_FEATURES_PER_ADAPTER (§5.4) — hard ceiling on total numeric
#                               features extracted from one analyser
#                               call; overflow is logged once.

MAX_DEPTH = 3
MAX_FANOUT = 32
MAX_FEATURES_PER_ADAPTER = 64


def _numeric_output(prefix: str, raw: Dict[str, Any]) -> Dict[str, float]:
    """
    Flatten analyzer outputs into numeric features.

    Bounded by ``MAX_DEPTH``, ``MAX_FANOUT`` and
    ``MAX_FEATURES_PER_ADAPTER`` (audit §3.5 + §5.4).
    """

    output: Dict[str, float] = {}
    truncated = False

    def flatten(key: str, value: Any, depth: int):
        nonlocal truncated

        if len(output) >= MAX_FEATURES_PER_ADAPTER:
            truncated = True
            return

        key = _sanitize_key(key)
        full_key = f"{prefix}{key}"

        if _is_number(value):
            output[full_key] = float(value)
            return

        if isinstance(value, dict):
            if depth >= MAX_DEPTH:
                truncated = True
                return
            for sub_k, sub_v in list(value.items())[:MAX_FANOUT]:
                if len(output) >= MAX_FEATURES_PER_ADAPTER:
                    truncated = True
                    return
                flatten(f"{key}_{sub_k}", sub_v, depth + 1)
            if len(value) > MAX_FANOUT:
                truncated = True
            return

        if isinstance(value, (list, tuple, set)):
            output[f"{full_key}_count"] = float(len(value))

    for k, v in list(raw.items())[:MAX_FANOUT]:
        if len(output) >= MAX_FEATURES_PER_ADAPTER:
            truncated = True
            break
        flatten(k, v, depth=1)

    if len(raw) > MAX_FANOUT:
        truncated = True

    if truncated:
        # Log once per adapter call rather than once per dropped key.
        logger.debug(
            "Analyzer flatten truncated under prefix=%r "
            "(MAX_DEPTH=%d MAX_FANOUT=%d MAX_FEATURES=%d)",
            prefix, MAX_DEPTH, MAX_FANOUT, MAX_FEATURES_PER_ADAPTER,
        )

    return output


# =========================================================
# BASE ADAPTER
# =========================================================

@dataclass
class _BaseAnalysisFeature(BaseFeature):

    group: str = "analysis"

    module_path: str = ""
    analyzer_class: str = ""
    key_prefix: str = ""
    cache_key: str = ""

    _analyzer: Any = field(default=None, init=False, repr=False)
    _load_failed: bool = field(default=False, init=False, repr=False)

    # -----------------------------------------------------

    def initialize(self) -> None:

        if self._analyzer is not None or self._load_failed:
            return

        try:
            module = importlib.import_module(self.module_path)
            analyzer_type = getattr(module, self.analyzer_class)
            self._analyzer = analyzer_type()

        except Exception as exc:
            self._load_failed = True
            logger.exception("Analyzer init failed for %s: %s", self.name, exc)
            # Audit fix §8 — when the operator opts in via the strict
            # flag, propagate so the failure is visible at boot or
            # CI time instead of being silently swallowed and the
            # extractor returning ``{}`` forever.
            if runtime_config.analysis_adapters_strict():
                raise

    # -----------------------------------------------------

    def _analyze(self, context: FeatureContext) -> Dict[str, Any]:

        if self._analyzer is None:
            return {}

        # 🔥 FIX: support both interfaces
        try:
            return self._analyzer.analyze(context)
        except TypeError:
            return self._analyzer.analyze(context.text)

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        if not isinstance(context.text, str) or not context.text.strip():
            return {}

        # -------------------------------------------------
        # CACHE HIT
        # -------------------------------------------------
        cached = context.get_cache(self.cache_key)
        if cached is not None:
            return _numeric_output(self.key_prefix, cached)

        if self._analyzer is None:
            self.initialize()

        if self._analyzer is None:
            return {}

        # -------------------------------------------------
        # EXECUTION
        # -------------------------------------------------
        try:
            raw = self._analyze(context)
        except Exception as exc:
            logger.exception("Runtime failure in %s: %s", self.name, exc)
            return {}

        if not isinstance(raw, dict):
            return {}

        # -------------------------------------------------
        # CACHE STORE
        # -------------------------------------------------
        context.set_cache(self.cache_key, raw)

        return _numeric_output(self.key_prefix, raw)


# =========================================================
# FEATURE ADAPTERS (UNCHANGED LOGIC, CLEAN PREFIXES)
# =========================================================

@dataclass
@register_feature
class AnalysisArgumentMiningFeature(_BaseAnalysisFeature):
    name: str = "analysis_argument_mining_feature"
    module_path: str = "src.analysis.argument_mining"
    analyzer_class: str = "ArgumentMiningAnalyzer"
    key_prefix: str = "analysis_argument_"
    cache_key: str = "analysis_argument"


@dataclass
@register_feature
class AnalysisDiscourseCoherenceFeature(_BaseAnalysisFeature):
    name: str = "analysis_discourse_coherence_feature"
    module_path: str = "src.analysis.discourse_coherence_analyzer"
    analyzer_class: str = "DiscourseCoherenceAnalyzer"
    key_prefix: str = "analysis_discourse_"
    cache_key: str = "analysis_discourse"


@dataclass
@register_feature
class AnalysisFramingFeature(_BaseAnalysisFeature):
    name: str = "analysis_framing_feature"
    module_path: str = "src.analysis.framing_analysis"
    analyzer_class: str = "FramingAnalyzer"
    key_prefix: str = "analysis_framing_"
    cache_key: str = "analysis_framing"


@dataclass
@register_feature
class AnalysisIdeologicalLanguageFeature(_BaseAnalysisFeature):
    name: str = "analysis_ideological_language_feature"
    module_path: str = "src.analysis.ideological_language_detector"
    analyzer_class: str = "IdeologicalLanguageDetector"
    key_prefix: str = "analysis_ideological_"
    cache_key: str = "analysis_ideological"