from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import numpy as np

from src.analysis.feature_context import FeatureContext

logger = logging.getLogger(__name__)


# =========================================================
# BASE ANALYZER (PRODUCTION GRADE)
# =========================================================

def _wrap_analyze_for_backward_compat(fn):
    """Decorator that allows analyzer.analyze(text_str) as a backward-compat
    shortcut.  If the first positional argument is a plain str, it is silently
    promoted to a ``FeatureContext`` before the real implementation is invoked.
    """
    import functools

    @functools.wraps(fn)
    def _wrapper(self, ctx_or_text=None, **kwargs):
        if isinstance(ctx_or_text, str):
            ctx_or_text = FeatureContext(text=ctx_or_text)
        return fn(self, ctx_or_text, **kwargs)

    return _wrapper


class BaseAnalyzer(ABC):
    """
    Abstract base class for all analyzers.

    Guarantees:
    - consistent interface
    - safe execution
    - validated outputs
    - shared cache compatibility
    - extensibility hooks
    """

    # -----------------------------------------------------
    # CONFIG (override in subclasses)
    # -----------------------------------------------------

    name: str = "base"
    expected_keys: Optional[set[str]] = None
    use_cache: bool = True  # enable per-context caching

    def __init_subclass__(cls, **kwargs):
        """Automatically wrap the concrete ``analyze`` method so that passing a
        plain string is accepted as backward-compatible shorthand for
        ``analyze(FeatureContext(text=…))``."""
        super().__init_subclass__(**kwargs)
        if "analyze" in cls.__dict__:
            cls.analyze = _wrap_analyze_for_backward_compat(cls.__dict__["analyze"])

    # =========================================================
    # PUBLIC API
    # =========================================================

    def __call__(self, ctx: FeatureContext, **kwargs: Any) -> Dict[str, float]:
        """
        Main execution wrapper.

        Handles:
        - validation
        - caching
        - safe execution
        - output normalization

        Forwards keyword arguments to ``analyze()`` so subclasses that accept
        extra inputs (e.g. narrative analyzers expecting hero/villain entity
        lists) can be invoked through the common ``__call__`` interface.
        """

        try:
            self._validate_context(ctx)

            # -------------------------------------------------
            # CACHE CHECK
            # -------------------------------------------------
            # Caching is keyed by analyzer name + ctx only. Skip the cache
            # path when extra kwargs are provided so different argument sets
            # don't silently return the same cached output.

            use_cache = self.use_cache and not kwargs

            if use_cache:
                cached = self._get_cached(ctx)
                if cached is not None:
                    return cached

            # -------------------------------------------------
            # CORE ANALYSIS
            # -------------------------------------------------

            features = self.analyze(ctx, **kwargs) if kwargs else self.analyze(ctx)

            # -------------------------------------------------
            # POSTPROCESS
            # -------------------------------------------------

            features = self._postprocess(features)

            # -------------------------------------------------
            # VALIDATE OUTPUT
            # -------------------------------------------------

            self._validate_output(features)

            # -------------------------------------------------
            # STORE CACHE
            # -------------------------------------------------

            if use_cache:
                self._set_cache(ctx, features)

            return features

        except Exception:
            logger.exception("[Analyzer Failure] name=%s", self.name)
            return self._fallback()

    # =========================================================
    # CORE METHOD (MUST IMPLEMENT)
    # =========================================================

    @abstractmethod
    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:
        raise NotImplementedError

    def analyze_doc(self, doc: Any) -> Dict[str, float]:
        """Convenience method: run analysis on a pre-processed spaCy Doc.

        Builds a FeatureContext from the Doc and delegates to analyze().
        """
        ctx = FeatureContext.from_doc(doc)
        return self.analyze(ctx)

    # =========================================================
    # CONTEXT VALIDATION
    # =========================================================

    def _validate_context(self, ctx: FeatureContext) -> None:
        if ctx is None:
            raise ValueError("FeatureContext cannot be None")

        if not isinstance(ctx.text, str):
            raise TypeError("FeatureContext.text must be a string")

        # shared cache setup (important)
        if not hasattr(ctx, "shared") or ctx.shared is None:
            ctx.shared = {}

        if not hasattr(ctx, "cache") or ctx.cache is None:
            ctx.cache = {}

        # CRIT-A6 / F16: a number of analyzers short-circuit when
        # `ctx.n_tokens == 0`. That property is computed lazily by
        # `ensure_tokens()`. Calling it here guarantees every analyzer
        # gets a populated token view regardless of whether upstream
        # construction (FeatureContext.from_doc, batch_processor, etc.)
        # has run it. Cheap when already populated.
        ensure = getattr(ctx, "ensure_tokens", None)
        if callable(ensure):
            try:
                ensure()
            except Exception:
                logger.debug(
                    "ensure_tokens() failed for %s; analyzer will see "
                    "whatever the context already exposes", self.name,
                )

    # =========================================================
    # OUTPUT VALIDATION
    # =========================================================

    def _validate_output(self, features: Dict[str, float]) -> None:

        if not isinstance(features, dict):
            raise TypeError("Analyzer output must be dict")

        for k, v in features.items():

            if not isinstance(k, str):
                raise TypeError(f"Feature key must be str: {k}")

            if not isinstance(v, (int, float)):
                raise TypeError(f"Feature value must be numeric: {k}")

            if not np.isfinite(v):
                raise ValueError(f"Feature contains NaN/Inf: {k}")

        # Optional strict schema enforcement
        if self.expected_keys:
            missing = self.expected_keys - set(features.keys())
            if missing:
                raise ValueError(f"Missing expected keys: {missing}")

    # =========================================================
    # POSTPROCESS
    # =========================================================

    def _postprocess(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize output values.
        """
        return {k: float(v) for k, v in features.items()}

    # =========================================================
    # CACHE SYSTEM
    # =========================================================

    def _get_cache_key(self) -> str:
        return f"analyzer::{self.name}"

    def _get_cached(self, ctx: FeatureContext) -> Optional[Dict[str, float]]:
        return ctx.cache.get(self._get_cache_key())

    def _set_cache(self, ctx: FeatureContext, features: Dict[str, float]) -> None:
        ctx.cache[self._get_cache_key()] = features

    # =========================================================
    # FAILSAFE
    # =========================================================

    def _fallback(self) -> Dict[str, float]:

        if self.expected_keys:
            return {k: 0.0 for k in self.expected_keys}

        return {}

    # =========================================================
    # VECTOR CONVERSION
    # =========================================================

    def to_vector(
        self,
        features: Dict[str, float],
        schema: list[str],
    ) -> np.ndarray:

        return np.array(
            [features.get(k, 0.0) for k in schema],
            dtype=np.float32,
        )

    # =========================================================
    # BATCH SUPPORT (NEW)
    # =========================================================

    def batch(
        self,
        contexts: list[FeatureContext],
    ) -> list[Dict[str, float]]:
        """
        Batch execution helper.
        """
        return [self(ctx) for ctx in contexts]