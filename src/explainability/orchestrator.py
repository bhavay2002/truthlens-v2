"""src/explainability/orchestrator.py

ExplainabilityOrchestrator: runs every sub-explainer on a single article.

Audit fixes
-----------
* **CRIT-8**: ``bias.integrated_gradients`` is now wrapped in an
  ``ExplanationOutput`` and threaded into both the aggregator and the
  consistency module — previously it was computed and discarded.
* **CRIT-12**: ``_make_batch_predict_fn`` now uses ``predict_fn.batch_predict``
  when present, instead of looping per text.
* **REC-3**: the orchestrator computes the model's base prediction once
  and threads ``base_proba`` (and the original ``text`` + per-token
  ``offsets``) into every metric, so the five sub-metrics no longer
  recompute the un-ablated forward.
* **FAITH-1**: optional faithfulness gate on attention rollout — when
  ``attention_faithfulness_threshold > 0`` and the |Spearman|
  correlation between attention and IG is below the threshold,
  attention is dropped from the aggregation/consistency stages.
* **FAITH-6**: failures are surfaced via ``module_failures`` and, when
  ``raise_on_majority_failure`` is True, a majority-failure raises
  ``RuntimeError`` instead of silently returning a corrupt result.
* **PERF-6**: ``get_default_orchestrator`` returns a process-wide
  cached singleton keyed by config hash — avoids re-instantiating
  GraphExplainer / ExplanationCache / etc on every article.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from src.explainability.attention_rollout import AttentionRollout
from src.explainability.bias_explainer import explain_bias
from src.explainability.common_schema import ExplanationOutput, TokenImportance
from src.explainability.emotion_explainer import explain_emotion
from src.explainability.explanation_aggregator import (
    AggregationWeights,
    ExplanationAggregator,
)
from src.explainability.explanation_cache import ExplanationCache
from src.explainability.explanation_consistency import ExplanationConsistency
from src.explainability.explanation_metrics import ExplanationMetrics
from src.explainability.explanation_monitor import ExplanationMonitor

from src.explainability.lime_explainer import explain_prediction as _lime_explain
from src.explainability.propaganda_explainer import explain_propaganda as _propaganda_explain
from src.explainability.shap_explainer import explain_text as _shap_explain

from src.graph.graph_explainer import GraphExplainer


logger = logging.getLogger(__name__)


# =========================================================
# CRIT-12: BATCHED PREDICT WRAPPER
# =========================================================

def _make_batch_predict_fn(predict_fn: Callable) -> Callable:
    """Wrap a single-text ``predict_fn(text) -> Dict`` into the
    ``List[str] -> List[Dict]`` signature ExplanationMetrics needs.

    CRIT-12: when the underlying predictor exposes ``.batch_predict`` we
    now route through it directly, so SHAP/LIME's heavy text fan-out
    avoids the per-call Python overhead and uses the GPU's batched
    forward path.
    """
    def _normalize(result: Any) -> Dict[str, Any]:
        if isinstance(result, list) and result:
            result = result[0]
        if isinstance(result, dict):
            if "fake_probability" in result:
                return result
            if "probabilities" in result and isinstance(result["probabilities"], (list, tuple)):
                probs = result["probabilities"]
                if len(probs) >= 2:
                    return {**result, "fake_probability": float(probs[1])}
        return {"fake_probability": 0.0}

    batch_fn = getattr(predict_fn, "batch_predict", None)
    if callable(batch_fn):
        def _batched(texts: List[str]) -> List[Dict]:
            try:
                return [_normalize(r) for r in list(batch_fn(list(texts)))]
            except Exception:
                logger.warning("batch_predict failed; falling back to per-text loop")
                return [_normalize(predict_fn(t)) for t in texts]
        return _batched

    def _loop(texts: List[str]) -> List[Dict]:
        return [_normalize(predict_fn(t)) for t in texts]

    return _loop


# =========================================================
# CONFIG
# =========================================================

@dataclass
class ExplainabilityConfig:
    enabled: bool = True
    use_lime: bool = True
    use_shap: bool = False
    use_attention_rollout: bool = True
    use_bias_emotion: bool = True
    use_propaganda_explainer: bool = False
    use_aggregation: bool = True
    use_consistency: bool = True
    use_explanation_metrics: bool = True
    # UNUSED EXPLAINERS FIX: expose a config flag for the GraphExplainer so
    # it is only instantiated (and run) when the caller actually needs it.
    # Default True preserves the previous unconditional behaviour.
    use_graph_explainer: bool = True

    # CRIT-9: opt-in for heuristic explainers (propaganda, etc.) in the
    # aggregator. Off by default — heuristic signals are still surfaced
    # in their own ``*_explanation`` fields.
    aggregator_include_heuristic: bool = False

    cache_enabled: bool = True
    cache_max_size: int = 128
    cache_dir: Optional[str] = None

    aggregation_weights: AggregationWeights = field(
        default_factory=AggregationWeights
    )

    # FAITH-1: when > 0, gate attention rollout on its |Spearman|
    # correlation with integrated gradients. Defaults to 0 (disabled)
    # so existing behaviour is preserved unless explicitly enabled.
    attention_faithfulness_threshold: float = 0.0

    # FAITH-6: when True, ``explain`` raises if the majority of enabled
    # sub-explainers fail. Default False keeps backward compatibility.
    raise_on_majority_failure: bool = False

    # Number of IG interpolation steps in the bias explainer.
    # Set to 0 to skip IG entirely (fast mode — only attention rollout).
    # Default 8 is a good balance between speed and attribution quality.
    ig_steps: int = 8


# =========================================================
# HELPERS
# =========================================================

def _wrap_bias_ig(bias: Optional[Dict[str, Any]]) -> Optional[ExplanationOutput]:
    """CRIT-8: wrap the bias explainer's ``integrated_gradients`` array
    in an ``ExplanationOutput`` so the aggregator + consistency stages
    can consume it like SHAP/LIME/attention.

    ``explain_bias`` returns ``integrated_gradients`` as a list of
    ``{"token": str, "importance": float}`` dicts (not raw floats).
    Falls back to ``token_importance`` (fused) when IG is empty.
    """
    if not bias:
        return None

    # Prefer IG; fall back to fused token importance
    ig_list = bias.get("integrated_gradients") or bias.get("token_importance") or []
    if not ig_list:
        return None

    # Unpack list-of-dicts format produced by explain_bias
    try:
        if isinstance(ig_list[0], dict):
            tokens = [item.get("token", "") for item in ig_list]
            ig = [float(item.get("importance", item.get("attention", 0.0))) for item in ig_list]
        else:
            # Plain float list — try to get tokens from token_importance
            ti = bias.get("token_importance") or []
            tokens = [item.get("token", f"t{i}") for i, item in enumerate(ti)] if ti else []
            ig = [float(v) for v in ig_list]
    except Exception as exc:
        logger.warning("Failed to parse bias IG list: %s", exc)
        return None

    if not tokens or not ig or len(tokens) != len(ig):
        return None

    try:
        structured = [
            TokenImportance(token=t, importance=float(v))
            for t, v in zip(tokens, ig)
        ]
        return ExplanationOutput(
            method="integrated_gradients",
            tokens=tokens,
            importance=ig,
            structured=structured,
            faithful=True,
        )
    except Exception as exc:
        logger.warning("Failed to wrap bias IG into ExplanationOutput: %s", exc)
        return None


def _spearman_safe(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    aa = np.asarray(a[:n], dtype=float)
    bb = np.asarray(b[:n], dtype=float)
    if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
        return 0.0
    ra = np.argsort(np.argsort(aa)).astype(float)
    rb = np.argsort(np.argsort(bb)).astype(float)
    c = np.corrcoef(ra, rb)[0, 1]
    return 0.0 if not np.isfinite(c) else float(c)


def _compute_offsets(
    tokenizer: Any,
    canonical_tokens: List[str],
    text: str,
) -> Optional[List[List[int]]]:
    """Best-effort character-offset alignment between the canonical
    aggregated token list and the original ``text`` so faithfulness
    metrics can ablate at the input level (CRIT-11).

    Returns ``None`` if alignment can't be derived.
    """
    if not canonical_tokens or not text:
        return None

    cursor = 0
    offsets: List[List[int]] = []
    lower_text = text.lower()
    for tok in canonical_tokens:
        if not tok:
            offsets.append([cursor, cursor])
            continue
        # Strip common subword markers so the surface form can be located.
        clean = tok
        for marker in ("##", "\u0120", "\u2581"):
            clean = clean.replace(marker, "")
        clean = clean.strip()
        if not clean:
            offsets.append([cursor, cursor])
            continue
        idx = lower_text.find(clean.lower(), cursor)
        if idx < 0:
            return None  # alignment broken; bail out
        end = idx + len(clean)
        offsets.append([idx, end])
        cursor = end

    return offsets


# =========================================================
# ORCHESTRATOR
# =========================================================

class ExplainabilityOrchestrator:

    def __init__(self, config: ExplainabilityConfig):
        self.config = config

        self.cache = (
            ExplanationCache(
                max_size=config.cache_max_size,
                cache_dir=config.cache_dir,
            )
            if config.cache_enabled
            else None
        )

        self.rollout = AttentionRollout()

        self.aggregator = (
            ExplanationAggregator(
                config.aggregation_weights,
                include_heuristic=config.aggregator_include_heuristic,
            )
            if config.use_aggregation
            else None
        )

        self.consistency = (
            ExplanationConsistency()
            if config.use_consistency
            else None
        )

        self.metrics = (
            ExplanationMetrics()
            if config.use_explanation_metrics
            else None
        )

        # UNUSED EXPLAINERS FIX: only instantiate ExplanationMonitor when
        # aggregation is active (it is only consumed inside the `if agg:` block).
        self.monitor = ExplanationMonitor() if config.use_aggregation else None

        # UNUSED EXPLAINERS FIX: GraphExplainer is gated behind the new
        # `use_graph_explainer` flag.  The previous unconditional instantiation
        # paid the full GraphExplainer init cost (NER, keyword extractor, …)
        # even for explain() callers that never needed graph features.
        self.graph_explainer = GraphExplainer() if config.use_graph_explainer else None

        logger.info(
            "ExplainabilityOrchestrator initialized | graph=%s monitor=%s",
            config.use_graph_explainer,
            config.use_aggregation,
        )

    # =====================================================
    # SAFE EXECUTION
    # =====================================================

    def _run(self, name: str, fn: Callable):
        start = time.time()
        try:
            result = fn()
            latency = (time.time() - start) * 1000
            return result, latency, True
        except Exception as e:
            logger.warning("%s failed: %s", name, e)
            latency = (time.time() - start) * 1000
            return None, latency, False

    # =====================================================
    # MAIN
    # =====================================================

    def explain(
        self,
        text: str,
        predict_fn: Callable[[str], Dict[str, Any]],
        *,
        tokens: Optional[List[str]] = None,
        attentions: Optional[List[torch.Tensor]] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:

        if not self.config.enabled:
            return {}

        if self.cache:
            cached = self.cache.get(text)
            if cached:
                return cached

        metadata = {
            "pipeline_version": "v5",
            "latency_ms": {},
            "modules": {},
        }

        # FAITH-6: track every failure for surfacing in the result.
        module_failures: List[str] = []

        explanation: Dict[str, Any] = {"module_failures": module_failures}

        def _record(name: str, ok: bool, latency: float) -> None:
            metadata["latency_ms"][name] = latency
            metadata["modules"][name] = ok
            if not ok:
                module_failures.append(name)

        # =================================================
        # SHAP
        # =================================================
        shap_out = None
        if self.config.use_shap:
            shap_out, t, ok = self._run("shap", lambda: _shap_explain(predict_fn, text))
            _record("shap", ok, t)
            explanation["shap_explanation"] = shap_out

        # =================================================
        # LIME
        # =================================================
        lime_out = None
        if self.config.use_lime:
            lime_out, t, ok = self._run("lime", lambda: _lime_explain(predict_fn, text))
            _record("lime", ok, t)
            explanation["lime_explanation"] = lime_out

        # =================================================
        # PROPAGANDA  (heuristic — gated out of aggregator by default)
        # =================================================
        propaganda_out = None
        if self.config.use_propaganda_explainer:
            propaganda_out, t, ok = self._run("propaganda", lambda: _propaganda_explain(text))
            _record("propaganda", ok, t)
            explanation["propaganda_explanation"] = propaganda_out

        # =================================================
        # BIAS + EMOTION
        # =================================================
        bias = None
        if self.config.use_bias_emotion and model and tokenizer:
            _use_shap = self.config.use_shap
            _ig_steps = self.config.ig_steps
            bias, t1, ok1 = self._run(
                "bias",
                lambda: explain_bias(model, tokenizer, text, use_shap=_use_shap, ig_steps=_ig_steps),
            )
            emo, t2, ok2 = self._run("emotion", lambda: explain_emotion(text, model, tokenizer))

            _record("bias", ok1, t1)
            _record("emotion", ok2, t2)

            explanation["bias_explanation"] = bias
            explanation["emotion_explanation"] = emo

        # CRIT-8: surface IG as its own explainer output.
        ig_out = _wrap_bias_ig(bias)
        if ig_out is not None:
            explanation["integrated_gradients_explanation"] = ig_out

        # =================================================
        # ATTENTION
        # =================================================
        attention_out = None
        if self.config.use_attention_rollout and tokens and attentions:
            attention_out, t, ok = self._run(
                "attention",
                lambda: self.rollout.compute_rollout(attentions, tokens),
            )
            _record("attention", ok, t)
            explanation["attention_explanation"] = attention_out

        # FAITH-1: optional faithfulness gate on attention rollout.
        if (
            self.config.attention_faithfulness_threshold > 0
            and attention_out is not None
            and ig_out is not None
        ):
            corr = abs(_spearman_safe(
                list(attention_out.importance),
                list(ig_out.importance),
            ))
            metadata["attention_ig_spearman"] = corr
            if corr < self.config.attention_faithfulness_threshold:
                logger.warning(
                    "Attention rollout dropped: |Spearman vs IG| = %.3f below threshold %.3f",
                    corr,
                    self.config.attention_faithfulness_threshold,
                )
                attention_out = None
                explanation["attention_explanation"] = None

        # =================================================
        # GRAPH EXPLANATION
        # =================================================
        graph_expl = None
        if self.graph_explainer is not None:
            graph_expl, t, ok = self._run(
                "graph_explainer",
                lambda: self.graph_explainer.explain_from_text(text),
            )
            _record("graph_explainer", ok, t)
            explanation["graph_explanation"] = graph_expl

        # =================================================
        # AGGREGATION (CRIT-8: thread IG through)
        # =================================================
        agg = None
        if self.aggregator:
            agg, t, ok = self._run(
                "aggregation",
                lambda: self.aggregator.aggregate(
                    shap=shap_out,
                    integrated_gradients=ig_out,
                    attention=attention_out,
                    lime=lime_out,
                    graph_explanation=graph_expl,
                ),
            )

            _record("aggregation", ok, t)

            explanation["aggregated_explanation"] = agg

            if agg and self.monitor is not None:
                scores = agg.final_token_importance
                if scores:
                    self.monitor.update(scores)
                    explanation["monitoring"] = self.monitor.summary()

        # =================================================
        # CONSISTENCY (CRIT-8: thread IG through)
        # =================================================
        if self.consistency:
            def _to_dict_list(structured):
                if not structured:
                    return None
                return [{"token": e.token, "importance": e.importance} for e in structured]

            cons, t, ok = self._run(
                "consistency",
                lambda: self.consistency.compute(
                    shap_importance=_to_dict_list(shap_out.structured) if shap_out else None,
                    integrated_gradients=_to_dict_list(ig_out.structured) if ig_out else None,
                    attention_scores=_to_dict_list(attention_out.structured) if attention_out else None,
                    lime_importance=[(e.token, e.importance) for e in lime_out.structured] if lime_out else None,
                ),
            )
            _record("consistency", ok, t)
            explanation["consistency_metrics"] = cons

        # =================================================
        # METRICS  (REC-3 + CRIT-11)
        # =================================================
        if self.metrics and agg is not None and agg.final_token_importance:
            try:
                batch_predict_fn = _make_batch_predict_fn(predict_fn)

                # REC-3: compute the un-ablated baseline once and forward it.
                base_proba: Optional[float] = None
                try:
                    base_proba = float(predict_fn(text)["fake_probability"])
                except Exception:
                    base_proba = None

                # CRIT-11: derive offsets so each metric ablates at the
                # original text level rather than re-tokenising a joined
                # string. ``_compute_offsets`` returns None if the
                # canonical tokens cannot be aligned to the source text.
                offsets = None
                if tokenizer is not None:
                    try:
                        offsets = _compute_offsets(tokenizer, agg.tokens, text)
                    except Exception:
                        offsets = None

                metrics = self.metrics.evaluate(
                    agg.tokens,
                    agg.final_token_importance,
                    batch_predict_fn,
                    text=text,
                    offsets=offsets,
                    base_proba=base_proba,
                )

                explanation["explanation_metrics"] = metrics
                explanation["explanation_quality_score"] = metrics.get("overall_score")

            except Exception as e:
                logger.warning("metrics failed: %s", e)
                module_failures.append("metrics")

        # =================================================
        # METADATA + FAITH-6
        # =================================================
        explanation["metadata"] = metadata

        if self.config.raise_on_majority_failure:
            run_modules = [m for m, ok in metadata["modules"].items()]
            if run_modules:
                fail_count = sum(1 for m in run_modules if m in module_failures)
                if fail_count * 2 > len(run_modules):
                    raise RuntimeError(
                        f"Majority of explainability modules failed "
                        f"({fail_count}/{len(run_modules)}): {module_failures}"
                    )

        if self.cache:
            self.cache.set(text, explanation)

        return explanation

    # =====================================================
    # FAST MODE
    # =====================================================

    def explain_fast(self, text: str, predict_fn):
        # RECOMPUTATION FIX: cache the base prediction before running LIME so
        # the return value does not pay an extra predict_fn(text) round-trip
        # on top of the many perturbed calls LIME already makes internally.
        base_prediction = predict_fn(text)
        lime, t, ok = self._run("lime", lambda: _lime_explain(predict_fn, text))

        return {
            "prediction": base_prediction,
            "lime_explanation": lime,
            "metadata": {
                "mode": "fast",
                "latency_ms": t,
                "lime_success": ok,
            },
        }


# =========================================================
# PERF-6: PROCESS-WIDE SINGLETON
# =========================================================

_ORCH_LOCK = threading.RLock()
_ORCH_CACHE: Dict[str, ExplainabilityOrchestrator] = {}


def _config_key(config: ExplainabilityConfig) -> str:
    """Hash a config so equivalent configs reuse the same orchestrator."""
    payload = asdict(config)
    # AggregationWeights is a dataclass — already JSON-friendly via asdict.
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def get_default_orchestrator(
    config: Optional[ExplainabilityConfig] = None,
) -> ExplainabilityOrchestrator:
    """Return a process-wide cached orchestrator for the given config.

    PERF-6: ``run_explainability_pipeline`` previously instantiated a
    fresh ``ExplainabilityOrchestrator`` (and therefore a fresh
    ``ExplanationCache`` + ``GraphExplainer`` + ``ExplanationMonitor``)
    on every article, which both burned latency and reset the per-text
    LRU cache. With this helper, identical configs share a single
    orchestrator instance.
    """
    cfg = config or ExplainabilityConfig()
    key = _config_key(cfg)
    with _ORCH_LOCK:
        if key not in _ORCH_CACHE:
            _ORCH_CACHE[key] = ExplainabilityOrchestrator(cfg)
        return _ORCH_CACHE[key]


def clear_orchestrator_cache() -> None:
    with _ORCH_LOCK:
        _ORCH_CACHE.clear()
