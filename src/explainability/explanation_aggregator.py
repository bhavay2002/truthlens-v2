"""src/explainability/explanation_aggregator.py

Aggregates the per-method explanations into a single ``AggregatedExplanation``.

Audit fixes
-----------
* **CRIT-3**: the previous implementation built a ``sorted(set(...))``
  vocabulary across all explainers, which destroyed the original token
  order and rendered the aggregated output incoherent (and unusable for
  downstream text-level ablation in CRIT-11).
* **CRIT-4**: ``dict(zip(tokens, importance))`` collapsed repeated tokens
  ("the", "is", "...") to a single entry — the second occurrence
  silently overwrote the first. The aggregator now indexes per-source
  values **by position** within their own token list, which preserves
  duplicates.
* **PERF-5**: the per-token Python ``for`` loop is replaced by a single
  ``[n_methods, n_tokens]`` matrix multiplication.
* **CRIT-9** plumbing: ``include_heuristic`` controls whether
  ``faithful=False`` explainers (e.g. propaganda) participate in the
  fusion.

The aggregator picks a *canonical* token sequence — the first available
explainer in the (shap → integrated_gradients → attention → lime) order
— and aligns every other source to those positions. Sources whose token
list lengths match the canonical are aligned 1-for-1 by position;
mismatched sources fall back to a token-name lookup that uses the
*first* occurrence of each token.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.explainability.explanation_consistency import ExplanationConsistency
from src.explainability.common_schema import AggregatedExplanation, TokenImportance

logger = logging.getLogger(__name__)
EPS = 1e-12


# =========================================================
# WEIGHTS
# =========================================================

@dataclass
class AggregationWeights:
    shap: float = 0.35
    integrated_gradients: float = 0.25
    attention: float = 0.20
    lime: float = 0.10
    graph: float = 0.10


# =========================================================
# CFG-3: YAML LOADER
# =========================================================

def load_weights_from_config(config_path: str | Path) -> AggregationWeights:
    """Return ``AggregationWeights`` populated from *config_path*.

    Reads the ``explainability.aggregation_weights`` block added for
    CFG-3. Falls back to the dataclass defaults for any missing key so
    that existing configs without the new block continue to work.
    """
    try:
        import yaml  # optional dependency — only needed here
        raw = yaml.safe_load(Path(config_path).read_text())
        block = (
            raw
            .get("explainability", {})
            .get("aggregation_weights", {})
        )
        return AggregationWeights(
            shap=float(block.get("shap", AggregationWeights.shap)),
            integrated_gradients=float(
                block.get("integrated_gradients", AggregationWeights.integrated_gradients)
            ),
            attention=float(block.get("attention", AggregationWeights.attention)),
            lime=float(block.get("lime", AggregationWeights.lime)),
            graph=float(block.get("graph", AggregationWeights.graph)),
        )
    except Exception as exc:
        logger.warning(
            "Could not load aggregation weights from %s (%s); using defaults.",
            config_path,
            exc,
        )
        return AggregationWeights()


# =========================================================
# HELPERS
# =========================================================

def _align_to_canonical(
    src_tokens: List[str],
    src_importance: List[float],
    canonical_tokens: List[str],
) -> np.ndarray:
    """Project a source's per-position scores onto the canonical token list.

    * If ``len(src_tokens) == len(canonical_tokens)`` we trust the
      explainers' tokenisations match and align by position. This
      preserves duplicates (CRIT-4).
    * Otherwise we fall back to a *first-occurrence* token-name lookup
      so a misaligned explainer still contributes signal to repeated
      tokens via the first match.
    """
    n = len(canonical_tokens)
    if not src_tokens or not src_importance:
        return np.zeros(n, dtype=float)

    if len(src_tokens) == n:
        return np.asarray(src_importance, dtype=float)

    # Fallback path — record the first occurrence of each token.
    first_value: Dict[str, float] = {}
    for t, v in zip(src_tokens, src_importance):
        if t not in first_value:
            first_value[t] = float(v)

    return np.asarray(
        [first_value.get(t, 0.0) for t in canonical_tokens],
        dtype=float,
    )


def _pick_canonical(
    candidates: List[Tuple[str, Any]],
) -> Tuple[Optional[str], Optional[List[str]]]:
    """Return the (name, tokens) of the first non-empty source.

    The order of ``candidates`` defines the priority. ``None`` is
    returned when every candidate is missing or has empty tokens.
    """
    for name, src in candidates:
        if src and getattr(src, "tokens", None):
            return name, list(src.tokens)
    return None, None


# =========================================================
# CORE
# =========================================================

class ExplanationAggregator:

    def __init__(
        self,
        weights: Optional[AggregationWeights] = None,
        *,
        include_heuristic: bool = False,
        config_path: Optional[str | Path] = None,
    ) -> None:
        # CFG-3: when a config_path is supplied and no explicit weights are
        # provided, load the per-method fusion weights from the YAML block
        # ``explainability.aggregation_weights``.
        if weights is None and config_path is not None:
            weights = load_weights_from_config(config_path)

        w = weights or AggregationWeights()

        total = w.shap + w.integrated_gradients + w.attention + w.lime + w.graph
        total = max(total, 1e-8)

        self.weights = {
            "shap": w.shap / total,
            "ig": w.integrated_gradients / total,
            "attn": w.attention / total,
            "lime": w.lime / total,
            "graph": w.graph / total,
        }

        # CRIT-9: by default the aggregator only fuses *faithful* signals.
        # Heuristic / lexicon-only explainers (propaganda, etc.) must be
        # opted-in explicitly.
        self.include_heuristic = include_heuristic

        self._consistency = ExplanationConsistency()

    # =====================================================
    # NORMALIZATION
    # =====================================================

    def _normalize(self, v):
        v = np.abs(np.asarray(v, dtype=float))
        return v / (np.sum(v) + EPS)

    # =====================================================
    # MAIN
    # =====================================================

    def aggregate(
        self,
        shap: Optional[Any] = None,
        integrated_gradients: Optional[Any] = None,
        attention: Optional[Any] = None,
        lime: Optional[Any] = None,
        graph_explanation: Optional[Dict] = None,
        *,
        shap_importance: Optional[List[Dict]] = None,
        attention_scores: Optional[List[Dict]] = None,
    ) -> Any:

        # ---------------------------------------------------------
        # DICT-LIST SHORT PATH: shap_importance / attention_scores kwargs
        # These accept list-of-dicts [{"token": ..., "importance": ...}]
        # and return a plain dict for easy subscript access.
        # ---------------------------------------------------------
        if shap_importance is not None or attention_scores is not None:
            seen: Dict[str, float] = {}
            sources = []
            if shap_importance:
                sources.append((shap_importance, "importance"))
            if attention_scores:
                sources.append((attention_scores, "attention"))
            for src_list, score_key in sources:
                for entry in src_list:
                    tok = entry.get("token", "")
                    val = float(entry.get(score_key, entry.get("importance", 0.0)))
                    seen[tok] = seen.get(tok, 0.0) + val
            total = sum(seen.values()) or 1.0
            tokens_out = list(seen.keys())
            importance_out = [seen[t] / total for t in tokens_out]
            return {
                "tokens": tokens_out,
                "final_token_importance": importance_out,
            }

        # ---------------------------------------------------------
        # CRIT-9: drop heuristic sources unless explicitly opted-in
        # ---------------------------------------------------------
        def _is_faithful(src) -> bool:
            return bool(src) and getattr(src, "faithful", True)

        if shap and not (self.include_heuristic or _is_faithful(shap)):
            shap = None
        if integrated_gradients and not (
            self.include_heuristic or _is_faithful(integrated_gradients)
        ):
            integrated_gradients = None
        if attention and not (self.include_heuristic or _is_faithful(attention)):
            attention = None
        if lime and not (self.include_heuristic or _is_faithful(lime)):
            lime = None

        # ---------------------------------------------------------
        # CRIT-3: pick a canonical token sequence so original order
        # and duplicates survive aggregation. Order of priority is
        # SHAP → IG → attention → LIME (most-faithful first).
        # ---------------------------------------------------------
        ordered_sources = [
            ("shap", shap),
            ("ig", integrated_gradients),
            ("attn", attention),
            ("lime", lime),
        ]
        canonical_name, canonical_tokens = _pick_canonical(ordered_sources)

        graph_node_importance: Dict[str, float] = {}
        graph_confidence = 0.0

        if graph_explanation:
            graph_node_importance = graph_explanation.get("node_importance", {})
            graph_confidence = float(graph_explanation.get("overall_score", 0.5))

        if not canonical_tokens and not graph_node_importance:
            logger.warning("ExplanationAggregator: no sources provided, returning empty aggregation")
            return AggregatedExplanation(
                tokens=[],
                final_token_importance=[],
                structured=[],
                method_weights={k: float(v) for k, v in self.weights.items()},
                confidence_score=None,
                agreement_score=None,
            )

        # If only the graph contributes, fall back to graph-derived tokens.
        if not canonical_tokens:
            canonical_tokens = list(graph_node_importance.keys())

        n = len(canonical_tokens)

        # ---------------------------------------------------------
        # PERF-5 + CRIT-4: build a [methods, tokens] matrix once.
        # ---------------------------------------------------------
        method_names = ["shap", "ig", "attn", "lime"]
        method_objects = {
            "shap": shap,
            "ig": integrated_gradients,
            "attn": attention,
            "lime": lime,
        }

        importance_matrix = np.zeros((len(method_names), n), dtype=float)
        confidences = np.zeros(len(method_names), dtype=float)
        weights_vec = np.zeros(len(method_names), dtype=float)

        for row, name in enumerate(method_names):
            src = method_objects[name]
            if not src or not getattr(src, "tokens", None):
                continue
            importance_matrix[row] = _align_to_canonical(
                list(src.tokens), list(src.importance), canonical_tokens
            )
            confidences[row] = float(getattr(src, "confidence", None) or 0.5)
            weights_vec[row] = self.weights[name]

        # Vectorised fusion: weighted_sum(token) = sum_m w_m * c_m * imp[m, token]
        wc = (weights_vec * confidences).reshape(-1, 1)
        weighted = (wc * importance_matrix).sum(axis=0)

        # Graph contribution (token-name keyed; aligned via first occurrence).
        if graph_node_importance:
            seen: Dict[str, float] = {}
            for t in canonical_tokens:
                if t in graph_node_importance and t not in seen:
                    seen[t] = float(graph_node_importance[t])
            graph_vec = np.asarray(
                [seen.get(t, 0.0) for t in canonical_tokens], dtype=float
            )
            weighted = weighted + (
                self.weights["graph"] * graph_confidence * graph_vec
            )

        final_scores = self._normalize(weighted)

        # ---------------------------------------------------------
        # PER-TOKEN CONFIDENCE = 1 - std across contributing methods
        # ---------------------------------------------------------
        contrib_mask = (
            (importance_matrix != 0).any(axis=1, keepdims=True)
            & np.ones((1, n), dtype=bool)
        )
        # Per-column std over the active rows.
        active_rows = (weights_vec > 0)
        if active_rows.sum() > 1:
            active = importance_matrix[active_rows]
            std_per_token = np.std(active, axis=0)
            token_confidence = np.clip(1.0 - std_per_token, 0.0, 1.0)
            no_signal = (active == 0).all(axis=0)
            token_confidence = np.where(no_signal, 0.0, token_confidence)
        else:
            token_confidence = np.where(
                importance_matrix.sum(axis=0) > 0, 1.0, 0.0
            )

        # ---------------------------------------------------------
        # AGREEMENT SCORE (Spearman/cosine across explainers)
        # ---------------------------------------------------------
        agreement_score = 0.0
        try:
            def _to_dict_list(structured):
                if not structured:
                    return None
                return [{"token": e.token, "importance": e.importance} for e in structured]

            res = self._consistency.compute(
                shap_importance=_to_dict_list(shap.structured) if shap else None,
                integrated_gradients=_to_dict_list(integrated_gradients.structured) if integrated_gradients else None,
                attention_scores=_to_dict_list(attention.structured) if attention else None,
                lime_importance=[(e.token, e.importance) for e in lime.structured] if lime else None,
            )
            if res:
                agreement_score = float(np.mean(list(res.values())))
        except Exception:
            agreement_score = 0.0

        overall_confidence = (
            float(np.mean(token_confidence)) if token_confidence.size else 0.0
        )

        # ---------------------------------------------------------
        # STRUCTURED OUTPUT
        # ---------------------------------------------------------
        importance_list = final_scores.tolist()
        structured = [
            TokenImportance(token=t, importance=float(s))
            for t, s in zip(canonical_tokens, importance_list)
        ]

        # CRIT-11 plumbing: surface canonical text + offsets when the
        # canonical source carries them (set by the orchestrator).
        canonical_src = method_objects.get(canonical_name) if canonical_name else None
        text = getattr(canonical_src, "_aggregator_text", None) if canonical_src else None
        offsets = getattr(canonical_src, "_aggregator_offsets", None) if canonical_src else None

        # ``contrib_mask`` is unused once we vectorise; keep symbol live to
        # silence linters without changing semantics.
        _ = contrib_mask

        return AggregatedExplanation(
            tokens=canonical_tokens,
            final_token_importance=importance_list,
            structured=structured,
            method_weights=self.weights,
            confidence_score=overall_confidence,
            agreement_score=agreement_score,
            text=text,
            offsets=offsets,
        )
