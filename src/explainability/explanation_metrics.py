"""src/explainability/explanation_metrics.py

Faithfulness / comprehensiveness / sufficiency / deletion / insertion
metrics computed against a model.

Audit fixes
-----------
* **CRIT-11**: previously every metric ablated tokens by ``" ".join(...)``-ing
  the per-explainer token list. After tokenisation by SHAP/LIME the
  "tokens" frequently include sub-word artefacts (``Ġthe``, ``##ing``,
  punctuation) and the joined string no longer matches the original
  text the model was trained on. Each metric now accepts an optional
  ``text`` + ``offsets`` pair; when provided, ablation happens at the
  *input text* level by replacing the character span ``text[start:end]``
  with a configurable ``mask_string`` (default: empty). The legacy
  ``" ".join(tokens)`` path is preserved as a fallback for callers that
  cannot supply offsets.
* **REC-3**: the base prediction (un-ablated text) is computed many times
  per article. ``evaluate`` now accepts a precomputed ``base_proba`` so
  the orchestrator can thread its single forward through every metric.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.explainability.utils_validation import validate_tokens_scores

logger = logging.getLogger(__name__)
EPS = 1e-12

PredictionFn = Callable[[List[str]], List[Dict[str, float]]]


class ExplanationMetrics:

    def __init__(self) -> None:
        logger.info("ExplanationMetrics initialized")

    # =====================================================
    # UTILS
    # =====================================================

    @staticmethod
    def _extract_fake_prob_batch(results):
        return np.array([r["fake_probability"] for r in results], dtype=float)

    @staticmethod
    def _sort_indices(scores):
        return list(np.argsort(np.asarray(scores))[::-1])

    @staticmethod
    def _normalize(x):
        x = np.asarray(x, dtype=float)

        if x.size == 0:
            return x

        mn, mx = np.min(x), np.max(x)
        if mx - mn < EPS:
            return np.zeros_like(x)

        return (x - mn) / (mx - mn + EPS)

    @staticmethod
    def _apply_confidence(value: float, confidence: Optional[float]) -> float:
        if confidence is None:
            return float(value)
        return float(value * np.clip(confidence, 0.0, 1.0))

    # =====================================================
    # CRIT-11: TEXT-LEVEL ABLATION HELPERS
    # =====================================================

    @staticmethod
    def _ablate_offsets(
        text: str,
        offsets: Sequence[Sequence[int]],
        mask_indices: Sequence[int],
        *,
        mask_string: str = "",
    ) -> str:
        """Ablate ``text`` by replacing the spans listed in
        ``mask_indices`` with ``mask_string``. Spans are merged + applied
        right-to-left so prior offsets stay valid.
        """
        if not mask_indices:
            return text

        spans = sorted(
            ((int(offsets[i][0]), int(offsets[i][1])) for i in mask_indices),
            key=lambda p: p[0],
            reverse=True,
        )

        out = text
        for start, end in spans:
            if start < 0 or end <= start or end > len(out):
                continue
            out = out[:start] + mask_string + out[end:]
        return out

    @classmethod
    def _build_baseline_text(
        cls,
        tokens: List[str],
        text: Optional[str],
    ) -> str:
        """Pick the text that *un-ablated* prediction is computed against.

        When the caller supplies ``text`` we always trust it (this is the
        original input the model is meant to score). Otherwise fall back
        to the legacy ``" ".join(tokens)`` joining.
        """
        if text is not None:
            return text
        return " ".join(tokens)

    # =====================================================
    # FAITHFULNESS (BATCHED)
    # =====================================================

    def faithfulness(
        self,
        tokens,
        scores,
        predict_fn,
        *,
        text: Optional[str] = None,
        offsets: Optional[Sequence[Sequence[int]]] = None,
        base_proba: Optional[float] = None,
        mask_string: str = "",
    ):

        validate_tokens_scores(tokens, scores, auto_fix=True)

        baseline_text = self._build_baseline_text(tokens, text)

        if base_proba is None:
            base = predict_fn([baseline_text])[0]["fake_probability"]
        else:
            base = float(base_proba)

        if text is not None and offsets is not None and len(offsets) == len(tokens):
            ablated = [
                self._ablate_offsets(text, offsets, [i], mask_string=mask_string)
                for i in range(len(tokens))
            ]
        else:
            ablated = [
                " ".join([t for j, t in enumerate(tokens) if j != i])
                for i in range(len(tokens))
            ]

        if not ablated:
            return 0.0

        preds = self._extract_fake_prob_batch(predict_fn(ablated))
        deltas = base - preds

        if len(deltas) < 2:
            return 0.0

        corr = np.corrcoef(scores, deltas)[0, 1]
        return 0.0 if np.isnan(corr) else float(corr)

    # =====================================================
    # COMPREHENSIVENESS
    # =====================================================

    def comprehensiveness(
        self,
        tokens,
        scores,
        predict_fn,
        top_k=5,
        *,
        text: Optional[str] = None,
        offsets: Optional[Sequence[Sequence[int]]] = None,
        base_proba: Optional[float] = None,
        mask_string: str = "",
    ):

        baseline_text = self._build_baseline_text(tokens, text)
        if base_proba is None:
            base = predict_fn([baseline_text])[0]["fake_probability"]
        else:
            base = float(base_proba)

        ranked = self._sort_indices(scores)[:top_k]

        if text is not None and offsets is not None and len(offsets) == len(tokens):
            perturbed = self._ablate_offsets(text, offsets, ranked, mask_string=mask_string)
        else:
            perturbed = " ".join([t for i, t in enumerate(tokens) if i not in ranked])
        new = predict_fn([perturbed])[0]["fake_probability"]

        return float(base - new)

    # =====================================================
    # SUFFICIENCY
    # =====================================================

    def sufficiency(
        self,
        tokens,
        scores,
        predict_fn,
        top_k=5,
        *,
        text: Optional[str] = None,
        offsets: Optional[Sequence[Sequence[int]]] = None,
        base_proba: Optional[float] = None,
        mask_string: str = "",
    ):

        baseline_text = self._build_baseline_text(tokens, text)
        if base_proba is None:
            base = predict_fn([baseline_text])[0]["fake_probability"]
        else:
            base = float(base_proba)

        ranked = self._sort_indices(scores)[:top_k]

        if text is not None and offsets is not None and len(offsets) == len(tokens):
            # Sufficiency needs the *complement* — keep only the top-k
            # spans, ablate everything else.
            keep = set(int(i) for i in ranked)
            drop = [i for i in range(len(tokens)) if i not in keep]
            kept_text = self._ablate_offsets(text, offsets, drop, mask_string=mask_string)
        else:
            kept_text = " ".join([tokens[i] for i in ranked])

        new = predict_fn([kept_text])[0]["fake_probability"]

        return float(base - new)

    # =====================================================
    # DELETION (BATCHED)
    # =====================================================

    def deletion_score(
        self,
        tokens,
        scores,
        predict_fn,
        *,
        text: Optional[str] = None,
        offsets: Optional[Sequence[Sequence[int]]] = None,
        base_proba: Optional[float] = None,
        mask_string: str = "",
    ):

        ranked = self._sort_indices(scores)

        if text is not None and offsets is not None and len(offsets) == len(tokens):
            removed: List[int] = []
            texts = []
            for idx in ranked:
                removed.append(int(idx))
                texts.append(
                    self._ablate_offsets(text, offsets, removed, mask_string=mask_string)
                )
        else:
            current = list(tokens)
            texts = []
            for idx in ranked:
                current[idx] = ""
                texts.append(" ".join([t for t in current if t]))

        if not texts:
            return 0.0
        preds = self._extract_fake_prob_batch(predict_fn(texts))

        if base_proba is not None:
            base = float(base_proba)
        else:
            base = preds[0] if len(preds) > 0 else 0.0
        return float(base - np.mean(preds))

    # =====================================================
    # INSERTION (BATCHED)
    # =====================================================

    def insertion_score(
        self,
        tokens,
        scores,
        predict_fn,
        *,
        text: Optional[str] = None,
        offsets: Optional[Sequence[Sequence[int]]] = None,
        mask_string: str = "",
        **_unused: Any,
    ):

        ranked = self._sort_indices(scores)

        if text is not None and offsets is not None and len(offsets) == len(tokens):
            # Insertion: start from a fully-ablated text and reveal the
            # ranked tokens one by one.
            kept: List[int] = []
            all_idx = set(range(len(tokens)))
            texts = []
            for idx in ranked:
                kept.append(int(idx))
                drop = sorted(all_idx - set(kept))
                texts.append(
                    self._ablate_offsets(text, offsets, drop, mask_string=mask_string)
                )
        else:
            slots = [""] * len(tokens)
            texts = []
            for idx in ranked:
                slots[idx] = tokens[idx]
                texts.append(" ".join([t for t in slots if t]))

        if not texts:
            return 0.0
        preds = self._extract_fake_prob_batch(predict_fn(texts))

        return float(np.trapz(preds))

    # =====================================================
    # VARIANCE
    # =====================================================

    def variance(self, scores: List[float]) -> float:
        arr = np.asarray(scores, dtype=float)
        if arr.size == 0:
            return 0.0
        return float(np.var(arr))

    # =====================================================
    # SINGLE EVALUATION
    # =====================================================

    def evaluate(
        self,
        tokens,
        scores,
        predict_fn,
        *,
        confidence: Optional[float] = None,
        text: Optional[str] = None,
        offsets: Optional[Sequence[Sequence[int]]] = None,
        base_proba: Optional[float] = None,
        mask_string: str = "",
    ) -> Dict[str, float]:

        validate_tokens_scores(tokens, scores, auto_fix=True)

        # Normalize scores to give flat LIME outputs proper spread before
        # computing faithfulness/comprehensiveness/sufficiency metrics.
        scores_arr = np.asarray(scores, dtype=float)
        std = float(np.std(scores_arr))
        if std > 1e-8:
            scores = ((scores_arr - scores_arr.mean()) / (std + 1e-8)).tolist()
        else:
            scores = scores_arr.tolist()

        # REC-3: compute the un-ablated baseline prediction once, here,
        # if the caller didn't already supply one. The five sub-metrics
        # below all reference the same number — passing it through as
        # ``base_proba`` collapses 5 redundant model forwards into 1.
        if base_proba is None:
            baseline_text = self._build_baseline_text(tokens, text)
            try:
                base_proba = float(predict_fn([baseline_text])[0]["fake_probability"])
            except Exception:
                base_proba = None

        common: Dict[str, Any] = dict(
            text=text,
            offsets=offsets,
            base_proba=base_proba,
            mask_string=mask_string,
        )

        raw = {
            "faithfulness": self.faithfulness(tokens, scores, predict_fn, **common),
            "comprehensiveness": self.comprehensiveness(tokens, scores, predict_fn, **common),
            "sufficiency": self.sufficiency(tokens, scores, predict_fn, **common),
            "deletion_score": self.deletion_score(tokens, scores, predict_fn, **common),
            "insertion_score": self.insertion_score(tokens, scores, predict_fn, **common),
        }

        # Sanitize raw metric values before confidence weighting.
        raw = {
            k: float(np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0))
            for k, v in raw.items()
        }

        weighted = {
            k: self._apply_confidence(v, confidence)
            for k, v in raw.items()
        }

        # SCALE-5: per-call min-max normalisation always yields mean=0.5
        # because the result spans exactly [0, 1] by construction. Instead,
        # map each confidence-weighted metric from [-1, 1] → [0, 1] via
        # (v + 1) / 2 and average. Metrics outside [-1, 1] (e.g. a large
        # insertion AUC) are clipped first so the formula stays bounded.
        values_arr = np.clip(
            np.nan_to_num(np.array(list(weighted.values()), dtype=float), nan=0.0),
            -1.0, 1.0,
        )
        norm_arr = (values_arr + 1.0) / 2.0

        result = {
            **weighted,
            "variance": self.variance(scores),
            "normalized": dict(zip(weighted.keys(), norm_arr.tolist())),
            "overall_score": float(np.mean(norm_arr)) if norm_arr.size > 0 else 0.0,
        }

        return result

    # =====================================================
    # BATCH EVALUATION
    # =====================================================

    def evaluate_batch(
        self,
        batch_tokens: List[List[str]],
        batch_scores: List[List[float]],
        predict_fn: PredictionFn,
        *,
        confidences: Optional[List[float]] = None,
    ) -> Dict[str, float]:

        results = []

        for i, (tokens, scores) in enumerate(zip(batch_tokens, batch_scores)):

            conf = confidences[i] if confidences and i < len(confidences) else None

            res = self.evaluate(
                tokens,
                scores,
                predict_fn,
                confidence=conf,
            )

            results.append(res["overall_score"])

        arr = np.asarray(results, dtype=float)

        return {
            "batch_mean": float(np.mean(arr)) if arr.size else 0.0,
            "batch_std": float(np.std(arr)) if arr.size else 0.0,
            "batch_min": float(np.min(arr)) if arr.size else 0.0,
            "batch_max": float(np.max(arr)) if arr.size else 0.0,
            "batch_size": int(arr.size),
        }
