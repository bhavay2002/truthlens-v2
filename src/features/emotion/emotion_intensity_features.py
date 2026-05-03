# src/features/emotion/emotion_intensity_features.py

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.base.numerics import EPS, MAX_CLIP, normalized_entropy
from src.features.base.tokenization import ensure_tokens_word
from src.features import runtime_config

from src.features.emotion.emotion_schema import (
    EMOTION_LABELS,
    EMOTION_TERMS,
)

logger = logging.getLogger(__name__)


# =========================================================
# TRANSFORMER SETUP — lazy (audit §5.2)
# =========================================================
# Previously the tokenizer + model were loaded at module import time
# inside a ``try/except``, which (a) blocked ``import src.features`` on
# ~300 ms of HF-tokenizer construction and ~600 ms of model download/
# weight load even for code paths that never call the emotion
# extractor, and (b) silently disabled the transformer whenever
# ``transformers`` was not installed in the test environment with no
# way to surface the failure.
#
# Lazy initialization solves both: the import is free, the first call
# pays the construction cost, and the second call onwards is amortized.
# A module-level lock makes the lazy init thread-safe under FastAPI's
# threadpool.

import threading

MODEL_NAME = os.environ.get(
    "TRUTHLENS_EMOTION_MODEL",
    "j-hartmann/emotion-english-distilroberta-base",
)

TRANSFORMER_LABELS = [
    "anger", "disgust", "fear",
    "joy", "neutral", "sadness", "surprise",
]

# Sentinel used by the lazy loader: ``None`` means "never tried", a
# truthy tuple means "ready", ``False`` means "import or weight load
# failed; do not try again this process".
_TRANSFORMER_STATE: Any = None
_TRANSFORMER_LOCK = threading.Lock()


def _lazy_load_transformer():
    """Return ``(tokenizer, model, device)`` or ``None`` on failure.

    Audit §5.2 (lazy load) + §6.1 (CUDA placement).
    """
    global _TRANSFORMER_STATE

    if _TRANSFORMER_STATE is False:
        return None
    if _TRANSFORMER_STATE is not None:
        return _TRANSFORMER_STATE

    with _TRANSFORMER_LOCK:
        if _TRANSFORMER_STATE is False:
            return None
        if _TRANSFORMER_STATE is not None:
            return _TRANSFORMER_STATE
        try:
            import torch  # noqa: WPS433
            from transformers import (  # noqa: WPS433
                AutoTokenizer,
                AutoModelForSequenceClassification,
            )
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

            # Audit §6.1 — actually move the model to CUDA when
            # available. Previously ``_model`` ran on CPU regardless of
            # GPU presence; on the standard 7-class DistilRoBERTa head
            # the GPU vs CPU difference is ~12× per batch.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device).eval()

            _TRANSFORMER_STATE = (tokenizer, model, device)
            logger.info(
                "Emotion transformer ready | model=%s device=%s",
                MODEL_NAME, device,
            )
            return _TRANSFORMER_STATE
        except Exception as exc:
            _TRANSFORMER_STATE = False
            logger.warning(
                "Emotion transformer unavailable, lexicon fallback only: %s",
                exc,
            )
            return None


def _transformer_runtime_available() -> bool:
    """Audit §6.6 — the runtime check.

    True iff:
      * the operator-facing flag ``runtime_config.transformer_enabled()``
        is on, AND
      * the lazy loader has either succeeded or has not yet been tried
        (we report ``True`` and let the actual call surface the failure
        via the empty-default path).
    """
    if not runtime_config.transformer_enabled():
        return False
    return _TRANSFORMER_STATE is not False


# =========================================================
# REVERSE LOOKUP (LEXICON)
# =========================================================

WORD_TO_EMOTION = {
    word: emotion
    for emotion, words in EMOTION_TERMS.items()
    for word in words
}


def _lexicon_emotions(tokens):
    counts = {emotion: 0 for emotion in EMOTION_LABELS}
    for token in tokens:
        emo = WORD_TO_EMOTION.get(token)
        if emo:
            counts[emo] += 1
    total_hits = sum(counts.values())
    total_tokens = len(tokens)
    return counts, total_hits, total_tokens


# =========================================================
# FEATURE EXTRACTOR
# =========================================================

@dataclass
@register_feature
class EmotionIntensityFeatures(BaseFeature):

    name: str = "emotion_intensity_features"
    group: str = "emotion"
    description: str = "Robust emotion intensity + hybrid modeling"

    # -----------------------------------------------------

    def _transformer_emotions(self, text: str) -> Dict[str, float]:
        # Single-text path delegates so the .cpu().numpy() copy and
        # softmax accounting are owned in exactly one place (audit §6.2).
        batched = self._transformer_emotions_batch([text])
        return batched[0] if batched else {e: 0.0 for e in EMOTION_LABELS}

    # -----------------------------------------------------

    def _transformer_emotions_batch(
        self, texts: List[str]
    ) -> List[Dict[str, float]]:
        """Batched HF inference with sliding-window overflow.

        Combines audit fixes:
          * §5.1 — ``stride=64, return_overflowing_tokens=True``: long
            documents are chunked at 256 tokens with 64-token stride
            and the per-window softmaxes are averaged into one
            distribution per document. This recovers signal from the
            opinion-piece tail that the previous ``max_length=512``
            truncation silently dropped.
          * §6.3 — single-sample path uses ``padding=False`` so it does
            not pay the ``B × L`` attention-mask cost for B=1.
          * §6.4 — ``torch.autocast("cuda", dtype=torch.float16)`` on
            CUDA — free 2× speedup for softmax classification.
          * §6.5 — chunks the caller batch into ``MAX_BATCH``-size
            slices to avoid the OOM explosion at large B × L.
          * §6.6 — checks the runtime flag at call time, not import
            time, so config / test toggles take effect without restart.
        """
        if not texts:
            return []

        empty_default = [
            {emotion: 0.0 for emotion in EMOTION_LABELS} for _ in texts
        ]

        if not _transformer_runtime_available():
            return empty_default

        loaded = _lazy_load_transformer()
        if loaded is None:
            return empty_default
        tokenizer, model, device = loaded

        try:
            import torch  # noqa: WPS433
        except ImportError:
            return empty_default

        max_batch = runtime_config.max_transformer_batch()
        chunk_len = runtime_config.transformer_chunk_length()
        stride = runtime_config.transformer_chunk_stride()

        results: List[Optional[Dict[str, float]]] = [None] * len(texts)

        # --- audit §6.5: chunk the caller batch ---
        for batch_start in range(0, len(texts), max_batch):
            batch_texts = texts[batch_start: batch_start + max_batch]

            # --- audit §6.3: skip padding when B == 1 ---
            tok_kwargs = dict(
                return_tensors="pt",
                truncation=True,
                max_length=chunk_len,
                return_overflowing_tokens=True,
                stride=stride,
            )
            if len(batch_texts) == 1:
                tok_kwargs["padding"] = False
            else:
                tok_kwargs["padding"] = True

            try:
                inputs = tokenizer(list(batch_texts), **tok_kwargs)
            except TypeError:
                # Older tokenizers without overflow support — fall back
                # to plain truncation at chunk_len.
                inputs = tokenizer(
                    list(batch_texts),
                    return_tensors="pt",
                    truncation=True,
                    max_length=chunk_len,
                    padding=(False if len(batch_texts) == 1 else True),
                )

            # ``overflow_to_sample_mapping`` tells us which original
            # text each window came from; without it (older HF) every
            # window maps 1:1 to its caller index.
            sample_map = inputs.pop("overflow_to_sample_mapping", None)
            if sample_map is None:
                sample_map = torch.arange(len(batch_texts))

            inputs = {
                k: v.to(device) for k, v in inputs.items()
                if isinstance(v, torch.Tensor)
            }

            # --- audit §6.1 + §6.4: inference_mode + autocast on CUDA ---
            with torch.inference_mode():
                if device == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)

                probs = torch.softmax(outputs.logits, dim=1)

            # ONE host copy per chunk-batch.
            probs_np = probs.float().cpu().numpy()
            sample_map_np = (
                sample_map.cpu().numpy()
                if hasattr(sample_map, "cpu")
                else np.asarray(sample_map)
            )

            # Average per-window softmaxes back to one row per text
            # (audit §5.1 — the "average per-window" half of the fix).
            num_labels = probs_np.shape[1]
            per_text = np.zeros(
                (len(batch_texts), num_labels), dtype=np.float32
            )
            counts = np.zeros(len(batch_texts), dtype=np.int32)
            for win_idx, src_idx in enumerate(sample_map_np):
                per_text[int(src_idx)] += probs_np[win_idx]
                counts[int(src_idx)] += 1
            counts = np.maximum(counts, 1).reshape(-1, 1)
            per_text = per_text / counts

            for j, row in enumerate(per_text):
                scores = {emotion: 0.0 for emotion in EMOTION_LABELS}
                for label, prob in zip(TRANSFORMER_LABELS, row):
                    if label in scores:
                        scores[label] = float(prob)
                results[batch_start + j] = scores

        return [
            r if r is not None else {e: 0.0 for e in EMOTION_LABELS}
            for r in results
        ]

    # -----------------------------------------------------

    def _hybrid_emotions(self, text: str, tokens):

        counts, hits, n_lex_tokens = _lexicon_emotions(tokens)

        lex_scores = (
            np.array([counts[e] for e in EMOTION_LABELS], dtype=np.float32)
            / (hits + EPS)
            if hits > 0 else np.zeros(len(EMOTION_LABELS), dtype=np.float32)
        )

        if _transformer_runtime_available():
            t_scores_dict = self._transformer_emotions(text)
            t_scores = np.array(
                [t_scores_dict[e] for e in EMOTION_LABELS],
                dtype=np.float32,
            )
        else:
            t_scores = np.zeros(len(EMOTION_LABELS), dtype=np.float32)

        alpha = 0.7 if t_scores.sum() > 0 else 0.0
        scores = alpha * t_scores + (1 - alpha) * lex_scores

        total = scores.sum()
        if total > 0:
            scores = scores / total

        return scores, hits, n_lex_tokens

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return self._empty()

        tokens = ensure_tokens_word(context, text)
        scores, hits, token_count = self._hybrid_emotions(text, tokens)

        token_count = max(token_count, 1)
        coverage = hits / token_count

        max_val = float(np.max(scores))
        mean_val = float(np.mean(scores))
        std_val = float(np.std(scores))
        range_val = float(np.max(scores) - np.min(scores))
        l2_intensity = float(np.linalg.norm(scores))
        entropy = normalized_entropy(scores)

        return {
            "emotion_intensity_max": self._safe(max_val),
            "emotion_intensity_mean": self._safe(mean_val),
            "emotion_intensity_std": self._safe(std_val),
            "emotion_intensity_range": self._safe(range_val),
            "emotion_intensity_l2": self._safe(l2_intensity),
            "emotion_intensity_entropy": self._safe(entropy),
            "emotion_coverage": self._safe(coverage),

            # Audit §11 + §6.6 — read at extract time so a config or
            # monkeypatch toggle is reflected per request.
            "emotion_transformer_available": (
                1.0 if _transformer_runtime_available() else 0.0
            ),
        }

    # -----------------------------------------------------
    # BATCH (audit fix §6.2)
    # -----------------------------------------------------

    def extract_batch(
        self,
        contexts: List[FeatureContext],
    ) -> List[Dict[str, float]]:

        if not contexts:
            return []

        texts: List[str] = []
        token_lists: List[Any] = []
        active_idx: List[int] = []

        results: List[Dict[str, float]] = [self._empty() for _ in contexts]

        for i, ctx in enumerate(contexts):
            text = (ctx.text or "").strip()
            if not text:
                continue
            # Audit §2.5 — read the cached tokens via ``ensure_tokens_word``
            # (cheap dict lookup if upstream extractor already populated
            # ``ctx.tokens_word``).
            tokens = ensure_tokens_word(ctx, text)
            token_lists.append(tokens)
            texts.append(text)
            active_idx.append(i)

        if not texts:
            return results

        if _transformer_runtime_available():
            t_batch = self._transformer_emotions_batch(texts)
        else:
            t_batch = [
                {emotion: 0.0 for emotion in EMOTION_LABELS}
                for _ in texts
            ]

        for j, dst_i in enumerate(active_idx):
            tokens = token_lists[j]
            counts, hits, n_lex_tokens = _lexicon_emotions(tokens)
            lex_scores = (
                np.array(
                    [counts[e] for e in EMOTION_LABELS], dtype=np.float32
                )
                / (hits + EPS)
                if hits > 0
                else np.zeros(len(EMOTION_LABELS), dtype=np.float32)
            )

            t_scores = np.array(
                [t_batch[j][e] for e in EMOTION_LABELS], dtype=np.float32
            )

            alpha = 0.7 if t_scores.sum() > 0 else 0.0
            scores = alpha * t_scores + (1 - alpha) * lex_scores
            total = scores.sum()
            if total > 0:
                scores = scores / total

            token_count = max(n_lex_tokens, 1)
            coverage = hits / token_count

            max_val = float(np.max(scores))
            mean_val = float(np.mean(scores))
            std_val = float(np.std(scores))
            range_val = float(np.max(scores) - np.min(scores))
            l2_intensity = float(np.linalg.norm(scores))
            entropy = normalized_entropy(scores)

            results[dst_i] = {
                "emotion_intensity_max": self._safe(max_val),
                "emotion_intensity_mean": self._safe(mean_val),
                "emotion_intensity_std": self._safe(std_val),
                "emotion_intensity_range": self._safe(range_val),
                "emotion_intensity_l2": self._safe(l2_intensity),
                "emotion_intensity_entropy": self._safe(entropy),
                "emotion_coverage": self._safe(coverage),
                "emotion_transformer_available": (
                    1.0 if _transformer_runtime_available() else 0.0
                ),
            }

        return results

    # -----------------------------------------------------

    def _empty(self) -> Dict[str, float]:
        return {
            "emotion_intensity_max": 0.0,
            "emotion_intensity_mean": 0.0,
            "emotion_intensity_std": 0.0,
            "emotion_intensity_range": 0.0,
            "emotion_intensity_l2": 0.0,
            "emotion_intensity_entropy": 0.0,
            "emotion_coverage": 0.0,
            "emotion_transformer_available": (
                1.0 if _transformer_runtime_available() else 0.0
            ),
        }

    def _fallback(self) -> Dict[str, float]:
        return self._empty()

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))
