from __future__ import annotations

import logging
import math
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

EPS = 1e-12


# =========================================================
# SECTION KEYWORDS
#
# TOK-AG-1: keywords are now matched against fully-recovered words
# (after subword merging) using exact equality, not naive substring
# containment. The previous `if any(k in clean for k in keys)` test
# credited "joy" to "enjoyed", "fear" to "fearless", "story" to
# "history" and so on — every wrong attribution flowed downstream
# into the section scores.
# =========================================================

SECTION_KEYWORDS: Dict[str, set] = {
    "bias":      {"bias", "biased", "opinion", "opinions", "subjective"},
    "emotion":   {"happy", "sad", "anger", "angry", "fear", "joy", "joyful"},
    "narrative": {"story", "claim", "claims", "event", "events"},
    "discourse": {"however", "therefore", "because"},
    "graph":     {"relation", "relations", "connection", "connections"},
    "ideology":  {"liberal", "conservative"},
    "analysis":  {"evidence", "analysis"},
}


# =========================================================
# UTILS
# =========================================================

def _normalize_importance(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    return scores / (np.sum(np.abs(scores)) + EPS)


def _entropy(probs):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, EPS, 1.0)
    return float(-np.sum(probs * np.log(probs)))


# =========================================================
# EXPLAINER
# =========================================================

class ScoreExplainer:

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer: Any = None,
        *,
        device: Optional[str] = None,
        steps: int = 32,
        method: Optional[str] = None,
        **_kwargs: Any,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.steps = steps
        self.method = method or "integrated_gradients"

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

    # =====================================================
    # FAST INTEGRATED GRADIENTS
    # =====================================================

    def _integrated_gradients(self, input_ids, attention_mask, task, target_idx):

        embeddings = self.model.encoder.embeddings(input_ids.to(self.device))
        baseline = torch.zeros_like(embeddings)

        # PERF-AG-2: build the alpha factor once and broadcast it
        # instead of materialising `steps` separate tensors followed by
        # a concat. On GPU this collapses 32 kernel launches + 32 small
        # allocations into a single fused multiply.
        alphas = torch.linspace(
            0.0,
            1.0,
            self.steps,
            device=embeddings.device,
            dtype=embeddings.dtype,
        ).view(-1, *([1] * embeddings.dim()))

        delta = (embeddings - baseline).unsqueeze(0)
        scaled = baseline.unsqueeze(0) + alphas * delta
        scaled = scaled.flatten(0, 1).detach().requires_grad_(True)

        # GPU-AG-2: zero any pre-existing gradient on the leaf tensor —
        # `model.zero_grad()` only zeroes the model parameters, not
        # this newly-created leaf, so without an explicit clear the
        # second call would accumulate gradients across invocations.
        if scaled.grad is not None:
            scaled.grad.zero_()

        att_mask_dev = attention_mask.to(self.device)
        att_mask_rep = att_mask_dev.repeat(self.steps, 1)

        outputs = self.model.encoder(
            inputs_embeds=scaled,
            attention_mask=att_mask_rep,
        )

        cls = outputs.last_hidden_state[:, 0]
        logits = self.model.heads[task](cls)

        target = logits[:, target_idx].sum()

        self.model.zero_grad()
        target.backward()

        grads = scaled.grad.view(self.steps, *embeddings.shape).mean(dim=0)

        integrated = (embeddings - baseline) * grads

        importance = integrated.sum(dim=-1).detach().cpu().numpy().squeeze()
        importance = np.atleast_1d(importance).astype(np.float64, copy=False)

        # TOK-AG-4: zero out attribution on padding positions so
        # downstream normalisation is not diluted by <pad> magnitudes.
        mask_np = (
            attention_mask[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        if mask_np.shape == importance.shape:
            importance = importance * mask_np

        return _normalize_importance(importance)

    # =====================================================
    # TOKEN → WORD GROUPING (TOK-AG-2 / TOK-AG-3)
    # =====================================================

    def _detok(self, token: str) -> str:
        """Return the surface form of a single subword token.

        Falls back to a manual strip when the tokenizer cannot
        round-trip a single piece.
        """
        if self.tokenizer is not None and hasattr(
            self.tokenizer, "convert_tokens_to_string"
        ):
            try:
                surface = self.tokenizer.convert_tokens_to_string([token])
                return surface.strip().lower()
            except Exception:  # pragma: no cover - defensive
                pass
        # Manual fallback strips both WordPiece (##) and BPE (Ġ / ▁).
        clean = token.replace("##", "")
        for marker in ("\u0120", "\u2581", " "):
            clean = clean.lstrip(marker)
        return clean.lower()

    def _is_word_start(self, token: str) -> bool:
        """Heuristic: does this piece begin a new word?"""
        if not token:
            return False
        if token.startswith("##"):
            return False  # WordPiece continuation
        # RoBERTa-BPE / SentencePiece word starts
        if token.startswith(("\u0120", "\u2581", " ")):
            return True
        # Special tokens (<s>, </s>, <pad>, [CLS] ...) are word starts.
        if token.startswith(("<", "[")):
            return True
        return True

    def _merge_subwords(
        self,
        tokens: List[str],
        importance: np.ndarray,
    ) -> List[Tuple[str, float]]:
        """Group subword pieces back into whole words.

        TOK-AG-3: long compound words are split into many subwords; if
        each subword keyword-matches a section, the section gets N×
        the importance instead of 1×. Group first, then match.
        """
        words: List[Tuple[str, float]] = []

        if len(tokens) == 0:
            return words

        cur_pieces: List[str] = []
        cur_score = 0.0

        for tok, imp in zip(tokens, importance):
            score = float(imp) if math.isfinite(float(imp)) else 0.0

            if self._is_word_start(tok) and cur_pieces:
                surface = self._detok("".join(cur_pieces))
                if surface:
                    words.append((surface, cur_score))
                cur_pieces = []
                cur_score = 0.0

            cur_pieces.append(tok)
            cur_score += score

        if cur_pieces:
            surface = self._detok("".join(cur_pieces))
            if surface:
                words.append((surface, cur_score))

        return words

    def _section_scores(self, tokens, importance):

        section_scores = {k: 0.0 for k in SECTION_KEYWORDS}

        if len(tokens) == 0:
            logger.debug(
                "[ScoreExplainer] empty token list — section scores zero"
            )
            return section_scores

        merged = self._merge_subwords(list(tokens), np.asarray(importance))

        for word, score in merged:
            for section, keys in SECTION_KEYWORDS.items():
                # TOK-AG-1: exact match (post-detok) instead of
                # substring-containment.
                if word in keys:
                    section_scores[section] += float(score)

        return section_scores

    # =====================================================
    # MAIN
    # =====================================================

    def explain_from_prediction(
        self,
        text: str,
        predictor_output: Dict[str, Any],
        *,
        top_k: int = 5,
    ) -> Dict[str, Any]:

        if text is None:
            raise ValueError(
                "ScoreExplainer.explain_from_prediction requires text"
            )

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        results = {}

        for task, output in predictor_output.items():

            if not isinstance(output, dict):
                continue

            logits = output.get("logits")
            probs = output.get("probabilities")

            if logits is None:
                continue

            logits_t = torch.as_tensor(logits)
            target_idx = int(torch.argmax(logits_t))

            importance = self._integrated_gradients(
                input_ids,
                attention_mask,
                task,
                target_idx,
            )

            section_scores = self._section_scores(tokens, importance)

            uncertainty = float(_entropy(probs)) if probs is not None else None

            # Surface top words (post-merge) so downstream consumers
            # see real surface forms instead of subword pieces.
            merged = self._merge_subwords(tokens, importance)
            merged.sort(key=lambda x: -abs(x[1]))

            results[task] = {
                "top_tokens": merged[:top_k],
                "section_scores": section_scores,
                "uncertainty": uncertainty,
            }

        return results

    # =====================================================
    # PROFILE MODE (REC-AG-4)
    # =====================================================

    def explain_profile(
        self,
        profile: Dict[str, Any],
        *,
        top_k: int = 5,
    ) -> Dict[str, Any]:

        contributions: List[Tuple[str, str, float]] = []
        section_scores: Dict[str, float] = {}

        # REC-AG-4: single pass instead of O(N·S). Build the section
        # totals while collecting contributions for the top-k.
        for section, payload in profile.items():
            if not isinstance(payload, dict):
                continue
            for k, v in payload.items():
                try:
                    val = float(v)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(val):
                    continue
                contributions.append((section, str(k), val))
                section_scores[section] = section_scores.get(section, 0.0) + val

        contributions.sort(key=lambda x: -abs(x[2]))

        return {
            "top_features": contributions[:top_k],
            "section_scores": section_scores,
        }
