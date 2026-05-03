"""src/explainability/emotion_explainer.py

Emotion explainer.

Audit fixes
-----------
* **CRIT-2**: ``fuse(lexicon, gradients)`` previously combined a word-level
  lexicon vector with a *subword*-level gradient vector. The two arrays
  have different lengths (and in the unlucky case where they happen to
  match, every position is silently misaligned). The simpler / faster
  fix recommended by the audit is taken: ``fuse`` no longer mixes the
  gradient term — it returns the (word-level) lexicon signal. The
  gradient signal, when computed, is exposed separately as a *subword*
  attribution so callers can use it without the position-mismatch.
* **FAITH-5**: per the audit, the lexicon path is treated as a *signal*
  rather than a faithful explanation. The returned dict now carries
  ``lexicon_signal`` (heuristic, word-level) and ``model_attribution``
  (faithful, subword-level) as distinct fields, plus a ``faithful`` flag.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.features.emotion.emotion_schema import EMOTION_TERMS

logger = logging.getLogger(__name__)
EPS = 1e-12


# =========================================================
# LOOKUP
# =========================================================

WORD_TO_EMOTION = {
    w.lower(): e
    for e, words in EMOTION_TERMS.items()
    for w in words
}


INTENSIFIERS = {
    "very", "extremely", "highly", "incredibly", "really", "so", "too",
    "completely", "totally", "deeply", "strongly",
}


# =========================================================
# DATA MODEL
# =========================================================

@dataclass
class EmotionExplanation:
    tokens: List[str]

    lexicon_intensity: List[float]
    gradient_importance: List[float]

    fused_importance: List[float]

    sentence_scores: List[Dict[str, float]]
    emotion_distribution: Dict[str, float]

    intensity_score: float

    # FAITH-5 / CRIT-2: explicit faithfulness marker. ``fused_importance``
    # is now a lexicon-derived heuristic; the gradient-based subword
    # attribution lives in ``model_attribution`` with its own token list.
    faithful: bool = False
    model_attribution: Dict[str, List] = field(default_factory=dict)


# =========================================================
# TOKENIZATION
# =========================================================

def tokenize(text: str):
    return re.findall(r"\b[a-z]+\b", text.lower())


def sentences(text: str):
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


# =========================================================
# LEXICON
# =========================================================

def compute_lexicon(tokens):

    values = []

    for t in tokens:
        val = 0.0

        if t in WORD_TO_EMOTION:
            val += 1.0

        if t in INTENSIFIERS:
            val += 0.5

        values.append(val)

    return np.asarray(values, dtype=float)


# =========================================================
# GRADIENT (faithful, subword-aligned)
# =========================================================

def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device("cpu")


def _is_multitask(model) -> bool:
    return hasattr(model, "encoder") and hasattr(model, "heads")


def compute_gradients(model, tokenizer, text, *, task: str = "emotion"):
    """Compute subword-level gradient×input on the multitask model.

    CRIT-2 / FAITH-5: this function returns a (subword-tokens, scores)
    tuple now so callers can keep the gradient signal aligned to its own
    token space rather than blindly mixing it with the word-level lexicon
    vector.
    """
    device = _model_device(model)
    enc = tokenizer(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    if _is_multitask(model):
        emb = model.encoder.embeddings(enc["input_ids"]).detach()
        emb.requires_grad_(True)
        encoder_kwargs = {"inputs_embeds": emb}
        if "attention_mask" in enc:
            encoder_kwargs["attention_mask"] = enc["attention_mask"]
        out = model.encoder(**encoder_kwargs)
        cls = out.last_hidden_state[:, 0]
        heads = model.heads
        head_name = task if task in heads else next(iter(heads.keys()))
        logits = heads[head_name](cls)
    else:
        emb = model.get_input_embeddings()(enc["input_ids"]).detach()
        emb.requires_grad_(True)
        out = model(inputs_embeds=emb)
        logits = getattr(out, "logits", None)
        if logits is None and isinstance(out, dict):
            logits = out.get("logits")
        if logits is None:
            raise RuntimeError("Model output does not expose logits")

    if hasattr(model, "zero_grad"):
        model.zero_grad()
    logits.max().backward()

    grads = emb.grad.abs().sum(dim=-1)[0].detach().cpu().numpy()
    subword_tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    return subword_tokens, grads


# =========================================================
# NORMALIZATION
# =========================================================

def normalize(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.maximum(x, 0)
    return x / (np.sum(x) + EPS)


# =========================================================
# FUSION (CRIT-2: lexicon-only — no cross-tokenization mixing)
# =========================================================

def fuse(lexicon, gradients=None):  # ``gradients`` kept for API compat
    """Return the (normalised) lexicon signal.

    The previous implementation combined this word-level vector with a
    subword-level gradient vector via ``0.6*lex + 0.4*grad`` — when the
    two had different lengths, NumPy raised; when they happened to match
    by coincidence, every position was silently misaligned. The audit
    explicitly recommends dropping the gradient term here. The gradient
    signal is exposed separately (in its own subword token space) so
    consumers that want a faithful per-token attribution can use it
    without the alignment hazard.
    """
    return normalize(lexicon)


# =========================================================
# SENTENCE LEVEL
# =========================================================

def compute_sentence_scores(text):

    results = []

    for s in sentences(text):
        toks = tokenize(s)
        vals = compute_lexicon(toks)

        score = float(np.mean(vals)) if len(vals) else 0.0

        results.append({
            "sentence": s,
            "emotion_intensity": score,
        })

    return results


# =========================================================
# DISTRIBUTION
# =========================================================

def emotion_distribution(tokens):

    counts: Dict[str, int] = {}

    for t in tokens:
        if t in WORD_TO_EMOTION:
            e = WORD_TO_EMOTION[t]
            counts[e] = counts.get(e, 0) + 1

    total = sum(counts.values()) + EPS

    return {k: v / total for k, v in counts.items()}


# =========================================================
# MAIN
# =========================================================

def explain_emotion(
    text: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> Dict[str, Any]:

    if not text or not text.strip():
        raise ValueError("text cannot be empty")

    tokens = tokenize(text)

    lexicon_vals = compute_lexicon(tokens)

    gradient_tokens: List[str] = []
    gradient_scores: np.ndarray = np.array([], dtype=float)
    if model is not None and tokenizer is not None:
        try:
            gradient_tokens, gradient_scores = compute_gradients(
                model, tokenizer, text
            )
        except Exception as exc:
            logger.warning("Gradient failed: %s", exc)
            gradient_tokens, gradient_scores = [], np.array([], dtype=float)

    fused = fuse(lexicon_vals)

    return EmotionExplanation(
        tokens=tokens,

        lexicon_intensity=normalize(lexicon_vals).tolist(),
        gradient_importance=[],  # back-compat: word-level grads no longer derived

        fused_importance=fused.tolist(),

        sentence_scores=compute_sentence_scores(text),
        emotion_distribution=emotion_distribution(tokens),

        intensity_score=float(np.mean(fused)) if fused.size else 0.0,

        faithful=False,  # FAITH-5: lexicon path is heuristic, not faithful.
        model_attribution={
            "tokens": list(gradient_tokens),
            "importance": normalize(gradient_scores).tolist()
            if gradient_scores.size
            else [],
            "faithful": True,
            "token_space": "subword",
        },
    ).__dict__
