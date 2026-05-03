from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np

from src.features.base.base_feature import FeatureContext
from src.features.base.tokenization import ensure_tokens_word, tokenize_words
from src.features.bias.bias_lexicon_features import BiasLexiconFeatures


# ---------------------------------------------------------
# Result schema
# ---------------------------------------------------------

@dataclass
class BiasResult:
    bias_score: float
    media_bias: str
    bias_intensity: float
    bias_entropy: float
    bias_subjectivity: float
    bias_certainty: float
    biased_tokens: List[str]
    sentence_heatmap: List[Dict[str, Any]]


# ---------------------------------------------------------
# Tokenization — audit fix §1.11
# ---------------------------------------------------------
# The previous ``[A-Za-z']+`` pattern was ASCII-only and silently
# stripped accented characters from non-English headlines (``café`` ->
# ``caf``). Every other extractor reads ``ensure_tokens_word`` from the
# per-context cache; this module now does the same.


# ---------------------------------------------------------
# Singleton extractor
#
# BiasLexiconFeatures previously got constructed on every call to
# compute_bias_features AND once per sentence in the heatmap loop.
# For long articles that meant dozens of full re-initializations per
# request.  Cache a single instance at module scope.
# ---------------------------------------------------------

_EXTRACTOR: BiasLexiconFeatures | None = None


def _get_extractor() -> BiasLexiconFeatures:
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = BiasLexiconFeatures()
    return _EXTRACTOR


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def compute_bias_features(text: str) -> BiasResult:

    extractor = _get_extractor()

    context = FeatureContext(text=text)
    tokens = ensure_tokens_word(context, text)

    features = extractor.extract(context)

    # -------------------------
    # Core signals
    # -------------------------

    bias_score = features.get("bias_density", 0.0)
    intensity = features.get("bias_intensity", 0.0)
    entropy = features.get("bias_entropy", 0.0)
    subjectivity = features.get("bias_subjectivity", 0.0)
    certainty = features.get("bias_certainty", 0.0)

    # -------------------------
    # Classification (IMPROVED)
    # -------------------------

    composite = 0.5 * bias_score + 0.3 * intensity + 0.2 * subjectivity

    if composite < 0.05:
        media_bias = "center"
    elif composite < 0.15:
        media_bias = "lean"
    else:
        media_bias = "strong"

    # -------------------------
    # Biased tokens (derived from features)
    # -------------------------

    biased_tokens = [
        t for t in tokens
        if features.get("bias_eval_ratio", 0) > 0
        or features.get("bias_assertive_ratio", 0) > 0
    ]

    # -------------------------
    # Sentence heatmap (CONSISTENT)
    # -------------------------

    sentences = [
        s.strip() for s in re.split(r"[.!?]+", text) if s.strip()
    ]

    sentence_heatmap = []

    for sent in sentences:

        # Audit fix §1.11 — same canonical Unicode tokenizer as the
        # rest of the codebase; previously this fell back to ASCII-only
        # regex which silently dropped non-English content.
        sent_tokens = tokenize_words(sent)

        if not sent_tokens:
            score = 0.0
        else:
            # reuse extractor logic locally
            sent_ctx = FeatureContext(text=sent, tokens_word=sent_tokens)
            sent_feat = extractor.extract(sent_ctx)

            score = sent_feat.get("bias_density", 0.0)

        sentence_heatmap.append({
            "sentence": sent,
            "bias_score": round(score, 4),
        })

    return BiasResult(
        bias_score=round(bias_score, 4),
        media_bias=media_bias,
        bias_intensity=round(intensity, 4),
        bias_entropy=round(entropy, 4),
        bias_subjectivity=round(subjectivity, 4),
        bias_certainty=round(certainty, 4),
        biased_tokens=biased_tokens,
        sentence_heatmap=sentence_heatmap,
    )