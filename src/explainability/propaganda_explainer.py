from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

import numpy as np

from src.explainability.common_schema import ExplanationOutput, TokenImportance
from src.explainability.explanation_calibrator import calibrate_explanation

logger = logging.getLogger(__name__)

EPS = 1e-12

# =========================================================
# PROPAGANDA TECHNIQUE LEXICON
# =========================================================

PROPAGANDA_PATTERNS: Dict[str, List[str]] = {
    "name_calling": [
        "terrorist", "criminal", "traitor", "monster", "radical", "extremist",
        "thug", "corrupt", "evil", "disgusting", "deplorable", "scum",
    ],
    "glittering_generalities": [
        "freedom", "democracy", "patriot", "justice", "liberty", "virtue",
        "honor", "truth", "faith", "glory", "heritage", "values",
    ],
    "fear_appeal": [
        "danger", "threat", "crisis", "catastrophe", "disaster", "collapse",
        "invasion", "attack", "destroy", "chaos", "panic", "emergency",
    ],
    "loaded_language": [
        "regime", "propaganda", "brainwash", "indoctrinate", "manipulate",
        "fake", "hoax", "conspiracy", "coverup", "lie", "fraud", "rigged",
    ],
    "false_dilemma": [
        "either", "only", "must", "never", "always", "impossible", "certain",
        "inevitable", "guaranteed", "no choice", "only option", "no alternative",
    ],
    "appeal_to_authority": [
        "experts say", "scientists confirm", "studies show", "research proves",
        "authorities claim", "officials state", "government declares",
    ],
    "bandwagon": [
        "everyone", "everybody", "nobody", "majority", "most people",
        "all Americans", "the public", "society agrees", "people believe",
    ],
    "repetition": [],
}

TECHNIQUE_WEIGHTS: Dict[str, float] = {
    "name_calling": 1.0,
    "glittering_generalities": 0.6,
    "fear_appeal": 0.9,
    "loaded_language": 1.0,
    "false_dilemma": 0.7,
    "appeal_to_authority": 0.5,
    "bandwagon": 0.6,
    "repetition": 0.8,
}

_TOKEN_RE = re.compile(r"\b[a-z]+\b")


# =========================================================
# TOKENIZE
# =========================================================

def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


# =========================================================
# SCORE TOKENS
# =========================================================

def _score_tokens(tokens: List[str]) -> List[float]:
    scores = [0.0] * len(tokens)

    for technique, patterns in PROPAGANDA_PATTERNS.items():
        w = TECHNIQUE_WEIGHTS.get(technique, 0.5)
        for i, tok in enumerate(tokens):
            if tok in patterns:
                scores[i] += w

    # repetition: boost tokens appearing more than twice
    from collections import Counter
    counts = Counter(tokens)
    rep_w = TECHNIQUE_WEIGHTS["repetition"]
    for i, tok in enumerate(tokens):
        if counts[tok] > 2:
            scores[i] += rep_w * min(counts[tok] / 5.0, 1.0)

    return scores


# =========================================================
# PHRASE MATCHING (multi-token patterns)
# =========================================================

def _apply_phrase_scores(tokens: List[str], scores: List[float]) -> List[float]:
    text = " ".join(tokens)
    for technique, patterns in PROPAGANDA_PATTERNS.items():
        w = TECHNIQUE_WEIGHTS.get(technique, 0.5)
        for phrase in patterns:
            if " " not in phrase:
                continue
            phrase_tokens = phrase.split()
            n = len(phrase_tokens)
            for i in range(len(tokens) - n + 1):
                if tokens[i:i + n] == phrase_tokens:
                    for j in range(i, i + n):
                        scores[j] += w
    return scores


# =========================================================
# MAIN EXPLAINER
# =========================================================

def explain_propaganda(
    text: str,
    *,
    top_k: Optional[int] = None,
) -> ExplanationOutput:

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text cannot be empty")

    tokens = _tokenize(text)

    if not tokens:
        return ExplanationOutput(
            method="propaganda",
            tokens=[],
            importance=[],
            structured=[],
            # CRIT-9: propaganda explainer is heuristic / lexicon-only.
            faithful=False,
        )

    raw_scores = _score_tokens(tokens)
    raw_scores = _apply_phrase_scores(tokens, raw_scores)

    arr = np.asarray(raw_scores, dtype=float)

    if np.sum(arr) < EPS:
        return ExplanationOutput(
            method="propaganda",
            tokens=[],
            importance=[],
            structured=[],
            faithful=False,
        )

    cal = calibrate_explanation(arr.tolist(), method="custom")
    scores = cal["scores"]
    confidence = cal["confidence"]
    entropy = cal["entropy"]

    if top_k is not None and top_k > 0:
        idx = np.argsort(scores)[-top_k:][::-1]
        tokens = [tokens[i] for i in idx]
        scores = scores[idx]

    structured = [
        TokenImportance(token=t, importance=float(s))
        for t, s in zip(tokens, scores)
    ]

    logger.info(
        "Propaganda explanation generated: %d tokens, confidence=%.3f",
        len(tokens),
        confidence,
    )

    return ExplanationOutput(
        method="propaganda",
        tokens=tokens,
        importance=scores.tolist(),
        structured=structured,
        confidence=confidence,
        entropy=entropy,
        # CRIT-9: pattern-matched lexicon — not a model attribution.
        faithful=False,
    )


# =========================================================
# TECHNIQUE SUMMARY
# =========================================================

def detect_techniques(text: str) -> Dict[str, List[str]]:
    tokens = _tokenize(text)
    found: Dict[str, List[str]] = {}

    for technique, patterns in PROPAGANDA_PATTERNS.items():
        matches = [t for t in tokens if t in patterns]
        if matches:
            found[technique] = list(set(matches))

    return found
