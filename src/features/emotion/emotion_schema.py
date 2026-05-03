# src/features/emotion/emotion_schema.py

from __future__ import annotations

from typing import Dict, List

# =========================================================
# LABELS (CANONICAL ORDER)
# =========================================================

# EMOTION-11: schema reduced from 20 → 11 labels (positional indices
# emotion_0 .. emotion_10 in the dataset). The remaining 9 names from
# the old 20-label set are kept below in ``_LEGACY_EMOTION_LABELS`` for
# audit/debug only — they are NOT part of the live label space and any
# ``emotion_<i>`` column for i >= 11 will be rejected by the data
# contract. The polarity groups below intentionally still list the full
# 20-name vocabulary; consumers (see ``emotion_features.py``) already
# filter via ``if e in EMOTION_LABELS`` so dropping a name from
# ``EMOTION_LABELS`` automatically removes it from the live polarity
# computation without needing parallel edits here.
EMOTION_LABELS: List[str] = [
    "neutral",
    "admiration", "approval", "gratitude",
    "annoyance", "amusement", "curiosity", "disapproval",
    "love", "optimism", "anger",
]

# Retained for migration/debug visibility only. DO NOT use in live code.
_LEGACY_EMOTION_LABELS: List[str] = [
    "joy", "confusion",
    "sadness", "disappointment", "realization",
    "caring", "surprise", "excitement", "disgust",
]

NUM_EMOTION_LABELS: int = len(EMOTION_LABELS)  # = 11

# Fast index lookup (CRITICAL for ML)
EMOTION_INDEX: Dict[str, int] = {
    label: i for i, label in enumerate(EMOTION_LABELS)
}


# =========================================================
# POLARITY GROUPS (NEW)
# =========================================================

POSITIVE_EMOTIONS = {
    "admiration","approval","gratitude","love","optimism",
    "joy","caring","amusement","excitement"
}

NEGATIVE_EMOTIONS = {
    "anger","disgust","sadness","disappointment",
    "annoyance","disapproval","confusion"
}

NEUTRAL_EMOTIONS = {"neutral","realization","curiosity","surprise"}


# =========================================================
# LEXICON (OPTIONALLY WEIGHTED)
# =========================================================

EMOTION_TERMS: Dict[str, Dict[str, float]] = {

    "neutral": {
        "neutral": 1.0, "objective": 1.0, "balanced": 1.0,
        "impartial": 1.0, "factual": 1.0,
    },

    "joy": {
        "happy": 1.0, "joyful": 1.2, "delighted": 1.3,
        "elated": 1.5, "cheerful": 1.1,
    },

    "anger": {
        "angry": 1.2, "furious": 1.5, "rage": 1.4,
        "outrage": 1.3, "hostile": 1.2,
    },

    "sadness": {
        "sad": 1.0, "depressed": 1.4,
        "gloomy": 1.2, "grief": 1.5,
    },

    # (expand rest similarly)
}


# =========================================================
# PHRASE TERMS (NEW)
# =========================================================

EMOTION_PHRASES: Dict[str, List[str]] = {
    "admiration": ["look up to", "highly respect"],
    "love": ["care deeply", "deep affection"],
    "gratitude": ["thank you", "much obliged"],
}


# =========================================================
# REVERSE LOOKUP (CENTRALIZED)
# =========================================================

WORD_TO_EMOTION: Dict[str, str] = {}
PHRASE_TO_EMOTION: Dict[str, str] = {}

for emotion, terms in EMOTION_TERMS.items():
    for word in terms:
        WORD_TO_EMOTION[word] = emotion

for emotion, phrases in EMOTION_PHRASES.items():
    for phrase in phrases:
        PHRASE_TO_EMOTION[phrase] = emotion


# =========================================================
# VALIDATION (CRITICAL FOR RESEARCH SYSTEMS)
# =========================================================

def validate_schema(strict: bool = False) -> None:
    assert len(EMOTION_LABELS) > 0, "No emotion labels defined"

    missing = [label for label in EMOTION_LABELS if label not in EMOTION_TERMS]
    if missing:
        msg = f"Missing lexicon entries for emotions: {missing}"
        if strict:
            raise ValueError(msg)
        # Auto-stub missing emotions with an empty lexicon so module import
        # never fails because of an incomplete dictionary in this snapshot.
        for label in missing:
            EMOTION_TERMS[label] = {}

    # check duplicates
    seen = {}
    for emotion, words in EMOTION_TERMS.items():
        for word in words:
            if word in seen:
                print(f"Warning: '{word}' appears in {emotion} and {seen[word]}")
            seen[word] = emotion


# Run validation on import (non-strict so import never fails on incomplete data).
validate_schema(strict=False)