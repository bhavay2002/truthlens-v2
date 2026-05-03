# src/analysis/emotion_lexicon.py

from __future__ import annotations

from typing import Dict, Set


# =========================================================
# EMOTION TAXONOMY (STANDARDIZED)
# =========================================================
# Inspired by:
# - Plutchik wheel
# - NRC Emotion Lexicon
# - Ekman basic emotions

EMOTION_TERMS: Dict[str, Set[str]] = {

    # -----------------------------------------------------
    # NEGATIVE HIGH-AROUSAL
    # -----------------------------------------------------

    "anger": {
        "anger", "angry", "furious", "rage", "outrage",
        "enraged", "irritated", "annoyed", "frustrated",
        "hostile", "aggressive", "resentment",
        "fury", "wrath", "hatred",
    },

    "fear": {
        "fear", "afraid", "scared", "terrified",
        "frightened", "panic", "panic attack",
        "anxious", "anxiety", "worry", "worried",
        "threat", "threatening", "unsafe",
    },

    "disgust": {
        "disgust", "disgusting", "repulsive",
        "gross", "nasty", "offensive",
        "sickening", "abhorrent",
    },

    # -----------------------------------------------------
    # NEGATIVE LOW-AROUSAL
    # -----------------------------------------------------

    "sadness": {
        "sad", "sadness", "unhappy", "depressed",
        "down", "miserable", "grief", "sorrow",
        "heartbroken", "hopeless", "despair",
        "melancholy",
    },

    "guilt": {
        "guilt", "guilty", "ashamed",
        "shame", "regret", "remorse",
    },

    # -----------------------------------------------------
    # POSITIVE
    # -----------------------------------------------------

    "joy": {
        "joy", "happy", "happiness", "delighted",
        "pleased", "glad", "excited", "thrilled",
        "cheerful", "content", "satisfied",
    },

    "trust": {
        "trust", "trustworthy", "reliable",
        "dependable", "faith", "confidence",
        "credible", "secure",
    },

    "love": {
        "love", "affection", "fond", "caring",
        "compassion", "empathy", "kindness",
    },

    # -----------------------------------------------------
    # SURPRISE / COGNITIVE
    # -----------------------------------------------------

    "surprise": {
        "surprise", "surprised", "shocked",
        "astonished", "unexpected",
        "suddenly", "out of nowhere",
    },

    "anticipation": {
        "anticipation", "expectation",
        "looking forward", "awaiting",
        "hopeful", "optimistic",
    },

    # -----------------------------------------------------
    # SOCIAL / POLITICAL SIGNALS (IMPORTANT FOR YOUR USE CASE)
    # -----------------------------------------------------

    "outrage": {
        "public outrage", "mass outrage",
        "backlash", "uproar", "storm of criticism",
        "condemnation", "widespread anger",
    },

    "victimhood": {
        "victim", "oppressed", "suffering",
        "marginalized", "targeted",
        "persecuted", "abused",
    },

    "pride": {
        "pride", "proud", "honor",
        "dignity", "self respect",
    },

    "hope": {
        "hope", "hopeful", "aspiration",
        "optimism", "positive outlook",
    },
}


# =========================================================
# INTENSITY WEIGHTS (CALIBRATED)
# =========================================================
# Purpose:
# - amplify strong emotional signals
# - stabilize weak signals
# - improve downstream learning

EMOTION_INTENSITY: Dict[str, float] = {

    # High arousal negative → strong signals
    "anger": 1.3,
    "fear": 1.25,
    "disgust": 1.3,
    "outrage": 1.4,

    # Medium intensity
    "sadness": 1.1,
    "guilt": 1.05,
    "surprise": 1.1,

    # Positive
    "joy": 1.1,
    "trust": 1.0,
    "love": 1.05,
    "hope": 1.05,
    "pride": 1.05,

    # Cognitive / future-oriented
    "anticipation": 1.0,

    # Social framing
    "victimhood": 1.2,
}


# =========================================================
# DEFAULT INTENSITY
# =========================================================

DEFAULT_INTENSITY: float = 1.0


# =========================================================
# UTILITIES
# =========================================================

def get_emotion_terms() -> Dict[str, Set[str]]:
    """
    Return a copy to prevent accidental mutation.
    """
    return {k: set(v) for k, v in EMOTION_TERMS.items()}


def get_emotion_intensity(emotion: str) -> float:
    return EMOTION_INTENSITY.get(emotion, DEFAULT_INTENSITY)


def normalize_term(term: str) -> str:
    return term.replace("_", " ").lower().strip()