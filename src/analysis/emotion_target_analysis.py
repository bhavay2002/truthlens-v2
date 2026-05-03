from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, DefaultDict, Tuple, Set, Optional

import numpy as np
from spacy.matcher import PhraseMatcher

from src.analysis.base_analyzer import BaseAnalyzer
from src.analysis.feature_context import FeatureContext
from src.analysis.feature_schema import EMOTION_TARGET_KEYS, make_vector
from src.analysis.spacy_loader import get_doc

from src.analysis.emotion_lexicon import (
    EMOTION_TERMS,
    EMOTION_INTENSITY,
    DEFAULT_INTENSITY,
)

logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTS
# =========================================================

EPS = 1e-8
MAX_CLIP = 1.0
MAX_EMOTION_TYPES = 20


# =========================================================
# ANALYZER
# =========================================================

class EmotionTargetAnalyzer(BaseAnalyzer):

    name = "emotion_target"
    expected_keys = set(EMOTION_TARGET_KEYS)

    def __init__(self):

        self.term_to_emotion: Dict[str, str] = {}
        self.term_weights: Dict[str, float] = {}

        # -------------------------
        # BUILD TERM MAPS
        # -------------------------
        for emotion, terms in EMOTION_TERMS.items():
            weight = EMOTION_INTENSITY.get(emotion, DEFAULT_INTENSITY)

            for t in terms:
                norm = t.replace("_", " ").lower()
                self.term_to_emotion[norm] = emotion
                self.term_weights[norm] = weight

        # PERF-A3 / CRIT-A6: build the PhraseMatcher eagerly in __init__
        # against the canonical NER-task Vocab. Lazy initialization on
        # the first call wasted the per-process build, leaked the import
        # into the hot path, and was not thread-safe under concurrent
        # requests. Building here is cheap (single-shot per process).
        self.matcher: Optional[PhraseMatcher] = None
        self._matcher_vocab = None
        self.phrase_to_emotion: Dict[Tuple[str, ...], str] = {}
        self.phrase_weights: Dict[Tuple[str, ...], float] = {}

        try:
            # VOCAB-FIX: use get_shared_nlp("safe") — the same nlp
            # instance that AnalysisPipeline/FeatureContext.from_doc use
            # to create and seed the shared "ner" doc slot. Using
            # get_task_nlp("ner") here created a *different* nlp instance
            # (different disable tuple → different cache key → different
            # Vocab object), so the "doc.vocab does not match PhraseMatcher
            # vocab" guard in analyze() always fired and phrase matching
            # was silently skipped on every article.
            from src.analysis.spacy_loader import get_shared_nlp
            nlp = get_shared_nlp("safe")

            self.matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
            self._matcher_vocab = nlp.vocab

            patterns = []

            for emotion, terms in EMOTION_TERMS.items():
                weight = EMOTION_INTENSITY.get(emotion, DEFAULT_INTENSITY)

                for term in terms:
                    if " " in term or "_" in term:
                        text = term.replace("_", " ")
                        span_doc = nlp.make_doc(text)

                        patterns.append(span_doc)

                        key = tuple(t.lower_ for t in span_doc)
                        self.phrase_to_emotion[key] = emotion
                        self.phrase_weights[key] = weight

            if patterns:
                self.matcher.add("EMOTION_PHRASES", patterns)

        except Exception:
            # Never let a startup matcher build crash analyzer
            # construction — fall back to token-only matching at runtime.
            logger.exception(
                "EmotionTargetAnalyzer: PhraseMatcher build failed; "
                "token-only matching will be used"
            )
            self.matcher = None
            self._matcher_vocab = None

        logger.info("EmotionTargetAnalyzer initialized")

    # =========================================================

    def analyze(self, ctx: FeatureContext) -> Dict[str, float]:

        # Section 4: defensive safe accessor.
        if ctx.safe_n_tokens() == 0:
            return self._empty_features()

        # 🔥 shared spaCy doc
        doc = get_doc(ctx, task="ner")

        entity_emotion_map: DefaultDict[str, float] = defaultdict(float)
        emotion_type_counter: DefaultDict[str, float] = defaultdict(float)

        emotion_score_total = 0.0
        matched_spans: Set[int] = set()

        # -----------------------------------------------------
        # PHRASE MATCHING
        # -----------------------------------------------------

        # CRIT-A4 guard: PhraseMatcher silently returns no matches when
        # called against a doc whose Vocab differs from the matcher's
        # Vocab. If that happens we skip phrase matching for this doc
        # (token-lemma matching below still works) and log once.
        vocab_ok = (
            self.matcher is not None
            and doc.vocab is self._matcher_vocab
        )
        if self.matcher is not None and not vocab_ok:
            logger.warning(
                "EmotionTarget: doc.vocab does not match PhraseMatcher vocab; "
                "falling back to token-only matching for this doc"
            )

        if self.matcher and vocab_ok:
            for _, start, end in self.matcher(doc):

                span = doc[start:end]
                key = tuple(t.lower_ for t in span)

                emotion = self.phrase_to_emotion.get(key)
                weight = self.phrase_weights.get(key, DEFAULT_INTENSITY)

                if not emotion:
                    continue

                matched_spans.update(range(start, end))

                emotion_score_total += weight
                emotion_type_counter[emotion] += weight

                target = self._resolve_target(span.root)
                if target:
                    entity_emotion_map[target] += weight

        # -----------------------------------------------------
        # TOKEN MATCHING
        # -----------------------------------------------------

        for i, token in enumerate(doc):

            if i in matched_spans:
                continue

            lemma = token.lemma_.lower()
            emotion = self.term_to_emotion.get(lemma)

            if not emotion:
                continue

            weight = self.term_weights.get(lemma, DEFAULT_INTENSITY)

            emotion_score_total += weight
            emotion_type_counter[emotion] += weight

            target = self._resolve_target(token)
            if target:
                entity_emotion_map[target] += weight

        # -----------------------------------------------------
        # FEATURE COMPUTATION
        # -----------------------------------------------------

        total_tokens = max(len(doc), 1)
        total_entity_weight = sum(entity_emotion_map.values())

        expression_ratio = emotion_score_total / (total_tokens + EPS)

        emotion_types = len(emotion_type_counter)

        dominant_emotion_strength = (
            max(emotion_type_counter.values(), default=0.0)
            / (emotion_score_total + EPS)
        )

        # diversity (log scaled)
        diversity = len(entity_emotion_map)
        diversity_norm = np.log1p(diversity) / np.log1p(20)

        # -----------------------------------------------------
        # OUTPUT
        # -----------------------------------------------------

        if total_entity_weight == 0:
            return {
                "emotion_target_diversity": self._safe(diversity_norm),
                "emotion_target_focus": 0.0,
                "emotion_expression_ratio": self._safe(expression_ratio),
                "emotion_type_diversity": self._safe(emotion_types / MAX_EMOTION_TYPES),
                "dominant_emotion_strength": self._safe(dominant_emotion_strength),
            }

        dominant_target = max(entity_emotion_map.values())
        focus_score = dominant_target / (total_entity_weight + EPS)

        return {
            "emotion_target_diversity": self._safe(diversity_norm),
            "emotion_target_focus": self._safe(focus_score),
            "emotion_expression_ratio": self._safe(expression_ratio),
            "emotion_type_diversity": self._safe(emotion_types / MAX_EMOTION_TYPES),
            "dominant_emotion_strength": self._safe(dominant_emotion_strength),
        }

    # =========================================================
    # TARGET RESOLUTION (IMPROVED)
    # =========================================================

    def _resolve_target(self, token) -> Optional[str]:

        # Named entity
        if token.ent_iob_ in {"B", "I"} and token.ent_type_:
            ent_span = next(
                (e for e in token.doc.ents if e.start <= token.i < e.end), None
            )
            if ent_span is not None:
                text = ent_span.text.lower().strip()
                if len(text) > 2:
                    return text

        # dependency relations
        for child in token.children:
            if child.dep_ in {"nsubj", "dobj", "pobj"}:
                return child.lemma_.lower()

        if token.head and token.dep_ in {"amod", "acomp"}:
            return token.head.lemma_.lower()

        return None

    # =========================================================

    def _safe(self, value: float) -> float:
        if not np.isfinite(value):
            return 0.0
        return float(np.clip(value, 0.0, MAX_CLIP))

    # =========================================================

    def _empty_features(self) -> Dict[str, float]:
        return {k: 0.0 for k in EMOTION_TARGET_KEYS}


# =========================================================
# VECTOR CONVERSION
# =========================================================

def emotion_target_vector(features: Dict[str, float]) -> np.ndarray:
    return make_vector(features, EMOTION_TARGET_KEYS)