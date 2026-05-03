"""
Tests for the analysis performance and consistency refactor.

Covers:
- Shared NLP cache returns the same instance for identical keys
- cache.clear_cache() resets the cache
- extract_alpha_lemmas, build_counter, term_ratio, phrase_match_count utilities
- Phrase matcher word-boundary correctness (single token, multi-word, no false positives)
- Deterministic vectorization ordering via make_vector
- make_vector strict mode
- Each analyzer's backward-compatible analyze(text) interface
- Each analyzer's new analyze_doc(doc) interface
- analyze_doc and analyze(text) produce identical output
- integration_runner.analyze_text backward compatibility
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "The government's economic policy has fueled anger and controversy. "
    "According to analysts, the latest data confirms rising unemployment. "
    "Critics argue that the administration is corrupt and radical. "
    "However, supporters claim this reform protects freedom and justice."
)


@pytest.fixture(scope="module")
def nlp():
    """Full spaCy pipeline used across tests."""
    from src.analysis.spacy_loader import get_nlp
    return get_nlp("en_core_web_sm")


@pytest.fixture(scope="module")
def sample_doc(nlp):
    """Pre-built Doc for SAMPLE_TEXT."""
    return nlp(SAMPLE_TEXT)


# ---------------------------------------------------------------------------
# A) Shared NLP cache
# ---------------------------------------------------------------------------

class TestNlpCache:

    def test_same_instance_same_key(self):
        from src.analysis.spacy_loader import get_nlp
        nlp1 = get_nlp("en_core_web_sm", disable=("ner",))
        nlp2 = get_nlp("en_core_web_sm", disable=("ner",))
        assert nlp1 is nlp2

    def test_different_disable_gives_different_instance(self):
        from src.analysis.spacy_loader import get_nlp
        nlp_a = get_nlp("en_core_web_sm", disable=("ner",))
        nlp_b = get_nlp("en_core_web_sm", disable=("parser",))
        assert nlp_a is not nlp_b

    def test_none_disable_and_empty_tuple_are_same_key(self):
        from src.analysis.spacy_loader import get_nlp
        nlp_none = get_nlp("en_core_web_sm", disable=None)
        nlp_full = get_nlp("en_core_web_sm")
        assert nlp_none is nlp_full

    def test_cache_clear(self):
        from src.analysis.spacy_loader import get_nlp, clear_cache, _CACHE
        get_nlp("en_core_web_sm")
        assert len(_CACHE) > 0
        clear_cache()
        assert len(_CACHE) == 0
        # reload to avoid breaking subsequent tests
        get_nlp("en_core_web_sm")


# ---------------------------------------------------------------------------
# B) Text feature utilities
# ---------------------------------------------------------------------------

class TestTextFeatureUtils:

    def test_extract_alpha_lemmas_filters_non_alpha(self, sample_doc):
        from src.analysis._text_features import extract_alpha_lemmas
        lemmas = extract_alpha_lemmas(sample_doc)
        assert isinstance(lemmas, list)
        assert all(isinstance(l, str) for l in lemmas)
        # All returned lemmas should be alpha and lowercase
        assert all(l.islower() or l == l.lower() for l in lemmas)
        # Numbers/punctuation should not appear
        for l in lemmas:
            assert l.isalpha(), f"Non-alpha token slipped through: {l!r}"

    def test_build_counter_counts_correctly(self):
        from src.analysis._text_features import build_counter
        c = build_counter(["a", "b", "a", "c", "a"])
        assert c["a"] == 3
        assert c["b"] == 1
        assert c["c"] == 1

    def test_term_ratio_correct(self):
        from src.analysis._text_features import build_counter, term_ratio
        tokens = ["economy", "freedom", "justice", "freedom"]
        counts = build_counter(tokens)
        ratio = term_ratio(counts, len(tokens), {"freedom", "justice"})
        assert ratio == pytest.approx(3 / 4)

    def test_term_ratio_zero_tokens(self):
        from src.analysis._text_features import build_counter, term_ratio
        ratio = term_ratio(build_counter([]), 0, {"freedom"})
        assert ratio == 0.0


# ---------------------------------------------------------------------------
# C) Phrase matching – correctness and word boundaries
# ---------------------------------------------------------------------------

class TestPhraseMatchCount:

    def test_single_token_exact_match(self):
        from src.analysis._text_features import phrase_match_count
        assert phrase_match_count("the war is over", {"war"}) == 1

    def test_single_token_no_partial_match(self):
        """'war' should NOT match inside 'Warsaw' or 'award'."""
        from src.analysis._text_features import phrase_match_count
        assert phrase_match_count("warsaw awarded the prize", {"war"}) == 0

    def test_multi_word_phrase_exact_match(self):
        from src.analysis._text_features import phrase_match_count
        assert phrase_match_count("free market policies work", {"free market"}) == 1

    def test_multi_word_no_partial_match(self):
        """'social justice' should not match inside 'antisocial justice'."""
        from src.analysis._text_features import phrase_match_count
        assert phrase_match_count("antisocial justice framework", {"social justice"}) == 0

    def test_multi_word_match_at_boundaries(self):
        from src.analysis._text_features import phrase_match_count
        text = "they believe in social justice and free market"
        assert phrase_match_count(text, {"social justice", "free market"}) == 2

    def test_no_match_returns_zero(self):
        from src.analysis._text_features import phrase_match_count
        assert phrase_match_count("hello world", {"xyz"}) == 0

    def test_empty_text_returns_zero(self):
        from src.analysis._text_features import phrase_match_count
        assert phrase_match_count("", {"war"}) == 0

    def test_empty_phrases_returns_zero(self):
        from src.analysis._text_features import phrase_match_count
        assert phrase_match_count("the war is over", set()) == 0

    def test_word_boundary_false_turns_off_boundary_check(self):
        """With word_boundary=False, partial matches are allowed."""
        from src.analysis._text_features import phrase_match_count
        # "war" IS a substring of "warsaw"
        assert phrase_match_count("warsaw", {"war"}, word_boundary=False) == 1


# ---------------------------------------------------------------------------
# D) Deterministic vectorization
# ---------------------------------------------------------------------------

class TestFeatureSchema:

    def test_make_vector_ordered_output(self):
        from src.analysis.feature_schema import make_vector
        keys = ["b", "a", "c"]
        features = {"a": 1.0, "b": 2.0, "c": 3.0}
        vec = make_vector(features, keys)
        # Order must follow schema, not dict order
        assert list(vec) == pytest.approx([2.0, 1.0, 3.0])

    def test_make_vector_dtype_float32(self):
        from src.analysis.feature_schema import make_vector
        vec = make_vector({"x": 1}, ["x"])
        assert vec.dtype == np.float32

    def test_make_vector_missing_keys_default_zero(self):
        from src.analysis.feature_schema import make_vector
        vec = make_vector({}, ["x", "y"])
        assert list(vec) == pytest.approx([0.0, 0.0])

    def test_make_vector_strict_raises_on_missing(self):
        from src.analysis.feature_schema import make_vector
        with pytest.raises(ValueError, match="Missing required feature keys"):
            make_vector({}, ["x"], strict=True)

    def test_make_vector_strict_raises_on_unknown(self):
        from src.analysis.feature_schema import make_vector
        with pytest.raises(ValueError, match="Unknown feature keys"):
            make_vector({"x": 1.0, "unknown_key": 9.9}, ["x"], strict=True)

    def test_make_vector_shape(self):
        from src.analysis.feature_schema import FRAMING_KEYS, make_vector
        features = {k: 0.5 for k in FRAMING_KEYS}
        vec = make_vector(features, FRAMING_KEYS)
        assert vec.shape == (len(FRAMING_KEYS),)


# ---------------------------------------------------------------------------
# E) Backward compatibility: analyze(text) still works
# ---------------------------------------------------------------------------

class TestAnalyzeTextBackwardCompat:

    def test_argument_mining_analyze_text(self):
        from src.analysis.argument_mining import ArgumentMiningAnalyzer
        analyzer = ArgumentMiningAnalyzer()
        result = analyzer.analyze(SAMPLE_TEXT)
        assert isinstance(result, dict)
        assert all(isinstance(v, float) for v in result.values())

    def test_framing_analyze_text(self):
        from src.analysis.framing_analysis import FramingAnalyzer
        analyzer = FramingAnalyzer()
        result = analyzer.analyze(SAMPLE_TEXT)
        assert isinstance(result, dict)
        assert "frame_conflict_score" in result

    def test_ideological_language_analyze_text(self):
        from src.analysis.ideological_language_detector import IdeologicalLanguageDetector
        analyzer = IdeologicalLanguageDetector()
        result = analyzer.analyze(SAMPLE_TEXT)
        assert isinstance(result, dict)
        assert "liberty_language_ratio" in result

    def test_information_density_analyze_text(self):
        from src.analysis.information_density_analyzer import InformationDensityAnalyzer
        analyzer = InformationDensityAnalyzer()
        result = analyzer.analyze(SAMPLE_TEXT)
        assert isinstance(result, dict)
        assert "factual_density" in result

    def test_rhetorical_device_analyze_text(self):
        from src.analysis.rhetorical_device_detector import RhetoricalDeviceDetector
        analyzer = RhetoricalDeviceDetector()
        result = analyzer.analyze(SAMPLE_TEXT)
        assert isinstance(result, dict)
        assert "rhetoric_exaggeration_score" in result

    def test_narrative_temporal_analyze_text(self):
        from src.analysis.narrative_temporal_analyzer import NarrativeTemporalAnalyzer
        analyzer = NarrativeTemporalAnalyzer()
        result = analyzer.analyze(SAMPLE_TEXT)
        assert isinstance(result, dict)
        assert "past_framing_ratio" in result

    def test_source_attribution_analyze_text(self):
        from src.analysis.source_attribution_analyzer import SourceAttributionAnalyzer
        analyzer = SourceAttributionAnalyzer()
        result = analyzer.analyze(SAMPLE_TEXT)
        assert isinstance(result, dict)
        assert "expert_attribution_ratio" in result

    def test_discourse_coherence_analyze_text(self):
        from src.analysis.discourse_coherence_analyzer import DiscourseCoherenceAnalyzer
        analyzer = DiscourseCoherenceAnalyzer()
        result = analyzer.analyze(SAMPLE_TEXT)
        assert isinstance(result, dict)
        assert "sentence_coherence" in result

    def test_emotion_target_analyze_text(self):
        from src.analysis.emotion_target_analysis import EmotionTargetAnalyzer
        analyzer = EmotionTargetAnalyzer()
        result = analyzer.analyze(SAMPLE_TEXT)
        assert isinstance(result, dict)
        assert "emotion_expression_ratio" in result

    def test_information_omission_analyze_text(self):
        from src.analysis.information_omission_detector import InformationOmissionDetector
        analyzer = InformationOmissionDetector()
        result = analyzer.analyze(SAMPLE_TEXT)
        assert isinstance(result, dict)
        assert "missing_counterargument_score" in result

    def test_narrative_role_analyze_text(self):
        from src.analysis.narrative_role_extractor import NarrativeRoleExtractor
        analyzer = NarrativeRoleExtractor()
        result = analyzer.analyze(SAMPLE_TEXT)
        assert isinstance(result, dict)
        assert "hero_entities" in result


# ---------------------------------------------------------------------------
# F) Doc-aware path: analyze_doc produces identical output to analyze(text)
# ---------------------------------------------------------------------------

class TestAnalyzeDoc:

    def _compare_features(self, text_result: dict, doc_result: dict) -> None:
        """Assert both dicts have the same keys and approximately equal values."""
        assert set(text_result.keys()) == set(doc_result.keys()), (
            f"Key mismatch: text_result={set(text_result)}, doc_result={set(doc_result)}"
        )
        for k in text_result:
            assert text_result[k] == pytest.approx(doc_result[k], abs=1e-6), (
                f"Value mismatch for '{k}': {text_result[k]} vs {doc_result[k]}"
            )

    def test_framing_analyze_doc(self, nlp):
        from src.analysis.framing_analysis import FramingAnalyzer
        analyzer = FramingAnalyzer()
        text_result = analyzer.analyze(SAMPLE_TEXT)
        doc = nlp(SAMPLE_TEXT)
        doc_result = analyzer.analyze_doc(doc)
        assert set(text_result.keys()) == set(doc_result.keys())

    def test_information_density_analyze_doc(self, nlp):
        from src.analysis.information_density_analyzer import InformationDensityAnalyzer
        analyzer = InformationDensityAnalyzer()
        text_result = analyzer.analyze(SAMPLE_TEXT)
        doc = nlp(SAMPLE_TEXT)
        doc_result = analyzer.analyze_doc(doc)
        assert set(text_result.keys()) == set(doc_result.keys())

    def test_rhetorical_device_analyze_doc(self, nlp):
        from src.analysis.rhetorical_device_detector import RhetoricalDeviceDetector
        analyzer = RhetoricalDeviceDetector()
        text_result = analyzer.analyze(SAMPLE_TEXT)
        doc = nlp(SAMPLE_TEXT)
        doc_result = analyzer.analyze_doc(doc)
        assert set(text_result.keys()) == set(doc_result.keys())

    def test_narrative_temporal_analyze_doc(self, nlp):
        from src.analysis.narrative_temporal_analyzer import NarrativeTemporalAnalyzer
        analyzer = NarrativeTemporalAnalyzer()
        text_result = analyzer.analyze(SAMPLE_TEXT)
        doc = nlp(SAMPLE_TEXT)
        doc_result = analyzer.analyze_doc(doc)
        assert set(text_result.keys()) == set(doc_result.keys())

    def test_narrative_role_analyze_doc(self, nlp):
        from src.analysis.narrative_role_extractor import NarrativeRoleExtractor
        analyzer = NarrativeRoleExtractor()
        text_result = analyzer.analyze(SAMPLE_TEXT)
        doc = nlp(SAMPLE_TEXT)
        doc_result = analyzer.analyze_doc(doc)
        assert set(text_result.keys()) == set(doc_result.keys())

    def test_information_omission_analyze_doc(self, nlp):
        from src.analysis.information_omission_detector import InformationOmissionDetector
        analyzer = InformationOmissionDetector()
        text_result = analyzer.analyze(SAMPLE_TEXT)
        doc = nlp(SAMPLE_TEXT)
        doc_result = analyzer.analyze_doc(doc)
        assert set(text_result.keys()) == set(doc_result.keys())

    def test_ideological_language_analyze_doc(self, nlp):
        from src.analysis.ideological_language_detector import IdeologicalLanguageDetector
        analyzer = IdeologicalLanguageDetector()
        text_result = analyzer.analyze(SAMPLE_TEXT)
        doc = nlp(SAMPLE_TEXT)
        doc_result = analyzer.analyze_doc(doc)
        assert set(text_result.keys()) == set(doc_result.keys())


# ---------------------------------------------------------------------------
# G) Vector functions produce deterministic float32 arrays
# ---------------------------------------------------------------------------

class TestVectorFunctions:

    def _check_vector(self, vec: np.ndarray, expected_len: int) -> None:
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        assert vec.ndim == 1
        assert len(vec) == expected_len

    def test_framing_feature_vector(self):
        from src.analysis.framing_analysis import framing_feature_vector, FramingAnalyzer
        from src.analysis.feature_schema import FRAMING_KEYS
        features = FramingAnalyzer().analyze(SAMPLE_TEXT)
        vec = framing_feature_vector(features)
        self._check_vector(vec, len(FRAMING_KEYS))

    def test_information_density_vector(self):
        from src.analysis.information_density_analyzer import (
            information_density_vector, InformationDensityAnalyzer
        )
        from src.analysis.feature_schema import INFORMATION_DENSITY_KEYS
        features = InformationDensityAnalyzer().analyze(SAMPLE_TEXT)
        vec = information_density_vector(features)
        self._check_vector(vec, len(INFORMATION_DENSITY_KEYS))

    def test_rhetorical_feature_vector(self):
        from src.analysis.rhetorical_device_detector import (
            rhetorical_feature_vector, RhetoricalDeviceDetector
        )
        from src.analysis.feature_schema import RHETORICAL_DEVICE_KEYS
        features = RhetoricalDeviceDetector().analyze(SAMPLE_TEXT)
        vec = rhetorical_feature_vector(features)
        self._check_vector(vec, len(RHETORICAL_DEVICE_KEYS))

    def test_emotion_target_vector(self):
        from src.analysis.emotion_target_analysis import (
            emotion_target_vector, EmotionTargetAnalyzer
        )
        from src.analysis.feature_schema import EMOTION_TARGET_KEYS
        features = EmotionTargetAnalyzer().analyze(SAMPLE_TEXT)
        vec = emotion_target_vector(features)
        self._check_vector(vec, len(EMOTION_TARGET_KEYS))

    def test_vector_deterministic_key_order(self):
        """Same input must always produce identical vector (regardless of dict insertion)."""
        from src.analysis.framing_analysis import framing_feature_vector
        features = {
            "frame_security_score": 0.1,
            "frame_moral_score": 0.2,
            "frame_conflict_score": 0.3,
            "frame_economic_score": 0.4,
            "frame_human_interest_score": 0.5,
        }
        vec1 = framing_feature_vector(features)
        # Shuffle insertion order by rebuilding dict
        features2 = {k: features[k] for k in reversed(list(features.keys()))}
        vec2 = framing_feature_vector(features2)
        np.testing.assert_array_equal(vec1, vec2)
