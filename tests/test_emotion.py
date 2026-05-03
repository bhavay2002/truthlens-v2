from __future__ import annotations

import pytest

from src.features.emotion.emotion_lexicon import EmotionLexiconAnalyzer, EmotionResult


class TestEmotionLexiconAnalyzer:
    def setup_method(self) -> None:
        self.analyzer = EmotionLexiconAnalyzer()

    def test_analyze_returns_emotion_result(self) -> None:
        result = self.analyzer.analyze("I am very happy and joyful today!")
        assert isinstance(result, EmotionResult)

    def test_analyze_returns_dominant_emotion(self) -> None:
        result = self.analyzer.analyze("I am terrified and afraid of this horrible news.")
        assert isinstance(result.dominant_emotion, str)
        assert len(result.dominant_emotion) > 0

    def test_analyze_returns_emotion_scores_dict(self) -> None:
        result = self.analyzer.analyze("Breaking news: scientists celebrate a great discovery.")
        assert isinstance(result.emotion_scores, dict)
        assert len(result.emotion_scores) > 0

    def test_analyze_scores_are_non_negative(self) -> None:
        result = self.analyzer.analyze("This is some neutral informational text about science.")
        for score in result.emotion_scores.values():
            assert score >= 0.0

    def test_emotion_distribution_sums_to_approximately_one(self) -> None:
        result = self.analyzer.analyze("The people were angry and fearful of the new policy.")
        total = sum(result.emotion_distribution.values())
        assert abs(total - 1.0) < 1e-3 or total == pytest.approx(1.0, abs=0.01)

    def test_distribution_keys_match_scores_keys(self) -> None:
        result = self.analyzer.analyze("Scientists made a joyful discovery this week.")
        assert set(result.emotion_distribution.keys()) == set(result.emotion_scores.keys())

    def test_dominant_emotion_is_in_distribution(self) -> None:
        result = self.analyzer.analyze("The athlete was filled with pride and joy after winning.")
        assert result.dominant_emotion in result.emotion_scores

    def test_analyze_with_empty_text_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            self.analyzer.analyze("")

    def test_analyze_with_long_text(self) -> None:
        long_text = (
            "The scientists were overjoyed as they celebrated their discovery. "
            "However, critics expressed anger and fear about the implications. "
            "The public remained surprised and uncertain about what this means. "
        ) * 5
        result = self.analyzer.analyze(long_text)
        assert isinstance(result, EmotionResult)
