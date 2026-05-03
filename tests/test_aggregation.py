from __future__ import annotations

import numpy as np
import pytest

from src.aggregation.score_normalizer import (
    clip_scores,
    normalize_minmax,
    normalize_pipeline,
    normalize_robust,
    normalize_zscore,
)
from src.aggregation.risk_assessment import (
    assess_risk_levels,
    assess_truthlens_risks,
    score_to_risk_level,
)


class TestScoreNormalizer:
    def test_minmax_produces_range_zero_to_one(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = normalize_minmax(values)
        assert float(np.min(result)) == pytest.approx(0.0)
        assert float(np.max(result)) == pytest.approx(1.0)

    def test_minmax_constant_values_returns_zeros(self) -> None:
        result = normalize_minmax([3.0, 3.0, 3.0])
        assert np.all(result == 0.0)

    def test_minmax_custom_feature_range(self) -> None:
        result = normalize_minmax([0.0, 1.0], feature_range=(-1.0, 1.0))
        assert float(np.min(result)) == pytest.approx(-1.0)
        assert float(np.max(result)) == pytest.approx(1.0)

    def test_zscore_mean_is_near_zero(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = normalize_zscore(values)
        assert float(np.mean(result)) == pytest.approx(0.0, abs=1e-5)

    def test_zscore_constant_values_returns_zeros(self) -> None:
        result = normalize_zscore([5.0, 5.0, 5.0])
        assert np.all(result == 0.0)

    def test_robust_median_centered(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = normalize_robust(values)
        assert result is not None
        assert len(result) == 5

    def test_robust_zero_iqr_returns_zeros(self) -> None:
        result = normalize_robust([2.0, 2.0, 2.0])
        assert np.all(result == 0.0)

    def test_clip_scores_bounds_values(self) -> None:
        result = clip_scores([-0.5, 0.5, 1.5], min_value=0.0, max_value=1.0)
        assert float(np.min(result)) >= 0.0
        assert float(np.max(result)) <= 1.0

    def test_normalize_pipeline_minmax(self) -> None:
        result = normalize_pipeline([1.0, 2.0, 3.0], method="minmax")
        assert float(np.max(result)) == pytest.approx(1.0)

    def test_normalize_pipeline_zscore(self) -> None:
        result = normalize_pipeline([10.0, 20.0, 30.0], method="zscore")
        assert float(np.mean(result)) == pytest.approx(0.0, abs=1e-5)

    def test_normalize_pipeline_robust(self) -> None:
        result = normalize_pipeline([1.0, 2.0, 3.0, 4.0, 5.0], method="robust")
        assert result is not None

    def test_normalize_pipeline_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported normalization method"):
            normalize_pipeline([1.0, 2.0], method="unknown")

    def test_empty_input_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_minmax([])


class TestRiskAssessment:
    def test_low_score_maps_to_low(self) -> None:
        assert score_to_risk_level(0.1) == "LOW"
        assert score_to_risk_level(0.0) == "LOW"

    def test_medium_score_maps_to_medium(self) -> None:
        assert score_to_risk_level(0.45) == "MEDIUM"
        assert score_to_risk_level(0.3) == "MEDIUM"

    def test_high_score_maps_to_high(self) -> None:
        assert score_to_risk_level(0.7) == "HIGH"
        assert score_to_risk_level(1.0) == "HIGH"

    def test_score_above_one_is_clamped(self) -> None:
        assert score_to_risk_level(1.5) == "HIGH"

    def test_score_below_zero_is_clamped(self) -> None:
        assert score_to_risk_level(-0.5) == "LOW"

    def test_non_numeric_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            score_to_risk_level("high")  # type: ignore[arg-type]

    def test_assess_risk_levels_maps_all_keys(self) -> None:
        scores = {"bias": 0.1, "propaganda": 0.55, "manipulation": 0.85}
        result = assess_risk_levels(scores)
        assert result["bias"] == "LOW"
        assert result["propaganda"] == "MEDIUM"
        assert result["manipulation"] == "HIGH"

    def test_assess_risk_levels_non_numeric_values_are_skipped(self) -> None:
        scores = {"bias": 0.2, "invalid": "not_a_number"}  # type: ignore[dict-item]
        result = assess_risk_levels(scores)
        assert "bias" in result
        assert "invalid" not in result

    def test_assess_risk_levels_requires_dict(self) -> None:
        with pytest.raises(ValueError):
            assess_risk_levels([0.1, 0.5])  # type: ignore[arg-type]

    def test_assess_truthlens_risks_all_keys(self) -> None:
        scores = {
            "truthlens_manipulation_risk": 0.8,
            "truthlens_credibility_score": 0.2,
            "truthlens_final_score": 0.5,
        }
        result = assess_truthlens_risks(scores)
        assert result["manipulation_risk"] == "HIGH"
        assert result["credibility_level"] == "LOW"
        assert result["overall_truthlens_rating"] == "MEDIUM"

    def test_assess_truthlens_risks_ignores_missing_keys(self) -> None:
        result = assess_truthlens_risks({"truthlens_final_score": 0.9})
        assert "overall_truthlens_rating" in result
        assert "manipulation_risk" not in result

    def test_assess_truthlens_risks_requires_dict(self) -> None:
        with pytest.raises(ValueError):
            assess_truthlens_risks("bad_input")  # type: ignore[arg-type]
