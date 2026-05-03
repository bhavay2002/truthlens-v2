import copy
import pytest

from src.aggregation.aggregation_pipeline import AggregationPipeline


def test_inject_analysis_does_not_mutate_input_profile():
    p = AggregationPipeline()
    profile = {"bias": {"x": 0.2}}
    original = copy.deepcopy(profile)

    _ = p._inject_analysis_sections(
        profile,
        {"framing": {"foo": 1.0}},
    )

    assert profile == original


def test_run_returns_distinct_raw_and_weighted_scores_when_weights_override_defaults():
    p = AggregationPipeline()

    # Override weights so weighted path diverges from default raw path.
    custom_weights = {
        "bias": 1.0,
        "emotion": 0.0,
        "narrative": 0.0,
        "analysis_influence_manipulation": 0.0,
        "discourse": 1.0,
        "graph": 0.0,
        "credibility_bias_penalty": 0.0,
        "analysis_influence_credibility": 0.0,
        "final_credibility": 1.0,
        "final_manipulation": 0.0,
        "final_ideology": 0.0,
    }
    p.weight_manager.weights = custom_weights

    profile = {
        "bias": {"b1": 0.8},
        "emotion": {"e1": 0.2},
        "narrative": {"n1": 0.4},
        "discourse": {"d1": 0.7},
        "graph": {"g1": 0.6},
        "ideology": {"i1": 0.3},
    }

    out = p.run(profile)

    assert "scores" in out
    assert "raw_scores" in out
    assert out["scores"] != out["raw_scores"]


def test_bool_values_are_not_treated_as_numeric_during_normalization():
    p = AggregationPipeline()
    profile = {
        "bias": {"numeric": 0.2, "flag": True},
    }
    norm = p.normalize_profile(profile)
    assert norm["bias"]["flag"] is True
    assert isinstance(norm["bias"]["numeric"], float)
