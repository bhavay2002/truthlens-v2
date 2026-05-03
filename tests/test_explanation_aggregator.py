import pytest

from src.explainability.explanation_aggregator import ExplanationAggregator


def test_aggregate_union_tokens():
    agg = ExplanationAggregator()
    out = agg.aggregate(
        shap_importance=[{"token": "a", "importance": 1.0}],
        attention_scores=[{"token": "b", "attention": 1.0}],
    )
    assert out["tokens"] == ["a", "b"]
    assert pytest.approx(sum(out["final_token_importance"]), 1e-6) == 1.0


def test_aggregate_no_sources():
    agg = ExplanationAggregator()
    with pytest.raises(ValueError):
        agg.aggregate()
