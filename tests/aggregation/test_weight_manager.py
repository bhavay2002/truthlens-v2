import pytest

from src.aggregation.weight_manager import WeightManager


def test_adjust_weight_allows_allowed_nondefault_key():
    wm = WeightManager()
    out = wm.adjust_weight("discourse", 0.5)
    assert "discourse" in out
    assert out["discourse"] >= 0.0


def test_grouped_normalization_sums_each_group_to_one():
    wm = WeightManager(
        weights={
            "bias": 4.0,
            "emotion": 3.0,
            "narrative": 2.0,
            "analysis_influence_manipulation": 1.0,
            "discourse": 5.0,
            "graph": 3.0,
            "credibility_bias_penalty": 1.0,
            "analysis_influence_credibility": 1.0,
            "final_credibility": 2.0,
            "final_manipulation": 1.0,
            "final_ideology": 1.0,
        }
    )
    w = wm.get_weights()

    manipulation_sum = (
        w["bias"]
        + w["emotion"]
        + w["narrative"]
        + w["analysis_influence_manipulation"]
    )
    credibility_sum = (
        w["discourse"]
        + w["graph"]
        + w["credibility_bias_penalty"]
        + w["analysis_influence_credibility"]
    )
    final_sum = w["final_credibility"] + w["final_manipulation"] + w["final_ideology"]

    assert pytest.approx(manipulation_sum, abs=1e-6) == 1.0
    assert pytest.approx(credibility_sum, abs=1e-6) == 1.0
    assert pytest.approx(final_sum, abs=1e-6) == 1.0


def test_load_weights_from_config_merges_instead_of_replacing(tmp_path):
    wm = WeightManager()
    cfg = tmp_path / "weights.json"
    cfg.write_text('{"discourse": 0.9}', encoding="utf-8")

    out = wm.load_weights_from_config(cfg)
    # Existing keys preserved
    assert "bias" in out
    # New key applied
    assert "discourse" in out


def test_adjust_weight_rejects_unknown_key():
    wm = WeightManager()
    with pytest.raises(KeyError):
        wm.adjust_weight("not_a_valid_key", 0.1)
