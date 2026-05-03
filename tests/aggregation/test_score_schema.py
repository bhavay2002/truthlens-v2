import pytest
from pydantic import ValidationError

from src.aggregation.score_schema import TruthLensScoreModel


def _valid_payload():
    return {
        "truthlens_bias_score": 0.1,
        "truthlens_emotion_score": 0.2,
        "truthlens_narrative_score": 0.3,
        "truthlens_discourse_score": 0.4,
        "truthlens_graph_score": 0.5,
        "truthlens_ideology_score": 0.6,
        "truthlens_manipulation_risk": 0.7,
        "truthlens_credibility_score": 0.8,
        "truthlens_final_score": 0.9,
    }


def test_score_model_accepts_valid_range():
    m = TruthLensScoreModel(**_valid_payload())
    assert m.truthlens_final_score == 0.9


@pytest.mark.parametrize(
    "field,value",
    [
        ("truthlens_bias_score", -0.01),
        ("truthlens_final_score", 1.01),
    ],
)
def test_score_model_rejects_out_of_range(field, value):
    payload = _valid_payload()
    payload[field] = value
    with pytest.raises(ValidationError):
        TruthLensScoreModel(**payload)
