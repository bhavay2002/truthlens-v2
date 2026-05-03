import numpy as np
import pytest

from src.aggregation.truthlens_score_calculator import truthlens_score_vector


def _scores_payload():
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


def test_truthlens_score_vector_is_ordered_and_stable():
    scores = _scores_payload()
    v = truthlens_score_vector(scores)
    assert isinstance(v, np.ndarray)
    assert v.shape == (9,)
    assert np.allclose(
        v,
        np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32),
    )


def test_truthlens_score_vector_raises_when_key_missing():
    scores = _scores_payload()
    del scores["truthlens_graph_score"]
    with pytest.raises(RuntimeError):
        truthlens_score_vector(scores)  # wrapped by function as RuntimeError
