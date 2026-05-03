import pytest
import torch

from src.explainability.attention_rollout import AttentionRollout


def test_compute_rollout_basic():
    ar = AttentionRollout()
    att = torch.ones(1, 2, 3, 3) / 3.0
    out = ar.compute_rollout([att, att], tokens=["[CLS]", "a", "b"], source_token_index=0)
    assert out["tokens"] == ["[CLS]", "a", "b"]
    assert len(out["rollout_scores"]) == 3
    assert abs(sum(out["rollout_scores"]) - 1.0) < 1e-6


def test_compute_rollout_bad_source_idx():
    ar = AttentionRollout()
    att = torch.ones(1, 2, 3, 3) / 3.0
    with pytest.raises(ValueError):
        ar.compute_rollout([att], tokens=["a"], source_token_index=5)
