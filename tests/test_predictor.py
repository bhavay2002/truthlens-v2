import types

import pytest
import torch

from src.inference import predict_api as predictor


class DummyModel:
    def __init__(self, logits, label2id=None, id2label=None):
        self._logits = logits
        self.config = types.SimpleNamespace(
            label2id=label2id or {},
            id2label=id2label or {},
        )

    def eval(self):
        return self

    def parameters(self):
        yield torch.zeros(1)

    def __call__(self, **kwargs):
        return types.SimpleNamespace(logits=self._logits)


class DummyTokenizer:
    def __call__(self, texts, **kwargs):
        return {"input_ids": torch.ones((len(texts), 4), dtype=torch.long)}


def test_predict_uses_dynamic_fake_index(monkeypatch):
    model = DummyModel(
        logits=torch.tensor([[4.0, 1.0]], dtype=torch.float32),
        label2id={"FAKE": 0, "REAL": 1},
    )
    tok = DummyTokenizer()
    monkeypatch.setattr(predictor, "load_model_and_tokenizer", lambda: (tok, model))

    out = predictor.predict("hello")
    assert out["label"] == "Fake"
    assert 0.0 <= out["fake_probability"] <= 1.0
    assert 0.0 <= out["confidence"] <= 1.0


def test_predict_batch_empty_returns_empty():
    assert predictor.predict_batch([]) == []


def test_extract_probs_invalid_shape_raises():
    bad = {"propaganda": {"probabilities": torch.tensor([0.2, 0.8])}}
    with pytest.raises(ValueError, match="Expected \\\[batch,>=2\\\]"):
        predictor._extract_probs(bad, model=None)


def test_extract_probs_missing_logits_raises():
    class NoLogits:
        pass

    with pytest.raises(ValueError, match="missing logits"):
        predictor._extract_probs(NoLogits(), model=None)


def test_resolve_fake_index_from_id2label(monkeypatch):
    model = DummyModel(
        logits=torch.tensor([[0.2, 0.8]], dtype=torch.float32),
        label2id={},
        id2label={0: "REAL", 1: "FAKE"},
    )
    tok = DummyTokenizer()
    monkeypatch.setattr(predictor, "load_model_and_tokenizer", lambda: (tok, model))
    out = predictor.predict("world")
    assert out["label"] == "Fake"
