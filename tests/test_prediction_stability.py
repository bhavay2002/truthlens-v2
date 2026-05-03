from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.inference import predict_api as predict_module


class _StableTokenizer:
    def __call__(self, text, **_kwargs):
        batch = len(text) if isinstance(text, list) else 1
        return {
            "input_ids": torch.ones((batch, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch, 4), dtype=torch.long),
        }


class _StableModel:
    config = SimpleNamespace(label2id={"REAL": 0, "FAKE": 1})

    def __call__(self, **inputs):
        batch = int(inputs["input_ids"].shape[0])
        return SimpleNamespace(logits=torch.tensor([[0.1, 2.0]] * batch))


@pytest.fixture
def stable_predictor(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        predict_module,
        "load_model_and_tokenizer",
        lambda: (_StableTokenizer(), _StableModel()),
    )
    monkeypatch.setattr(
        predict_module,
        "_prepare_texts_for_inference",
        lambda texts: [str(t) for t in texts],
    )
    return predict_module


def test_prediction_is_deterministic_across_calls(stable_predictor) -> None:
    text = "Breaking news: new technology released for public use this week."
    pred1 = stable_predictor.predict(text)
    pred2 = stable_predictor.predict(text)

    assert pred1["label"] == pred2["label"]
    assert pred1["fake_probability"] == pred2["fake_probability"]
    assert pred1["confidence"] == pred2["confidence"]


def test_prediction_schema_is_stable(stable_predictor) -> None:
    text = "The government announced major infrastructure investments for the nation."
    result = stable_predictor.predict(text)

    assert set(result.keys()) == {"label", "fake_probability", "confidence"}
    assert result["label"] in {"Fake", "Real"}
    assert 0.0 <= float(result["fake_probability"]) <= 1.0
    assert 0.0 <= float(result["confidence"]) <= 1.0


def test_different_texts_produce_consistent_schema(stable_predictor) -> None:
    texts = [
        "Scientists discover cure for previously untreatable disease.",
        "Political leaders debate economic policies this session.",
        "Sports team wins national championship after long season.",
    ]
    for text in texts:
        result = stable_predictor.predict(text)
        assert "label" in result
        assert "confidence" in result


def test_batch_returns_probability_list(stable_predictor) -> None:
    texts = [
        "Economy shows signs of recovery after difficult period.",
        "New climate agreement signed by world leaders today.",
    ]
    batch = stable_predictor.predict_batch(texts)
    assert len(batch) == len(texts)
    for row in batch:
        assert isinstance(row, list)
        assert all(0.0 <= p <= 1.0 for p in row)


def test_empty_text_raises_value_error(stable_predictor) -> None:
    with pytest.raises(ValueError):
        stable_predictor.predict("   ")
