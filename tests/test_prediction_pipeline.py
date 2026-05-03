from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.inference import predict_api as predict_module


class _DummyTokenizer:
    def __call__(self, text, **_kwargs):
        batch_size = len(text) if isinstance(text, list) else 1
        return {
            "input_ids": torch.ones((batch_size, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 4), dtype=torch.long),
        }


class _DummyModel:
    config = SimpleNamespace(label2id={"REAL": 0, "FAKE": 1})

    def __call__(self, **inputs):
        batch_size = int(inputs["input_ids"].shape[0])
        logits = torch.tensor([[0.1, 2.0]] * batch_size)
        return SimpleNamespace(logits=logits)


def test_predict_returns_expected_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        predict_module,
        "load_model_and_tokenizer",
        lambda: (_DummyTokenizer(), _DummyModel()),
    )
    monkeypatch.setattr(
        predict_module,
        "_prepare_texts_for_inference",
        lambda texts: [str(t) for t in texts],
    )

    result = predict_module.predict("The government announced a new policy today.")

    assert set(result.keys()) == {"label", "fake_probability", "confidence"}
    assert result["label"] in {"Fake", "Real"}
    assert 0.0 <= float(result["fake_probability"]) <= 1.0
    assert 0.0 <= float(result["confidence"]) <= 1.0


def test_predict_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        predict_module.predict("   ")
