from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from pipelines import prediction_pipeline as pipeline_module


class _DummyTokenizer:
    def __init__(self) -> None:
        self.last_text = None

    def __call__(self, text, **_kwargs):
        self.last_text = text
        return {
            "input_ids": torch.ones((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        }


class _DummyModel:
    config = SimpleNamespace(id2label={0: "REAL", 1: "FAKE"})

    def __call__(self, **_inputs):
        return SimpleNamespace(logits=torch.tensor([[0.2, 1.8]]))


def test_predict_text_returns_expected_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _DummyTokenizer()
    model = _DummyModel()
    monkeypatch.setattr(
        pipeline_module,
        "_get_assets",
        lambda: (model, tokenizer, None),
    )

    result = pipeline_module.predict_text(
        "Breaking report: market conditions changed overnight."
    )

    assert set(result.keys()) == {"prediction", "confidence", "probabilities"}
    assert result["prediction"] in {"REAL", "FAKE"}
    assert 0.0 <= float(result["confidence"]) <= 1.0
    assert set(result["probabilities"].keys()) == {"REAL", "FAKE"}


def test_predict_text_uses_feature_pipeline_when_vectorizer_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _DummyTokenizer()
    model = _DummyModel()

    monkeypatch.setattr(
        pipeline_module,
        "_get_assets",
        lambda: (model, tokenizer, object()),
    )
    monkeypatch.setattr(
        pipeline_module,
        "transform_feature_pipeline",
        lambda _df, **_kwargs: pd.DataFrame({"engineered_text": ["engineered text"]}),
    )

    pipeline_module.predict_text("Breaking report: market conditions changed overnight.")

    assert tokenizer.last_text == "engineered text"


def test_predict_batch_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        pipeline_module.predict_batch([])

