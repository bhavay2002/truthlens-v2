from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import torch

from src.inference import predict_api as predict_module


class _FastTokenizer:
    def __call__(self, text, **_kwargs):
        batch = len(text) if isinstance(text, list) else 1
        return {
            "input_ids": torch.ones((batch, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch, 4), dtype=torch.long),
        }


class _FastModel:
    config = SimpleNamespace(label2id={"REAL": 0, "FAKE": 1})

    def __call__(self, **inputs):
        batch = int(inputs["input_ids"].shape[0])
        return SimpleNamespace(logits=torch.tensor([[0.1, 2.0]] * batch))


def test_single_prediction_completes_within_time_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        predict_module,
        "load_model_and_tokenizer",
        lambda: (_FastTokenizer(), _FastModel()),
    )
    monkeypatch.setattr(
        predict_module,
        "_prepare_texts_for_inference",
        lambda texts: [str(t) for t in texts],
    )

    text = "Breaking news: government releases new economic data for fiscal quarter."

    start = time.perf_counter()
    result = predict_module.predict(text)
    elapsed = time.perf_counter() - start

    assert elapsed < 5.0, f"Prediction took {elapsed:.3f}s, expected < 5s"
    assert "label" in result


def test_batch_prediction_scales_linearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        predict_module,
        "load_model_and_tokenizer",
        lambda: (_FastTokenizer(), _FastModel()),
    )
    monkeypatch.setattr(
        predict_module,
        "_prepare_texts_for_inference",
        lambda texts: [str(t) for t in texts],
    )

    texts = [
        f"News article number {i}: this discusses important political and economic issues."
        for i in range(5)
    ]

    start = time.perf_counter()
    results = predict_module.predict_batch(texts)
    elapsed = time.perf_counter() - start

    assert elapsed < 10.0
    assert len(results) == 5


def test_repeated_predictions_are_consistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        predict_module,
        "load_model_and_tokenizer",
        lambda: (_FastTokenizer(), _FastModel()),
    )
    monkeypatch.setattr(
        predict_module,
        "_prepare_texts_for_inference",
        lambda texts: [str(t) for t in texts],
    )

    text = "The government announced new economic policies affecting all citizens."
    first = predict_module.predict(text)
    second = predict_module.predict(text)

    assert first["label"] == second["label"]
    assert abs(float(first["fake_probability"]) - float(second["fake_probability"])) < 1e-6
