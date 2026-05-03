from __future__ import annotations

from pathlib import Path

import pytest

from src.models.registry import model_registry


class _DummyModelLoader:
    @staticmethod
    def from_pretrained(path: str | Path):
        return {"loaded_model_path": str(path)}


class _DummyTokenizerLoader:
    @staticmethod
    def from_pretrained(path: str | Path):
        return {"loaded_tokenizer_path": str(path)}


def test_load_model_includes_vectorizer_when_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    vectorizer_path = tmp_path / "tfidf_vectorizer.joblib"
    vectorizer_path.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(model_registry, "MODEL_DIR", model_dir)
    monkeypatch.setattr(model_registry, "VECTORIZER_PATH", vectorizer_path)
    monkeypatch.setattr(
        model_registry,
        "RobertaForSequenceClassification",
        _DummyModelLoader,
    )
    monkeypatch.setattr(model_registry, "RobertaTokenizer", _DummyTokenizerLoader)
    monkeypatch.setattr(
        model_registry.joblib,
        "load",
        lambda _path: {"vectorizer_loaded": True},
    )

    assets = model_registry.ModelRegistry.load_model()

    assert assets["model"] == {"loaded_model_path": str(model_dir)}
    assert assets["tokenizer"] == {"loaded_tokenizer_path": str(model_dir)}
    assert assets["vectorizer"] == {"vectorizer_loaded": True}


def test_load_model_returns_none_vectorizer_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    missing_vectorizer_path = tmp_path / "missing_vectorizer.joblib"

    monkeypatch.setattr(model_registry, "MODEL_DIR", model_dir)
    monkeypatch.setattr(model_registry, "VECTORIZER_PATH", missing_vectorizer_path)
    monkeypatch.setattr(
        model_registry,
        "RobertaForSequenceClassification",
        _DummyModelLoader,
    )
    monkeypatch.setattr(model_registry, "RobertaTokenizer", _DummyTokenizerLoader)

    assets = model_registry.ModelRegistry.load_model()

    assert assets["model"] == {"loaded_model_path": str(model_dir)}
    assert assets["tokenizer"] == {"loaded_tokenizer_path": str(model_dir)}
    assert assets["vectorizer"] is None


def test_get_model_delegates_to_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_assets = {"model": object(), "tokenizer": object(), "vectorizer": None}
    monkeypatch.setattr(
        model_registry.ModelRegistry,
        "load_model",
        staticmethod(lambda: expected_assets),
    )

    assert model_registry.get_model() == expected_assets

