from __future__ import annotations

import pytest

from src.explainability import bias_explainer


ACTUAL_EXPLAIN_BIAS_KEYS = {
    "token_importance",
    "integrated_gradients",
    "biased_tokens",
    "sentence_bias_scores",
    "attention_scores",
    "bias_heatmap",
    "bias_intensity",
}


def _patch_bias_explainer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bias_explainer,
        "compute_shap_importance",
        lambda *_args, **_kwargs: [{"token": "shocking", "importance": 0.8}],
    )
    monkeypatch.setattr(
        bias_explainer,
        "compute_integrated_gradients",
        lambda *_args, **_kwargs: [{"token": "shocking", "importance": 0.3}],
    )
    monkeypatch.setattr(
        bias_explainer,
        "compute_attention_scores",
        lambda *_args, **_kwargs: [{"token": "shocking", "attention": 0.4}],
    )
    monkeypatch.setattr(
        bias_explainer,
        "compute_sentence_bias",
        lambda _text: [
            {
                "sentence": _text,
                "bias_score": 0.7,
                "biased_tokens": ["shocking"],
            }
        ],
    )


def test_explain_bias_returns_expected_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "This shocking news will destroy the country!"
    _patch_bias_explainer(monkeypatch)

    result = bias_explainer.explain_bias(model=object(), tokenizer=object(), text=text)

    assert set(result.keys()) == ACTUAL_EXPLAIN_BIAS_KEYS
    assert isinstance(result["biased_tokens"], list)


def test_explain_bias_token_importance_contains_expected_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_bias_explainer(monkeypatch)

    result = bias_explainer.explain_bias(
        model=object(),
        tokenizer=object(),
        text="This shocking news destroys trust.",
    )

    assert len(result["token_importance"]) > 0
    assert result["token_importance"][0]["token"] == "shocking"


def test_explain_bias_sentence_scores_are_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_bias_explainer(monkeypatch)

    result = bias_explainer.explain_bias(
        model=object(),
        tokenizer=object(),
        text="Incredible bias found in this shocking report.",
    )

    assert "sentence_bias_scores" in result
    assert len(result["sentence_bias_scores"]) > 0


def test_explain_bias_attention_scores_are_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_bias_explainer(monkeypatch)

    result = bias_explainer.explain_bias(
        model=object(),
        tokenizer=object(),
        text="This is extremely manipulative political rhetoric.",
    )

    assert "attention_scores" in result
    assert result["attention_scores"][0]["token"] == "shocking"
