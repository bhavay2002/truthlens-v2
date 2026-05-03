from __future__ import annotations

from types import SimpleNamespace

from src.explainability import shap_explainer


class _FakeExplainer:
    def __init__(self, *_args, **_kwargs):
        pass


def test_shap_explainer_cache_is_bounded(monkeypatch) -> None:
    monkeypatch.setattr(
        shap_explainer,
        "shap",
        SimpleNamespace(
            maskers=SimpleNamespace(Text=lambda: object()),
            Explainer=_FakeExplainer,
        ),
    )

    shap_explainer._EXPLAINER_CACHE.clear()

    cache_limit = shap_explainer._MAX_EXPLAINER_CACHE_SIZE
    for idx in range(cache_limit + 5):
        predict_fn = lambda _text, i=idx: {
            "fake_probability": i / 100.0,
            "label": "Fake",
            "confidence": 0.5,
        }
        shap_explainer.get_explainer(predict_fn)

    assert len(shap_explainer._EXPLAINER_CACHE) <= cache_limit
    shap_explainer._EXPLAINER_CACHE.clear()


def test_shap_explainer_reuses_named_predictor(monkeypatch) -> None:
    monkeypatch.setattr(
        shap_explainer,
        "shap",
        SimpleNamespace(
            maskers=SimpleNamespace(Text=lambda: object()),
            Explainer=_FakeExplainer,
        ),
    )

    shap_explainer._EXPLAINER_CACHE.clear()

    def stable_predict(_text):
        return {"fake_probability": 0.5, "label": "Fake", "confidence": 0.5}

    first = shap_explainer.get_explainer(stable_predict)
    second = shap_explainer.get_explainer(stable_predict)

    assert first is second
    assert len(shap_explainer._EXPLAINER_CACHE) == 1
    shap_explainer._EXPLAINER_CACHE.clear()
