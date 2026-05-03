def test_full_pipeline_contract_serializable(monkeypatch):
    from src.explainability import model_explainer as m

    def pred(_):
        return {"fake_probability": 0.4}

    monkeypatch.setattr(
        m,
        "explain_shap_text",
        lambda p, t: {"text": t, "token_importance": [{"token": "x", "importance": 1.0}]},
    )
    monkeypatch.setattr(
        m,
        "explain_lime_prediction",
        lambda p, t: {"text": t, "important_features": [("x", 0.5)]},
    )

    out = m.explain_prediction_full("abc", pred, model=None, tokenizer=None)
    assert "prediction" in out
    assert isinstance(out["shap_explanation"], dict)
