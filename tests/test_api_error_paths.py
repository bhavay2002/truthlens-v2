from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.app as api_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(api_app.app)


def test_predict_returns_503_when_model_assets_missing(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_app,
        "predict",
        lambda _text: (_ for _ in ()).throw(FileNotFoundError("missing model")),
    )

    response = client.post(
        "/predict",
        json={"text": "Breaking news: policy updates announced for all states."},
    )

    assert response.status_code == 503
    assert "Model not available" in response.json()["detail"]


def test_predict_returns_400_for_value_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_app,
        "predict",
        lambda _text: (_ for _ in ()).throw(ValueError("bad payload")),
    )

    response = client.post(
        "/predict",
        json={"text": "Breaking news: policy updates announced for all states."},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "bad payload"


def test_predict_returns_500_for_unexpected_errors(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_app,
        "predict",
        lambda _text: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    response = client.post(
        "/predict",
        json={"text": "Breaking news: policy updates announced for all states."},
    )

    assert response.status_code == 500
    assert "Internal server error" in response.json()["detail"]


def test_analyze_returns_503_when_model_assets_missing(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_app,
        "predict",
        lambda _text: (_ for _ in ()).throw(FileNotFoundError("missing model")),
    )

    response = client.post(
        "/analyze",
        json={"text": "Breaking news: policy updates announced for all states."},
    )

    assert response.status_code == 503
    assert "Model not available" in response.json()["detail"]

