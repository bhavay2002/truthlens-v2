import pandas as pd
import pytest

from src.inference.batch_inference import BatchInferenceConfig, BatchInferenceEngine


class _DummyArtifacts:
    bias_model = None
    ideology_model = None
    emotion_model = None
    feature_schema = ["text_length"]
    feature_scaler = None
    feature_selector = None


def test_batch_run_without_title_source(monkeypatch, tmp_path):
    dataset = tmp_path / "dataset.csv"
    pd.DataFrame({"text": ["a", "b"]}).to_csv(dataset, index=False)

    cfg = BatchInferenceConfig(
        dataset_path=str(dataset),
        text_column="text",
        output_path=str(tmp_path / "out.json"),
        num_workers=0,
    )

    monkeypatch.setattr(
        "src.inference.batch_inference.ModelLoader.load_all",
        lambda self: _DummyArtifacts(),
    )
    monkeypatch.setattr(
        "src.inference.batch_inference.PredictionPipeline.predict",
        lambda self, t: {
            "bias": ["non_bias", "bias"],
            "emotion": [{}, {}],
            "credibility_score": [0.9, 0.2],
            "ideology": ["center", "left"],
            "propaganda_probability": [0.1, 0.8],
            "credibility_explanation": [{}, {}],
        },
    )
    monkeypatch.setattr(
        "src.inference.batch_inference.FeaturePreparer.prepare_batch",
        lambda self, rows: [[len(str(r.get("text_length", 0)))] for r in rows],
    )
    monkeypatch.setattr(
        "src.inference.batch_inference.AnalysisIntegrationRunner.analyze_text",
        lambda self, text: {},
    )
    monkeypatch.setattr(
        "src.inference.batch_inference.GraphPipeline.run",
        lambda self, text: {},
    )
    monkeypatch.setattr(
        "src.inference.batch_inference.ReportGenerator.generate_report",
        lambda self, **kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        "src.inference.batch_inference.ResultFormatter.format_api_response",
        lambda self, payload: payload,
    )

    engine = BatchInferenceEngine(cfg)
    results = engine.run()
    assert len(results) == 2
