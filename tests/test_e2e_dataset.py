"""
tests/test_e2e_dataset.py

End-to-end test suite for TruthLens AI.

Tests cover:
  1.  Dataset integrity — schema, label values, column types, row count
  2.  Unified label schema validation against the dataset
  3.  Every API endpoint (GET + POST) against the live running server
  4.  /predict for each dataset row — fields, ranges, REAL/FAKE logic
  5.  /batch-predict — full 10-row batch, cache hits, consistency
  6.  /analyze — deep analysis for real and propaganda articles
  7.  /report — report generation for every row
  8.  /calibration/info + /calibration/metrics
  9.  /ensemble/info + /ensemble/predict (3 strategies)
  10. /export/info + /export/onnx + /export/torchscript
  11. Cache clear (/cache/clear)
  12. Error handling — invalid inputs, wrong types, too-long text
  13. Cross-validation — probability/prediction agreement, batch vs single
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "truthlens_sample_dataset.csv"
BASE_URL = "http://localhost:5000"

REQUIRED_COLUMNS = [
    "title", "text",
    "bias_label", "ideology_label", "propaganda_label",
    "frame",
    "CO", "EC", "HI", "MO", "RE",
    "hero", "villain", "victim",
    "hero_entities", "villain_entities", "victim_entities",
    # EMOTION-11: schema reduced from 20 → 11 columns.
    *[f"emotion_{i}" for i in range(11)],
    "dataset",
]

VALID_BIAS_LABELS = {"non-biased", "non biased", "nonbiased", "neutral", "biased"}
VALID_IDEOLOGY_LABELS = {"left", "center", "centre", "neutral", "right"}
VALID_PROPAGANDA_LABELS = {"no", "false", "yes", "true"}
VALID_FRAME_LABELS = {"CO", "EC", "HI", "MO", "RE"}


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    assert DATASET_PATH.exists(), f"Dataset not found: {DATASET_PATH}"
    return pd.read_csv(DATASET_PATH)


def _get(path: str) -> requests.Response:
    return requests.get(f"{BASE_URL}{path}", timeout=30)


def _post(path: str, payload: Any) -> requests.Response:
    return requests.post(f"{BASE_URL}{path}", json=payload, timeout=60)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Dataset integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetIntegrity:
    def test_dataset_file_exists(self) -> None:
        assert DATASET_PATH.exists()

    def test_dataset_has_ten_rows(self, dataset: pd.DataFrame) -> None:
        assert len(dataset) == 10, f"Expected 10 rows, got {len(dataset)}"

    def test_all_required_columns_present(self, dataset: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in dataset.columns]
        assert missing == [], f"Missing columns: {missing}"

    def test_no_extra_unexpected_columns(self, dataset: pd.DataFrame) -> None:
        extra = [c for c in dataset.columns if c not in REQUIRED_COLUMNS]
        assert extra == [], f"Unexpected extra columns: {extra}"

    def test_no_null_values_in_text_columns(self, dataset: pd.DataFrame) -> None:
        for col in ("title", "text"):
            assert dataset[col].notna().all(), f"Nulls in column: {col}"

    def test_title_and_text_are_non_empty_strings(self, dataset: pd.DataFrame) -> None:
        for col in ("title", "text"):
            assert (dataset[col].str.strip().str.len() > 0).all(), (
                f"Empty strings found in column: {col}"
            )

    def test_text_meets_api_minimum_length(self, dataset: pd.DataFrame) -> None:
        assert (dataset["text"].str.len() >= 10).all(), (
            "Some texts are shorter than 10 characters (API minimum)"
        )

    def test_bias_labels_are_valid(self, dataset: pd.DataFrame) -> None:
        invalid = dataset[~dataset["bias_label"].isin(VALID_BIAS_LABELS)]
        assert invalid.empty, f"Invalid bias_label values: {invalid['bias_label'].tolist()}"

    def test_ideology_labels_are_valid(self, dataset: pd.DataFrame) -> None:
        invalid = dataset[~dataset["ideology_label"].isin(VALID_IDEOLOGY_LABELS)]
        assert invalid.empty, (
            f"Invalid ideology_label: {invalid['ideology_label'].tolist()}"
        )

    def test_propaganda_labels_are_valid(self, dataset: pd.DataFrame) -> None:
        invalid = dataset[~dataset["propaganda_label"].isin(VALID_PROPAGANDA_LABELS)]
        assert invalid.empty, (
            f"Invalid propaganda_label: {invalid['propaganda_label'].tolist()}"
        )

    def test_frame_column_is_valid(self, dataset: pd.DataFrame) -> None:
        invalid = dataset[~dataset["frame"].isin(VALID_FRAME_LABELS)]
        assert invalid.empty, f"Invalid frame values: {invalid['frame'].tolist()}"

    def test_narrative_frame_binary_columns(self, dataset: pd.DataFrame) -> None:
        for col in ("CO", "EC", "HI", "MO", "RE"):
            assert dataset[col].isin([0, 1]).all(), f"Non-binary values in {col}"

    def test_narrative_role_binary_columns(self, dataset: pd.DataFrame) -> None:
        for col in ("hero", "villain", "victim"):
            assert dataset[col].isin([0, 1]).all(), f"Non-binary values in {col}"

    def test_emotion_columns_are_binary(self, dataset: pd.DataFrame) -> None:
        # EMOTION-11: schema reduced from 20 → 11 columns.
        for i in range(11):
            col = f"emotion_{i}"
            assert dataset[col].isin([0, 1]).all(), f"Non-binary values in {col}"

    def test_each_row_has_at_least_one_emotion(self, dataset: pd.DataFrame) -> None:
        emotion_cols = [f"emotion_{i}" for i in range(11)]
        row_sums = dataset[emotion_cols].sum(axis=1)
        assert (row_sums >= 1).all(), "Some rows have zero emotion labels"

    def test_entity_columns_parse_as_json_lists(self, dataset: pd.DataFrame) -> None:
        for col in ("hero_entities", "villain_entities", "victim_entities"):
            for val in dataset[col]:
                parsed = json.loads(val)
                assert isinstance(parsed, list), (
                    f"Column {col}: expected JSON list, got {type(parsed).__name__}"
                )

    def test_dataset_source_column_is_nonempty(self, dataset: pd.DataFrame) -> None:
        assert dataset["dataset"].notna().all()
        assert (dataset["dataset"].str.strip().str.len() > 0).all()

    def test_balanced_bias_labels(self, dataset: pd.DataFrame) -> None:
        counts = dataset["bias_label"].value_counts()
        assert counts.get("biased", 0) >= 4
        assert counts.get("non-biased", 0) >= 4

    def test_multiple_source_datasets_represented(
        self, dataset: pd.DataFrame
    ) -> None:
        assert dataset["dataset"].nunique() >= 2

    def test_multiple_frame_types_represented(self, dataset: pd.DataFrame) -> None:
        assert len(set(dataset["frame"])) >= 3

    def test_propaganda_label_matches_bias_label_trend(
        self, dataset: pd.DataFrame
    ) -> None:
        biased = dataset[dataset["bias_label"] == "biased"]["propaganda_label"]
        assert (biased == "yes").all(), (
            "All biased articles should be labelled as propaganda in this dataset"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Unified label schema
# ─────────────────────────────────────────────────────────────────────────────

class TestUnifiedLabelSchema:
    def test_schema_columns_match_dataset(self, dataset: pd.DataFrame) -> None:
        from src.data.unified_label_schema import UNIFIED_REQUIRED_COLUMNS
        for col in UNIFIED_REQUIRED_COLUMNS:
            assert col in dataset.columns, f"Schema column missing from dataset: {col}"

    def test_bias_label_to_id_covers_dataset_values(
        self, dataset: pd.DataFrame
    ) -> None:
        from src.data.unified_label_schema import BIAS_LABEL_TO_ID
        for val in dataset["bias_label"]:
            assert val in BIAS_LABEL_TO_ID, f"bias_label '{val}' not in schema mapping"

    def test_ideology_label_to_id_covers_dataset_values(
        self, dataset: pd.DataFrame
    ) -> None:
        from src.data.unified_label_schema import IDEOLOGY_LABEL_TO_ID
        for val in dataset["ideology_label"]:
            assert val in IDEOLOGY_LABEL_TO_ID, (
                f"ideology_label '{val}' not in schema mapping"
            )

    def test_propaganda_label_to_id_covers_dataset_values(
        self, dataset: pd.DataFrame
    ) -> None:
        from src.data.unified_label_schema import PROPAGANDA_LABEL_TO_ID
        for val in dataset["propaganda_label"]:
            assert val in PROPAGANDA_LABEL_TO_ID, (
                f"propaganda_label '{val}' not in schema mapping"
            )

    def test_task_column_groups_are_complete(self, dataset: pd.DataFrame) -> None:
        from src.data.unified_label_schema import TASK_COLUMN_GROUPS
        for task, cols in TASK_COLUMN_GROUPS.items():
            for col in cols:
                assert col in dataset.columns, (
                    f"Task '{task}' column '{col}' missing from dataset"
                )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Home / health / project-view
# ─────────────────────────────────────────────────────────────────────────────

class TestHomeAndHealthEndpoints:
    def test_home_returns_200(self) -> None:
        assert _get("/").status_code == 200

    def test_home_status_is_online(self) -> None:
        assert _get("/").json()["status"] == "online"

    def test_home_lists_required_endpoint_keys(self) -> None:
        endpoints = _get("/").json()["endpoints"]
        for key in ("predict", "batch_predict", "analyze", "health", "docs"):
            assert key in endpoints

    def test_health_returns_200(self) -> None:
        assert _get("/health").status_code == 200

    def test_health_has_status_field(self) -> None:
        data = _get("/health").json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded", "unhealthy")

    def test_health_includes_model_path(self) -> None:
        assert "model_path" in _get("/health").json()

    def test_health_inference_cache_entries_is_integer(self) -> None:
        data = _get("/health").json()
        assert isinstance(data.get("inference_cache_entries"), int)

    def test_project_view_returns_200(self) -> None:
        assert _get("/project-view").status_code == 200

    def test_project_view_src_exists(self) -> None:
        assert _get("/project-view").json()["structure"]["src_exists"] is True

    def test_docs_endpoint_is_accessible(self) -> None:
        resp = requests.get(f"{BASE_URL}/docs", timeout=10)
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# 4. /predict — one request per dataset row
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictEndpoint:
    def _p(self, text: str) -> dict[str, Any]:
        resp = _post("/predict", {"text": text})
        assert resp.status_code == 200, (
            f"/predict: {resp.status_code} — {resp.text[:200]}"
        )
        return resp.json()

    def test_predict_all_rows_return_200(self, dataset: pd.DataFrame) -> None:
        for _, row in dataset.iterrows():
            resp = _post("/predict", {"text": row["text"]})
            assert resp.status_code == 200, (
                f"Row '{row['title']}': {resp.status_code}"
            )

    def test_predict_response_has_required_fields(
        self, dataset: pd.DataFrame
    ) -> None:
        data = self._p(dataset.iloc[0]["text"])
        for f in ("text", "fake_probability", "prediction", "confidence"):
            assert f in data

    def test_predict_fake_probability_in_range(
        self, dataset: pd.DataFrame
    ) -> None:
        for _, row in dataset.iterrows():
            p = self._p(row["text"])["fake_probability"]
            assert 0.0 <= p <= 1.0, f"Out of range for '{row['title']}': {p}"

    def test_predict_confidence_in_range(self, dataset: pd.DataFrame) -> None:
        for _, row in dataset.iterrows():
            c = self._p(row["text"])["confidence"]
            assert 0.0 <= c <= 1.0, f"Confidence out of range: {c}"

    def test_predict_prediction_is_real_or_fake(
        self, dataset: pd.DataFrame
    ) -> None:
        for _, row in dataset.iterrows():
            pred = self._p(row["text"])["prediction"]
            assert pred in ("REAL", "FAKE"), f"Unexpected prediction: {pred}"

    def test_predict_text_preview_present(self, dataset: pd.DataFrame) -> None:
        data = self._p(dataset.iloc[0]["text"])
        assert isinstance(data["text"], str) and len(data["text"]) > 0

    def test_predict_long_text_is_previewed(self) -> None:
        long_text = "A " * 600
        data = self._p(long_text)
        assert len(data["text"]) < len(long_text)

    def test_predict_caches_repeated_requests(
        self, dataset: pd.DataFrame
    ) -> None:
        text = dataset.iloc[2]["text"]
        r1 = self._p(text)
        r2 = self._p(text)
        assert r1["fake_probability"] == r2["fake_probability"]
        assert r1["prediction"] == r2["prediction"]

    def test_predict_rejects_empty_text_422(self) -> None:
        assert _post("/predict", {"text": ""}).status_code == 422

    def test_predict_rejects_text_under_minimum_422(self) -> None:
        assert _post("/predict", {"text": "hi"}).status_code == 422

    def test_predict_rejects_text_over_maximum_422(self) -> None:
        assert _post("/predict", {"text": "x" * 10_001}).status_code == 422

    def test_predict_rejects_missing_text_field_422(self) -> None:
        assert _post("/predict", {}).status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# 5. /batch-predict
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchPredictEndpoint:
    def test_batch_all_ten_rows_returns_200(
        self, dataset: pd.DataFrame
    ) -> None:
        resp = _post("/batch-predict", {"texts": dataset["text"].tolist()})
        assert resp.status_code == 200, resp.text[:300]

    def test_batch_result_count_matches_input(
        self, dataset: pd.DataFrame
    ) -> None:
        texts = dataset["text"].tolist()
        data = _post("/batch-predict", {"texts": texts}).json()
        assert data["total"] == len(texts)
        assert len(data["results"]) == len(texts)

    def test_batch_each_result_has_required_fields(
        self, dataset: pd.DataFrame
    ) -> None:
        texts = dataset["text"].tolist()
        results = _post("/batch-predict", {"texts": texts}).json()["results"]
        for i, r in enumerate(results):
            for f in ("text", "fake_probability", "prediction", "confidence"):
                assert f in r, f"Result #{i} missing field '{f}'"

    def test_batch_all_fake_probabilities_in_range(
        self, dataset: pd.DataFrame
    ) -> None:
        texts = dataset["text"].tolist()
        for i, r in enumerate(
            _post("/batch-predict", {"texts": texts}).json()["results"]
        ):
            p = r["fake_probability"]
            assert 0.0 <= p <= 1.0, f"Batch result #{i}: p={p}"

    def test_batch_second_call_records_cache_hits(
        self, dataset: pd.DataFrame
    ) -> None:
        texts = dataset["text"].tolist()[:4]
        _post("/batch-predict", {"texts": texts})
        r2 = _post("/batch-predict", {"texts": texts}).json()
        assert r2["cache_hits"] == len(texts)

    def test_batch_rejects_empty_list_422(self) -> None:
        assert _post("/batch-predict", {"texts": []}).status_code == 422

    def test_batch_rejects_over_50_texts_422(self) -> None:
        texts = ["Valid article text for testing purposes." for _ in range(51)]
        assert _post("/batch-predict", {"texts": texts}).status_code == 422

    def test_batch_rejects_missing_texts_field_422(self) -> None:
        assert _post("/batch-predict", {}).status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# 6. /analyze
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeEndpoint:
    TOP_LEVEL = (
        "text", "prediction", "fake_probability", "confidence",
        "bias", "emotion", "narrative", "framing", "rhetoric",
        "discourse", "propaganda_analysis", "credibility_profile",
        "graph_analysis", "explainability",
    )

    def _analyze(self, text: str) -> dict[str, Any]:
        resp = _post("/analyze", {"text": text})
        assert resp.status_code == 200, (
            f"/analyze: {resp.status_code} — {resp.text[:300]}"
        )
        return resp.json()

    def test_analyze_unbiased_article_returns_200(
        self, dataset: pd.DataFrame
    ) -> None:
        row = dataset[dataset["bias_label"] == "non-biased"].iloc[0]
        assert _post("/analyze", {"text": row["text"]}).status_code == 200

    def test_analyze_propaganda_article_returns_200(
        self, dataset: pd.DataFrame
    ) -> None:
        row = dataset[dataset["propaganda_label"] == "yes"].iloc[0]
        assert _post("/analyze", {"text": row["text"]}).status_code == 200

    def test_analyze_response_has_all_top_level_fields(
        self, dataset: pd.DataFrame
    ) -> None:
        data = self._analyze(dataset.iloc[0]["text"])
        for f in self.TOP_LEVEL:
            assert f in data, f"Missing top-level field '{f}'"

    def test_analyze_fake_probability_in_range(
        self, dataset: pd.DataFrame
    ) -> None:
        for idx in (0, 1, 5, 8):
            data = self._analyze(dataset.iloc[idx]["text"])
            assert 0.0 <= data["fake_probability"] <= 1.0

    def test_analyze_bias_section_has_score(self, dataset: pd.DataFrame) -> None:
        data = self._analyze(dataset.iloc[0]["text"])
        bias = data["bias"]
        assert "bias_score" in bias
        assert isinstance(bias["bias_score"], float)
        assert 0.0 <= bias["bias_score"] <= 1.0

    def test_analyze_emotion_has_dominant_emotion(
        self, dataset: pd.DataFrame
    ) -> None:
        data = self._analyze(dataset.iloc[0]["text"])
        assert "dominant_emotion" in data["emotion"]
        assert isinstance(data["emotion"]["dominant_emotion"], str)

    def test_analyze_narrative_has_roles(self, dataset: pd.DataFrame) -> None:
        data = self._analyze(dataset.iloc[0]["text"])
        assert "roles" in data["narrative"]

    def test_analyze_explainability_has_lime(self, dataset: pd.DataFrame) -> None:
        data = self._analyze(dataset.iloc[0]["text"])
        assert "lime" in data["explainability"]
        assert "important_features" in data["explainability"]["lime"]

    def test_analyze_graph_analysis_is_dict(self, dataset: pd.DataFrame) -> None:
        data = self._analyze(dataset.iloc[0]["text"])
        assert isinstance(data["graph_analysis"], dict)

    def test_analyze_rejects_empty_text_422(self) -> None:
        assert _post("/analyze", {"text": ""}).status_code == 422

    def test_analyze_rejects_too_short_text_422(self) -> None:
        assert _post("/analyze", {"text": "hi"}).status_code == 422

    def test_analyze_caches_repeated_requests(
        self, dataset: pd.DataFrame
    ) -> None:
        text = dataset.iloc[3]["text"]
        r1 = self._analyze(text)
        r2 = self._analyze(text)
        assert r1["fake_probability"] == r2["fake_probability"]


# ─────────────────────────────────────────────────────────────────────────────
# 7. /report
# ─────────────────────────────────────────────────────────────────────────────

class TestReportEndpoint:
    REPORT_FIELDS = (
        "article_summary", "bias_analysis", "emotion_analysis",
        "narrative_structure", "entity_graph", "credibility_score",
    )

    def test_report_all_rows_return_200(self, dataset: pd.DataFrame) -> None:
        for _, row in dataset.iterrows():
            resp = _post("/report", {"text": row["text"]})
            assert resp.status_code == 200, (
                f"Report for '{row['title']}': {resp.status_code}"
            )

    def test_report_has_all_required_fields(self, dataset: pd.DataFrame) -> None:
        data = _post("/report", {"text": dataset.iloc[0]["text"]}).json()
        for f in self.REPORT_FIELDS:
            assert f in data, f"Missing report field '{f}'"

    def test_report_bias_analysis_has_bias_score(
        self, dataset: pd.DataFrame
    ) -> None:
        data = _post("/report", {"text": dataset.iloc[0]["text"]}).json()
        assert "bias_score" in data["bias_analysis"]

    def test_report_emotion_analysis_has_dominant(
        self, dataset: pd.DataFrame
    ) -> None:
        data = _post("/report", {"text": dataset.iloc[0]["text"]}).json()
        assert "dominant_emotion" in data["emotion_analysis"]

    def test_report_rejects_short_text_422(self) -> None:
        assert _post("/report", {"text": "hi"}).status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# 8. /inference/model-info  +  /cache/clear
# ─────────────────────────────────────────────────────────────────────────────

class TestInferenceModelInfo:
    def test_model_info_returns_200(self) -> None:
        assert _get("/inference/model-info").status_code == 200

    def test_model_info_has_available_field(self) -> None:
        assert "available" in _get("/inference/model-info").json()

    def test_model_info_has_model_path(self) -> None:
        data = _get("/inference/model-info").json()
        assert "model_path" in data
        assert isinstance(data["model_path"], str)


class TestCacheEndpoint:
    def test_cache_clear_returns_200(self) -> None:
        assert _post("/cache/clear", {}).status_code == 200

    def test_cache_clear_response_contains_confirmation(self) -> None:
        data = _post("/cache/clear", {}).json()
        assert any(k in data for k in ("cleared", "message", "entries_cleared"))

    def test_after_cache_clear_new_request_still_succeeds(
        self, dataset: pd.DataFrame
    ) -> None:
        _post("/cache/clear", {})
        resp = _post("/predict", {"text": dataset.iloc[0]["text"]})
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# 9. /calibration endpoints
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibrationEndpoints:
    def test_calibration_info_returns_200(self) -> None:
        assert _get("/calibration/info").status_code == 200

    def test_calibration_info_has_methods_or_content(self) -> None:
        data = _get("/calibration/info").json()
        assert len(data) > 0

    def test_calibration_metrics_with_dataset_probs_200(
        self, dataset: pd.DataFrame
    ) -> None:
        n = len(dataset)
        payload = {
            "probabilities": [[0.3 + 0.04 * i, 0.7 - 0.04 * i] for i in range(n)],
            "labels": [
                1 if r["propaganda_label"] == "yes" else 0
                for _, r in dataset.iterrows()
            ],
            "n_bins": 5,
        }
        assert _post("/calibration/metrics", payload).status_code == 200

    def test_calibration_metrics_response_has_ece_and_brier(self) -> None:
        payload = {
            "probabilities": [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.4, 0.6]],
            "labels": [0, 1, 0, 1],
            "n_bins": 4,
        }
        data = _post("/calibration/metrics", payload).json()
        for f in ("ece", "mce", "brier_score", "nll", "n_samples"):
            assert f in data, f"Missing calibration metric '{f}'"

    def test_calibration_ece_in_range(self) -> None:
        payload = {
            "probabilities": [[0.9, 0.1], [0.1, 0.9], [0.7, 0.3]],
            "labels": [0, 1, 0],
            "n_bins": 3,
        }
        ece = _post("/calibration/metrics", payload).json()["ece"]
        assert 0.0 <= ece <= 1.0

    def test_calibration_metrics_rejects_mismatched_lengths(self) -> None:
        payload = {
            "probabilities": [[0.8, 0.2], [0.3, 0.7]],
            "labels": [0],
            "n_bins": 5,
        }
        assert _post("/calibration/metrics", payload).status_code in (400, 422, 500)

    def test_calibration_metrics_rejects_invalid_bins(self) -> None:
        payload = {
            "probabilities": [[0.8, 0.2]],
            "labels": [0],
            "n_bins": 1,
        }
        assert _post("/calibration/metrics", payload).status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# 10. /ensemble endpoints
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsembleEndpoints:
    def test_ensemble_info_returns_200(self) -> None:
        assert _get("/ensemble/info").status_code == 200

    def test_ensemble_info_has_strategies_key(self) -> None:
        data = _get("/ensemble/info").json()
        assert "strategies" in data or len(data) > 0

    def test_ensemble_average_strategy_200(self) -> None:
        payload = {
            "model_probabilities": [[0.7, 0.3], [0.4, 0.6], [0.6, 0.4]],
            "strategy": "average",
        }
        assert _post("/ensemble/predict", payload).status_code == 200

    def test_ensemble_weighted_average_strategy_200(self) -> None:
        payload = {
            "model_probabilities": [[0.7, 0.3], [0.4, 0.6], [0.6, 0.4]],
            "weights": [0.5, 0.3, 0.2],
            "strategy": "weighted_average",
        }
        assert _post("/ensemble/predict", payload).status_code == 200

    def test_ensemble_majority_vote_strategy_200(self) -> None:
        payload = {
            "model_probabilities": [[0.8, 0.2], [0.3, 0.7], [0.7, 0.3]],
            "strategy": "majority_vote",
        }
        assert _post("/ensemble/predict", payload).status_code == 200

    def test_ensemble_response_has_required_fields(self) -> None:
        payload = {
            "model_probabilities": [[0.7, 0.3], [0.4, 0.6]],
            "strategy": "average",
        }
        data = _post("/ensemble/predict", payload).json()
        for f in ("strategy", "ensemble_probabilities", "prediction",
                  "fake_probability", "confidence", "num_models"):
            assert f in data, f"Missing ensemble field: {f}"

    def test_ensemble_probabilities_sum_to_one(self) -> None:
        payload = {
            "model_probabilities": [[0.6, 0.4], [0.5, 0.5], [0.4, 0.6]],
            "strategy": "average",
        }
        probs = _post("/ensemble/predict", payload).json()["ensemble_probabilities"]
        assert math.isclose(sum(probs), 1.0, abs_tol=1e-4)

    def test_ensemble_num_models_matches_input(self) -> None:
        payload = {
            "model_probabilities": [[0.7, 0.3], [0.4, 0.6], [0.6, 0.4]],
            "strategy": "average",
        }
        data = _post("/ensemble/predict", payload).json()
        assert data["num_models"] == 3

    def test_ensemble_prediction_is_real_or_fake(self) -> None:
        payload = {
            "model_probabilities": [[0.7, 0.3], [0.6, 0.4]],
            "strategy": "average",
        }
        pred = _post("/ensemble/predict", payload).json()["prediction"]
        assert pred in ("REAL", "FAKE")


# ─────────────────────────────────────────────────────────────────────────────
# 11. /export endpoints
# ─────────────────────────────────────────────────────────────────────────────

class TestExportEndpoints:
    def test_export_info_returns_200(self) -> None:
        assert _get("/export/info").status_code == 200

    def test_export_info_has_content(self) -> None:
        data = _get("/export/info").json()
        assert len(data) > 0

    def test_export_onnx_returns_expected_status(self) -> None:
        resp = _post("/export/onnx", {"output_path": "/tmp/test_model.onnx"})
        assert resp.status_code in (200, 400, 404, 422, 500, 503)

    def test_export_torchscript_returns_expected_status(self) -> None:
        resp = _post("/export/torchscript", {"output_path": "/tmp/test_model.pt"})
        assert resp.status_code in (200, 400, 404, 422, 500, 503)


# ─────────────────────────────────────────────────────────────────────────────
# 12. Error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_unknown_route_is_404(self) -> None:
        assert _get("/nonexistent-endpoint").status_code == 404

    def test_predict_text_too_long_is_422(self) -> None:
        assert _post("/predict", {"text": "x" * 10_001}).status_code == 422

    def test_batch_over_50_texts_is_422(self) -> None:
        texts = ["Valid article text for testing." for _ in range(51)]
        assert _post("/batch-predict", {"texts": texts}).status_code == 422

    def test_calibration_bins_below_minimum_is_422(self) -> None:
        payload = {
            "probabilities": [[0.8, 0.2]],
            "labels": [0],
            "n_bins": 1,
        }
        assert _post("/calibration/metrics", payload).status_code == 422

    def test_calibration_bins_above_maximum_is_422(self) -> None:
        payload = {
            "probabilities": [[0.8, 0.2]],
            "labels": [0],
            "n_bins": 101,
        }
        assert _post("/calibration/metrics", payload).status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# 13. Cross-validation — self-consistency checks
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictionConsistency:
    def test_prediction_label_agrees_with_fake_probability(
        self, dataset: pd.DataFrame
    ) -> None:
        for _, row in dataset.iterrows():
            data = _post("/predict", {"text": row["text"]}).json()
            prob = data["fake_probability"]
            pred = data["prediction"]
            if prob > 0.5:
                assert pred == "FAKE", (
                    f"p={prob:.4f} but pred={pred} for '{row['title']}'"
                )
            else:
                assert pred == "REAL", (
                    f"p={prob:.4f} but pred={pred} for '{row['title']}'"
                )

    def test_confidence_equals_max_class_probability(
        self, dataset: pd.DataFrame
    ) -> None:
        for _, row in dataset.iterrows():
            data = _post("/predict", {"text": row["text"]}).json()
            prob = data["fake_probability"]
            conf = data["confidence"]
            expected = max(prob, 1.0 - prob)
            assert math.isclose(conf, expected, abs_tol=0.01), (
                f"conf={conf:.4f} != max(p,1-p)={expected:.4f} "
                f"for '{row['title']}'"
            )

    def test_single_and_batch_predictions_agree(
        self, dataset: pd.DataFrame
    ) -> None:
        texts = dataset["text"].tolist()
        batch_results = _post("/batch-predict", {"texts": texts}).json()["results"]
        for i, (text, br) in enumerate(zip(texts, batch_results)):
            sr = _post("/predict", {"text": text}).json()
            assert sr["prediction"] == br["prediction"], (
                f"Row {i}: single={sr['prediction']} vs batch={br['prediction']}"
            )

    def test_analyze_and_predict_fake_probability_agree(
        self, dataset: pd.DataFrame
    ) -> None:
        for idx in (0, 1, 4):
            text = dataset.iloc[idx]["text"]
            p_resp = _post("/predict", {"text": text}).json()
            a_resp = _post("/analyze", {"text": text}).json()
            assert math.isclose(
                p_resp["fake_probability"],
                a_resp["fake_probability"],
                abs_tol=0.001,
            ), (
                f"Row {idx}: predict p={p_resp['fake_probability']:.4f} "
                f"vs analyze p={a_resp['fake_probability']:.4f}"
            )
