from __future__ import annotations

import pytest

from src.evaluation.evaluate_model import evaluate
from src.evaluation.metrics_engine import compute_classification_metrics


class TestEvaluateFunction:
    def test_binary_evaluate_returns_accuracy(self) -> None:
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 0]
        results = evaluate(y_true, y_pred)
        assert "accuracy" in results
        assert 0 <= results["accuracy"] <= 1

    def test_multiclass_evaluate_returns_expected_fields(self) -> None:
        y_true = [0, 1, 2, 1, 2, 0]
        y_pred = [0, 2, 2, 1, 0, 0]
        results = evaluate(y_true, y_pred)
        assert results["metric_average"] == "macro"
        assert len(results["confusion_matrix"]) == 3
        assert results["dataset_stats"]["num_classes"] == 3
        assert "class_counts" in results["dataset_stats"]

    def test_perfect_predictions_give_accuracy_one(self) -> None:
        y_true = [0, 1, 0, 1, 1]
        y_pred = [0, 1, 0, 1, 1]
        results = evaluate(y_true, y_pred)
        assert results["accuracy"] == pytest.approx(1.0)

    def test_all_wrong_predictions_give_accuracy_zero(self) -> None:
        y_true = [0, 0, 0, 0]
        y_pred = [1, 1, 1, 1]
        results = evaluate(y_true, y_pred)
        assert results["accuracy"] == pytest.approx(0.0)


class TestComputeClassificationMetricsIntegration:
    def test_binary_metrics_keys(self) -> None:
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 1]
        result = compute_classification_metrics(y_true, y_pred)
        assert "accuracy" in result
        assert "f1" in result
        assert "precision" in result
        assert "recall" in result
        assert "mcc" in result
        assert "confusion_matrix" in result

    def test_balanced_accuracy_within_range(self) -> None:
        y_true = [0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 0, 1]
        result = compute_classification_metrics(y_true, y_pred)
        assert 0.0 <= result["balanced_accuracy"] <= 1.0

    def test_roc_auc_returned_with_probabilities(self) -> None:
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 1]
        y_proba = [0.1, 0.9, 0.2, 0.8]
        result = compute_classification_metrics(y_true, y_pred, y_proba=y_proba)
        assert "roc_auc" in result
        assert 0.0 <= result["roc_auc"] <= 1.0

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises((ValueError, RuntimeError)):
            compute_classification_metrics([0, 1, 0], [1, 0])
