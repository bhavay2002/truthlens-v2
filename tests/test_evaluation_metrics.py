from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics_engine import (
    compute_classification_metrics,
    compute_multilabel_metrics,
)
from src.evaluation.uncertainty import (
    confidence_scores,
    predictive_entropy,
    uncertainty_statistics,
)


class TestComputeClassificationMetrics:
    def test_binary_returns_expected_keys(self) -> None:
        y_true = [0, 1, 0, 1, 1]
        y_pred = [0, 1, 0, 0, 1]
        result = compute_classification_metrics(y_true, y_pred)
        for key in ("accuracy", "precision", "recall", "f1", "mcc", "confusion_matrix"):
            assert key in result

    def test_perfect_prediction_gives_accuracy_one(self) -> None:
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 1]
        result = compute_classification_metrics(y_true, y_pred)
        assert result["accuracy"] == pytest.approx(1.0)

    def test_multiclass_returns_metrics(self) -> None:
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 2, 0, 1, 1]
        result = compute_classification_metrics(y_true, y_pred)
        assert 0.0 <= result["accuracy"] <= 1.0
        assert len(result["confusion_matrix"]) == 3

    def test_with_proba_binary_returns_roc_auc(self) -> None:
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 1]
        y_proba = [0.1, 0.9, 0.2, 0.8]
        result = compute_classification_metrics(y_true, y_pred, y_proba=y_proba)
        assert "roc_auc" in result
        assert 0.0 <= result["roc_auc"] <= 1.0

    def test_shape_mismatch_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_classification_metrics([0, 1, 0], [1, 0])

    def test_empty_input_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_classification_metrics([], [])

    def test_accuracy_within_range(self) -> None:
        y_true = [0, 1, 0, 1, 1, 0]
        y_pred = [1, 1, 0, 0, 1, 0]
        result = compute_classification_metrics(y_true, y_pred)
        assert 0.0 <= result["accuracy"] <= 1.0


class TestComputeMultilabelMetrics:
    def _make_labels(self, n: int = 4, k: int = 3) -> tuple:
        y_true = np.eye(n, k, dtype=int).tolist()
        y_pred = np.eye(n, k, dtype=int).tolist()
        return y_true, y_pred

    def test_perfect_multilabel_returns_expected_keys(self) -> None:
        y_true, y_pred = self._make_labels()
        result = compute_multilabel_metrics(y_true, y_pred)
        for key in (
            "subset_accuracy",
            "element_accuracy",
            "f1_micro",
            "f1_macro",
            "hamming_loss",
        ):
            assert key in result

    def test_perfect_multilabel_subset_accuracy_is_one(self) -> None:
        y_true, y_pred = self._make_labels()
        result = compute_multilabel_metrics(y_true, y_pred)
        assert result["subset_accuracy"] == pytest.approx(1.0)

    def test_shape_mismatch_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_multilabel_metrics([[0, 1], [1, 0]], [[0, 1]])


class TestPredictiveEntropy:
    def _uniform_probs(self, n: int = 3) -> list:
        p = 1.0 / n
        return [[p] * n, [p] * n]

    def test_returns_array_with_correct_length(self) -> None:
        probs = [[0.8, 0.2], [0.6, 0.4]]
        result = predictive_entropy(probs)
        assert len(result) == 2

    def test_uniform_distribution_has_highest_entropy(self) -> None:
        uniform = self._uniform_probs(n=4)
        skewed = [[0.99, 0.01, 0.0, 0.0], [0.95, 0.05, 0.0, 0.0]]
        e_uniform = np.mean(predictive_entropy(uniform))
        e_skewed = np.mean(predictive_entropy(skewed))
        assert e_uniform > e_skewed

    def test_non_2d_input_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            predictive_entropy([0.5, 0.5])

    def test_empty_array_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            predictive_entropy(np.empty((0, 2)))


class TestConfidenceScores:
    def test_returns_max_probability_per_row(self) -> None:
        probs = [[0.8, 0.2], [0.3, 0.7]]
        result = confidence_scores(probs)
        assert result[0] == pytest.approx(0.8)
        assert result[1] == pytest.approx(0.7)

    def test_array_length_matches_input(self) -> None:
        probs = [[0.5, 0.5], [0.9, 0.1], [0.6, 0.4]]
        result = confidence_scores(probs)
        assert len(result) == 3


class TestUncertaintyStatistics:
    def test_returns_expected_stats_keys(self) -> None:
        probs = [[0.8, 0.2], [0.6, 0.4], [0.5, 0.5]]
        result = uncertainty_statistics(probs)
        for key in (
            "mean_entropy",
            "std_entropy",
            "max_entropy",
            "min_entropy",
            "mean_confidence",
            "std_confidence",
            "max_confidence",
            "min_confidence",
        ):
            assert key in result

    def test_confidence_stats_are_in_valid_range(self) -> None:
        probs = [[0.9, 0.1], [0.7, 0.3], [0.6, 0.4]]
        result = uncertainty_statistics(probs)
        assert 0.0 <= result["min_confidence"] <= result["max_confidence"] <= 1.0

    def test_entropy_values_are_non_negative(self) -> None:
        probs = [[0.8, 0.2], [0.5, 0.5]]
        result = uncertainty_statistics(probs)
        assert result["min_entropy"] >= 0.0
