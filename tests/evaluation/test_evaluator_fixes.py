import numpy as np
import pytest

from src.evaluation.evaluator import Evaluator
from src.evaluation.evaluate_model import evaluate
from src.evaluation.task_correlation import compute_task_correlation


def test_feature_importance_ablation_no_nameerror():
    class DummyModel:
        def predict(self, X):
            return np.zeros(X.shape[0], dtype=int)

    X = np.random.rand(6, 3)
    y = np.array([0, 1, 0, 1, 0, 1])
    feature_names = ["f1", "f2", "f3"]

    try:
        Evaluator.feature_importance_ablation(
            model=DummyModel(),
            X=X,
            y=y,
            feature_names=feature_names,
        )
    except NameError as exc:
        pytest.fail(f"Unexpected NameError: {exc}")


def test_feature_importance_shap_rejects_zero_max_samples():
    class DummyModel:
        def predict(self, X):
            return np.zeros(X.shape[0], dtype=int)

    X = np.random.rand(5, 2)
    with pytest.raises(ValueError):
        Evaluator.feature_importance_shap(
            model=DummyModel(),
            X=X,
            feature_names=["a", "b"],
            max_samples=0,
        )


def test_evaluate_checks_y_proba_length():
    y_true = [0, 1, 0]
    y_pred = [0, 1, 1]
    y_proba = [0.2, 0.9]
    with pytest.raises(ValueError):
        evaluate(y_true, y_pred, y_proba=y_proba)


def test_task_correlation_rejects_non_numeric_only():
    preds = {"bias": ["a", "b", "c"], "frame": ["x", "y", "z"]}
    with pytest.raises(RuntimeError):
        compute_task_correlation(preds)
