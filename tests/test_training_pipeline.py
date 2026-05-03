from __future__ import annotations

import pandas as pd

from src.training.cross_validation import cross_validate_model
from src.training.hyperparameter_tuning import run_optuna


class _DummyTrainer:
    """
    Minimal trainer used to simulate model training behaviour.
    """

    def __init__(self, score: float) -> None:
        self._score = score

    def evaluate(self, _dataset):
        return {
            "eval_loss": self._score,
            "eval_accuracy": 1.0 - self._score,
        }


def _dummy_train_function(
    train_df: pd.DataFrame,
    *,
    params=None,
    text_column: str = "text",
    validation_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
):
    """
    Fake training function used for testing cross-validation
    and hyperparameter tuning without heavy model training.
    """

    _ = text_column
    _ = test_df

    effective_eval_df = validation_df if validation_df is not None else train_df

    epochs = int((params or {}).get("epochs", 1))

    score = (1.0 / max(len(train_df), 1)) + (epochs * 0.0001)

    return _DummyTrainer(score), effective_eval_df


def _sample_df(rows: int = 20) -> pd.DataFrame:
    """
    Generate a synthetic dataset for testing.
    """

    texts = [f"sample text {i}" for i in range(rows)]
    labels = [i % 2 for i in range(rows)]

    return pd.DataFrame({
        "text": texts,
        "label": labels,
    })


def _sample_unified_df(rows: int = 20) -> pd.DataFrame:
    texts = [f"sample text {i}" for i in range(rows)]
    bias_labels = [i % 2 for i in range(rows)]

    return pd.DataFrame(
        {
            "title": ["" for _ in range(rows)],
            "text": texts,
            "bias_label": bias_labels,
            "dataset": ["unit_test" for _ in range(rows)],
        }
    )


# ---------------------------------------------------------
# Cross Validation Tests
# ---------------------------------------------------------

def test_cross_validate_model_returns_summary():

    df = _sample_df(24)

    results = cross_validate_model(
        df,
        _dummy_train_function,
        n_splits=3,
        text_column="text",
        metric_name="eval_loss",
    )

    assert results["metric_name"] == "eval_loss"
    assert results["n_splits"] == 3
    assert len(results["fold_scores"]) == 3
    assert isinstance(results["mean_score"], float)
    assert isinstance(results["std_score"], float)


# ---------------------------------------------------------
# Optuna Hyperparameter Tuning Tests
# ---------------------------------------------------------

def test_run_optuna_accepts_dataframe():

    df = _sample_df(30)

    results = run_optuna(
        df,
        train_function=_dummy_train_function,
        text_column="text",
        n_trials=2,
        metric_name="eval_loss",
        direction="minimize",
    )

    assert "best_params" in results
    assert "best_value" in results
    assert isinstance(results["best_params"], dict)
    assert isinstance(results["best_value"], float)


def test_cross_validate_model_supports_unified_label_column():
    df = _sample_unified_df(24)

    results = cross_validate_model(
        df,
        _dummy_train_function,
        n_splits=3,
        text_column="text",
        label_column="bias_label",
        metric_name="eval_loss",
    )

    assert results["label_column"] == "bias_label"
    assert len(results["fold_scores"]) == 3


def test_run_optuna_supports_unified_label_column():
    df = _sample_unified_df(30)

    results = run_optuna(
        df,
        train_function=_dummy_train_function,
        text_column="text",
        label_column="bias_label",
        n_trials=2,
        metric_name="eval_loss",
        direction="minimize",
    )

    assert results["label_column"] == "bias_label"
    assert isinstance(results["best_value"], float)
