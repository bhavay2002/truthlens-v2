import pytest

from src.explainability.utils_validation import validate_tokens_scores


def test_validate_tokens_scores_ok():
    validate_tokens_scores(["a", "b"], [0.1, 0.2])


def test_validate_tokens_scores_nan():
    with pytest.raises(ValueError):
        validate_tokens_scores(["a"], [float("nan")])


def test_validate_tokens_scores_mismatch():
    with pytest.raises(ValueError):
        validate_tokens_scores(["a"], [0.1, 0.2])
