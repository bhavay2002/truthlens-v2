import pytest

from src.explainability.token_alignment import align_tokens


def test_wordpiece_merge():
    mt, ms = align_tokens(["play", "##ing"], [0.2, 0.4], "wordpiece")
    assert mt == ["playing"]
    assert ms == [0.3]


def test_invalid_tokenizer_type():
    with pytest.raises(ValueError):
        align_tokens(["a"], [0.1], "bpe")
