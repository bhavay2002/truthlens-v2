from __future__ import annotations

from src.pipelines.baseline_training import tokenize_function


class _DummyTokenizer:
    def __call__(self, text, truncation, padding, max_length):
        assert truncation is True
        assert padding == "max_length"
        assert max_length > 0
        return {
            "input_ids": [101, 200, 102],
            "attention_mask": [1, 1, 1],
        }


def test_tokenize_function_returns_expected_keys() -> None:
    output = tokenize_function(
        {"text": "This is a news article."},
        tokenizer=_DummyTokenizer(),
        text_column="text",
    )

    assert set(output.keys()) == {"input_ids", "attention_mask"}
    assert len(output["input_ids"]) == len(output["attention_mask"])
