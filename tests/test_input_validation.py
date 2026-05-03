from __future__ import annotations

import pandas as pd
import pytest

from src.utils.input_validation import (
    ensure_dataframe,
    ensure_non_empty_text,
    ensure_non_empty_text_column,
    ensure_non_empty_text_list,
    ensure_positive_int,
)


class TestEnsurePositiveInt:
    def test_valid_value_is_returned(self) -> None:
        assert ensure_positive_int(5, name="value", min_value=1) == 5

    def test_zero_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            ensure_positive_int(0, name="value", min_value=1)

    def test_negative_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            ensure_positive_int(-3, name="value", min_value=1)

    def test_float_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            ensure_positive_int(1.5, name="value", min_value=1)  # type: ignore[arg-type]

    def test_bool_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            ensure_positive_int(True, name="value", min_value=1)  # type: ignore[arg-type]

    def test_custom_min_value(self) -> None:
        assert ensure_positive_int(10, name="value", min_value=10) == 10

    def test_below_custom_min_raises(self) -> None:
        with pytest.raises(ValueError):
            ensure_positive_int(9, name="value", min_value=10)


class TestEnsureDataframe:
    def test_valid_dataframe_passes(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        ensure_dataframe(df, required_columns=["a"])

    def test_non_dataframe_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            ensure_dataframe([1, 2], name="df")  # type: ignore[arg-type]

    def test_missing_required_column_raises_value_error(self) -> None:
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="missing required columns"):
            ensure_dataframe(df, required_columns=["a", "b"])

    def test_empty_dataframe_raises_value_error(self) -> None:
        df = pd.DataFrame({"a": []})
        with pytest.raises(ValueError, match="at least 1 row"):
            ensure_dataframe(df)

    def test_min_rows_respected(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError):
            ensure_dataframe(df, min_rows=5)


class TestEnsureNonEmptyText:
    def test_valid_text_is_returned(self) -> None:
        assert ensure_non_empty_text("hello world") == "hello world"

    def test_whitespace_only_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            ensure_non_empty_text("   ")

    def test_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            ensure_non_empty_text("")

    def test_non_string_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            ensure_non_empty_text(42)  # type: ignore[arg-type]


class TestEnsureNonEmptyTextColumn:
    def test_valid_text_column_passes(self) -> None:
        df = pd.DataFrame({"text": ["hello", "world"]})
        ensure_non_empty_text_column(df, "text")

    def test_missing_column_raises_value_error(self) -> None:
        df = pd.DataFrame({"other": ["hello"]})
        with pytest.raises(ValueError, match="does not contain text column"):
            ensure_non_empty_text_column(df, "text")

    def test_all_empty_strings_raises_value_error(self) -> None:
        df = pd.DataFrame({"text": ["", "  ", ""]})
        with pytest.raises(ValueError, match="cannot be entirely empty"):
            ensure_non_empty_text_column(df, "text")


class TestEnsureNonEmptyTextList:
    def test_valid_list_is_returned(self) -> None:
        result = ensure_non_empty_text_list(["hello", "world"])
        assert result == ["hello", "world"]

    def test_empty_list_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            ensure_non_empty_text_list([])

    def test_none_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            ensure_non_empty_text_list(None)  # type: ignore[arg-type]

    def test_all_whitespace_items_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            ensure_non_empty_text_list(["  ", "", "   "])

    def test_single_string_is_wrapped(self) -> None:
        result = ensure_non_empty_text_list("hello")
        assert isinstance(result, list)
        assert result[0] == "hello"
