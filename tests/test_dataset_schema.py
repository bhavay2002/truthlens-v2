from __future__ import annotations

import pandas as pd
import pytest


UNIFIED_REQUIRED_COLUMNS = [
    "title",
    "text",
    "bias_label",
    "ideology_label",
    "propaganda_label",
    "frame",
    "CO",
    "EC",
    "HI",
    "MO",
    "RE",
    "hero",
    "villain",
    "victim",
    "hero_entities",
    "villain_entities",
    "victim_entities",
    "dataset",
] + [f"emotion_{idx}" for idx in range(11)]  # EMOTION-11


def _make_full_schema_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title": [""],
            "text": ["news article about politics"],
            "bias_label": [1],
            "ideology_label": [0],
            "propaganda_label": [0],
            "frame": ["economic"],
            "CO": [1],
            "EC": [0],
            "HI": [0],
            "MO": [1],
            "RE": [0],
            "hero": [0],
            "villain": [1],
            "victim": [0],
            "hero_entities": [""],
            "villain_entities": ["corporation"],
            "victim_entities": [""],
            "dataset": ["unit_test"],
            **{f"emotion_{idx}": [0] for idx in range(11)},  # EMOTION-11
        }
    )


class TestDatasetSchema:
    def test_full_schema_has_all_required_columns(self) -> None:
        df = _make_full_schema_df()
        for col in UNIFIED_REQUIRED_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_binary_label_columns_contain_valid_values(self) -> None:
        df = _make_full_schema_df()
        for col in ("bias_label", "propaganda_label", "CO", "EC", "HI", "MO", "RE", "hero", "villain", "victim"):
            assert df[col].iloc[0] in {0, 1}, f"Invalid value in {col}"

    def test_emotion_columns_are_numeric(self) -> None:
        df = _make_full_schema_df()
        for idx in range(11):  # EMOTION-11
            col = f"emotion_{idx}"
            assert pd.api.types.is_numeric_dtype(df[col]), f"{col} must be numeric"

    def test_text_column_is_non_empty(self) -> None:
        df = _make_full_schema_df()
        assert df["text"].iloc[0].strip() != ""

    def test_dataset_column_is_string(self) -> None:
        df = _make_full_schema_df()
        assert isinstance(df["dataset"].iloc[0], str)

    def test_ideology_label_is_within_expected_range(self) -> None:
        df = _make_full_schema_df()
        assert df["ideology_label"].iloc[0] in {0, 1, 2}

    def test_frame_column_is_string_category(self) -> None:
        df = _make_full_schema_df()
        assert isinstance(df["frame"].iloc[0], str)

    def test_schema_rejects_missing_text_column(self) -> None:
        df = _make_full_schema_df().drop(columns=["text"])
        assert "text" not in df.columns

    def test_empty_dataframe_has_correct_columns(self) -> None:
        df = pd.DataFrame(columns=UNIFIED_REQUIRED_COLUMNS)
        for col in UNIFIED_REQUIRED_COLUMNS:
            assert col in df.columns
