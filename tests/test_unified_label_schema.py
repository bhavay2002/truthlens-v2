from __future__ import annotations

import pandas as pd
import pytest

from src.data.unified_label_schema import (
    EMOTION_COLUMNS,
    NARRATIVE_FRAME_COLUMNS,
    UNIFIED_REQUIRED_COLUMNS,
    normalize_unified_labels,
    validate_unified_labels,
)


def test_normalize_unified_labels_maps_expected_values_and_aliases() -> None:
    df = pd.DataFrame(
        {
            "text": ["sample one", "sample two"],
            "bias": ["Non-biased", "Biased"],
            "ideology": ["Left", "Right"],
            "propaganda": ["No", "Yes"],
            "frame": ["economic", "security"],
            "co": [1, 0],
            "ec": [0, 1],
            "hi": ["yes", "no"],
            "mo": ["false", "true"],
            "re": [True, False],
            "narrative_hero": [1, 0],
            "narrative_villain": [0, 1],
            "narrative_victim": [1, 0],
            "hero_entities": ["citizens", "government"],
            "villain_entities": ["none", "opposition"],
            "victim_entities": ["workers", "voters"],
            "dataset_source": ["babe", "allsides"],
            "emotion": [3, pd.NA],
        }
    )

    normalized = normalize_unified_labels(df)

    assert list(normalized.columns[: len(UNIFIED_REQUIRED_COLUMNS)]) == list(
        UNIFIED_REQUIRED_COLUMNS
    )
    assert normalized["title"].tolist() == ["", ""]
    assert normalized["dataset"].tolist() == ["babe", "allsides"]
    assert normalized["bias_label"].tolist() == [0, 1]
    assert normalized["ideology_label"].tolist() == [0, 2]
    assert normalized["propaganda_label"].tolist() == [0, 1]
    assert normalized["hero"].tolist() == [1, 0]
    assert normalized["villain"].tolist() == [0, 1]
    assert normalized["victim"].tolist() == [1, 0]
    assert normalized["CO"].tolist() == [1, 0]
    assert normalized["EC"].tolist() == [0, 1]
    assert normalized["HI"].tolist() == [1, 0]
    assert normalized["MO"].tolist() == [0, 1]
    assert normalized["RE"].tolist() == [1, 0]
    assert int(normalized.loc[0, "emotion_3"]) == 1
    assert int(normalized.loc[0, "emotion_0"]) == 0
    assert pd.isna(normalized.loc[1, "emotion_3"])


def test_validate_unified_labels_reports_invalid_range() -> None:
    df = pd.DataFrame(
        {
            "text": ["example"],
            "dataset": ["unit_test"],
            "bias_label": [5],
            "CO": [2],
            "emotion_0": [3],
        }
    )

    errors = validate_unified_labels(df)

    assert errors
    assert "Unified label normalization failed" in errors[0]
    assert "bias_label: invalid values" in errors[0]
    assert "CO: invalid values" in errors[0]
    assert "emotion_0: invalid values" in errors[0]


def test_normalize_unified_labels_supports_sparse_task_labels() -> None:
    df = pd.DataFrame(
        {
            "text": ["only text is required for content"],
            "dataset": ["sparse_source"],
        }
    )

    normalized = normalize_unified_labels(df)

    assert pd.isna(normalized.loc[0, "bias_label"])
    assert pd.isna(normalized.loc[0, "frame"])
    for column in NARRATIVE_FRAME_COLUMNS:
        assert pd.isna(normalized.loc[0, column])
    for column in EMOTION_COLUMNS:
        assert pd.isna(normalized.loc[0, column])


def test_normalize_unified_labels_raises_for_missing_text() -> None:
    df = pd.DataFrame({"dataset": ["missing_text"]})

    with pytest.raises(ValueError, match="Missing required unified dataset column: 'text'"):
        normalize_unified_labels(df)
