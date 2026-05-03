from __future__ import annotations

import pandas as pd
import pytest

from training import train_transformer_model


def _training_df(rows: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": list(range(rows)),
            "text": [f"sample training text {i}" for i in range(rows)],
            "label": [i % 2 for i in range(rows)],
        }
    )


def test_split_train_val_test_produces_disjoint_partitions() -> None:
    df = _training_df(100)

    train_df, val_df, test_df = train_transformer_model._split_train_val_test(df)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert set(train_df["id"]).isdisjoint(val_df["id"])
    assert set(train_df["id"]).isdisjoint(test_df["id"])
    assert set(val_df["id"]).isdisjoint(test_df["id"])
    assert set(train_df.columns) >= {"text", "label"}


def test_validate_split_df_requires_expected_columns() -> None:
    missing_text_df = pd.DataFrame({"label": [0, 1]})

    with pytest.raises(ValueError):
        train_transformer_model._validate_split_df(missing_text_df, "df", "text")


def test_validate_split_df_rejects_empty_text_column() -> None:
    empty_text_df = pd.DataFrame({"text": ["", "   "], "label": [0, 1]})

    with pytest.raises(ValueError):
        train_transformer_model._validate_split_df(empty_text_df, "df", "text")


def test_validate_split_df_supports_custom_label_column() -> None:
    df = pd.DataFrame({"text": ["alpha", "beta"], "bias_label": [0, 1]})

    train_transformer_model._validate_split_df(
        df,
        "df",
        "text",
        label_column="bias_label",
    )


def test_split_train_val_test_supports_custom_label_column() -> None:
    df = pd.DataFrame(
        {
            "id": list(range(120)),
            "text": [f"sample training text {i}" for i in range(120)],
            "bias_label": [i % 2 for i in range(120)],
        }
    )

    train_df, val_df, test_df = train_transformer_model._split_train_val_test(
        df,
        label_column="bias_label",
    )

    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert set(train_df.columns) >= {"text", "bias_label"}


def test_compute_checkpoint_save_steps_uses_ten_percent_progress() -> None:
    # 100 examples, batch size 8, grad accumulation 2, epochs 3
    # forward steps/epoch=13, optimizer steps/epoch=7, total=21
    # 10% cadence => ceil(2.1)=3
    save_steps = train_transformer_model._compute_checkpoint_save_steps(
        train_examples=100,
        batch_size=8,
        gradient_accumulation_steps=2,
        epochs=3,
    )

    assert save_steps == 3


def test_compute_checkpoint_save_steps_has_minimum_of_one() -> None:
    save_steps = train_transformer_model._compute_checkpoint_save_steps(
        train_examples=4,
        batch_size=8,
        gradient_accumulation_steps=2,
        epochs=1,
    )

    assert save_steps == 1


def test_training_paths_follow_settings_configuration() -> None:
    settings = train_transformer_model.SETTINGS

    assert train_transformer_model.MODELS_DIR == settings.paths.models_dir
    assert train_transformer_model.LOGS_DIR == settings.paths.logs_dir
    assert train_transformer_model.GOOGLE_DRIVE_REPORTS_DIR == settings.paths.reports_dir
    assert (
        train_transformer_model.GOOGLE_DRIVE_CHECKPOINTS_DIR
        == settings.paths.models_dir / "checkpoints"
    )
