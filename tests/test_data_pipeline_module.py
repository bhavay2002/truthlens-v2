from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data import data_pipeline as pipeline


def _base_config(tmp_path: Path) -> dict:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    splits_dir = tmp_path / "splits"

    return {
        "dataset": {
            "raw_data_dir": str(raw_dir),
            "fake_news_file": "fake.csv",
            "real_news_file": "real.csv",
            "text_column": "body",
            "title_column": "headline",
            "label_column": "category",
            "labels": {"fake": 9, "real": 3},
        },
        "validation": {
            "required_columns": ["body", "category"],
            "max_null_ratio": 0.2,
            "max_duplicate_ratio": 0.2,
            "min_class_ratio": 0.1,
            "min_text_length": 5,
        },
        "cleaning": {"min_word_count": 1},
        "balancing": {
            "enabled": False,
            "method": "oversample",
            "random_state": 42,
        },
        "augmentation": {"enabled": False, "multiplier": 2},
        "split": {
            "train_ratio": 0.6,
            "validation_ratio": 0.2,
            "test_ratio": 0.2,
            "random_state": 11,
            "stratified": True,
        },
        "profiling": {"enabled": False, "report_dir": str(tmp_path / "reports")},
        "output": {
            "processed_data_dir": str(processed_dir),
            "splits_dir": str(splits_dir),
        },
    }


def test_validate_config_requires_title_column(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    del config["dataset"]["title_column"]

    with pytest.raises(KeyError):
        pipeline._validate_config(config)


def test_run_data_pipeline_uses_dataset_config_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _base_config(tmp_path)
    captured: dict[str, object] = {}

    sample_df = pd.DataFrame(
        {
            "headline": ["h1", "h2", "h3", "h4", "h5"],
            "body": [
                "text one sample",
                "text two sample",
                "text three sample",
                "text four sample",
                "text five sample",
            ],
            "category": [9, 3, 9, 3, 9],
        }
    )

    def fake_merge_datasets(fake_path, real_path, **kwargs):
        captured["fake_path"] = Path(fake_path)
        captured["real_path"] = Path(real_path)
        captured["merge_kwargs"] = kwargs
        return sample_df.copy()

    class _Validator:
        def __init__(self, *args, **kwargs):
            captured["validator_kwargs"] = kwargs

        def validate(self, _df):
            return {"all_passed": True}

    def fake_clean_dataframe(df, **kwargs):
        captured["clean_kwargs"] = kwargs
        return df

    def fake_split_dataset(df, **kwargs):
        captured["split_kwargs"] = kwargs
        return df.iloc[:3], df.iloc[3:4], df.iloc[4:]

    def fake_save_splits(train_df, val_df, test_df, output_dir):
        captured["splits_output_dir"] = Path(output_dir)
        captured["split_sizes"] = (len(train_df), len(val_df), len(test_df))

    monkeypatch.setattr(pipeline, "load_config", lambda _path: config)
    monkeypatch.setattr(pipeline, "merge_datasets", fake_merge_datasets)
    monkeypatch.setattr(pipeline, "DataValidator", _Validator)
    monkeypatch.setattr(pipeline, "clean_dataframe", fake_clean_dataframe)
    monkeypatch.setattr(pipeline, "split_dataset", fake_split_dataset)
    monkeypatch.setattr(pipeline, "save_splits", fake_save_splits)

    result = pipeline.run_data_pipeline(config_path="ignored.yaml")

    merge_kwargs = captured["merge_kwargs"]
    assert merge_kwargs == {
        "text_column": "body",
        "title_column": "headline",
        "label_column": "category",
        "fake_label": 9,
        "real_label": 3,
    }
    assert captured["clean_kwargs"] == {
        "text_column": "body",
        "title_column": "headline",
        "min_len": 1,
    }
    assert captured["split_kwargs"]["label_column"] == "category"
    assert captured["splits_output_dir"] == Path(config["output"]["splits_dir"])
    assert captured["split_sizes"] == (3, 1, 1)
    assert result["processed_rows"] == 5
    assert result["train_rows"] == 3
    assert result["val_rows"] == 1
    assert result["test_rows"] == 1


def test_run_data_pipeline_supports_unified_dataset_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _base_config(tmp_path)
    config["dataset"]["unified_dataset_file"] = "unified.csv"
    config["dataset"]["label_column"] = "bias_label"
    config["dataset"]["label_columns"] = ["bias_label", "ideology_label"]
    captured: dict[str, object] = {}

    sample_df = pd.DataFrame(
        {
            "headline": ["h1", "h2", "h3", "h4", "h5"],
            "body": [
                "text one sample",
                "text two sample",
                "text three sample",
                "text four sample",
                "text five sample",
            ],
            "bias_label": [1, 0, 1, 0, 1],
            "ideology_label": [0, 1, 0, 1, 0],
        }
    )

    def fake_load_csv(path):
        captured["loaded_path"] = Path(path)
        return sample_df.copy()

    def fake_normalize_schema(df, **kwargs):
        captured["normalize_kwargs"] = kwargs
        return df.copy()

    def fail_merge_datasets(*_args, **_kwargs):
        raise AssertionError("merge_datasets should not be called for unified_dataset_file")

    class _Validator:
        def __init__(self, *args, **kwargs):
            captured["validator_kwargs"] = kwargs

        def validate(self, _df):
            return {"all_passed": True}

    def fake_clean_dataframe(df, **kwargs):
        captured["clean_kwargs"] = kwargs
        return df

    def fake_split_dataset(df, **kwargs):
        captured["split_kwargs"] = kwargs
        return df.iloc[:3], df.iloc[3:4], df.iloc[4:]

    def fake_save_splits(train_df, val_df, test_df, output_dir):
        captured["splits_output_dir"] = Path(output_dir)
        captured["split_sizes"] = (len(train_df), len(val_df), len(test_df))

    monkeypatch.setattr(pipeline, "load_config", lambda _path: config)
    monkeypatch.setattr(pipeline, "load_csv", fake_load_csv)
    monkeypatch.setattr(pipeline, "normalize_schema", fake_normalize_schema)
    monkeypatch.setattr(pipeline, "merge_datasets", fail_merge_datasets)
    monkeypatch.setattr(pipeline, "DataValidator", _Validator)
    monkeypatch.setattr(pipeline, "clean_dataframe", fake_clean_dataframe)
    monkeypatch.setattr(pipeline, "split_dataset", fake_split_dataset)
    monkeypatch.setattr(pipeline, "save_splits", fake_save_splits)

    result = pipeline.run_data_pipeline(config_path="ignored.yaml")

    assert captured["loaded_path"] == Path(config["dataset"]["raw_data_dir"]) / "unified.csv"
    assert captured["normalize_kwargs"] == {
        "text_column": "body",
        "title_column": "headline",
        "label_columns": ["bias_label", "ideology_label"],
    }
    assert captured["split_kwargs"]["label_column"] == "bias_label"
    assert captured["split_sizes"] == (3, 1, 1)
    assert result["processed_rows"] == 5
