from __future__ import annotations

import pytest

from src.utils.config_loader import load_config
from src.utils.settings import load_settings


class TestLoadConfig:
    def test_config_has_top_level_sections(self) -> None:
        config = load_config("config/config.yaml")
        for section in ("model", "training", "data", "api"):
            assert section in config

    def test_model_section_has_name_and_path(self) -> None:
        config = load_config("config/config.yaml")
        model = config["model"]
        assert "name" in model or "path" in model

    def test_training_section_has_hyperparameters(self) -> None:
        config = load_config("config/config.yaml")
        training = config["training"]
        assert "epochs" in training or "batch_size" in training

    def test_api_section_has_title_and_version(self) -> None:
        config = load_config("config/config.yaml")
        api = config["api"]
        assert "title" in api
        assert "version" in api

    def test_missing_config_file_raises(self) -> None:
        with pytest.raises(Exception):
            load_config("config/nonexistent_config.yaml")


class TestLoadSettings:
    def test_settings_api_title_is_non_empty_string(self) -> None:
        settings = load_settings()
        assert isinstance(settings.api.title, str)
        assert len(settings.api.title) > 0

    def test_settings_api_version_follows_semver(self) -> None:
        settings = load_settings()
        parts = settings.api.version.split(".")
        assert len(parts) >= 2

    def test_settings_text_preview_chars_is_positive(self) -> None:
        settings = load_settings()
        assert settings.api.text_preview_chars > 0

    def test_settings_batch_size_is_power_or_positive(self) -> None:
        settings = load_settings()
        assert settings.training.batch_size >= 1

    def test_settings_learning_rate_in_valid_range(self) -> None:
        settings = load_settings()
        assert 0.0 < settings.training.learning_rate < 1.0

    def test_settings_validation_size_fraction(self) -> None:
        settings = load_settings()
        assert 0.0 < settings.training.validation_size < 1.0
