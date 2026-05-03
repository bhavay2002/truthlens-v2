from __future__ import annotations

import importlib
from pathlib import Path


class TestProjectStructure:
    def test_top_level_directories_exist(self) -> None:
        for directory in ("src", "api", "models", "data", "config", "tests"):
            assert Path(directory).exists(), f"Missing directory: {directory}"

    def test_source_subdirectories_exist(self) -> None:
        for subdir in (
            "src/data",
            "src/models",
            "src/features",
            "src/evaluation",
            "src/aggregation",
            "src/explainability",
            "src/pipelines",
            "src/training",
            "src/utils",
            "src/graph",
            "src/inference",
            "src/analysis",
        ):
            assert Path(subdir).exists(), f"Missing src subdir: {subdir}"

    def test_config_files_exist(self) -> None:
        assert Path("config/config.yaml").exists()
        assert Path("requirements.txt").exists()

    def test_api_package_exists(self) -> None:
        assert Path("api/__init__.py").exists()
        assert Path("api/app.py").exists()

    def test_models_subpackages_exist(self) -> None:
        for subpackage in (
            "src/models/emotion",
            "src/models/encoder",
            "src/models/ideology",
            "src/models/multitask",
            "src/models/narrative",
            "src/models/propaganda",
        ):
            assert Path(subpackage).exists(), f"Missing: {subpackage}"
            assert (Path(subpackage) / "__init__.py").exists(), f"Missing __init__: {subpackage}"

    def test_key_source_modules_importable(self) -> None:
        modules = [
            "src.utils.settings",
            "src.utils.config_loader",
            "src.utils.logging_utils",
            "src.utils.seed_utils",
            "src.utils.time_utils",
            "src.utils.json_utils",
            "src.utils.device_utils",
            "src.utils.input_validation",
            "src.aggregation.score_normalizer",
            "src.aggregation.risk_assessment",
            "src.evaluation.metrics",
            "src.evaluation.uncertainty",
        ]
        for module_name in modules:
            mod = importlib.import_module(module_name)
            assert mod is not None, f"Could not import: {module_name}"

    def test_model_compat_packages_importable(self) -> None:
        modules = [
            "models.inference.predictor",
            "models.checkpointing.checkpoint_manager",
            "models.registry.model_registry",
            "models.utils.model_utils",
        ]
        for module_name in modules:
            mod = importlib.import_module(module_name)
            assert mod is not None

    def test_training_modules_importable(self) -> None:
        modules = [
            "src.training.cross_validation",
            "src.training.hyperparameter_tuning",
            "src.training.checkpointing",
            "src.training.optimizer_factory",
            "src.training.scheduler_factory",
        ]
        for module_name in modules:
            mod = importlib.import_module(module_name)
            assert mod is not None
