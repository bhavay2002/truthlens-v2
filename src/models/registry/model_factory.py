from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Dict, Any

import torch.nn as nn

from ..config import ModelConfigLoader, MultiTaskModelConfig
from ..encoder.encoder_config import EncoderConfig

from ..tasks.bias.bias_classifier import BiasClassifier, BiasClassifierConfig
from ..tasks.ideology.ideology_classifier import (
    IdeologyClassifier,
    IdeologyClassifierConfig,
)
from ..tasks.propaganda.propaganda_detector import (
    PropagandaDetector,
    PropagandaDetectorConfig,
)
from ..tasks.narrative.narrative_detector import (
    NarrativeDetector,
    NarrativeDetectorConfig,
)
from ..tasks.emotion.emotion_classifier import (
    EmotionClassifier,
    EmotionClassifierConfig,
)
from ..multitask.multitask_truthlens_model import (
    MultiTaskTruthLensModel,
    MultiTaskTruthLensConfig,
)

logger = logging.getLogger(__name__)


class ModelFactory:

    SUPPORTED_MODELS = {
        "bias_classifier",
        "ideology_classifier",
        "propaganda_detector",
        "narrative_detector",
        "emotion_classifier",
        "multitask_truthlens",
    }

    # =====================================================
    # INTERNAL RESOLUTION
    # =====================================================

    @staticmethod
    def _resolve_encoder_fields(config: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(config, dict):
            return {}

        raw = config.get("encoder_config")

        if isinstance(raw, EncoderConfig):
            return {
                "model_name": raw.model_name,
                "pooling": raw.pooling,
                "device": raw.device,
            }

        if isinstance(raw, dict):
            cfg = EncoderConfig(**raw)
            return {
                "model_name": cfg.model_name,
                "pooling": cfg.pooling,
                "device": cfg.device,
            }

        return {}

    @staticmethod
    def _resolve_regression_fields(config: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(config, dict):
            return {}

        raw = config.get("regression_config")
        if not isinstance(raw, dict):
            return {}

        return {
            "use_regression_head": bool(raw.get("enabled", False)),
            "regression_output_dim": raw.get("output_dim", 1),
            "regression_hidden_dim": raw.get("hidden_dim"),
            "regression_activation": raw.get("activation", "gelu"),
        }

    # =====================================================
    # CORE CREATE
    # =====================================================

    @staticmethod
    def _filter_for_dataclass(
        cls: type, merged: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Keep only keys that are valid fields of the given dataclass.

        Callers (notably the training pipeline) hand the factory a single
        ``params`` dict that mixes model-config fields (``model_name``,
        ``dropout`` …) with optimizer / loop fields (``lr``, ``epochs``,
        ``tokenizer`` …). The task-config dataclasses are strict and
        reject unknown keys, so we drop the non-model keys here rather
        than asking every call site to split the dict.
        """
        valid = {f.name for f in dataclasses.fields(cls)}
        return {k: v for k, v in merged.items() if k in valid}

    @staticmethod
    def create(model_type: str, config: Dict[str, Any]) -> nn.Module:

        if not isinstance(model_type, str) or not model_type.strip():
            raise ValueError("model_type must be non-empty")

        model_type = model_type.lower().strip()

        if model_type not in ModelFactory.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model_type: {model_type}")

        import copy
        merged = copy.deepcopy(config)

        merged.update(ModelFactory._resolve_encoder_fields(config))
        merged.update(ModelFactory._resolve_regression_fields(config))

        logger.info("[MODEL FACTORY] Creating: %s", model_type)

        _filter = ModelFactory._filter_for_dataclass

        try:

            if model_type == "bias_classifier":
                model = BiasClassifier(
                    BiasClassifierConfig(**_filter(BiasClassifierConfig, merged))
                )

            elif model_type == "ideology_classifier":
                model = IdeologyClassifier(
                    IdeologyClassifierConfig(
                        **_filter(IdeologyClassifierConfig, merged)
                    )
                )

            elif model_type == "propaganda_detector":
                model = PropagandaDetector(
                    PropagandaDetectorConfig(
                        **_filter(PropagandaDetectorConfig, merged)
                    )
                )

            elif model_type == "narrative_detector":
                model = NarrativeDetector(
                    NarrativeDetectorConfig(
                        **_filter(NarrativeDetectorConfig, merged)
                    )
                )

            elif model_type == "emotion_classifier":
                model = EmotionClassifier(
                    EmotionClassifierConfig(
                        **_filter(EmotionClassifierConfig, merged)
                    )
                )

            elif model_type == "multitask_truthlens":
                # P4: MultiTaskTruthLensModel.__init__ accepts the
                # convenience-config exclusively as a *keyword* arg
                # (so the (encoder, task_heads) positional contract
                # stays unambiguous). Passing it positionally would
                # bind to ``encoder`` and explode at validation time.
                model = MultiTaskTruthLensModel(
                    config=MultiTaskTruthLensConfig(
                        **_filter(MultiTaskTruthLensConfig, merged)
                    )
                )

            else:
                raise RuntimeError("Invalid model type")

        except TypeError as e:
            raise ValueError(f"Invalid config for {model_type}: {e}") from e

        device = merged.get("device")
        if device:
            model.to(device)

        return model

    # =====================================================
    # WRAPPERS
    # =====================================================

    @staticmethod
    def create_wrapper(
        model_type: str,
        config: Dict[str, Any],
        *,
        device: str | None = None,
    ):
        from src.models.inference.model_wrapper import ModelWrapper

        model = ModelFactory.create(model_type, config)
        return ModelWrapper(model=model, device=device)

    @staticmethod
    def create_predictor(
        model_type: str,
        config: Dict[str, Any],
        *,
        device: str | None = None,
    ):
        from src.models.inference.predictor import Predictor

        model = ModelFactory.create(model_type, config)
        return Predictor(model=model, device=device)

    # =====================================================
    # MULTITASK CONFIG
    # =====================================================

    @staticmethod
    def create_from_model_config(
        model_config: MultiTaskModelConfig,
    ) -> nn.Module:

        logger.info(
            "[MODEL FACTORY] multitask from config | encoder=%s",
            model_config.encoder.model_name,
        )

        return MultiTaskTruthLensModel.from_model_config(model_config)

    @staticmethod
    def create_task_from_model_config(
        task_name: str,
        model_config: MultiTaskModelConfig,
    ) -> nn.Module:

        mapping = {
            "bias": BiasClassifier,
            "ideology": IdeologyClassifier,
            "propaganda": PropagandaDetector,
            "narrative": NarrativeDetector,
            "emotion": EmotionClassifier,
        }

        if task_name not in mapping:
            raise ValueError(f"Unsupported task: {task_name}")

        return mapping[task_name].from_model_config(model_config)

    # =====================================================
    # YAML
    # =====================================================

    @staticmethod
    def create_from_yaml(yaml_path: str | Path) -> nn.Module:

        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            raise FileNotFoundError(f"Config not found: {yaml_path}")

        config = ModelConfigLoader.load_multitask_config(yaml_path)

        logger.info("[MODEL FACTORY] from yaml: %s", yaml_path)

        return ModelFactory.create_from_model_config(config)

    # =====================================================
    # CHECKPOINT
    # =====================================================

    @staticmethod
    def create_from_checkpoint(
        model_dir: str | Path,
        device: str | None = None,
    ) -> nn.Module:

        from ..checkpointing.model_loader import ModelLoader

        loaded = ModelLoader(model_dir=model_dir, device=device).load()

        model = loaded.get("model")

        if not isinstance(model, nn.Module):
            raise RuntimeError("Invalid checkpoint model")

        return model


_TASK_TO_MODEL_TYPE = {
    "bias": "bias_classifier",
    "ideology": "ideology_classifier",
    "propaganda": "propaganda_detector",
    "narrative": "narrative_detector",
    "emotion": "emotion_classifier",
    "multitask": "multitask_truthlens",
    "multitask_truthlens": "multitask_truthlens",
    "bias_classifier": "bias_classifier",
    "ideology_classifier": "ideology_classifier",
    "propaganda_detector": "propaganda_detector",
    "narrative_detector": "narrative_detector",
    "emotion_classifier": "emotion_classifier",
}


def build_model(task: str, config: Dict[str, Any]) -> nn.Module:

    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")

    if not isinstance(config, dict):
        raise TypeError("config must be a dict")

    key = task.lower().strip()
    model_type = _TASK_TO_MODEL_TYPE.get(key)

    if model_type is None:
        raise ValueError(
            f"Unknown task '{task}'. "
            f"Supported: {sorted(_TASK_TO_MODEL_TYPE.keys())}"
        )

    return ModelFactory.create(model_type=model_type, config=config)