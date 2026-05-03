"""
Utility loader for EmotionClassifier checkpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any

import torch

from ..tasks.emotion.emotion_classifier import (
    EmotionClassifier,
    EmotionClassifierConfig,
)

logger = logging.getLogger(__name__)


class EmotionModelLoader:
    """
    Production-grade loader for EmotionClassifier checkpoints.

    Supports:
    - raw state_dict checkpoints
    - training checkpoints with metadata
    - automatic device placement
    - strict validation + compatibility fallback
    - optional config extraction from checkpoint
    """

    # -----------------------------------------------------

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:

        if device is not None:
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    # -----------------------------------------------------

    @staticmethod
    def _extract_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, Any]:

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        return checkpoint

    # -----------------------------------------------------

    @staticmethod
    def _extract_config(
        checkpoint: Dict[str, Any],
        override: Optional[EmotionClassifierConfig],
    ) -> EmotionClassifierConfig:

        if override is not None:
            return override

        if "config" in checkpoint and isinstance(checkpoint["config"], dict):
            try:
                return EmotionClassifierConfig(**checkpoint["config"])
            except Exception:
                logger.warning("[CONFIG] Failed to load config from checkpoint")

        return EmotionClassifierConfig()

    # -----------------------------------------------------

    @staticmethod
    def _validate_load_result(load_result) -> None:

        if load_result.missing_keys:
            raise RuntimeError(
                f"[CHECKPOINT ERROR] Missing keys: {load_result.missing_keys}"
            )

        if load_result.unexpected_keys:
            raise RuntimeError(
                f"[CHECKPOINT ERROR] Unexpected keys: {load_result.unexpected_keys}"
            )

    # -----------------------------------------------------

    @staticmethod
    def load(
        model_path: Union[str, Path],
        config: Optional[EmotionClassifierConfig] = None,
        device: Optional[str] = None,
    ) -> EmotionClassifier:

        device_obj = EmotionModelLoader._resolve_device(device)

        path_obj = Path(model_path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Emotion model checkpoint not found: {model_path}")

        logger.info("[MODEL LOAD] Loading emotion model: %s", path_obj)

        checkpoint = torch.load(
            path_obj,
            map_location=device_obj,
            weights_only=False,
        )

        resolved_config = EmotionModelLoader._extract_config(
            checkpoint,
            config,
        )

        model = EmotionClassifier(resolved_config)

        state_dict = EmotionModelLoader._extract_state_dict(checkpoint)

        load_result = model.load_state_dict(state_dict, strict=False)

        EmotionModelLoader._validate_load_result(load_result)

        model.to(device_obj)
        model.eval()

        logger.info(
            "[MODEL LOAD] Emotion model loaded successfully on device: %s",
            device_obj,
        )

        return model


__all__ = ["EmotionModelLoader"]