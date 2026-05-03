from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from .transformer_encoder import TransformerEncoder
from .encoder_config import EncoderConfig
from ..config import (
    EncoderConfig as ModelEncoderConfig,
    ModelConfigLoader,
    MultiTaskModelConfig,
)

logger = logging.getLogger(__name__)


class EncoderFactory:

    SUPPORTED_ENCODERS = {"transformer"}

    # =====================================================
    # CORE
    # =====================================================

    @staticmethod
    def create_transformer_encoder(config: EncoderConfig) -> TransformerEncoder:

        if not isinstance(config, EncoderConfig):
            raise TypeError("config must be EncoderConfig")

        config.validate()

        device = EncoderFactory.detect_device(config.device)

        logger.info(
            "Creating encoder | model=%s | pooling=%s | device=%s",
            config.model_name,
            config.pooling,
            device,
        )

        encoder = TransformerEncoder(
            model_name=config.model_name,
            pooling=config.pooling,
            device=device,
            freeze_encoder=config.freeze_encoder,
            gradient_checkpointing=config.gradient_checkpointing,
            output_hidden_states=config.output_hidden_states,
            use_amp=config.use_amp,
            amp_dtype=config.amp_dtype,
            use_compile=config.use_compile,
            compile_mode=config.compile_mode,
            max_length=config.max_length,
            init_from_config_only=config.init_from_config_only,
            **config.extra_kwargs,
        )

        return encoder

    # =====================================================
    # GENERIC FACTORY
    # =====================================================

    @staticmethod
    def create_from_name(
        encoder_type: str,
        config: EncoderConfig,
    ) -> TransformerEncoder:

        if not isinstance(encoder_type, str) or not encoder_type.strip():
            raise ValueError("encoder_type invalid")

        encoder_type = encoder_type.lower()

        if encoder_type not in EncoderFactory.SUPPORTED_ENCODERS:
            raise ValueError(f"Unsupported encoder: {encoder_type}")

        if encoder_type == "transformer":
            return EncoderFactory.create_transformer_encoder(config)

        raise RuntimeError("Unexpected encoder type")

    # =====================================================
    # MODEL CONFIG
    # =====================================================

    @staticmethod
    def create_from_model_config(
        model_config: MultiTaskModelConfig,
        freeze_encoder: Optional[bool] = None,
    ) -> TransformerEncoder:

        effective_freeze = (
            freeze_encoder
            if freeze_encoder is not None
            else getattr(model_config.encoder, "freeze_encoder", False)
        )

        cfg = EncoderConfig(
            model_type="transformer",
            model_name=model_config.encoder.model_name,
            pooling=model_config.encoder.pooling,
            device=model_config.encoder.device,
            freeze_encoder=effective_freeze,
            output_hidden_states=False,
            gradient_checkpointing=getattr(
                model_config.encoder, "gradient_checkpointing", False
            ),
            use_amp=getattr(model_config.encoder, "use_amp", True),
            amp_dtype=getattr(model_config.encoder, "amp_dtype", "bf16"),
            use_compile=getattr(model_config.encoder, "use_compile", False),
            compile_mode=getattr(model_config.encoder, "compile_mode", "default"),
            max_length=getattr(model_config.encoder, "max_length", 512),
            init_from_config_only=getattr(
                model_config.encoder, "init_from_config_only", False
            ),
            extra_kwargs={},
        )

        return EncoderFactory.create_transformer_encoder(cfg)

    # =====================================================
    # DIRECT CONFIG
    # =====================================================

    @staticmethod
    def create_from_encoder_config(
        encoder_config: ModelEncoderConfig,
        freeze_encoder: bool = False,
    ) -> TransformerEncoder:

        cfg = EncoderConfig(
            model_type="transformer",
            model_name=encoder_config.model_name,
            pooling=encoder_config.pooling,
            device=encoder_config.device,
            freeze_encoder=freeze_encoder,
            output_hidden_states=False,
            gradient_checkpointing=getattr(
                encoder_config, "gradient_checkpointing", False
            ),
            use_amp=getattr(encoder_config, "use_amp", True),
            amp_dtype=getattr(encoder_config, "amp_dtype", "bf16"),
            use_compile=getattr(encoder_config, "use_compile", False),
            compile_mode=getattr(encoder_config, "compile_mode", "default"),
            max_length=getattr(encoder_config, "max_length", 512),
            init_from_config_only=getattr(
                encoder_config, "init_from_config_only", False
            ),
            extra_kwargs={},
        )

        return EncoderFactory.create_transformer_encoder(cfg)

    # =====================================================
    # YAML
    # =====================================================

    @staticmethod
    def create_from_yaml(
        yaml_path: str | Path,
        freeze_encoder: bool = False,
    ) -> TransformerEncoder:

        model_config = ModelConfigLoader.load_multitask_config(yaml_path)

        logger.info("Loading encoder from YAML: %s", yaml_path)

        return EncoderFactory.create_from_model_config(
            model_config,
            freeze_encoder=freeze_encoder,
        )

    # =====================================================
    # DEVICE
    # =====================================================

    @staticmethod
    def detect_device(device: Optional[str] = None) -> torch.device:
        # A5.1: delegate to the single source of truth so CUDA / MPS /
        # CPU fallback is identical across every device-picking site.
        from src.models._device import detect_device

        return detect_device(device)