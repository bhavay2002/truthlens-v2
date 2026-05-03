from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTS
# =========================================================

VALID_POOLING_STRATEGIES = {"cls", "mean", "max", "attention"}
VALID_MODEL_TYPES = {"transformer"}


# =========================================================
# CONFIG
# =========================================================

@dataclass
class EncoderConfig:

    model_type: str = "transformer"
    model_name: str = "roberta-base"

    pooling: str = "cls"

    device: Optional[str] = None

    freeze_encoder: bool = False

    output_hidden_states: bool = False

    gradient_checkpointing: bool = False

    use_amp: bool = True
    amp_dtype: str = "bf16"  # fp16 | bf16

    # P2.1: ``use_compile`` is now tri-state.
    #
    #   • ``True``  — always wrap the encoder with ``torch.compile``.
    #   • ``False`` — never wrap (explicit opt-out).
    #   • ``None``  — auto-detect (default): enable on CUDA, disable on
    #     CPU. The auto-on default brings the 30–50 % step-time
    #     speed-up that ``torch.compile`` delivers on every supported
    #     GPU without forcing every config to mention it explicitly.
    #     CPU runs stay opt-in because the compile overhead does not
    #     pay back on small CPU workloads.
    use_compile: Optional[bool] = None
    compile_mode: str = "default"

    init_from_config_only: bool = False

    max_length: int = 512

    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    # =====================================================
    # VALIDATION
    # =====================================================

    def validate(self) -> None:

        if self.model_type not in VALID_MODEL_TYPES:
            raise ValueError(f"Invalid model_type: {self.model_type}")

        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("model_name invalid")

        if self.pooling not in VALID_POOLING_STRATEGIES:
            raise ValueError(f"Invalid pooling: {self.pooling}")

        if self.device is not None and not isinstance(self.device, str):
            raise ValueError("device must be str or None")

        if not isinstance(self.freeze_encoder, bool):
            raise ValueError("freeze_encoder must be bool")

        if self.amp_dtype not in {"fp16", "bf16"}:
            raise ValueError("amp_dtype must be fp16 or bf16")

        if self.compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
            raise ValueError("Invalid compile_mode")

        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

        if not isinstance(self.extra_kwargs, dict):
            raise ValueError("extra_kwargs must be dict")

        logger.debug("EncoderConfig validated")

    # =====================================================
    # FACTORY
    # =====================================================

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EncoderConfig":

        if not isinstance(config_dict, dict):
            raise TypeError("config_dict must be dict")

        cfg = cls(**config_dict)
        cfg.validate()

        return cfg

    # =====================================================
    # EXPORT
    # =====================================================

    def to_dict(self) -> Dict[str, Any]:

        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "pooling": self.pooling,
            "device": self.device,
            "freeze_encoder": self.freeze_encoder,
            "output_hidden_states": self.output_hidden_states,
            "gradient_checkpointing": self.gradient_checkpointing,
            "use_amp": self.use_amp,
            "amp_dtype": self.amp_dtype,
            "use_compile": self.use_compile,
            "compile_mode": self.compile_mode,
            "init_from_config_only": self.init_from_config_only,
            "max_length": self.max_length,
            "extra_kwargs": self.extra_kwargs,
        }

    # =====================================================
    # SUMMARY
    # =====================================================

    def summary(self) -> Dict[str, Any]:

        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "pooling": self.pooling,
            "freeze_encoder": self.freeze_encoder,
            "amp": self.use_amp,
        }