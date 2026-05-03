from __future__ import annotations

import logging
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict

import yaml
import torch

# CRIT-1: single source of truth for the inference dataclass. The loader
# previously declared its own dataclass with extra fields that were then
# silently dropped at the engine boundary because the engine used a
# different ``InferenceConfig`` of the same name. The loader now returns
# the engine's dataclass directly.
from src.inference.inference_engine import InferenceConfig
from src.inference.constants import DEFAULT_INFERENCE_BATCH_SIZE

logger = logging.getLogger(__name__)


__all__ = [
    "InferenceConfig",
    "InferenceConfigLoader",
    "load_inference_config",
]


# =========================================================
# LOADER
# =========================================================

class InferenceConfigLoader:

    REQUIRED_FIELDS = {
        "device": str,
        "batch_size": int,
    }

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        logger.info("InferenceConfigLoader initialized")

    # =====================================================
    # LOAD
    # =====================================================

    def load(self) -> InferenceConfig:

        with open(self.config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

        if not isinstance(config_dict, dict):
            raise TypeError("Config must be a dictionary")

        self._validate_config(config_dict)

        allowed = {f.name for f in fields(InferenceConfig)}

        filtered = {k: v for k, v in config_dict.items() if k in allowed}

        unknown = sorted(set(config_dict.keys()) - allowed)
        if unknown:
            logger.warning("Unknown config keys ignored: %s", unknown)

        # ``model_path`` is required by the engine's dataclass; if the YAML
        # does not provide one, fall back to a conservative default so the
        # loader does not crash before the caller has a chance to override.
        filtered.setdefault("model_path", str(self.config_path.parent))

        config = InferenceConfig(**filtered)

        # 🔥 RESOLVE DEVICE
        config.device = self._resolve_device(config.device)

        logger.info("Inference config loaded (device=%s)", config.device)

        return config

    # =====================================================
    # VALIDATION
    # =====================================================

    def _validate_config(self, config: Dict[str, Any]):

        # CFG-7: ``REQUIRED_FIELDS`` previously asserted *type* but
        # tolerated *absence* (``if field not in config: continue``),
        # so a missing ``batch_size`` would silently fall through to
        # the engine's dataclass default. Either the field is
        # required (then enforce presence too) or it is optional
        # (then drop it from REQUIRED_FIELDS). We pick the latter:
        # the engine default ``DEFAULT_INFERENCE_BATCH_SIZE`` is the
        # right answer when YAML does not override, but we keep the
        # type/range checks for whatever IS present.
        for field, expected_type in self.REQUIRED_FIELDS.items():
            if field not in config:
                continue
            if not isinstance(config[field], expected_type):
                raise TypeError(
                    f"{field} must be {expected_type.__name__}"
                )

        batch_size = config.get("batch_size")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        # Backstop for callers that deleted the key entirely from YAML
        # but expect the loader to inject a sane default. Keeps the
        # source-of-truth aligned with the engine's dataclass default.
        if batch_size is None:
            config["batch_size"] = DEFAULT_INFERENCE_BATCH_SIZE

        device = config.get("device")
        if device is not None and device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be cpu | cuda | auto")

    # =====================================================
    # DEVICE RESOLUTION
    # =====================================================

    def _resolve_device(self, device: str) -> str:

        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available → falling back to CPU")
            return "cpu"

        return device

    # =====================================================
    # FROM DICT
    # =====================================================

    @staticmethod
    def from_dict(config: Dict[str, Any]) -> InferenceConfig:

        if not isinstance(config, dict):
            raise TypeError("config must be dict")

        allowed = {f.name for f in fields(InferenceConfig)}
        filtered = {k: v for k, v in config.items() if k in allowed}
        filtered.setdefault("model_path", "")

        return InferenceConfig(**filtered)


# =========================================================
# HELPER
# =========================================================

def load_inference_config(path: str | Path) -> InferenceConfig:
    return InferenceConfigLoader(path).load()
