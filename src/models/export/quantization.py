from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Type

import torch
import torch.nn as nn
import torch.quantization as quantization

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class QuantizationConfig:
    method: str = "dynamic"  # dynamic | static | qat
    dtype: torch.dtype = torch.qint8
    backend: str = "fbgemm"
    device: str = "cpu"

    def __post_init__(self):
        if self.method not in {"dynamic", "static", "qat"}:
            raise ValueError("Invalid quantization method")


# =========================================================
# ENGINE
# =========================================================

class QuantizationEngine:

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()

        if self.config.backend not in torch.backends.quantized.supported_engines:
            raise ValueError(f"Unsupported backend: {self.config.backend}")

        torch.backends.quantized.engine = self.config.backend

    # -----------------------------------------------------

    def _validate(self, model: nn.Module):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be nn.Module")

    # -----------------------------------------------------
    # DYNAMIC
    # -----------------------------------------------------

    def dynamic(
        self,
        model: nn.Module,
        layers: Optional[Iterable[Type[nn.Module]]] = None,
    ) -> nn.Module:

        self._validate(model)

        layers = layers or {nn.Linear}

        logger.info("Dynamic quantization")

        return quantization.quantize_dynamic(
            model,
            layers,
            dtype=self.config.dtype,
        )

    # -----------------------------------------------------
    # STATIC
    # -----------------------------------------------------

    def static_prepare(self, model: nn.Module) -> nn.Module:

        self._validate(model)

        model.eval()
        model.qconfig = quantization.get_default_qconfig(self.config.backend)

        logger.info("Static prepare")

        return quantization.prepare(model, inplace=False)

    def static_convert(self, model: nn.Module) -> nn.Module:

        model.eval()

        logger.info("Static convert")

        return quantization.convert(model, inplace=False)

    # -----------------------------------------------------
    # QAT
    # -----------------------------------------------------

    def qat_prepare(self, model: nn.Module) -> nn.Module:

        self._validate(model)

        model.train()
        model.qconfig = quantization.get_default_qat_qconfig(self.config.backend)

        logger.info("QAT prepare")

        return quantization.prepare_qat(model, inplace=False)

    # -----------------------------------------------------
    # APPLY
    # -----------------------------------------------------

    def apply(
        self,
        model: nn.Module,
        calibration_data: Optional[Iterable[torch.Tensor]] = None,
    ) -> nn.Module:

        method = self.config.method

        if method == "dynamic":
            return self.dynamic(model)

        if method == "static":

            if calibration_data is None:
                raise ValueError("Calibration data required")

            prepared = self.static_prepare(model)

            logger.info("Calibrating")

            with torch.no_grad():
                for batch in calibration_data:
                    prepared(batch)

            return self.static_convert(prepared)

        if method == "qat":
            return self.qat_prepare(model)

        raise RuntimeError("Invalid method")