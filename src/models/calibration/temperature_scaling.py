"""Temperature scaling calibration.

This module owns the numerical implementation of
:class:`TemperatureScaler` (and its small dataclass config).

CFG6: previously the class lived in ``src.evaluation.calibration`` and
was re-exported from here, which forced the ``models`` layer to import
from the ``evaluation`` layer — a backwards dependency arrow. The
class has now been moved here so the arrow runs ``evaluation -> models``
like every other production-stack import; ``src.evaluation.calibration``
re-exports the class from this module to keep the public API stable.

Public API:
  • :class:`TemperatureScaler`
  • :class:`TemperatureScalingConfig`
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Floor for the temperature divisor — matches the EPS used elsewhere
# in the calibration stack so the two paths stay numerically aligned.
_TEMPERATURE_EPS: float = 1e-12


# =========================================================
# SCALER
# =========================================================

class TemperatureScaler(nn.Module):
    """A single learnable scalar that divides the logits.

    CFG6 / G4: the scalar is a calibration parameter and must NOT be
    updated by the main training optimizer — see
    :meth:`BaseModel.get_calibration_parameters`. It is fitted post-hoc
    on held-out validation logits via :func:`fit_temperature` (in
    ``src.evaluation.calibration``) or an LBFGS pass over this module.
    """

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / torch.clamp(self.temperature, min=_TEMPERATURE_EPS)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class TemperatureScalingConfig:
    """Optimizer hyper-parameters for fitting a temperature scaler."""

    max_iter: int = 50
    lr: float = 0.01


__all__ = ["TemperatureScaler", "TemperatureScalingConfig"]
