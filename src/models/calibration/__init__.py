"""
Package: src.models.calibration
Description:
    Post-hoc calibration utilities for probabilistic classification models.

    Each public class is now defined in the file whose name advertises
    it (the historical file-name swap has been undone):

      • :class:`CalibrationMetrics` / :class:`CalibrationMetricConfig`
        — ECE, MCE, Brier, NLL — defined in ``calibration_metrics.py``.
      • :class:`TemperatureScaler` / :class:`TemperatureScalingConfig`
        — single-parameter logit scaling — defined in
        ``temperature_scaling.py`` (numerical body re-exported from
        ``src.evaluation.calibration``).
      • :class:`IsotonicCalibrator` / :class:`IsotonicCalibrationConfig`
        — isotonic regression calibration — defined in
        ``isotonic_calibration.py``.
"""

from src.models.calibration.calibration_metrics import (
    CalibrationMetricConfig,
    CalibrationMetrics,
)
from src.models.calibration.temperature_scaling import (
    TemperatureScaler,
    TemperatureScalingConfig,
)
from src.models.calibration.isotonic_calibration import (
    IsotonicCalibrationConfig,
    IsotonicCalibrator,
)


__all__ = [
    "CalibrationMetricConfig",
    "CalibrationMetrics",
    "IsotonicCalibrationConfig",
    "IsotonicCalibrator",
    "TemperatureScalingConfig",
    "TemperatureScaler",
]
