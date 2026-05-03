from __future__ import annotations

import logging
import os
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn as nn

from src.models.calibration import IsotonicCalibrator, TemperatureScaler
from src.models.inference.prediction_output import PredictionOutput
from src.models.multitask.multitask_output import MultiTaskOutput
from src.utils import get_device, move_to_device

logger = logging.getLogger(__name__)


DEFAULT_FAKE_INDEX = 1
FAKE_LABEL_CANDIDATES = {"fake", "false", "misleading"}
_FAKE_HEAD_KEYS = ("fake_logits", "fakenews_logits", "misinformation_logits")


# GPU-3 (v13/v14 audit): honour the same TRUTHLENS_AMP_DTYPE env var
# that ``PredictionPipeline`` already reads. Previously this Predictor
# hard-selected bf16 vs fp16 from ``torch.cuda.is_bf16_supported()`` and
# ignored the operator's choice entirely — long articles ran in
# whatever-the-card-supported even when the operator explicitly picked
# fp32 for numerical-stability debugging or fp16 to match the trained
# checkpoint. The helper mirrors ``inference_pipeline._resolve_amp_dtype``
# so both orchestrators interpret the env var identically.
def _resolve_amp_dtype_from_env(default: str = "float16") -> torch.dtype:
    requested = (os.environ.get("TRUTHLENS_AMP_DTYPE") or default).lower()
    if requested in ("bf16", "bfloat16"):
        return torch.bfloat16
    if requested in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


# =========================================================
# UTILS
# =========================================================

def _find_tensor_by_keys(data: Dict[str, Any], keys: tuple[str, ...]) -> Optional[torch.Tensor]:
    for k in keys:
        v = data.get(k)
        if isinstance(v, torch.Tensor):
            return v
    return None


# =========================================================
# PREDICTOR
# =========================================================

class Predictor:

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        temperature_scaler: Optional[TemperatureScaler] = None,
        isotonic_calibrator: Optional[IsotonicCalibrator] = None,
        ensemble_model: Optional[nn.Module] = None,
    ):

        if not isinstance(model, nn.Module):
            raise TypeError("model must be nn.Module")

        self.device = torch.device(device) if device else get_device(prefer_gpu=True)

        self.model = model.to(self.device).eval()

        self.ensemble_model = ensemble_model
        if self.ensemble_model:
            self.ensemble_model.to(self.device).eval()

        self.temperature_scaler = temperature_scaler
        self.isotonic_calibrator = isotonic_calibrator

        logger.info("Predictor initialized on %s", self.device)

    # =====================================================
    # CONFIG
    # =====================================================

    def set_temperature_scaler(self, scaler: TemperatureScaler):
        self.temperature_scaler = scaler

    def set_isotonic_calibrator(self, calibrator: IsotonicCalibrator):
        self.isotonic_calibrator = calibrator

    def set_ensemble_model(self, ensemble_model: nn.Module):
        self.ensemble_model = ensemble_model.to(self.device).eval()

    # =====================================================
    # PUBLIC
    # =====================================================

    @torch.inference_mode()
    def predict_batch(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        return_structured: bool = False,
    ) -> Dict[str, Any]:

        batch = move_to_device(batch, self.device)

        outputs = self._forward(batch)
        formatted = self._format_outputs(outputs)

        if return_structured:
            return PredictionOutput.from_flat(formatted).to_dict()

        return formatted

    @torch.inference_mode()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        return_structured: bool = False,
    ) -> Dict[str, Any]:

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        result = self.predict_batch(batch, return_structured=return_structured)
        return self._squeeze(result)

    @torch.inference_mode()
    def predict_batch_structured(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> PredictionOutput:

        return PredictionOutput.from_flat(self.predict_batch(batch))

    # =====================================================
    # FORWARD
    # =====================================================

    def _forward(self, batch: Dict[str, torch.Tensor]) -> Any:

        if self.ensemble_model:
            return self._run_ensemble(batch)

        use_amp = self.device.type == "cuda"

        if use_amp:
            # GPU-3: env-driven dtype. Falls back to bf16 if the env
            # var is unset (matches PredictionPipeline default); a
            # bf16 request on a card without bf16 support is silently
            # demoted to fp16 to avoid an autocast crash.
            dtype = _resolve_amp_dtype_from_env()
            if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
                dtype = torch.float16
            with torch.autocast(device_type="cuda", dtype=dtype):
                return self.model(**batch)

        return self.model(**batch)

    # =====================================================
    # FORMAT
    # =====================================================

    def _format_outputs(self, outputs: Any) -> Dict[str, Any]:

        if isinstance(outputs, MultiTaskOutput):
            return outputs.to_flat_prediction_dict()

        if isinstance(outputs, dict):

            mt = outputs.get("multitask_output")
            if isinstance(mt, MultiTaskOutput):
                return mt.to_flat_prediction_dict()

            formatted = {}

            # =====================================================
            # INFERENCE-CONTRACT-FIX V7 — per-task dict entries
            #
            # ``MultiTaskTruthLensModel.forward`` returns a dict shaped
            # like::
            #
            #     {
            #       "bias":         {"logits": Tensor, ...},
            #       "ideology":     {"logits": Tensor, ...},
            #       ...,
            #       "task_logits":  {"bias": Tensor, "ideology": Tensor, ...},
            #     }
            #
            # — i.e. each *per-task* entry is itself a dict (the
            # ``BaseHead`` contract requires it), and ``"task_logits"``
            # is a parallel dict-of-tensors view consumed by the
            # training loop.
            #
            # Pre-V7 ``_format_outputs`` only flattened entries where
            # ``v`` was a *tensor* whose key ended in ``"_logits"``. No
            # such key exists in the multi-task forward output: the
            # per-task entries are dicts, ``"task_logits"`` is itself a
            # dict, and so the flattening branch *never fired*. The
            # downstream aggregator (``WeightManager`` /
            # ``TruthLensScoreCalculator``) therefore saw zero
            # confidence / entropy / probabilities and silently
            # produced an aggregate score of 0.0.
            #
            # We now handle three shapes explicitly:
            #
            #   1) Tensor whose key ends in ``"_logits"`` —
            #      legacy single-task / pre-flattened path.
            #
            #   2) Per-task dict containing ``"logits"`` — the
            #      multi-task contract. The task name is the dict key
            #      itself (``"bias"`` / ``"ideology"`` / ...). We
            #      flatten into ``<task>_logits``,
            #      ``<task>_probabilities``, ``<task>_confidence``,
            #      ``<task>_entropy``, ``<task>_predictions`` so the
            #      aggregator pipeline (which keys on those exact
            #      suffixes) sees real values.
            #
            #   3) ``"task_logits"`` dict-of-tensors — flatten each
            #      ``{task: tensor}`` entry the same way we handle (2).
            #      This is the back-compat path for any caller that
            #      still relies on the parallel view.
            #
            # Calibration is applied per-task via ``_calibrate`` so
            # temperature / isotonic adjustments stay scoped to each
            # head's logits.
            # =====================================================

            def _flatten_task_logits(
                base: str,
                logits_tensor: torch.Tensor,
            ) -> None:
                logits = torch.nan_to_num(logits_tensor)
                probs = torch.softmax(logits, dim=-1)
                probs = self._calibrate(logits, probs)
                preds = torch.argmax(probs, dim=-1)

                # Confidence = max class probability per row;
                # entropy   = Shannon entropy of the calibrated
                # distribution. Both are consumed by the aggregator
                # weight manager (``src.aggregation.weight_manager``)
                # as per-task scalars keyed on these exact suffixes.
                confidence = probs.max(dim=-1).values
                # Use ``clamp_min`` to avoid ``log(0)`` → -inf when a
                # calibrator collapses a class to exactly 0.
                safe_probs = probs.clamp_min(1e-12)
                entropy = -(probs * safe_probs.log()).sum(dim=-1)

                formatted[f"{base}_logits"] = logits
                formatted[f"{base}_probabilities"] = probs
                formatted[f"{base}_predictions"] = preds
                formatted[f"{base}_confidence"] = confidence
                formatted[f"{base}_entropy"] = entropy

            for k, v in outputs.items():

                # Shape (1): legacy tensor-with-_logits-suffix path.
                if isinstance(v, torch.Tensor) and k.endswith("_logits"):
                    _flatten_task_logits(k[:-7], v)
                    continue

                # Shape (2): per-task dict carrying ``"logits"``. The
                # key is the task name (no ``_logits`` suffix). We
                # also keep the original dict under ``k`` for any
                # downstream consumer that already speaks the
                # nested shape.
                if (
                    isinstance(v, dict)
                    and "logits" in v
                    and isinstance(v["logits"], torch.Tensor)
                ):
                    _flatten_task_logits(k, v["logits"])
                    formatted[k] = v
                    continue

                # Shape (3): ``"task_logits"`` parallel view —
                # ``{task: tensor}``. Flatten each entry and keep
                # the dict itself for back-compat.
                if k == "task_logits" and isinstance(v, dict):
                    for task_name, task_logits in v.items():
                        if isinstance(task_logits, torch.Tensor):
                            # Shape (2) above already produced
                            # ``<task>_logits`` for every per-task
                            # dict entry. Skip the duplicate work
                            # but still expose the parallel view so
                            # callers that read ``task_logits`` see
                            # the same tensor.
                            formatted.setdefault(
                                f"{task_name}_logits",
                                torch.nan_to_num(task_logits),
                            )
                    formatted[k] = v
                    continue

                formatted[k] = v

            return formatted

        raise RuntimeError("Unsupported output")

    # =====================================================
    # CALIBRATION
    # =====================================================

    def _calibrate(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:

        if self.temperature_scaler:
            try:
                scaled = self.temperature_scaler.predict_proba(logits)
                return scaled.to(self.device)
            except Exception:
                pass

        if self.isotonic_calibrator:
            try:
                np_probs = probs.detach().cpu().numpy()
                calibrated = self.isotonic_calibrator.predict_proba(np_probs)
                return torch.tensor(calibrated, device=self.device, dtype=probs.dtype)
            except Exception:
                pass

        return probs

    # =====================================================
    # FAKE DETECTION
    # =====================================================

    def build_fake_real_output(self, formatted: Dict[str, Any]) -> Dict[str, Any]:

        probs = self._extract_fake_probs(formatted)

        if probs is None:
            raise RuntimeError("No fake head found")

        fake_idx = self._resolve_fake_index()

        probs = probs.mean(dim=0) if probs.dim() > 1 else probs

        fake_prob = float(probs[fake_idx])
        pred_idx = int(torch.argmax(probs))

        return {
            "label": "Fake" if pred_idx == fake_idx else "Real",
            "fake_probability": float(np.clip(fake_prob, 0.0, 1.0)),
            "confidence": float(torch.max(probs)),
        }

    def _extract_fake_probs(self, formatted: Dict[str, Any]) -> Optional[torch.Tensor]:

        for key in _FAKE_HEAD_KEYS:
            p = formatted.get(key.replace("_logits", "_probabilities"))
            if isinstance(p, torch.Tensor):
                return p

        logits = _find_tensor_by_keys(formatted, _FAKE_HEAD_KEYS)

        if isinstance(logits, torch.Tensor):
            return torch.softmax(torch.nan_to_num(logits), dim=-1)

        if "logits" in formatted:
            logits = formatted["logits"]
            return torch.softmax(torch.nan_to_num(logits), dim=-1)

        return None

    def _resolve_fake_index(self) -> int:

        config = getattr(self.model, "config", None)

        if config:

            id2label = getattr(config, "id2label", None)
            if isinstance(id2label, dict):
                for idx, label in id2label.items():
                    if str(label).lower() in FAKE_LABEL_CANDIDATES:
                        return int(idx)

        return DEFAULT_FAKE_INDEX

    # =====================================================
    # UTILS
    # =====================================================

    def _squeeze(self, data: Dict[str, Any]) -> Dict[str, Any]:

        return {
            k: v.squeeze(0) if isinstance(v, torch.Tensor) and v.dim() > 0 else v
            for k, v in data.items()
        }

    def _run_ensemble(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:

        out = self.ensemble_model(**batch)

        if isinstance(out, torch.Tensor):
            return {"ensemble_logits": out}

        if isinstance(out, dict):
            if "logits" in out:
                return {"ensemble_logits": out["logits"]}
            raise RuntimeError("Invalid ensemble output")

        raise RuntimeError("Unsupported ensemble output")