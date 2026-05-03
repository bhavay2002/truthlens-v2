from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any

import torch

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class TorchScriptExportConfig:
    method: str = "trace"  # trace | script
    device: str = "cpu"
    verify_export: bool = True
    strict_trace: bool = True
    atol: float = 1e-4

    def __post_init__(self):
        if self.method not in {"trace", "script"}:
            raise ValueError("Invalid method")


# =========================================================
# EXPORTER
# =========================================================

class TorchScriptExporter:

    def __init__(self, config: Optional[TorchScriptExportConfig] = None):
        self.config = config or TorchScriptExportConfig()

    # -----------------------------------------------------

    def _validate(self, model: torch.nn.Module, example_input: torch.Tensor):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be nn.Module")
        if not isinstance(example_input, torch.Tensor):
            raise TypeError("example_input must be Tensor")
        if example_input.ndim == 0:
            raise ValueError("Invalid input shape")

    # -----------------------------------------------------

    def _extract_output(self, out: Any) -> torch.Tensor:
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, dict):
            if "logits" in out:
                return out["logits"]
            return next(iter(out.values()))
        if isinstance(out, tuple):
            return out[0]
        raise TypeError("Unsupported output type")

    # -----------------------------------------------------

    def _trace(self, model, example_input):
        return torch.jit.trace(
            model,
            example_input,
            strict=self.config.strict_trace,
        )

    def _script(self, model):
        return torch.jit.script(model)

    # -----------------------------------------------------

    def export(
        self,
        model: torch.nn.Module,
        example_input: torch.Tensor,
        output_path: str | Path,
    ) -> Path:

        self._validate(model, example_input)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        device = torch.device(self.config.device)

        model = model.to(device)
        example_input = example_input.to(device)

        model.eval()

        logger.info("TorchScript export (%s)", self.config.method)

        if self.config.method == "trace":
            ts_model = self._trace(model, example_input)
        else:
            ts_model = self._script(model)

        ts_model.save(str(output_path))

        logger.info("Saved TorchScript: %s", output_path)

        if self.config.verify_export:
            ok, diff = self.verify(model, ts_model, example_input)
            if not ok:
                raise RuntimeError(f"Verification failed diff={diff}")
            logger.info("Verified diff=%.6f", diff)

        return output_path

    # -----------------------------------------------------

    def verify(
        self,
        model: torch.nn.Module,
        ts_model: torch.jit.ScriptModule,
        example_input: torch.Tensor,
    ) -> Tuple[bool, float]:

        model.eval()
        ts_model.eval()

        with torch.no_grad():
            pt_out = self._extract_output(model(example_input))
            ts_out = self._extract_output(ts_model(example_input))

        diff = torch.abs(pt_out - ts_out)
        max_diff = float(diff.max().item())

        ok = max_diff <= self.config.atol

        logger.info("Verification max_diff=%.6f ok=%s", max_diff, ok)

        return ok, max_diff