from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

try:
    import onnx
except ImportError:
    onnx = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None


logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class ONNXExportConfig:
    opset_version: int = 17
    dynamic_batch: bool = True
    export_params: bool = True
    do_constant_folding: bool = True
    input_name: str = "input"
    output_name: str = "output"
    verify_export: bool = True
    device: str = "cpu"
    atol: float = 1e-4

    def __post_init__(self):
        if self.opset_version < 9:
            raise ValueError("opset_version must be >= 9")


# =========================================================
# EXPORTER
# =========================================================

class ONNXExporter:

    def __init__(self, config: Optional[ONNXExportConfig] = None):
        self.config = config or ONNXExportConfig()

    # -----------------------------------------------------

    def _extract_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, dict):
            if "logits" in output:
                return output["logits"]
            return next(iter(output.values()))
        if isinstance(output, tuple):
            return output[0]
        raise TypeError("Unsupported output type")

    # -----------------------------------------------------

    def _dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        if not self.config.dynamic_batch:
            return {}
        return {
            self.config.input_name: {0: "batch"},
            self.config.output_name: {0: "batch"},
        }

    # -----------------------------------------------------

    def export(
        self,
        model: torch.nn.Module,
        example_input: torch.Tensor,
        output_path: str | Path,
    ) -> Path:

        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be nn.Module")

        if not isinstance(example_input, torch.Tensor):
            raise TypeError("example_input must be Tensor")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        device = torch.device(self.config.device)

        model = model.to(device)
        example_input = example_input.to(device)

        model.eval()

        logger.info("Exporting ONNX: %s", output_path)

        torch.onnx.export(
            model,
            example_input,
            str(output_path),
            export_params=self.config.export_params,
            opset_version=self.config.opset_version,
            do_constant_folding=self.config.do_constant_folding,
            input_names=[self.config.input_name],
            output_names=[self.config.output_name],
            dynamic_axes=self._dynamic_axes() or None,
        )

        logger.info("ONNX export complete")

        if self.config.verify_export:
            ok, diff = self.verify(output_path, model, example_input)
            if not ok:
                raise RuntimeError(f"ONNX verification failed diff={diff}")
            logger.info("ONNX verified diff=%.6f", diff)

        return output_path

    # -----------------------------------------------------

    def verify(
        self,
        onnx_path: str | Path,
        model: torch.nn.Module,
        example_input: torch.Tensor,
    ) -> Tuple[bool, float]:

        if onnx is None or ort is None:
            raise ImportError("onnx + onnxruntime required")

        onnx_path = Path(onnx_path)

        if not onnx_path.exists():
            raise FileNotFoundError(onnx_path)

        logger.info("Verifying ONNX: %s", onnx_path)

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )

        device = next(model.parameters()).device
        example_input = example_input.to(device)

        with torch.no_grad():
            pt_out = self._extract_output(
                model(example_input)
            ).detach().cpu().numpy()

        ort_inputs = {
            session.get_inputs()[0].name: example_input.cpu().numpy()
        }

        ort_out = session.run(None, ort_inputs)[0]

        diff = float(np.max(np.abs(pt_out - ort_out)))
        ok = diff <= self.config.atol

        logger.info("Verification diff=%.6f ok=%s", diff, ok)

        return ok, diff