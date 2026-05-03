from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import torch
import torch.distributed as dist

# NEW MODULAR SYSTEM
from src.models.checkpointing.io_utils import atomic_save, ensure_dir
from src.models.checkpointing.integrity import attach_integrity_metadata
from src.models.checkpointing.metadata import save_metadata
from src.models.checkpointing.schema import attach_schema
from src.models.checkpointing.validator import validate_checkpoint

from src.models.export import (
    ONNXExportConfig,
    ONNXExporter,
    QuantizationConfig,
    QuantizationEngine,
    TorchScriptExportConfig,
    TorchScriptExporter,
)

from src.models.metadata.model_card import ModelCard
from src.models.metadata.model_metadata import ModelMetadata
from src.models.metadata.model_versioning import ModelVersionInfo, ModelVersionRegistry

logger = logging.getLogger(__name__)


# =========================================================
# ASYNC WRITER (FIXED)
# =========================================================

class AsyncCheckpointWriter:

    def __init__(self, max_queue_size: int = 8) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._closed = False
        self.last_error: Optional[Exception] = None
        self._thread.start()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()

            if item is None:
                self._queue.task_done()
                break

            path, obj = item

            try:
                atomic_save(obj, path)  # 🔥 USE SAFE SAVE
            except Exception as e:
                self.last_error = e
                logger.exception("Async save failed")
            finally:
                self._queue.task_done()

    def save(self, path: Path, obj: Any) -> None:
        if self._closed:
            raise RuntimeError("Writer closed")

        # C1.9: under back-pressure we MUST NOT silently drop the oldest
        # queued checkpoint — that is what most often holds the best
        # model so far and losing it corrupts run history. Drop the
        # *incoming* save instead and surface a loud WARNING so the
        # operator knows checkpoint cadence is exceeding I/O bandwidth.
        try:
            self._queue.put_nowait((path, obj))
        except queue.Full:
            logger.warning(
                "AsyncCheckpointWriter queue full (max=%d); "
                "DROPPING incoming save: %s. Increase max_queue_size "
                "or reduce checkpoint frequency.",
                self._queue.maxsize,
                path,
            )

    def flush(self) -> None:
        self._queue.join()

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._closed = True
        self._queue.put(None)
        self._thread.join()


# =========================================================
# ARTIFACT MANAGER (V2)
# =========================================================

class ArtifactManager:

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        ensure_dir(self.artifact_dir)

        self._writer = AsyncCheckpointWriter()

    # =====================================================
    # DISTRIBUTED SAFETY
    # =====================================================

    @staticmethod
    def _is_primary() -> bool:
        return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0

    # =====================================================
    # MODEL SAVE
    # =====================================================

    def save_model(
        self,
        model: torch.nn.Module,
        name: str = "model.pt",
    ) -> Path:

        path = self.artifact_dir / name

        if not self._is_primary():
            return path

        state = model.state_dict()

        validate_checkpoint(state, strict=False)

        # -------------------------
        # SCHEMA
        # -------------------------

        payload = attach_schema({
            "model_state_dict": state,
            "timestamp": time.time(),
        })

        # -------------------------
        # SAVE
        # -------------------------

        self._writer.save(path, payload)

        # -------------------------
        # METADATA + INTEGRITY
        # -------------------------

        integrity = attach_integrity_metadata(path)

        save_metadata(self.artifact_dir, {
            "type": "model",
            "file": name,
            **integrity,
        })

        return path

    # =====================================================
    # CHECKPOINT SAVE
    # =====================================================

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        step: Optional[int] = None,
    ) -> Path:

        name = f"checkpoint_{step}.pt" if step else "checkpoint.pt"
        path = self.artifact_dir / name

        if not self._is_primary():
            return path

        payload = attach_schema({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "step": step,
            "timestamp": time.time(),
        })

        validate_checkpoint(payload["model_state_dict"], strict=False)

        self._writer.save(path, payload)

        integrity = attach_integrity_metadata(path)

        save_metadata(self.artifact_dir, {
            "type": "checkpoint",
            "step": step,
            **integrity,
        })

        return path

    # =====================================================
    # EXPORT (UNCHANGED - ALREADY GOOD)
    # =====================================================

    def export_onnx(self, model, example_input, name="model.onnx", config=None):
        path = self.artifact_dir / name
        ONNXExporter(config).export(model, example_input, path)
        return path

    def export_torchscript(self, model, example_input, name="model.ts.pt", config=None):
        path = self.artifact_dir / name
        TorchScriptExporter(config).export(model, example_input, path)
        return path

    def export_quantized(self, model, name="model.quantized.pt", config=None):
        config = config or QuantizationConfig(method="dynamic")
        q_model = QuantizationEngine(config).apply(model)
        path = self.artifact_dir / name
        torch.save(q_model, path)
        return path

    # =====================================================
    # AUX
    # =====================================================

    def save_tokenizer(self, tokenizer, name="tokenizer"):
        path = self.artifact_dir / name
        path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(path)
        return path

    def save_vectorizer(self, vectorizer, name="vectorizer.joblib"):
        path = self.artifact_dir / name
        joblib.dump(vectorizer, path)
        return path

    # =====================================================
    # LOAD
    # =====================================================

    def load_model(self, name="model.pt") -> Dict[str, Any]:

        path = self.artifact_dir / name

        if not path.exists():
            raise FileNotFoundError(path)

        # PyTorch ≥2.6 changed the default of `weights_only` from False to
        # True and now warns loudly on the implicit-default code path. Pin
        # CPU mapping (we only need the state dict) and request the safer
        # `weights_only=True` deserializer explicitly so we neither emit
        # the warning nor fall victim to a future default flip back to
        # arbitrary-pickle execution.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

        validate_checkpoint(checkpoint.get("model_state_dict", {}), strict=False)

        return checkpoint

    # =====================================================
    # LIFECYCLE
    # =====================================================

    def flush(self):
        self._writer.flush()

    def close(self):
        self._writer.close()