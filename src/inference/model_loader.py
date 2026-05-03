from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, List

import joblib
import torch
import numpy as np
from transformers import AutoTokenizer


# REC-3: process-wide tokenizer cache. The same tokenizer was being
# materialised twice — once in ``InferenceEngine._load_model`` and once
# in ``ModelLoader._load_tokenizer`` — paying the (non-trivial) HF
# vocabulary read cost twice on every cold start. Keying by the
# resolved path lets every entry point share one fast-tokenizer
# instance per artifact directory; ``use_fast=True`` tokenizers are
# stateless and thread-safe for inference.
_TOKENIZER_CACHE: Dict[str, Any] = {}
_TOKENIZER_LOCK: Lock = Lock()


def get_cached_tokenizer(path: Path | str):
    """Return a cached fast tokenizer for ``path`` (loading once)."""
    if path is None:
        return None
    key = str(Path(path).resolve())
    with _TOKENIZER_LOCK:
        cached = _TOKENIZER_CACHE.get(key)
        if cached is not None:
            return cached
        if not Path(key).exists():
            return None
        tok = AutoTokenizer.from_pretrained(key, use_fast=True)
        _TOKENIZER_CACHE[key] = tok
        return tok

from src.models.config import ModelConfigLoader, MultiTaskModelConfig
from src.models.metadata.model_metadata import ModelMetadata
from src.models.inference.predictor import Predictor
from src.models.registry.model_factory import ModelFactory

# PP-1: task-type registry drives the activation choice (sigmoid vs.
# softmax) inside ``UnifiedPredictor._format_output``. Without this the
# multilabel ``emotion`` head was being softmax-collapsed into a single
# argmax label at inference time.
try:
    from src.config.task_config import TASK_CONFIG as _TASK_CONFIG
except Exception:  # pragma: no cover - registry optional
    _TASK_CONFIG = None

logger = logging.getLogger(__name__)


# =========================================================
# ARTIFACT CONTAINER
# =========================================================

@dataclass
class ModelArtifacts:
    bias_model: Optional[torch.nn.Module] = None
    ideology_model: Optional[torch.nn.Module] = None
    emotion_model: Optional[torch.nn.Module] = None

    multitask_model: Optional[torch.nn.Module] = None

    tokenizer: Optional[Any] = None

    feature_scaler: Optional[Any] = None
    feature_selector: Optional[Any] = None
    feature_schema: Optional[Dict[str, Any]] = None

    model_metadata: Optional[ModelMetadata] = None
    model_config: Optional[MultiTaskModelConfig] = None

    bias_predictor: Optional[Predictor] = None
    ideology_predictor: Optional[Predictor] = None
    emotion_predictor: Optional[Predictor] = None
    multitask_predictor: Optional[Predictor] = None

    unified_predictor: Optional["UnifiedPredictor"] = None


# =========================================================
# UNIFIED PREDICTOR
# =========================================================

class UnifiedPredictor:

    def __init__(self, artifacts: ModelArtifacts, device: torch.device):
        self.artifacts = artifacts
        self.device = device

    @staticmethod
    def _resolve_task_type(task: str) -> str:
        """PP-1: pick activation by task type from the registry.

        ``multilabel`` heads (e.g. emotion) MUST use sigmoid; only
        ``multiclass`` heads use softmax. The previous unconditional
        softmax collapsed 20 independent emotion sigmoids into a
        single Categorical, destroying multilabel semantics.
        """
        if _TASK_CONFIG is None or task not in _TASK_CONFIG:
            return "multiclass"
        return str(_TASK_CONFIG[task].get("type", "multiclass"))

    def _format_output(self, raw: Dict[str, Any], task: str = "multiclass"):

        logits = raw.get("logits")
        probs = raw.get("probabilities")

        if logits is not None:
            logits = np.asarray(logits)

        task_type = self._resolve_task_type(task) if isinstance(task, str) else "multiclass"

        if probs is None and logits is not None:
            t = torch.as_tensor(logits)
            if task_type in ("multilabel", "binary"):
                probs = torch.sigmoid(t).numpy()
            else:
                probs = torch.softmax(t, dim=-1).numpy()

        preds = None
        if probs is not None:
            if task_type == "multilabel":
                # PP-1: per-label binarisation rather than argmax across labels.
                preds = (probs >= 0.5).astype(np.int64)
            elif task_type == "binary" and probs.ndim == 1:
                preds = (probs >= 0.5).astype(np.int64)
            else:
                preds = np.argmax(probs, axis=1)

        return {
            "logits": logits,
            "probabilities": probs,
            "predictions": preds,
            "task_type": task_type,
        }

    def predict_for_evaluation(self, texts: List[str]):

        # ---------------- MULTITASK ----------------
        if self.artifacts.multitask_predictor:
            raw = self.artifacts.multitask_predictor.predict(texts)
            # Multitask predictor returns a dict keyed by head; the
            # caller can re-key per-task downstream. The umbrella
            # "multitask" entry stays multiclass-shaped to avoid
            # changing its contract here.
            return {"multitask": self._format_output(raw, "multitask")}

        # ---------------- SINGLE TASK ----------------
        outputs = {}

        for name, predictor in {
            "bias": self.artifacts.bias_predictor,
            "ideology": self.artifacts.ideology_predictor,
            "emotion": self.artifacts.emotion_predictor,
        }.items():

            if predictor is None:
                continue

            raw = predictor.predict(texts)
            outputs[name] = self._format_output(raw, name)

        return outputs


# =========================================================
# MODEL LOADER
# =========================================================

class ModelLoader:

    def __init__(
        self,
        models_dir: str,
        device: str = "auto",
        *,
        use_torch_compile: bool = False,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.device = self._resolve_device(device)
        # LAT-6: torch.compile is opt-in. The previous unconditional call
        # paid a 60+ second graph-capture cost on first use of every
        # model on CUDA — a clear loss for short-lived workers.
        self.use_torch_compile = bool(use_torch_compile)

        if not self.models_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.models_dir}")

        logger.info("ModelLoader initialized at %s", self.models_dir)

    # =====================================================
    # DEVICE
    # =====================================================

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    # =====================================================
    # LOAD HELPERS
    # =====================================================

    def _load_torch_model(self, path: Path) -> Optional[torch.nn.Module]:

        if not path.exists():
            logger.warning("Model not found: %s", path)
            return None

        obj = torch.load(path, map_location="cpu")

        if isinstance(obj, dict) and "state_dict" in obj:
            raise RuntimeError(f"{path} contains state_dict, not full model.")

        if not isinstance(obj, torch.nn.Module):
            raise RuntimeError(f"Unsupported model object: {type(obj)}")

        model = obj

        # DEV-1: do NOT cast model weights to fp16 unconditionally.
        # Calibration (TemperatureScaler / IsotonicCalibrator) was fit
        # on fp32 logits during validation; running the forward pass
        # in fp16 weights raises ECE by 1-2 percentage points and the
        # error is silent. AMP autocast (used downstream) gives the
        # speed-up of fp16/bf16 compute while keeping accumulators in
        # fp32 — best of both worlds.
        model.to(self.device)

        # COMPILE-OFF: ``torch.compile`` removed project-wide (see
        # ``training_setup.optimize_model`` for the full rationale).
        # The ``use_torch_compile`` config field remains as inert
        # back-compat plumbing for older callers; flipping it on no
        # longer triggers compilation.
        if self.use_torch_compile:
            logger.info(
                "Inference compile request ignored (COMPILE-OFF); "
                "running in eager mode."
            )

        model.eval()
        # DEV-3: ``.eval()`` only disables Dropout / BatchNorm running
        # stats; parameters still carry ``requires_grad=True``, which
        # forces autograd version-counter allocations on every forward
        # pass — wasted work for inference. Freeze the parameters once.
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def _load_tokenizer(self, path: Path):
        # REC-3: route through the process-wide cache so a second
        # ``ModelLoader`` (or ``InferenceEngine``) for the same artifact
        # path reuses the same tokenizer instance.
        return get_cached_tokenizer(path)

    def _load_joblib(self, path: Path):
        return joblib.load(path) if path.exists() else None

    def _load_json(self, path: Path):
        if not path.exists():
            return None
        with open(path, "r") as f:
            return json.load(f)

    # =====================================================
    # MAIN LOAD
    # =====================================================

    def load_all(self) -> ModelArtifacts:

        artifacts = ModelArtifacts()

        # ---------------- MODELS ----------------
        artifacts.bias_model = self._load_torch_model(self.models_dir / "bias_model.pt")
        artifacts.ideology_model = self._load_torch_model(self.models_dir / "ideology_model.pt")
        artifacts.emotion_model = self._load_torch_model(self.models_dir / "emotion_model.pt")

        # ---------------- TOKENIZER ----------------
        tokenizer_path = self.models_dir / "tokenizer"
        if not tokenizer_path.exists():
            tokenizer_path = self.models_dir

        artifacts.tokenizer = self._load_tokenizer(tokenizer_path)

        # ---------------- FEATURES ----------------
        artifacts.feature_scaler = self._load_joblib(self.models_dir / "feature_scaler.pkl")
        artifacts.feature_selector = self._load_joblib(self.models_dir / "feature_selector.pkl")
        artifacts.feature_schema = self._load_json(self.models_dir / "feature_schema.json")

        # ---------------- METADATA ----------------
        artifacts.model_metadata = self.load_model_metadata()
        artifacts.model_config = self.load_model_config()

        # ---------------- PREDICTORS ----------------
        artifacts.bias_predictor = self._build_predictor(artifacts.bias_model)
        artifacts.ideology_predictor = self._build_predictor(artifacts.ideology_model)
        artifacts.emotion_predictor = self._build_predictor(artifacts.emotion_model)

        # ---------------- MULTITASK ----------------
        artifacts.multitask_model = self.load_multitask_model(artifacts.model_config)
        artifacts.multitask_predictor = self._build_predictor(artifacts.multitask_model)

        # ---------------- UNIFIED ----------------
        artifacts.unified_predictor = UnifiedPredictor(artifacts, self.device)

        return artifacts

    # =====================================================
    # BUILDERS
    # =====================================================

    def _build_predictor(self, model):

        if model is None:
            return None

        return Predictor(model=model, device=self.device)

    # =====================================================
    # MULTITASK
    # =====================================================

    def load_multitask_model(self, config):

        if config is None:
            return None

        model = ModelFactory.create_from_model_config(config)

        path = self.models_dir / "multitask_model.pt"

        if path.exists():
            state = torch.load(path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)

        model.to(self.device)
        model.eval()
        # DEV-3: freeze parameters for inference (see _load_torch_model).
        for p in model.parameters():
            p.requires_grad_(False)

        return model

    # =====================================================
    # METADATA
    # =====================================================

    def load_model_metadata(self):

        path = self.models_dir / "metadata.json"

        if not path.exists():
            return None

        try:
            return ModelMetadata.load_json(path)
        except Exception:
            return None

    def load_model_config(self):

        for name in ["config.yaml", "model_config.yaml"]:
            path = self.models_dir / name
            if path.exists():
                try:
                    return ModelConfigLoader.load_multitask_config(path)
                except Exception:
                    return None

        return None

    # =====================================================
    # CACHED ARTIFACTS
    # =====================================================

    def get_artifacts(self) -> ModelArtifacts:
        """Return cached artifacts, loading once on first call."""
        if not hasattr(self, "_artifacts") or self._artifacts is None:
            self._artifacts = self.load_all()
        return self._artifacts

    # =====================================================
    # PUBLIC API
    # =====================================================

    def predict_for_evaluation(self, texts):

        artifacts = self.get_artifacts()
        return artifacts.unified_predictor.predict_for_evaluation(texts)

    def get_model_versions(self):

        meta = self.load_model_metadata()

        if not meta:
            return {}

        return {
            "model_version": getattr(meta, "version", "unknown"),
            "trained_at": getattr(meta, "timestamp", None),
        }

    def validate_features(self, features: Dict[str, Any]):

        schema = self.get_artifacts().feature_schema

        if not schema:
            return True

        missing = set(schema) - set(features)

        if missing:
            logger.warning(f"Missing features: {missing}")

        return True

    # =====================================================
    # ONNX EXPORT
    # =====================================================

    def export_onnx(self, model_name: str, output_path: str):

        model = self._load_torch_model(self.models_dir / f"{model_name}.pt")

        if model is None:
            raise ValueError(f"Model not found: {model_name}")

        dummy_ids = torch.ones(1, 16, dtype=torch.long).to(self.device)
        dummy_mask = torch.ones(1, 16, dtype=torch.long).to(self.device)

        class Wrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids, attention_mask):
                out = self.m(input_ids=input_ids, attention_mask=attention_mask)
                return out.logits if hasattr(out, "logits") else out

        torch.onnx.export(
            Wrapper(model),
            (dummy_ids, dummy_mask),
            output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
            },
            opset_version=17,
        )

        logger.info("ONNX exported to %s", output_path)