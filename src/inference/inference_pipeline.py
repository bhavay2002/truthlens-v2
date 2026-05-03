from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from src.aggregation.aggregation_pipeline import AggregationPipeline
from src.config.task_config import TASK_CONFIG
from src.inference.postprocessing import Postprocessor, PostprocessingConfig
from src.explainability.orchestrator import ExplainabilityOrchestrator
from src.graph.graph_pipeline import GraphPipeline, get_default_pipeline


# DEV-2: training writes AMP dtype via TRUTHLENS_AMP_DTYPE; inference must
# read the same knob. fp16 (the previous hardcoded default) has a narrower
# dynamic range than bf16 — the unbalanced bias head will overflow for
# some logits when the trainer used bf16. Map "bf16"/"fp16"/"fp32" to the
# matching torch dtype; default to bf16 if env is unset and CUDA is
# available, else fp16 (legacy CUDA-only behaviour).
def _resolve_amp_dtype(default: str = "float16") -> torch.dtype:
    requested = (os.environ.get("TRUTHLENS_AMP_DTYPE") or default).lower()
    if requested in ("bf16", "bfloat16"):
        return torch.bfloat16
    if requested in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32

from src.monitoring.feature_logger import (
    log_request_latency,
    log_failure,
    time_block,
)

import time

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class PredictionPipelineConfig:
    # CFG-4: previously defaulted to "cpu", which silently masked GPU
    # availability whenever a caller forgot to pass ``device``. "auto"
    # mirrors every other inference entry point (engine, loader,
    # predict_api) so a single GPU box uses the GPU by default.
    device: str = "auto"
    return_probabilities: bool = True
    # PP-2: optional path to a JSON {task: float} of per-task thresholds
    # produced by training's threshold optimiser. None falls back to 0.5.
    task_thresholds_path: Optional[str] = None
    # PP-2: explicit per-task threshold overrides (take precedence over
    # the file). Useful for ad-hoc tuning at deploy time.
    task_thresholds: Optional[Dict[str, float]] = None


ExplainabilityLayer = ExplainabilityOrchestrator


# =========================================================
# PIPELINE
# =========================================================

class PredictionPipeline:

    def __init__(
        self,
        config: PredictionPipelineConfig,
        bias_model: Optional[torch.nn.Module] = None,
        ideology_model: Optional[torch.nn.Module] = None,
        propaganda_model: Optional[torch.nn.Module] = None,
        emotion_model: Optional[torch.nn.Module] = None,
        explainability_layer: Optional[ExplainabilityLayer] = None,
        aggregation_pipeline: Optional[AggregationPipeline] = None,
    ) -> None:

        self.config = config
        # CFG-4: resolve "auto" → "cuda" if available, else "cpu". Without
        # this, ``torch.device("auto")`` would raise.
        resolved = config.device
        if resolved == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(resolved)

        self.bias_model = bias_model
        self.ideology_model = ideology_model
        self.propaganda_model = propaganda_model
        self.emotion_model = emotion_model

        self.explainability_layer = explainability_layer
        self.aggregation_pipeline = aggregation_pipeline or AggregationPipeline()

        # PP-2: build a postprocessor whose ``task_thresholds`` is
        # populated from (a) a JSON file emitted by training and (b)
        # explicit overrides on the config — in that order so overrides
        # win. Without this, every multilabel head silently used 0.5.
        pp_config = PostprocessingConfig()
        self.postprocessor = Postprocessor(pp_config)
        if config.task_thresholds_path:
            try:
                self.postprocessor.load_task_thresholds(config.task_thresholds_path)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Failed to load thresholds from %s: %s",
                    config.task_thresholds_path, exc,
                )
        if config.task_thresholds:
            existing = dict(self.postprocessor.config.task_thresholds or {})
            existing.update(config.task_thresholds)
            self.postprocessor.config.task_thresholds = existing

        # DEV-2: cache the AMP dtype once so ``_forward_all`` does not
        # re-read the env on every call.
        self._amp_dtype = _resolve_amp_dtype()

        # 🔥 NEW — G-R1: share the process-wide singleton.
        self.graph_pipeline = get_default_pipeline()

        for name in ["bias_model", "ideology_model", "propaganda_model", "emotion_model"]:
            model = getattr(self, name)
            if model:
                model.to(self.device)
                model.eval()

        logger.info("PredictionPipeline initialized")

    # =====================================================
    # CORE FORWARD
    # =====================================================

    def _forward_all(self, features: torch.Tensor) -> Dict[str, Any]:

        outputs = {}

        # DEV-2: dtype matches the training-time TRUTHLENS_AMP_DTYPE
        # (default bf16). Hardcoded fp16 here would overflow tail-class
        # logits trained under bf16.
        ctx = (
            torch.autocast("cuda", dtype=self._amp_dtype)
            if self.device.type == "cuda"
            else nullcontext()
        )

        with torch.no_grad():
            with ctx:

                if self.bias_model:
                    outputs["bias"] = self.bias_model(features)

                if self.ideology_model:
                    outputs["ideology"] = self.ideology_model(features)

                if self.propaganda_model:
                    outputs["propaganda"] = self.propaganda_model(features)

                if self.emotion_model:
                    outputs["emotion"] = self.emotion_model(features)

        return outputs

    def _extract_logits(self, out):

        if isinstance(out, dict) and "logits" in out:
            return out["logits"]

        if hasattr(out, "logits"):
            return out.logits

        if isinstance(out, torch.Tensor):
            return out

        raise RuntimeError("Invalid model output")

    # =====================================================
    # MULTI-TASK
    # =====================================================
    #
    # CRIT-3: per-task output type is driven by ``TASK_CONFIG`` (i.e. the
    # ``tasks:`` block in ``config/config.yaml``) — never by hardcoded
    # constants in this file. The previous ``_BINARY_TASKS = {"propaganda"}``
    # contradicted the YAML (which marks propaganda as ``multiclass``) and
    # silently produced wrong predictions.

    _MULTILABEL_THRESHOLD = 0.5

    def _resolve_task_type(self, task: str) -> str:
        try:
            return str(TASK_CONFIG[task]["type"])
        except (KeyError, TypeError):
            logger.warning("No task type registered for %s; defaulting to multiclass", task)
            return "multiclass"

    def _resolve_threshold(self, task: str) -> float:
        """PP-2: per-task threshold lookup.

        Order: postprocessor.config.task_thresholds → task registry's
        configured threshold → 0.5 default.
        """
        thresholds = self.postprocessor.config.task_thresholds or {}
        if task in thresholds:
            return float(thresholds[task])
        try:
            registered = TASK_CONFIG[task].get("threshold")
            if registered is not None:
                return float(registered)
        except (KeyError, TypeError):
            pass
        return self._MULTILABEL_THRESHOLD

    def predict_multitask(self, features: torch.Tensor) -> Dict[str, Any]:

        outputs = self._forward_all(features)

        results = {}

        for task, out in outputs.items():

            logits = self._extract_logits(out)

            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            task_type = self._resolve_task_type(task)
            threshold = self._resolve_threshold(task)

            if task_type == "multilabel":
                probs = torch.sigmoid(logits)
                preds = (probs >= threshold).int()
            elif task_type == "binary":
                probs = torch.sigmoid(logits)
                if logits.shape[-1] == 1:
                    preds = (probs >= threshold).int().squeeze(-1)
                else:
                    preds = torch.argmax(probs, dim=-1)
            else:
                # multiclass
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

            results[task] = {
                "logits": logits.detach().cpu().numpy(),
                "probabilities": probs.detach().cpu().numpy(),
                "predictions": preds.detach().cpu().numpy(),
                # PP-4: surface task_type so downstream uncertainty
                # computation can pick the correct entropy formula.
                "task_type": task_type,
            }

        return results

    # =====================================================
    # POSTPROCESSING
    # =====================================================

    def predict_with_postprocessing(
        self,
        features: torch.Tensor,
        *,
        task_types: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:

        raw = self.predict_multitask(features)

        # CRIT-8: drive the postprocessor's task types from the task
        # registry whenever the caller does not override them, so the
        # multilabel/multiclass branches stay in lockstep with the YAML.
        if task_types is None:
            task_types = {task: self._resolve_task_type(task) for task in raw}

        return self.postprocessor.process(
            raw,
            task_types=task_types,
        )

    # =====================================================
    # MAIN PREDICT
    # =====================================================

    def predict(self, features: torch.Tensor) -> Dict[str, Any]:

        processed = self.predict_with_postprocessing(features)

        batch_size = int(features.shape[0])

        result = {
            "bias": [],
            "ideology": [],
            "propaganda_probability": [],
            "emotion": [],
        }

        for i in range(batch_size):

            for task, out in processed.items():

                if task == "bias":
                    result["bias"].append(out["labels"][i])

                elif task == "ideology":
                    result["ideology"].append(out["labels"][i])

                elif task == "propaganda":
                    prob = out["probabilities"][i]
                    result["propaganda_probability"].append(float(prob[1]))

                elif task == "emotion":
                    result["emotion"].append(out["probabilities"][i].tolist())

        return result

    # =====================================================
    #  NEW FULL OUTPUT (SERVICE READY)
    # =====================================================

    def predict_full(
        self,
        features: torch.Tensor,
        *,
        text: Optional[str] = None,
        predict_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
       ) -> Dict[str, Any]:
    
        start_total = time.time()
    
        try:
            # ---------------- BASE PREDICTION ----------------
            with time_block("model_prediction", task="inference"):
                prediction = self.predict(features)
    
            # ---------------- AGGREGATION ----------------
            with time_block("aggregation", task="inference"):
                profile = self.aggregation_pipeline.build_profile_from_prediction(prediction)
                aggregation = self.aggregation_pipeline.run(profile, text=text)
    
            scores = aggregation.get("raw_scores", {})
    
            # ---------------- GRAPH ----------------
            graph_output = None
            if text:
                try:
                    with time_block("graph_pipeline", task="inference"):
                        graph_output = self.graph_pipeline.run(text)
                except Exception as e:
                    log_failure(e, context={"stage": "graph_pipeline"})
                    logger.warning("Graph pipeline failed: %s", e)
    
            # ---------------- EXPLAINABILITY ----------------
            explanation = {}
            if self.explainability_layer and text and predict_fn:
                try:
                    with time_block("explainability", task="inference"):
                        explanation = self.explainability_layer.explain(
                            text=text,
                            predict_fn=predict_fn,
                        )
                except Exception as e:
                    log_failure(e, context={"stage": "explainability"})
                    logger.warning("Explainability failed: %s", e)
    
            # ---------------- TOTAL LATENCY ----------------
            total_latency = time.time() - start_total
    
            log_request_latency(
                total_latency,
                task="full_inference",
            )
    
            # ---------------- FINAL OUTPUT ----------------
            return {
                "prediction": prediction,
                "scores": scores,
                "analysis_modules": {
                    "graph": graph_output,
                    "graph_explanation": explanation.get("graph_explanation"),
                },
                "explanation": explanation,
                "meta": {
                    "total_latency": round(total_latency, 4),
                },
            }
    
        except Exception as e:
            log_failure(
                e,
                context={
                    "stage": "predict_full",
                    "has_text": text is not None,
                },
            )
            raise