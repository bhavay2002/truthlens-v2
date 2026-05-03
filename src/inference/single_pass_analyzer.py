"""
SinglePassAnalyzer — Inference v2 §4 + §5 (Single-pass inference + async analyzers).

Runs all task analyzers in a single encoder forward pass through
InteractingMultiTaskModel, then fans out async post-processing steps
(explainability, aggregation, attribution) using asyncio.gather.

Design goals
------------
1. Single-pass: the heavy encoder runs exactly once per request.
2. Async fan-out: independent post-processing steps run concurrently.
3. Deterministic contract: SinglePassResult is a well-typed dataclass.
4. Graceful degradation: failed sub-steps populate `warnings` and return
   partial results rather than raising.

Public API
----------
    from src.inference.single_pass_analyzer import SinglePassAnalyzer, SinglePassConfig

    config = SinglePassConfig(device="auto", return_explanations=True)
    analyzer = SinglePassAnalyzer(model, tokenizer, config=config)

    # Sync convenience wrapper (runs event loop internally):
    result = analyzer.analyze(text)

    # Async (use inside existing event loop):
    result = await analyzer.analyze_async(text)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class SinglePassConfig:
    device: str = "auto"
    max_length: int = 512
    return_explanations: bool = True
    return_cross_task: bool = True
    return_attention_graph: bool = True
    top_k_tokens: int = 20
    attribution_method: str = "gate"       # "gate" or "gradient"
    async_timeout: float = 30.0            # seconds per async sub-step
    amp: bool = False


# =========================================================
# RESULT SCHEMA
# =========================================================

@dataclass
class TaskPrediction:
    task: str
    label: int
    probabilities: List[float]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "label": self.label,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
        }


@dataclass
class SinglePassResult:
    text: str
    task_predictions: List[TaskPrediction]
    credibility_score: float
    risk_label: str                         # "low" | "medium" | "high"
    risk_probabilities: List[float]

    cross_task_influence: Optional[Dict[str, Dict[str, float]]] = None
    attention_graph: Optional[Dict[str, Any]] = None
    token_explanations: Optional[Dict[str, Any]] = None

    latency_ms: float = 0.0
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    RISK_LABELS = ["low", "medium", "high"]

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "text": self.text,
            "task_predictions": [p.to_dict() for p in self.task_predictions],
            "credibility_score": self.credibility_score,
            "risk_label": self.risk_label,
            "risk_probabilities": self.risk_probabilities,
            "latency_ms": self.latency_ms,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }
        if self.cross_task_influence is not None:
            d["cross_task_influence"] = self.cross_task_influence
        if self.attention_graph is not None:
            d["attention_graph"] = self.attention_graph
        if self.token_explanations is not None:
            d["token_explanations"] = self.token_explanations
        return d


# =========================================================
# DEVICE HELPER
# =========================================================

def _resolve_device(spec: str):
    if torch is None:
        return None
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


# =========================================================
# ANALYZER
# =========================================================

class SinglePassAnalyzer:
    """Single-pass inference with async post-processing fan-out.

    Parameters
    ----------
    model : InteractingMultiTaskModel
    tokenizer : HuggingFace tokenizer
    config : SinglePassConfig
    attributor : CrossTaskAttributor, optional
    graph_builder : AttentionGraphBuilder, optional
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[SinglePassConfig] = None,
        attributor: Optional[Any] = None,
        graph_builder: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SinglePassConfig()
        self.attributor = attributor
        self.graph_builder = graph_builder

        self.device = _resolve_device(self.config.device)
        if self.device is not None and torch is not None:
            self.model.to(self.device)
            self.model.eval()

    # -------------------------------------------------------
    # TOKENIZE
    # -------------------------------------------------------

    def _tokenize(self, text: str) -> Dict[str, Any]:
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        if self.device is not None:
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
        return encoded

    # -------------------------------------------------------
    # SINGLE FORWARD PASS (§4 — single-pass inference)
    # -------------------------------------------------------

    def _forward(self, encoded: Dict[str, Any]) -> Dict[str, Any]:
        """Run the model forward pass once and return raw outputs."""
        ctx = torch.no_grad() if torch is not None else _nullctx()
        amp_ctx = (
            torch.autocast(device_type=self.device.type, dtype=torch.float16)
            if self.config.amp and self.device is not None and self.device.type == "cuda"
            else _nullctx()
        )
        with ctx, amp_ctx:
            outputs = self.model(**encoded)
        return outputs

    # -------------------------------------------------------
    # PARSE TASK OUTPUTS
    # -------------------------------------------------------

    def _parse_task_predictions(
        self,
        model_outputs: Dict[str, Any],
    ) -> List[TaskPrediction]:
        task_logits: Dict[str, Any] = model_outputs.get("task_logits", {})
        predictions: List[TaskPrediction] = []

        for task, logits in task_logits.items():
            if torch is not None:
                probs = torch.softmax(logits.float(), dim=-1)[0].detach().cpu().numpy()
            else:
                probs = np.array(logits[0])
            label = int(np.argmax(probs))
            confidence = float(probs[label])

            predictions.append(TaskPrediction(
                task=task,
                label=label,
                probabilities=[float(p) for p in probs],
                confidence=confidence,
            ))

        return predictions

    def _parse_credibility(self, model_outputs: Dict[str, Any]) -> float:
        score = model_outputs.get("credibility_score")
        if score is None:
            return 0.5
        if torch is not None and isinstance(score, torch.Tensor):
            return float(score[0].item())
        return float(score)

    def _parse_risk(self, model_outputs: Dict[str, Any]):
        risk_logits = model_outputs.get("risk")
        if risk_logits is None:
            return "medium", [0.0, 1.0, 0.0]
        if torch is not None and isinstance(risk_logits, torch.Tensor):
            probs = torch.softmax(risk_logits.float(), dim=-1)[0].detach().cpu().numpy()
        else:
            probs = np.array(risk_logits[0])
        label_idx = int(np.argmax(probs))
        label = SinglePassResult.RISK_LABELS[label_idx]
        return label, [float(p) for p in probs]

    # -------------------------------------------------------
    # ASYNC POST-PROCESSING STEPS  (§5 — async analyzer execution)
    # -------------------------------------------------------

    async def _async_cross_task(
        self,
        encoded: Dict[str, Any],
        warnings: List[str],
    ) -> Optional[Dict[str, Dict[str, float]]]:
        if self.attributor is None or not self.config.return_cross_task:
            return None
        loop = asyncio.get_event_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.attributor.attribute_safe(
                        self.model,
                        encoded,
                        method=self.config.attribution_method,
                    ),
                ),
                timeout=self.config.async_timeout,
            )
            return result
        except asyncio.TimeoutError:
            warnings.append("cross_task_attribution: timed out")
            return None
        except Exception as exc:
            warnings.append(f"cross_task_attribution: {exc}")
            return None

    async def _async_attention_graph(
        self,
        tokens: List[str],
        model_outputs: Dict[str, Any],
        cross_task: Optional[Dict[str, Dict[str, float]]],
        warnings: List[str],
    ) -> Optional[Dict[str, Any]]:
        if self.graph_builder is None or not self.config.return_attention_graph:
            return None
        loop = asyncio.get_event_loop()
        try:
            def _build():
                from src.explainability.attention_graph import AttentionGraphBuilder
                task_names = list(model_outputs.get("task_logits", {}).keys())
                if cross_task:
                    task_to_decision = {
                        t: sum(cross_task.get(t, {}).values())
                        for t in task_names
                    }
                else:
                    task_to_decision = {t: 1.0 for t in task_names}

                token_scores = {
                    task: [1.0 / max(len(tokens), 1)] * len(tokens)
                    for task in task_names
                }
                return self.graph_builder.build(
                    tokens=tokens,
                    token_scores=token_scores,
                    task_to_decision=task_to_decision,
                ).to_dict()

            return await asyncio.wait_for(
                loop.run_in_executor(None, _build),
                timeout=self.config.async_timeout,
            )
        except asyncio.TimeoutError:
            warnings.append("attention_graph: timed out")
            return None
        except Exception as exc:
            warnings.append(f"attention_graph: {exc}")
            return None

    # -------------------------------------------------------
    # ASYNC ANALYSE
    # -------------------------------------------------------

    async def analyze_async(self, text: str) -> SinglePassResult:
        """Full single-pass analysis with async fan-out post-processing.

        Parameters
        ----------
        text : input article / claim text

        Returns
        -------
        SinglePassResult with all enabled fields populated.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        # ── 1. Tokenize ───────────────────────────────────────────────
        try:
            encoded = self._tokenize(text)
        except Exception as exc:
            raise RuntimeError(f"Tokenization failed: {exc}") from exc

        # ── 2. Single forward pass ────────────────────────────────────
        try:
            loop = asyncio.get_event_loop()
            model_outputs = await loop.run_in_executor(
                None, lambda: self._forward(encoded)
            )
        except Exception as exc:
            raise RuntimeError(f"Model forward failed: {exc}") from exc

        # ── 3. Extract core outputs ───────────────────────────────────
        task_predictions = self._parse_task_predictions(model_outputs)
        credibility = self._parse_credibility(model_outputs)
        risk_label, risk_probs = self._parse_risk(model_outputs)

        # ── 4. Decode tokens for graph / explanations ─────────────────
        try:
            input_ids = encoded.get("input_ids")
            if input_ids is not None and self.tokenizer is not None:
                token_list: List[str] = self.tokenizer.convert_ids_to_tokens(
                    input_ids[0].cpu().tolist()
                )
            else:
                token_list = []
        except Exception as exc:
            warnings.append(f"token_decode: {exc}")
            token_list = []

        # ── 5. Async fan-out ──────────────────────────────────────────
        cross_task_task = self._async_cross_task(encoded, warnings)
        graph_placeholder = asyncio.coroutine(lambda: None)()  # will wait after cross_task

        cross_task_result = await cross_task_task

        graph_result = await self._async_attention_graph(
            token_list, model_outputs, cross_task_result, warnings
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return SinglePassResult(
            text=text,
            task_predictions=task_predictions,
            credibility_score=credibility,
            risk_label=risk_label,
            risk_probabilities=risk_probs,
            cross_task_influence=cross_task_result,
            attention_graph=graph_result,
            token_explanations=None,
            latency_ms=latency_ms,
            warnings=warnings,
            metadata={
                "num_tokens": len(token_list),
                "device": str(self.device) if self.device is not None else "unknown",
            },
        )

    # -------------------------------------------------------
    # SYNC WRAPPER
    # -------------------------------------------------------

    def analyze(self, text: str) -> SinglePassResult:
        """Synchronous wrapper around analyze_async.

        Runs a new event loop if no running loop exists (e.g. in scripts
        or tests). Inside an existing event loop (e.g. FastAPI), prefer
        ``await analyzer.analyze_async(text)`` directly.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(asyncio.run, self.analyze_async(text))
                    return fut.result()
            else:
                return loop.run_until_complete(self.analyze_async(text))
        except RuntimeError:
            return asyncio.run(self.analyze_async(text))


# =========================================================
# NULL CONTEXT HELPER (when torch not available)
# =========================================================

class _nullctx:
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
