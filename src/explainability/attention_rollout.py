from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch

from src.explainability.explanation_calibrator import calibrate_explanation
from src.explainability.common_schema import ExplanationOutput, TokenImportance

logger = logging.getLogger(__name__)

EPS = 1e-12


class AttentionRollout:

    def __init__(self) -> None:
        logger.info("AttentionRollout initialized")

    # =====================================================
    # VALIDATION
    # =====================================================

    @staticmethod
    def _validate_inputs(
        attentions: List[torch.Tensor],
        tokens: List[str],
        sample_index: int,
        source_token_index: int,
    ) -> int:

        if not attentions:
            raise ValueError("attentions list cannot be empty")

        if not isinstance(tokens, list) or not tokens:
            raise ValueError("tokens must be non-empty list")

        if sample_index < 0:
            raise ValueError("sample_index must be >= 0")

        seq_len = None
        batch_size = None

        for tensor in attentions:

            if tensor.ndim != 4:
                raise ValueError("attention must be (batch, heads, seq, seq)")

            b, _, s1, s2 = tensor.shape

            if s1 != s2:
                raise ValueError("attention matrices must be square")

            if batch_size is None:
                batch_size = b
            elif b != batch_size:
                raise ValueError("inconsistent batch size")

            if seq_len is None:
                seq_len = s1
            elif s1 != seq_len:
                raise ValueError("inconsistent seq_len")

        if sample_index >= batch_size:
            raise ValueError("sample_index out of range")

        if len(tokens) > seq_len:
            raise ValueError("tokens exceed seq_len")

        if not (0 <= source_token_index < seq_len):
            raise ValueError("invalid source_token_index")

        return int(seq_len)

    # =====================================================
    # CORE OPS
    # =====================================================

    @staticmethod
    def _aggregate_heads(attention: torch.Tensor, sample_index: int) -> torch.Tensor:
        return attention.mean(dim=1)[sample_index]

    @staticmethod
    def _add_residual(attention: torch.Tensor) -> torch.Tensor:
        seq_len = attention.shape[0]
        identity = torch.eye(seq_len, device=attention.device, dtype=attention.dtype)
        attention = attention + identity
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(EPS)
        return attention

    @staticmethod
    def _stack_add_residual_normalize(
        attentions_per_layer: List[torch.Tensor],
    ) -> torch.Tensor:
        """PERF-4: stack the per-layer head-averaged attentions and apply
        the residual + row-normalisation in a single fused vectorised
        pass instead of a Python ``for layer in attentions`` loop. On a
        12-layer transformer this collapses 24 small kernel launches into
        2 large ones.
        """
        stacked = torch.stack(attentions_per_layer, dim=0)
        if stacked.dtype in (torch.float16, torch.bfloat16):
            stacked = stacked.to(torch.float32)

        seq_len = stacked.shape[-1]
        identity = torch.eye(
            seq_len, device=stacked.device, dtype=stacked.dtype
        )
        stacked = stacked + identity  # broadcast over the layer dim
        denom = stacked.sum(dim=-1, keepdim=True).clamp_min(EPS)
        return stacked / denom

    # =====================================================
    # MAIN (FINAL)
    # =====================================================

    def compute_rollout(
        self,
        attentions: List[torch.Tensor],
        tokens: List[str],
        *,
        sample_index: int = 0,
        source_token_index: int = 0,
        mask_tokens: Optional[List[str]] = None,
        layer_weights: Optional[List[float]] = None,
        top_k: Optional[int] = None,
    ) -> ExplanationOutput:

        self._validate_inputs(attentions, tokens, sample_index, source_token_index)

        try:
            with torch.no_grad():

                # PERF-4: head-aggregate every layer first, then stack and
                # apply residual + normalise in a single vectorised op.
                head_avg = [
                    self._aggregate_heads(layer_attention, sample_index)
                    for layer_attention in attentions
                ]
                processed_stack = self._stack_add_residual_normalize(head_avg)

                if layer_weights:
                    n = min(len(layer_weights), processed_stack.shape[0])
                    weights = torch.ones(
                        processed_stack.shape[0],
                        device=processed_stack.device,
                        dtype=processed_stack.dtype,
                    )
                    weights[:n] = torch.tensor(
                        [float(w) for w in layer_weights[:n]],
                        device=processed_stack.device,
                        dtype=processed_stack.dtype,
                    )
                    processed_stack = processed_stack * weights.view(-1, 1, 1)

                processed: List[torch.Tensor] = list(
                    processed_stack.unbind(dim=0)
                )

                rollout = torch.linalg.multi_dot(processed[::-1])

                scores = rollout[source_token_index]
                scores = scores.detach().cpu().numpy().astype(np.float32)

                scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
                scores = np.maximum(scores, 0.0)

                # mask tokens
                if mask_tokens:
                    for i, t in enumerate(tokens):
                        if t in mask_tokens:
                            scores[i] = 0.0

                # =====================================================
                # 🔥 CALIBRATION
                # =====================================================
                cal = calibrate_explanation(scores.tolist(), method="attention")

                scores = cal["scores"]
                confidence = cal["confidence"]
                entropy = cal["entropy"]

                # =====================================================
                # TOP-K
                # =====================================================
                if top_k is not None and top_k > 0:
                    idx = np.argsort(scores)[-top_k:][::-1]
                    tokens = [tokens[i] for i in idx]
                    scores = scores[idx]

                structured = [
                    TokenImportance(token=t, importance=float(s))
                    for t, s in zip(tokens, scores)
                ]

                importance_list = scores.tolist()
                return ExplanationOutput(
                    method="attention",
                    tokens=list(tokens),
                    importance=importance_list,
                    structured=structured,
                    confidence=confidence,
                    entropy=entropy,
                    faithful=True,
                )

        except Exception as exc:
            logger.exception("Attention rollout computation failed")
            raise RuntimeError("Attention rollout failed") from exc