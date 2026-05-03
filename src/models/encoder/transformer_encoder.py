"""Canonical ``TransformerEncoder`` implementation (audit 3.2).

This module used to be a re-export shim that pointed at
``src.models.inference.model_wrapper``. The audit flipped the convention:
the encoder package is the natural home for an encoder, so the real
class lives here and ``model_wrapper.py`` is the back-compat shim.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from src.models.base.base_model import BaseModel

logger = logging.getLogger(__name__)


# =========================================================
# OUTPUT
# =========================================================

@dataclass
class EncoderOutput:
    sequence_output: torch.Tensor
    pooled_output: torch.Tensor


# =========================================================
# ENCODER
# =========================================================

class TransformerEncoder(BaseModel):

    VALID_POOLING = {"cls", "mean", "attention"}

    def __init__(
        self,
        model_name: str,
        pooling: str = "cls",
        device: Optional[str] = None,
        freeze_encoder: bool = False,
        gradient_checkpointing: bool = False,
        output_hidden_states: bool = False,
        use_amp: bool = True,
        amp_dtype: str = "float16",
        # P2.1: tri-state — None means "auto: on for CUDA, off for CPU".
        # See ``EncoderConfig.use_compile`` for the rationale.
        use_compile: Optional[bool] = None,
        compile_mode: str = "default",
        max_length: int = 512,
        init_from_config_only: bool = False,
        **kwargs,
    ) -> None:

        super().__init__()

        if pooling not in self.VALID_POOLING:
            raise ValueError(f"Invalid pooling: {pooling}")

        self.model_name = model_name
        self.pooling = pooling
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.max_length = max_length
        self.output_hidden_states = output_hidden_states

        self._encoder_frozen = False

        # A5.1: route through the centralised detector so MPS is honoured
        # and the resolution rules cannot drift from the EncoderFactory /
        # benchmark / model_utils path.
        from src.models._device import detect_device

        device_obj = detect_device(device)

        try:
            self.config = AutoConfig.from_pretrained(model_name)

            if hasattr(self.config, "add_pooling_layer"):
                self.config.add_pooling_layer = False

            if init_from_config_only:
                self.encoder = AutoModel.from_config(self.config)
            else:
                self.encoder = AutoModel.from_pretrained(
                    model_name,
                    config=self.config,
                )

        except Exception as e:
            logger.exception("Encoder init failed")
            raise RuntimeError from e

        self.hidden_size = self.config.hidden_size

        if gradient_checkpointing:
            self.gradient_checkpointing_enable()

        if freeze_encoder:
            self.freeze()

        # COMPILE-OFF: ``torch.compile`` removed project-wide (see
        # ``training_setup.optimize_model`` for the full rationale —
        # instability across environments + spurious bf16 overflow
        # warnings + not needed for current training stability). The
        # ``use_compile`` and ``compile_mode`` parameters are still
        # accepted by ``__init__`` for back-compat with existing callers
        # but are now no-ops.
        if use_compile:
            logger.info(
                "Encoder compile request ignored (COMPILE-OFF); "
                "running in eager mode."
            )

        self.set_device(device_obj)

        # P2.7: cache the resolved device on a plain attribute so
        # ``forward`` can compare against it without walking
        # ``next(self.parameters()).device`` on every call (which is a
        # measurable per-batch overhead on small inputs).
        self._cached_device = device_obj

        logger.info(
            "Encoder ready | model=%s | hidden=%d",
            model_name,
            self.hidden_size,
        )

    # =====================================================
    # FORWARD
    # =====================================================

    # =====================================================
    # INFERENCE-CONTRACT-FIX V7 — embeddings access
    #
    # The explainability layer (``src.explainability.bias_explainer``,
    # ``src.explainability.emotion_explainer``,
    # ``src.aggregation.score_explainer``) builds Integrated-Gradients
    # / gradient×input attributions by walking through ``model.encoder``
    # directly: it grabs the embedding layer with
    # ``model.encoder.embeddings(input_ids)`` and then re-runs the
    # encoder on the resulting embedding tensor with
    # ``model.encoder(inputs_embeds=..., attention_mask=...)``.
    #
    # Pre-V7 the wrapper only exposed ``self.encoder`` (the underlying
    # HuggingFace ``AutoModel``) but had no top-level ``embeddings``
    # attribute, and the wrapper's ``forward`` had a strict
    # ``(input_ids, attention_mask)`` signature with no
    # ``inputs_embeds`` parameter. Both code paths therefore raised
    # before any attribution could be produced — the IG step in
    # ``score_explainer._integrated_gradients`` died at the
    # ``model.encoder.embeddings(...)`` line and the gradient×input
    # step in ``bias_explainer.compute_gradients`` died one line later
    # at ``model.encoder(inputs_embeds=...)``.
    #
    # We surface the HF embedding module via ``self.embeddings`` (so
    # it is callable on ``input_ids`` exactly as the explainers expect)
    # and accept ``inputs_embeds`` as an alternate, mutually-exclusive
    # entry point in ``forward``.
    # =====================================================

    @property
    def embeddings(self):
        """Expose the underlying HF embedding module.

        Used by Integrated-Gradients / gradient×input attribution
        helpers that need to (a) run ``input_ids`` through the
        embedding layer to materialise a leaf tensor and then
        (b) re-enter the encoder with ``inputs_embeds=...``.
        """
        return self.encoder.embeddings

    # =====================================================
    # INFERENCE-CONTRACT-FIX V7 — encoder output container
    #
    # The wrapper has historically returned a plain ``dict`` from
    # ``forward`` (``{"sequence_output", "pooled_output"}``). The
    # explainability / inference call sites split into two camps:
    #
    #   * Internal callers (``MultiTaskTruthLensModel._extract_pooled``,
    #     training step, etc.) use *dict* access:
    #     ``encoder_outputs.get("pooled_output")``.
    #
    #   * Explainability callers (``bias_explainer._forward_logits``,
    #     ``score_explainer._integrated_gradients``) use *attribute*
    #     access in the HF style: ``out.last_hidden_state[:, 0]``.
    #     Pre-V7 this raised ``AttributeError: 'dict' object has no
    #     attribute 'last_hidden_state'`` and was the second failure
    #     mode in the explainability pipeline (the first being the
    #     missing ``model.heads`` / ``get_input_embeddings``).
    #
    # We make the return value a ``dict`` *subclass* that also
    # honours ``__getattr__`` so both call styles work without
    # forcing every caller to migrate. Subclassing ``dict`` (as
    # opposed to using ``transformers.modeling_outputs.BaseModelOutput``
    # / ``SimpleNamespace``) keeps every existing ``isinstance(...,
    # dict)`` / ``dict.get`` / ``**outputs`` site green.
    # =====================================================

    class _EncoderOutput(dict):
        """Dict that also exposes its keys as attributes (HF-style)."""

        def __getattr__(self, name: str):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        # INFERENCE-CONTRACT-FIX V7: accept ``inputs_embeds`` as a
        # mutually-exclusive alternative to ``input_ids`` so the
        # explainability IG path can re-enter the encoder with a
        # custom embedding tensor (see the ``embeddings`` property
        # comment block above).
        if input_ids is None and inputs_embeds is None:
            raise TypeError(
                "TransformerEncoder.forward requires either "
                "`input_ids` or `inputs_embeds`."
            )
        if input_ids is not None and inputs_embeds is not None:
            raise TypeError(
                "TransformerEncoder.forward accepts `input_ids` OR "
                "`inputs_embeds`, not both."
            )
        if attention_mask is None:
            raise TypeError(
                "TransformerEncoder.forward requires `attention_mask`."
            )

        # P2.7 + A3.3: ``self._cached_device`` is a plain attribute set
        # in ``__init__`` / ``set_device``; the ``device`` property's
        # fast path also returns ``self._device`` directly, but holding
        # a local reference avoids the property dispatch on the per-batch
        # hot path.
        device = getattr(self, "_cached_device", None)
        if device is None:
            device = self.device
            self._cached_device = device

        if input_ids is not None and input_ids.device != device:
            input_ids = input_ids.to(device)

        if inputs_embeds is not None and inputs_embeds.device != device:
            inputs_embeds = inputs_embeds.to(device)

        if attention_mask.device != device:
            attention_mask = attention_mask.to(device)

        autocast_dtype = (
            torch.bfloat16
            if self.amp_dtype in {"bf16", "bfloat16"}
            else torch.float16
        )

        # INFERENCE-CONTRACT-FIX V7: build the kwargs once so the
        # ``input_ids`` vs ``inputs_embeds`` branch only lives in one
        # place. HF transformers reject passing both as not-None even
        # if one is just a left-over default, so we have to be strict.
        encoder_kwargs: Dict[str, Any] = {
            "attention_mask": attention_mask,
            "return_dict": True,
            "output_hidden_states": self.output_hidden_states,
        }
        if input_ids is not None:
            encoder_kwargs["input_ids"] = input_ids
        else:
            encoder_kwargs["inputs_embeds"] = inputs_embeds

        with torch.autocast(
            device_type=device.type,
            enabled=self.use_amp and device.type == "cuda",
            dtype=autocast_dtype,
        ):
            if self._encoder_frozen:
                with torch.no_grad():
                    outputs = self.encoder(**encoder_kwargs)
            else:
                outputs = self.encoder(**encoder_kwargs)

        sequence_output = outputs.last_hidden_state
        pooled_output = self._pool(sequence_output, attention_mask)

        # INFERENCE-CONTRACT-FIX V7: surface ``last_hidden_state`` so
        # callers that prefer the HF-style attribute access (e.g.
        # ``score_explainer._integrated_gradients`` does
        # ``outputs.last_hidden_state[:, 0]``) keep working when they
        # invoke the wrapper's ``forward`` directly. The dict already
        # carries the same tensor under ``sequence_output``; the new
        # key is purely a back-compat alias. We wrap the return in
        # ``_EncoderOutput`` (a dict subclass with attribute access)
        # so HF-style ``out.last_hidden_state`` works too — see the
        # block-comment above ``_EncoderOutput``.
        return self._EncoderOutput(
            sequence_output=sequence_output,
            pooled_output=pooled_output,
            last_hidden_state=sequence_output,
        )

    # =====================================================
    # POOLING
    # =====================================================

    def _pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        if self.pooling == "cls":
            return hidden_states[:, 0]

        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            summed = torch.sum(hidden_states * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            return summed / counts

        if self.pooling == "attention":
            weights = torch.softmax(hidden_states.mean(dim=-1), dim=1)
            return torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)

        raise RuntimeError("Invalid pooling")

    # =====================================================
    # DEVICE  (P2.7: keep cached device in sync)
    # =====================================================

    def set_device(self, device):
        super().set_device(device)
        # ``BaseModel.set_device`` normalises strings into ``torch.device``
        # via ``self._device`` — surface that resolved value into the
        # ``_cached_device`` slot so ``forward`` keeps using the fast path.
        self._cached_device = self._device

    # =====================================================
    # GRAD CKPT
    # =====================================================

    def gradient_checkpointing_enable(self):

        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):

        if hasattr(self.encoder, "gradient_checkpointing_disable"):
            self.encoder.gradient_checkpointing_disable()

    # =====================================================
    # FREEZE
    # =====================================================

    def freeze(self):

        for p in self.encoder.parameters():
            p.requires_grad = False

        self._encoder_frozen = True

    def unfreeze(self):

        for p in self.encoder.parameters():
            p.requires_grad = True

        self._encoder_frozen = False

    # =====================================================
    # UTILS
    # =====================================================

    def get_hidden_size(self) -> int:
        return self.hidden_size


__all__ = ["TransformerEncoder", "EncoderOutput"]
