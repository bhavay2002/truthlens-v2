"""Shared encoder components for model heads."""

# NOTE: In this snapshot of the repo, ``transformer_encoder.py`` actually
# contains ``EncoderFactory`` (file naming was swapped during development).
# The real ``TransformerEncoder`` lives under ``src.models.inference.model_wrapper``.
from src.models.inference.model_wrapper import TransformerEncoder

__all__ = ["TransformerEncoder"]
