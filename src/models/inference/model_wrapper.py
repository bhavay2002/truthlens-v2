"""Re-export shim for the canonical :class:`TransformerEncoder` (A3.2).

The audit moved the real implementation into the encoder package
(``src.models.encoder.transformer_encoder``) — its natural home.
Existing call-sites that imported from
``src.models.inference.model_wrapper`` keep working via this shim;
new code should import directly from
``src.models.encoder.transformer_encoder``.
"""

from __future__ import annotations

from src.models.encoder.transformer_encoder import (
    EncoderOutput,
    TransformerEncoder,
)

__all__ = ["EncoderOutput", "TransformerEncoder"]
