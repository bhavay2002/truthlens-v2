from __future__ import annotations

import torch
import torch.nn as nn


class CLSPooling(nn.Module):
    """
    CLS token pooling layer.

    Extracts the representation of the first token (typically [CLS] or <s>)
    from transformer outputs. Optionally applies normalization and dropout.

    Inputs:
        hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim)

    Outputs:
        pooled: Tensor of shape (batch_size, hidden_dim)
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()

        self.use_layernorm = use_layernorm

        if use_layernorm:
            self.norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: (B, T, H)

        Returns:
            pooled: (B, H)
        """

        # CLS token is first token
        cls_token = hidden_states[:, 0, :]  # (B, H)

        if self.use_layernorm:
            cls_token = self.norm(cls_token)

        cls_token = self.dropout(cls_token)

        return cls_token