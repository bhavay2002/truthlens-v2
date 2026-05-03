from __future__ import annotations

import torch
import torch.nn as nn


class MeanPooling(nn.Module):
    """
    Mean pooling layer for transformer outputs.

    Computes the average of token embeddings, optionally masked to ignore padding.

    Inputs:
        hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim)
        attention_mask: Optional Tensor of shape (batch_size, seq_len)

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            hidden_states: (B, T, H)
            attention_mask: (B, T)

        Returns:
            pooled: (B, H)
        """

        if attention_mask is None:
            pooled = hidden_states.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
            masked_hidden = hidden_states * mask

            sum_hidden = masked_hidden.sum(dim=1)  # (B, H)
            denom = mask.sum(dim=1).clamp(min=1e-6)  # (B, 1)

            pooled = sum_hidden / denom

        if self.use_layernorm:
            pooled = self.norm(pooled)

        pooled = self.dropout(pooled)

        return pooled