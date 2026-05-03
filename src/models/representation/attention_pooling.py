from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Attention-based pooling layer for sequence representations.

    Computes a weighted sum of token embeddings where attention weights
    are learned dynamically.

    Inputs:
        hidden_states: Tensor of shape (batch_size, seq_len, hidden_dim)
        attention_mask: Optional Tensor of shape (batch_size, seq_len)

    Outputs:
        pooled: Tensor of shape (batch_size, hidden_dim)
        weights: Tensor of shape (batch_size, seq_len)
    """

    def __init__(
        self,
        hidden_dim: int,
        attention_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        attention_dim = attention_dim or hidden_dim

        self.proj = nn.Linear(hidden_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # (B, T, H) → (B, T, A)
        x = torch.tanh(self.proj(hidden_states))
        x = self.dropout(x)

        # (B, T, 1) → (B, T)
        scores = self.score(x).squeeze(-1)

        if attention_mask is not None:
            # G1: ``float("-inf")`` overflows fp16 / bf16 to NaN once it
            # is added to anything finite (and ``softmax`` of an all-NaN
            # row is itself all-NaN). ``torch.finfo(scores.dtype).min``
            # is the largest negative finite value representable in the
            # current dtype, so masked positions reliably get zero
            # softmax weight without poisoning the row.
            mask_bool = attention_mask == 0
            scores = scores.masked_fill(
                mask_bool, torch.finfo(scores.dtype).min
            )

            # G1: if a row is *entirely* masked, every position got the
            # finfo.min sentinel and ``softmax`` would produce a uniform
            # distribution over padding — silently mixing pad tokens
            # into the pooled output. Detect that case and emit a
            # genuine all-zero attention vector + all-zero pooled
            # representation, which is the only mathematically defined
            # answer for "no input tokens". The all-masked rows are a
            # data-pipeline bug, but we refuse to pretend otherwise.
            all_masked = mask_bool.all(dim=-1, keepdim=True)
            weights = F.softmax(scores, dim=-1)
            weights = torch.where(
                all_masked, torch.zeros_like(weights), weights
            )
        else:
            weights = F.softmax(scores, dim=-1)

        # (B, T, H)
        weights_expanded = weights.unsqueeze(-1)

        # weighted sum → (B, H)
        pooled = torch.sum(hidden_states * weights_expanded, dim=1)

        return pooled, weights