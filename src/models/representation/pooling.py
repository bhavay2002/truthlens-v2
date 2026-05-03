from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


# =========================================================
# BASE POOLING
# =========================================================

class BasePooling(nn.Module):
    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


# =========================================================
# CLS POOLING
# =========================================================

class CLSPooling(BasePooling):
    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return last_hidden_state[:, 0]


# =========================================================
# MEAN POOLING
# =========================================================

class MeanPooling(BasePooling):
    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if attention_mask is None:
            return last_hidden_state.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)

        return summed / counts


# =========================================================
# MAX POOLING
# =========================================================

class MaxPooling(BasePooling):
    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if attention_mask is None:
            return torch.max(last_hidden_state, dim=1).values

        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        # Use the dtype-correct minimum so the mask still wins over the
        # smallest representable activation under fp16/bf16 (where -1e9
        # silently flushes to -inf or worse, NaN). (A3)
        fill_value = torch.finfo(last_hidden_state.dtype).min
        masked = last_hidden_state.masked_fill(mask == 0, fill_value)

        return torch.max(masked, dim=1).values


# =========================================================
# ATTENTION POOLING
# =========================================================

class AttentionPooling(BasePooling):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        scores = self.attention(last_hidden_state).squeeze(-1)

        if attention_mask is not None:
            # Mirror MaxPooling: use the dtype-aware floor so masked
            # positions get exactly zero softmax weight under fp16/bf16
            # without underflow / NaN propagation. (A3)
            fill_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(attention_mask == 0, fill_value)

        weights = torch.softmax(scores, dim=1)
        weights = weights.unsqueeze(-1)

        return torch.sum(last_hidden_state * weights, dim=1)


# =========================================================
# POOLING FACTORY
# =========================================================

class PoolingFactory:

    @staticmethod
    def create(pooling_type: str, hidden_size: int) -> BasePooling:

        pooling_type = pooling_type.lower()

        if pooling_type == "cls":
            return CLSPooling()

        if pooling_type == "mean":
            return MeanPooling()

        if pooling_type == "max":
            return MaxPooling()

        if pooling_type == "attention":
            return AttentionPooling(hidden_size)

        raise ValueError(f"Unsupported pooling type: {pooling_type}")