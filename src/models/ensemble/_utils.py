from __future__ import annotations

from typing import Any

import torch


def extract_logits(output: Any) -> torch.Tensor:
    """
    Normalize model outputs to logits tensor.

    Supports:
    - torch.Tensor
    - dict with "logits" or common alternatives
    - tuple/list (first element assumed logits)
    - objects with .logits attribute (HF-style)
    """

    # -------------------------
    # DIRECT TENSOR
    # -------------------------
    if isinstance(output, torch.Tensor):
        return output

    # -------------------------
    # DICTIONARY
    # -------------------------
    if isinstance(output, dict):

        for key in ("logits", "output", "outputs"):
            val = output.get(key)
            if isinstance(val, torch.Tensor):
                return val

        raise TypeError("Dict output must contain tensor logits-like key")

    # -------------------------
    # HUGGINGFACE STYLE OBJECT
    # -------------------------
    if hasattr(output, "logits"):
        logits = getattr(output, "logits")
        if isinstance(logits, torch.Tensor):
            return logits

    # -------------------------
    # TUPLE / LIST
    # -------------------------
    if isinstance(output, (tuple, list)) and len(output) > 0:
        return extract_logits(output[0])

    # -------------------------
    # FAILURE
    # -------------------------
    raise TypeError(
        f"Unsupported model output type: {type(output)}"
    )