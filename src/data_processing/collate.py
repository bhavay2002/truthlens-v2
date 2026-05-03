"""
Generic collate functions.

The pad-token id is *not* hardcoded — RoBERTa uses 1, BERT uses 0, etc.
``build_collate_fn(pad_token_id=...)`` returns a closure with the correct
padding value baked in. ``collate_fn`` is the legacy helper kept for
back-compat (defaults to pad-id 0; suitable for BERT/DeBERTa only).
"""

from __future__ import annotations

from functools import partial
from typing import List, Dict, Any, Callable

import torch
from torch.nn.utils.rnn import pad_sequence


# =========================================================
# CORE
# =========================================================

def _collate(
    batch: List[Dict[str, Any]],
    *,
    pad_token_id: int,
    safety_check: bool = True,
) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Empty batch")

    # ----- inputs -----
    input_ids = pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    attention_mask = pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )

    out: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    # ----- offsets (optional) -----
    if "offset_mapping" in batch[0]:
        out["offset_mapping"] = pad_sequence(
            [item["offset_mapping"] for item in batch],
            batch_first=True,
            padding_value=0,
        )

    # ----- labels -----
    out["labels"] = torch.stack([item["labels"] for item in batch])

    # ----- task -----
    task = batch[0]["task"]
    if safety_check:
        for item in batch:
            if item["task"] != task:
                raise RuntimeError("Mixed-task batch detected")
    out["task"] = task

    return out


def build_collate_fn(
    *,
    pad_token_id: int,
    safety_check: bool = True,
) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Build a collate function bound to a specific tokenizer pad-token id.

    Usage:
        collate = build_collate_fn(pad_token_id=tokenizer.pad_token_id)
        DataLoader(..., collate_fn=collate)
    """
    return partial(_collate, pad_token_id=pad_token_id, safety_check=safety_check)


# =========================================================
# LEGACY (BERT-default pad id = 0)
# =========================================================

import warnings


def _legacy_collate_warning(name: str) -> None:
    # UNUSED-D4: surface the foot-gun. The legacy helpers default to
    # ``pad_token_id=0`` which is the BERT/DeBERTa pad id — RoBERTa-family
    # models use ``pad_token_id=1`` and silently train on garbage padding
    # if these are ever wired to a RoBERTa loader. Fire a one-shot
    # DeprecationWarning per process so callers migrate to
    # ``build_collate_fn(pad_token_id=tokenizer.pad_token_id)``.
    warnings.warn(
        f"{name} is deprecated; use build_collate_fn(pad_token_id=tokenizer.pad_token_id). "
        "The legacy default pad_token_id=0 is unsafe for RoBERTa-family tokenizers.",
        DeprecationWarning,
        stacklevel=3,
    )


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Back-compat collate (pad_token_id=0). Prefer ``build_collate_fn``."""
    _legacy_collate_warning("collate_fn")
    return _collate(batch, pad_token_id=0, safety_check=True)


def fast_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Back-compat fast collate (no safety check, pad_token_id=0)."""
    _legacy_collate_warning("fast_collate_fn")
    return _collate(batch, pad_token_id=0, safety_check=False)
