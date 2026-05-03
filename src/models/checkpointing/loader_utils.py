from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =========================================================
# CORE FILTER
# =========================================================

def filter_state_dict(
    state_dict: Dict[str, torch.Tensor],
    *,
    include_prefixes: Optional[Iterable[str]] = None,
    exclude_prefixes: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Filter state_dict keys using include/exclude prefixes.

    Parameters
    ----------
    include_prefixes : list[str] | None
        Keep only these prefixes

    exclude_prefixes : list[str] | None
        Remove these prefixes
    """

    result = {}

    for k, v in state_dict.items():

        if include_prefixes:
            if not any(k.startswith(p) for p in include_prefixes):
                continue

        if exclude_prefixes:
            if any(k.startswith(p) for p in exclude_prefixes):
                continue

        result[k] = v

    return result


# =========================================================
# SAFE LOAD CORE
# =========================================================

def safe_load_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    strict: bool = False,
    verbose: bool = True,
) -> None:
    """
    Load state dict with detailed logging.

    Handles:
    - missing keys
    - unexpected keys
    """

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if verbose:
        if missing:
            logger.warning("Missing keys: %d", len(missing))
            for k in missing[:10]:
                logger.debug("  MISSING: %s", k)

        if unexpected:
            logger.warning("Unexpected keys: %d", len(unexpected))
            for k in unexpected[:10]:
                logger.debug("  UNEXPECTED: %s", k)

        if not missing and not unexpected:
            logger.info("State dict loaded cleanly")

    return


# =========================================================
# PARTIAL LOADING API
# =========================================================

def load_encoder_only(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    strict: bool = False,
) -> None:
    """
    Load ONLY encoder weights.
    """
    filtered = filter_state_dict(state_dict, include_prefixes=["encoder"])
    safe_load_state_dict(model, filtered, strict=strict)

    logger.info("Loaded encoder weights")


def load_heads_only(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    head_names: Iterable[str],
    strict: bool = False,
) -> None:
    """
    Load specific task heads.

    Example:
        head_names = ["bias_head", "emotion_head"]
    """

    prefixes = list(head_names)

    filtered = filter_state_dict(state_dict, include_prefixes=prefixes)

    safe_load_state_dict(model, filtered, strict=strict)

    logger.info("Loaded heads: %s", prefixes)


def load_excluding_heads(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    head_names: Iterable[str],
    strict: bool = False,
) -> None:
    """
    Load everything EXCEPT specified heads.
    """

    filtered = filter_state_dict(
        state_dict,
        exclude_prefixes=list(head_names),
    )

    safe_load_state_dict(model, filtered, strict=strict)

    logger.info("Loaded model excluding heads: %s", list(head_names))


# =========================================================
# PREFIX-BASED GENERIC LOADER
# =========================================================

def load_by_prefix(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    prefixes: Iterable[str],
    strict: bool = False,
) -> None:
    """
    Generic loader using arbitrary prefixes.
    """

    filtered = filter_state_dict(
        state_dict,
        include_prefixes=list(prefixes),
    )

    safe_load_state_dict(model, filtered, strict=strict)

    logger.info("Loaded prefixes: %s", list(prefixes))


# =========================================================
# DEVICE ALIGNMENT (OPTIMIZER SAFE)
# =========================================================

def move_optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """
    Move optimizer state tensors to correct device.
    (CRITICAL for resume training)
    """

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    logger.debug("Optimizer moved to device: %s", device)