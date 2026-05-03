from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================
#
# A valid TruthLens checkpoint must contain (a) an `encoder.*` submodule
# and (b) at least one task-head submodule. Heads may be exposed either
# as a single `task_heads.*` `nn.ModuleDict` (the canonical
# `MultiTaskTruthLensModel` / `MultiTaskBaseModel` layout) or as
# individually-named top-level modules ending in `_head` (the legacy
# `HybridTruthLensModel` layout). Custom callers may still pass an
# explicit `required_prefixes` to override these defaults.

REQUIRED_PREFIXES = ["encoder", "task_heads"]


def _state_has_any_head(state_dict_keys: Iterable[str]) -> bool:
    """Return True if any key looks like a head submodule.

    Accepts both the canonical `task_heads.*` ModuleDict layout and the
    legacy `<task>_head.*` per-head layout.
    """

    for key in state_dict_keys:
        if key.startswith("task_heads.") or ".task_heads." in key:
            return True

        # Top-level head submodule, e.g. "bias_head.weight" or
        # "narrative_frame_head.bias".
        first_segment = key.split(".", 1)[0]
        if first_segment.endswith("_head"):
            return True

    return False


# =========================================================
# CORE VALIDATION
# =========================================================

def validate_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    *,
    required_prefixes: Optional[Iterable[str]] = None,
    strict: bool = True,
    check_shapes: bool = False,
    check_dtypes: bool = False,
    expected_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> None:
    """
    Validate checkpoint integrity.

    Parameters
    ----------
    state_dict : dict
        Model weights

    required_prefixes : list[str]
        Expected module prefixes

    strict : bool
        Raise error on missing components

    check_shapes : bool
        Validate tensor shapes

    check_dtypes : bool
        Validate dtype consistency

    expected_shapes : dict[str, tuple]
        Optional exact shape expectations
    """

    # -----------------------------------------------------
    # BASIC VALIDATION
    # -----------------------------------------------------

    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Invalid or empty state_dict")

    user_supplied_prefixes = required_prefixes is not None
    prefixes = list(required_prefixes or REQUIRED_PREFIXES)

    # -----------------------------------------------------
    # FINITE CHECK
    # -----------------------------------------------------

    for name, tensor in state_dict.items():

        if not torch.is_tensor(tensor):
            continue

        if tensor.is_floating_point():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite values detected in: {name}")

    # -----------------------------------------------------
    # STRUCTURE CHECK
    # -----------------------------------------------------

    keys = list(state_dict.keys())
    missing = []

    if user_supplied_prefixes:
        missing = [
            p for p in prefixes
            if not any(k.startswith(p) for k in keys)
        ]
    else:
        # Default contract: require an encoder and at least one head
        # (either `task_heads.*` ModuleDict or a `<task>_head.*` module).
        if not any(k.startswith("encoder") or ".encoder." in k for k in keys):
            missing.append("encoder")

        if not _state_has_any_head(keys):
            missing.append("task_heads | <task>_head")

    if missing:
        msg = f"Missing required components: {missing}"

        if strict:
            raise ValueError(msg)

        logger.warning(msg)

    # -----------------------------------------------------
    # SHAPE CHECK
    # -----------------------------------------------------

    if check_shapes:

        for name, tensor in state_dict.items():

            if not torch.is_tensor(tensor):
                continue

            if tensor.numel() == 0:
                raise ValueError(f"Empty tensor detected: {name}")

            if any(dim <= 0 for dim in tensor.shape):
                raise ValueError(f"Invalid shape in {name}: {tensor.shape}")

    # -----------------------------------------------------
    # EXACT SHAPE MATCHING (ADVANCED)
    # -----------------------------------------------------

    if expected_shapes:

        for name, expected_shape in expected_shapes.items():

            if name not in state_dict:
                continue

            actual = tuple(state_dict[name].shape)

            if actual != expected_shape:
                raise ValueError(
                    f"Shape mismatch for {name}: "
                    f"expected={expected_shape}, got={actual}"
                )

    # -----------------------------------------------------
    # DTYPE CHECK (OPTIONAL)
    # -----------------------------------------------------

    if check_dtypes:

        dtypes = {
            tensor.dtype
            for tensor in state_dict.values()
            if torch.is_tensor(tensor)
        }

        if len(dtypes) > 1:
            logger.warning(f"Mixed dtypes detected: {dtypes}")

    # -----------------------------------------------------
    # DEVICE CHECK (DEBUGGING)
    # -----------------------------------------------------

    devices = {
        str(tensor.device)
        for tensor in state_dict.values()
        if torch.is_tensor(tensor)
    }

    if len(devices) > 1:
        logger.warning(f"Mixed devices in checkpoint: {devices}")

    logger.info(
        "Checkpoint validation passed | tensors=%d | prefixes=%d",
        len(state_dict),
        len(prefixes),
    )