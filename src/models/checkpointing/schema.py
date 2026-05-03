from __future__ import annotations

import logging
from typing import Dict, Any

import torch

logger = logging.getLogger(__name__)


# =========================================================
# SCHEMA VERSIONING
# =========================================================

SCHEMA_VERSION = 2

SUPPORTED_VERSIONS = {1, 2}


# =========================================================
# REQUIRED FIELDS
# =========================================================

REQUIRED_KEYS_V2 = {
    "model_state_dict",
    "optimizer_state_dict",
    "epoch",
}

OPTIONAL_KEYS = {
    "scheduler_state_dict",
    "scaler_state_dict",
    "metrics",
    "config",
    "step",
    "loss",
}


# =========================================================
# ATTACH SCHEMA
# =========================================================

def attach_schema(
    checkpoint: Dict[str, Any],
    *,
    version: int = SCHEMA_VERSION,
) -> Dict[str, Any]:
    """
    Attach schema version to checkpoint.
    """

    checkpoint["_schema_version"] = version
    checkpoint["_pytorch_version"] = torch.__version__

    return checkpoint


# =========================================================
# VALIDATION
# =========================================================

def validate_schema(
    checkpoint: Dict[str, Any],
    *,
    strict: bool = True,
) -> int:
    """
    Validate checkpoint schema.

    Returns
    -------
    int
        schema version
    """

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary")

    version = checkpoint.get("_schema_version", 1)

    if version not in SUPPORTED_VERSIONS:
        msg = f"Unsupported schema version: {version}"
        if strict:
            raise RuntimeError(msg)
        logger.warning(msg)

    # -----------------------------------------------------
    # V2 VALIDATION
    # -----------------------------------------------------

    if version == 2:

        missing = REQUIRED_KEYS_V2 - checkpoint.keys()

        if missing:
            msg = f"Missing required keys (v2): {missing}"
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg)

    logger.info("Checkpoint schema validated | version=%d", version)

    return version


# =========================================================
# MIGRATION
# =========================================================

def migrate_checkpoint(
    checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Upgrade checkpoint to latest schema version.
    """

    version = checkpoint.get("_schema_version", 1)

    if version == SCHEMA_VERSION:
        return checkpoint

    logger.info("Migrating checkpoint | from v%d → v%d", version, SCHEMA_VERSION)

    # -----------------------------------------------------
    # V1 → V2 MIGRATION
    # -----------------------------------------------------

    if version == 1:

        # Rename common legacy keys
        if "model" in checkpoint:
            checkpoint["model_state_dict"] = checkpoint.pop("model")

        if "optimizer" in checkpoint:
            checkpoint["optimizer_state_dict"] = checkpoint.pop("optimizer")

        # Add missing defaults
        checkpoint.setdefault("epoch", 0)

        checkpoint["_schema_version"] = 2

        logger.info("Migration complete: v1 → v2")

    return checkpoint


# =========================================================
# SAFE PREPARE (LOAD PIPELINE)
# =========================================================

def prepare_checkpoint(
    checkpoint: Dict[str, Any],
    *,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Full schema pipeline:
    1. Validate
    2. Migrate (if needed)
    3. Re-validate

    Returns
    -------
    Dict[str, Any]
    """

    version = validate_schema(checkpoint, strict=strict)

    if version != SCHEMA_VERSION:
        checkpoint = migrate_checkpoint(checkpoint)
        validate_schema(checkpoint, strict=strict)

    return checkpoint


# =========================================================
# DEBUG UTILITIES
# =========================================================

def summarize_schema(
    checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return schema summary for debugging/logging.
    """

    return {
        "schema_version": checkpoint.get("_schema_version", 1),
        "pytorch_version": checkpoint.get("_pytorch_version", "unknown"),
        "keys": list(checkpoint.keys()),
    }