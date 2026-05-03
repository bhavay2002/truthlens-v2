from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


# =========================================================
# CONSTANTS
# =========================================================

METADATA_FILENAME = "metadata.json"


# =========================================================
# SAVE
# =========================================================

def save_metadata(
    directory: str | Path,
    metadata: Dict[str, Any],
    *,
    filename: str = METADATA_FILENAME,
    indent: int = 2,
) -> Path:
    """
    Save metadata to JSON file.

    Features
    --------
    - atomic write (safe replace)
    - readable formatting
    - consistent schema
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / filename
    tmp_path = path.with_suffix(".tmp")

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=indent)

        tmp_path.replace(path)

        logger.debug("Metadata saved: %s", path)

    except Exception:
        logger.exception("Failed to save metadata: %s", path)

        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

        raise

    return path


# =========================================================
# LOAD
# =========================================================

def load_metadata(
    path: str | Path,
    *,
    filename: str = METADATA_FILENAME,
) -> Dict[str, Any]:
    """
    Load metadata from JSON.

    Supports:
    - directory → loads metadata.json
    - file → loads directly
    """

    path = Path(path)

    if path.is_dir():
        path = path / filename

    if not path.exists():
        raise FileNotFoundError(path)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    except Exception as e:
        logger.exception("Failed to load metadata: %s", path)
        raise RuntimeError(f"Metadata load failed: {path}") from e


# =========================================================
# UPDATE
# =========================================================

def update_metadata(
    directory: str | Path,
    updates: Dict[str, Any],
    *,
    filename: str = METADATA_FILENAME,
) -> Path:
    """
    Update metadata file (merge strategy).
    """

    directory = Path(directory)

    try:
        current = {}

        try:
            current = load_metadata(directory, filename=filename)
        except FileNotFoundError:
            pass

        current.update(updates)

        return save_metadata(directory, current, filename=filename)

    except Exception:
        logger.exception("Metadata update failed")
        raise


# =========================================================
# VALIDATION (LIGHTWEIGHT)
# =========================================================

def validate_metadata(
    metadata: Dict[str, Any],
    *,
    required_keys: Optional[list[str]] = None,
) -> None:
    """
    Lightweight metadata validation.
    """

    if not isinstance(metadata, dict):
        raise ValueError("metadata must be dict")

    if required_keys:
        missing = [k for k in required_keys if k not in metadata]
        if missing:
            raise ValueError(f"Missing metadata keys: {missing}")


# =========================================================
# STANDARD BUILDER
# =========================================================

def build_metadata(
    *,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    integrity: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build standardized metadata dictionary.

    This ensures consistency across:
    - checkpoints
    - models
    - experiments
    """

    meta: Dict[str, Any] = {}

    if step is not None:
        meta["step"] = step

    if epoch is not None:
        meta["epoch"] = epoch

    if metrics:
        meta["metrics"] = metrics

    if config:
        meta["config"] = config

    if integrity:
        meta.update(integrity)

    if extra:
        meta.update(extra)

    return meta


# =========================================================
# DEBUG UTIL
# =========================================================

def summarize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return compact summary for logging/debugging.
    """

    return {
        "step": metadata.get("step"),
        "epoch": metadata.get("epoch"),
        "metrics": list((metadata.get("metrics") or {}).keys()),
        "has_checksum": "checksum" in metadata,
    }