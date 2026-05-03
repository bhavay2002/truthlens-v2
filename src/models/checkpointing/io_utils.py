from __future__ import annotations

import os
import gzip
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


# =========================================================
# ATOMIC SAVE (WITH FSYNC)
# =========================================================

def atomic_save(
    obj: Any,
    path: Path,
    *,
    compress: bool = False,
    use_zip_serialization: bool = True,
) -> Path:
    """
    Atomically save a PyTorch object to disk with fsync durability.

    Steps:
    1. Write to temporary file
    2. Flush + fsync (disk guarantee)
    3. Rename → atomic replace

    Parameters
    ----------
    obj : Any
        Object to save (state_dict / checkpoint / tensor)

    path : Path
        Final destination path

    compress : bool
        If True, saves using gzip compression

    use_zip_serialization : bool
        PyTorch zip serialization flag

    Returns
    -------
    Path
        Final saved path
    """

    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        # -------------------------
        # WRITE
        # -------------------------
        if compress:
            with gzip.open(tmp_path, "wb") as f:
                torch.save(obj, f, _use_new_zipfile_serialization=use_zip_serialization)
        else:
            with open(tmp_path, "wb") as f:
                torch.save(obj, f, _use_new_zipfile_serialization=use_zip_serialization)

                # -------------------------
                # CRITICAL: fsync
                # -------------------------
                f.flush()
                os.fsync(f.fileno())

        # -------------------------
        # ATOMIC REPLACE
        # -------------------------
        tmp_path.replace(path)

        logger.debug("Atomic save complete: %s", path)

    except Exception:
        logger.exception("Atomic save failed: %s", path)

        # cleanup tmp file if exists
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                logger.warning("Failed to remove temp file: %s", tmp_path)

        raise

    return path


# =========================================================
# SAFE LOAD
# =========================================================

def safe_load(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Any:
    """
    Safe torch.load wrapper with error handling.

    Parameters
    ----------
    path : Path
        File to load

    map_location : device
        Device mapping

    Returns
    -------
    Any
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    try:
        if str(path).endswith(".gz"):
            import gzip
            with gzip.open(path, "rb") as f:
                return torch.load(f, map_location=map_location)
        else:
            return torch.load(path, map_location=map_location)

    except Exception as e:
        logger.exception("Failed to load checkpoint: %s", path)
        raise RuntimeError(f"Checkpoint load failed: {path}") from e


# =========================================================
# FILE UTILITIES
# =========================================================

def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_size(path: Path) -> int:
    """
    Get file size in bytes.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path.stat().st_size


def exists(path: Path) -> bool:
    """
    Safe exists check.
    """
    return Path(path).exists()


# =========================================================
# FSYNC DIRECTORY (ADVANCED SAFETY)
# =========================================================

def fsync_dir(path: Path) -> None:
    """
    Ensure directory metadata is flushed to disk.

    Important after atomic rename for full durability.
    """
    path = Path(path)

    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        logger.warning("Directory fsync failed: %s", path)