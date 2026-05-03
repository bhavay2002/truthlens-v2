#src\utils\helper_functions.py

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


# =========================================================
# PATH UTILITIES
# =========================================================

def to_path(path: PathLike) -> Path:
    if isinstance(path, Path):
        return path.resolve()

    if not isinstance(path, str):
        raise TypeError(f"Expected str or Path, got {type(path).__name__}")

    return Path(path).resolve()


# =========================================================
# DIRECTORY CREATION (DDP SAFE)
# =========================================================

def create_folder(path: PathLike, retries: int = 3) -> Path:
    path_obj = to_path(path)

    for attempt in range(retries):
        try:
            path_obj.mkdir(parents=True, exist_ok=True)
            return path_obj
        except OSError as e:
            logger.warning("Retry mkdir (%d/%d): %s", attempt + 1, retries, path_obj)
            if attempt == retries - 1:
                raise RuntimeError(f"Failed to create directory: {path_obj}") from e

    return path_obj


def ensure_directories(paths: Iterable[PathLike]) -> list[Path]:
    return [create_folder(p) for p in paths]


# =========================================================
# FILE VALIDATION
# =========================================================

def ensure_file_exists(path: PathLike) -> Path:
    path_obj = to_path(path)

    if not path_obj.exists() or not path_obj.is_file():
        raise FileNotFoundError(f"File not found: {path_obj}")

    return path_obj


def ensure_files_exist(paths: Iterable[PathLike]) -> list[Path]:
    return [ensure_file_exists(p) for p in paths]


# =========================================================
# ATOMIC WRITE (CRITICAL FOR ML SYSTEMS)
# =========================================================

def atomic_write(file_path: PathLike, data: bytes) -> Path:
    path = to_path(file_path)
    tmp_dir = path.parent

    create_folder(tmp_dir)

    with tempfile.NamedTemporaryFile(delete=False, dir=tmp_dir) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)

    tmp_path.replace(path)  # atomic rename

    return path


# =========================================================
# FILE SIZE + DISK UTILITIES
# =========================================================

def get_file_size(path: PathLike) -> int:
    file_path = ensure_file_exists(path)
    return file_path.stat().st_size


def get_directory_size(path: PathLike) -> int:
    path_obj = to_path(path)

    if not path_obj.exists():
        return 0

    total = 0
    for p in path_obj.rglob("*"):
        if p.is_file():
            total += p.stat().st_size

    return total


# =========================================================
# SAFE DELETE
# =========================================================

def safe_delete(path: PathLike) -> None:
    path_obj = to_path(path)

    if not path_obj.exists():
        return

    if path_obj.is_file():
        path_obj.unlink()
        logger.debug("Deleted file: %s", path_obj)
        return

    shutil.rmtree(path_obj)
    logger.debug("Deleted directory: %s", path_obj)


# =========================================================
# TEMP DIRECTORY (FOR PIPELINES)
# =========================================================

def create_temp_dir(prefix: str = "tmp_") -> Path:
    path = Path(tempfile.mkdtemp(prefix=prefix))
    return path.resolve()


# =========================================================
# PATH VALIDATION (STRICT)
# =========================================================

def assert_is_directory(path: PathLike) -> Path:
    p = to_path(path)
    if not p.exists() or not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {p}")
    return p


def assert_is_file(path: PathLike) -> Path:
    p = to_path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Not a file: {p}")
    return p