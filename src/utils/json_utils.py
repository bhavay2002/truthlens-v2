#src\utils\json_utils.py

from __future__ import annotations

import json
import logging
import os
import tempfile
import gzip
from pathlib import Path
from typing import Iterable, Any
from contextlib import contextmanager

import portalocker  

logger = logging.getLogger(__name__)


# =========================================================
# FILE LOCK (CROSS-PLATFORM SAFE)
# =========================================================

@contextmanager
def file_lock(path: Path):
    """
    Cross-platform file lock using portalocker.
    Works on Windows, Linux, Mac.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")

    with open(lock_path, "w") as lock_file:
        try:
            portalocker.lock(lock_file, portalocker.LOCK_EX)
            yield
        finally:
            portalocker.unlock(lock_file)


# =========================================================
# ATOMIC WRITE
# =========================================================

def _atomic_write(path: Path, data: bytes):
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=path.parent) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
            tmp = Path(f.name)

        # Atomic replace
        os.replace(tmp, path)

    finally:
        if tmp and tmp.exists():
            tmp.unlink(missing_ok=True)


# =========================================================
# SAVE JSON
# =========================================================

def save_json(data: dict, path: str | Path, indent: int = 2) -> Path:
    path = Path(path)

    payload = json.dumps(data, indent=indent, ensure_ascii=False).encode("utf-8")

    with file_lock(path):
        _atomic_write(path, payload)

    return path


# =========================================================
# LOAD JSON (LOCK SAFE)
# =========================================================

def load_json(path: str | Path) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with file_lock(path):
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)


# =========================================================
# APPEND JSONL
# =========================================================

def append_json(entry: dict, path: str | Path) -> Path:
    path = Path(path)

    line = json.dumps(entry, ensure_ascii=False) + "\n"

    with file_lock(path):
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    return path


# =========================================================
# BATCH APPEND (HIGH PERFORMANCE)
# =========================================================

def append_json_batch(entries: Iterable[dict], path: str | Path) -> Path:
    path = Path(path)

    lines = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries)

    with file_lock(path):
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("a", encoding="utf-8") as f:
            f.write(lines)
            f.flush()
            os.fsync(f.fileno())

    return path


# =========================================================
# COMPRESSED JSONL (GZIP)
# =========================================================

def append_json_gz(entry: dict, path: str | Path) -> Path:
    path = Path(path)

    with file_lock(path):
        path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(path, "at", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return path