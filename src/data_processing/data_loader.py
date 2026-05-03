"""
Unified file loader for CSV / JSON(L) / Parquet.

Encoding fallback is logged loudly so silent corruption does not slip
through.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_ENCODING = "utf-8"
FALLBACK_ENCODING = "latin-1"


# =========================================================
# CORE LOADERS
# =========================================================
#
# UNUSED-D2: ``compute_md5`` was removed. The cache layer now uses
# ``data_cache._file_fingerprint`` which hashes full content via
# SHA-256 for files ≤ 2 MB and head+tail for larger files (CRIT-D2).
# MD5 was both weaker and unused outside a dead ``compute_hash=True``
# branch in ``load_dataframe``.
#
# UNUSED-D3: ``load_csv_in_chunks`` was removed. Nothing in
# ``run_data_pipeline`` (or anywhere else) consumed the chunk iterator;
# pandas can stream chunks via ``pd.read_csv(..., chunksize=…)``
# directly when a future >1 GB CSV path needs it. Keeping a thin
# unexported wrapper just so it shows up in dir() is dead weight.

def load_csv(
    path: Path,
    *,
    usecols: Optional[List[str]] = None,
    dtype: Optional[Dict[str, Any]] = None,
    encoding: str = DEFAULT_ENCODING,
    na_values: Optional[List[str]] = None,
) -> pd.DataFrame:
    common = dict(usecols=usecols, dtype=dtype, na_values=na_values, low_memory=False)
    try:
        return pd.read_csv(path, encoding=encoding, **common)
    except UnicodeDecodeError:
        logger.warning(
            "Encoding fallback %s → %s for %s",
            encoding, FALLBACK_ENCODING, path,
        )
        return pd.read_csv(path, encoding=FALLBACK_ENCODING, **common)


def load_json(
    path: Path,
    *,
    usecols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load JSON or JSONL — sniffs the file shape (CRIT-D4).

    A standard JSON-array file with .json suffix would previously raise
    ``ValueError: Trailing data`` because lines=True was hardcoded.
    Now the first non-whitespace byte chooses the format.

    ``usecols`` is honoured post-load (pandas read_json has no native
    column-projection arg), which still avoids paying the downstream
    feature/encoding cost for unused columns. (CRIT-D3)
    """
    with open(path, "rb") as f:
        # Skip whitespace to find the first real byte
        first = b""
        while True:
            b = f.read(1)
            if not b:
                break
            if not b.isspace():
                first = b
                break
    is_lines = first != b"["
    df = pd.read_json(path, lines=is_lines)
    if usecols:
        keep = [c for c in usecols if c in df.columns]
        df = df[keep]
    return df


def load_parquet(
    path: Path,
    *,
    usecols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Parquet loader with native column projection (CRIT-D3).

    Forwarding ``usecols`` to ``pd.read_parquet(columns=...)`` cuts
    memory ~3-5× for narrative/emotion frames that carry large
    auxiliary columns (e.g. ``*_entities`` text blobs).
    """
    return pd.read_parquet(path, columns=usecols) if usecols else pd.read_parquet(path)


# =========================================================
# GENERIC LOADER
# =========================================================

def load_dataframe(
    path: Path,
    *,
    usecols: Optional[List[str]] = None,
    dtype: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    logger.info("Loading dataset: %s", path)

    if suffix == ".csv":
        df = load_csv(path, usecols=usecols, dtype=dtype)
    elif suffix in (".json", ".jsonl"):
        df = load_json(path, usecols=usecols)
    elif suffix == ".parquet":
        df = load_parquet(path, usecols=usecols)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    logger.info("Loaded %d rows × %d cols", len(df), len(df.columns))
    return df


# =========================================================
# COLUMN GUARD
# =========================================================

def enforce_required_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
