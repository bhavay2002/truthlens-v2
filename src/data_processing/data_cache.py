"""
Dataset cache.

Improvements vs the original:
- Module no longer calls ``load_settings()`` at import time (so importing
  this module does not require the data CSVs to exist).
- File fingerprint uses ``(size, sha256(first 1MB) + sha256(last 1MB))``
  instead of ``mtime``, so ``cp -p`` / ``git checkout`` does not spuriously
  invalidate the cache.
- ``get_cache_key`` accepts arbitrary extra inputs (tokenizer name,
  max_length, cleaning/augmentation config) so changing any of them
  invalidates the cache.
- Cache load/save logs the failed file when corruption is detected.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Any

import pandas as pd

logger = logging.getLogger(__name__)

# CACHE-D2 — Auto-derived cache version.
# ``_BASE_VERSION`` is bumped manually when pipeline-level semantics
# change in a way that's not visible from the source of the two
# functions below (e.g. a different label dtype downstream).
# ``_LOGIC_FINGERPRINT`` is the md5 of the source of
# ``_file_fingerprint`` and ``get_cache_key`` themselves — so any
# change to either function (e.g. switching SHA-256 → BLAKE2b, or
# adjusting the head/tail boundary) auto-invalidates every cache
# entry without needing a coordinated PR to bump a string literal.
_BASE_VERSION = "v4"


def _derive_logic_fingerprint() -> str:
    # Module-level helper — the references to the two functions are
    # resolved lazily inside the body so this can be called at import
    # time after both have been defined (see end of file).
    src = inspect.getsource(_file_fingerprint) + inspect.getsource(get_cache_key)
    return hashlib.md5(src.encode("utf-8")).hexdigest()[:8]


# Set at the very end of the module once the dependent functions exist.
CACHE_VERSION: str = _BASE_VERSION  # filled in below

_CACHE_DIR: Optional[Path] = None


# =========================================================
# LAZY SETTINGS
# =========================================================

def _get_cache_dir() -> Path:
    global _CACHE_DIR
    if _CACHE_DIR is None:
        # imported lazily so importing data_cache does not trigger
        # filesystem validation in settings_loader
        from src.config.settings_loader import load_settings
        _CACHE_DIR = load_settings().paths.cache_dir / "data"
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


# =========================================================
# HASHING
# =========================================================

def _hash_dict(obj: Dict) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _file_fingerprint(path: Path) -> Dict[str, Any]:
    """Stable, mtime-free fingerprint: (size, sha256(content)). (CRIT-D2)

    Files ≤ 2 MB are hashed in full so two files of identical size +
    identical first 1 MB but different tail cannot collide. Larger
    files use head + tail to keep the cost bounded — this is the band
    where a content collision is statistically negligible anyway.
    """
    if not path.exists():
        return {"missing": True}

    size = path.stat().st_size
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if size <= (2 << 20):
            # Small / medium file → hash everything (fixes 1 MB < size ≤ 2 MB blind spot)
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        else:
            # Large file → head + tail
            h.update(f.read(1 << 20))
            f.seek(-(1 << 20), 2)
            h.update(f.read())
    return {"size": size, "sha": h.hexdigest()}


def _hash_files(file_paths: Dict[str, Dict[str, Path]]) -> str:
    fingerprint: Dict[str, Dict[str, Any]] = {}
    for task, splits in file_paths.items():
        fingerprint[task] = {
            split: _file_fingerprint(Path(p)) for split, p in splits.items()
        }
    return _hash_dict(fingerprint)


# =========================================================
# CACHE KEY
# =========================================================

def get_cache_key(
    data_config: Dict,
    file_paths: Dict[str, Dict[str, Path]],
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Cache key derived from:
        - data_config dict
        - file fingerprints (size + content sha)
        - cache version
        - any extra dict (tokenizer, max_length, cleaning, augmentation, …)
    """
    return _hash_dict({
        "config": data_config,
        "files": _hash_files(file_paths),
        "version": CACHE_VERSION,
        "extra": extra or {},
    })


# =========================================================
# SAVE
# =========================================================

def save_cached_datasets(
    datasets: Dict[str, Dict[str, pd.DataFrame]],
    cache_key: str,
) -> None:
    """Atomically write a dataset cache (CACHE-D5).

    Previously this function wrote each parquet file directly into the
    final ``{cache_key}/`` directory and only flushed ``meta.json`` at
    the end. A crash mid-write left a half-populated dir that
    ``load_cached_datasets`` happily returned (no meta validation), so
    a subsequent run would train on a partial dataset and silently
    miss whole splits. Fix: stage everything in ``{cache_key}.tmp/``
    and ``os.replace`` the directory into place once ``meta.json`` has
    been fsync'd. ``os.replace`` is atomic on POSIX and on Windows
    (when src + dst are on the same filesystem), which holds for our
    cache layout.
    """
    cache_root = _get_cache_dir()
    final = cache_root / cache_key
    tmp = cache_root / f"{cache_key}.tmp"

    # Clear any leftover tmp from a previous interrupted run before we
    # start writing — otherwise stale shards could leak in.
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    logger.info("Saving dataset cache → %s (staging in %s)", final, tmp)

    meta: Dict[str, int] = {}
    for task, splits in datasets.items():
        for split, df in splits.items():
            file = tmp / f"{task}__{split}.parquet"
            df.to_parquet(file, index=False)
            meta[f"{task}__{split}"] = len(df)

    meta_path = tmp / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # Some filesystems (tmpfs in CI containers) don't support
            # fsync — log and continue; rename is still atomic.
            logger.debug("fsync(meta.json) skipped — fs does not support it")

    # Atomic swap. If ``final`` already exists (e.g. cache hit
    # repopulated by a parallel process) clear it first, since
    # ``os.replace`` on a non-empty dir raises ``OSError`` on POSIX.
    if final.exists():
        shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)

    logger.info("Cache saved (%d frames)", len(meta))


# =========================================================
# LOAD
# =========================================================

def load_cached_datasets(cache_key: str) -> Optional[Dict[str, Dict[str, pd.DataFrame]]]:
    base = _get_cache_dir() / cache_key
    if not base.exists():
        logger.info("Cache miss: %s", cache_key[:12])
        return None

    # CACHE-D5: meta.json is the atomic-write commit marker. If it's
    # missing the directory is from a crashed save and is unsafe to
    # consume — even if it has parquet files in it.
    meta_path = base / "meta.json"
    if not meta_path.exists():
        logger.warning(
            "Cache directory missing meta.json (incomplete save?), invalidating: %s",
            base,
        )
        return None

    files = list(base.glob("*.parquet"))
    if not files:
        logger.warning("Empty cache directory, ignoring: %s", base)
        return None

    logger.info("Loading cached dataset → %s", base)
    datasets: Dict[str, Dict[str, pd.DataFrame]] = {}

    for file in files:
        name = file.stem
        if "__" not in name:
            continue
        task, split = name.rsplit("__", 1)

        try:
            df = pd.read_parquet(file)
        except Exception as e:
            logger.warning("Cache corruption at %s: %s — invalidating", file, e)
            return None

        datasets.setdefault(task, {})[split] = df

    logger.info("Cache loaded (%d frames)", sum(len(v) for v in datasets.values()))
    return datasets


# =========================================================
# PRUNE  (CACHE-D4 — disk eviction)
#
# ``_get_cache_dir`` writes ``{cache_key}/{task}__{split}.parquet`` and
# previously had no max-bytes / max-age policy, so a long-running
# experiment with many cleaning-config sweeps would accumulate cache
# directories indefinitely. Mirror what
# ``src/features/cache/cache_manager.prune_all`` does for the feature
# cache: scan every entry under the cache root and evict by total
# byte budget (LRU by mtime) and/or age in days. Safe to call at
# process startup; missing dirs and OS errors are logged and skipped,
# never raised.
# =========================================================

def _entry_size_bytes(entry: Path) -> int:
    total = 0
    for p in entry.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def prune_cache(
    *,
    max_bytes: Optional[int] = None,
    max_age_days: Optional[float] = None,
) -> Dict[str, int]:
    """Evict stale dataset-cache entries.

    Args:
        max_bytes: keep total cache size below this byte budget by
            removing the oldest (mtime) entries first. ``None`` →
            no byte cap.
        max_age_days: also remove entries whose mtime is older than
            this many days. ``None`` → no age cap.

    Returns:
        ``{"scanned": N, "removed_age": A, "removed_bytes": B,
           "bytes_freed": F}``
    """
    stats: Dict[str, int] = {
        "scanned": 0,
        "removed_age": 0,
        "removed_bytes": 0,
        "bytes_freed": 0,
    }

    cache_root = _get_cache_dir()
    if not cache_root.exists():
        return stats

    entries = []
    for child in cache_root.iterdir():
        if not child.is_dir():
            continue
        # Skip in-flight tmp dirs from concurrent saves.
        if child.name.endswith(".tmp"):
            continue
        try:
            mtime = child.stat().st_mtime
            size = _entry_size_bytes(child)
        except OSError as exc:
            logger.warning("prune_cache: stat failed for %s: %s", child, exc)
            continue
        entries.append((child, mtime, size))
        stats["scanned"] += 1

    # 1) Age-based pass.
    if max_age_days is not None:
        cutoff = time.time() - (max_age_days * 86400.0)
        survivors = []
        for entry, mtime, size in entries:
            if mtime < cutoff:
                try:
                    shutil.rmtree(entry)
                    stats["removed_age"] += 1
                    stats["bytes_freed"] += size
                    logger.info("prune_cache: removed stale entry %s (age)", entry.name)
                except OSError as exc:
                    logger.warning("prune_cache: rmtree(%s) failed: %s", entry, exc)
                    survivors.append((entry, mtime, size))
            else:
                survivors.append((entry, mtime, size))
        entries = survivors

    # 2) Byte-budget pass — drop oldest first until under cap.
    if max_bytes is not None:
        total = sum(s for _, _, s in entries)
        if total > max_bytes:
            entries.sort(key=lambda e: e[1])  # oldest first
            for entry, _mtime, size in entries:
                if total <= max_bytes:
                    break
                try:
                    shutil.rmtree(entry)
                    stats["removed_bytes"] += 1
                    stats["bytes_freed"] += size
                    total -= size
                    logger.info(
                        "prune_cache: removed entry %s (byte cap)", entry.name
                    )
                except OSError as exc:
                    logger.warning("prune_cache: rmtree(%s) failed: %s", entry, exc)

    return stats


# =========================================================
# CACHE_VERSION FINALISATION  (CACHE-D2)
# =========================================================
# Now that ``_file_fingerprint`` and ``get_cache_key`` are defined we
# can fold their source fingerprint into ``CACHE_VERSION``. Any future
# edit to either function automatically invalidates stale caches —
# the manual ``_BASE_VERSION`` literal becomes a coarse "I changed
# something the source-fingerprint can't see" override.
CACHE_VERSION = f"{_BASE_VERSION}-{_derive_logic_fingerprint()}"
