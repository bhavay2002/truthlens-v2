from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import gzip
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import numpy as np

from src.inference.constants import INFERENCE_CACHE_VERSION

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class InferenceCacheConfig:
    cache_dir: str = "cache"
    enable_disk_cache: bool = True
    ttl_seconds: Optional[int] = None
    enable_memory_cache: bool = True
    max_memory_items: int = 1024
    max_disk_items: Optional[int] = None

    # CFG-2: previously defaulted to "v1" while ``predict_api`` defaulted
    # to "v2" — every entry point built a divergent cache namespace.
    # Both now share ``INFERENCE_CACHE_VERSION`` so the cache is one
    # logical pool keyed by a single bumpable constant.
    cache_version: str = INFERENCE_CACHE_VERSION
    enable_compression: bool = True


# =========================================================
# SERIALIZATION (CRITICAL)
# =========================================================

def _serialize(obj: Any):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def _deserialize(obj: Any):
    return obj


# =========================================================
# MAIN CACHE
# =========================================================

class InferenceCache:

    def __init__(self, config: InferenceCacheConfig):

        self.config = config
        self.memory_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = Lock()
        self._inflight: Dict[str, Lock] = {}

        self.cache_dir = Path(config.cache_dir)

        if config.enable_disk_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"InferenceCache initialized (version={config.cache_version})")

    # =====================================================
    # HASH (UPGRADED 🔥)
    # =====================================================

    def _hash_input(self, data: Any) -> str:

        try:
            def default(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, set):
                    return sorted(obj)
                return repr(obj)

            payload = (
                data
                if isinstance(data, str)
                else json.dumps(data, sort_keys=True, default=default)
            )

            # 🔥 VERSION AWARE HASH
            payload = f"{self.config.cache_version}:{payload}"

            return hashlib.sha256(payload.encode("utf-8")).hexdigest()

        except Exception as exc:
            raise RuntimeError("Cache key generation failed") from exc

    # =====================================================
    # PATH
    # =====================================================

    def _cache_path(self, key: str) -> Path:
        suffix = ".json.gz" if self.config.enable_compression else ".json"
        return self.cache_dir / f"{key}{suffix}"

    # =====================================================
    # IO
    # =====================================================

    def _safe_write(self, path: Path, payload: str):

        # MEM-3: ``path.with_suffix(".tmp")`` only replaces the LAST
        # suffix, so ``cache.json.gz`` became ``cache.json.tmp`` —
        # different stem from the final filename and a moving target if
        # compression mode toggled. Use ``with_name`` so the temp file
        # is unambiguously the final path + ``.tmp``.
        tmp = path.with_name(path.name + ".tmp")

        if self.config.enable_compression:
            # MEM-3: fsync the gzip path too. The previous code only
            # fsynced the uncompressed branch; a crash mid-write of a
            # compressed cache produced a partial gz that the reader
            # silently deleted, losing the work.
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except (AttributeError, OSError):
                    pass
        else:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())

        os.replace(tmp, path)

    def _safe_read(self, path: Path):

        if self.config.enable_compression:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    # =====================================================
    # TTL
    # =====================================================

    def _is_expired(self, ts: float):
        if self.config.ttl_seconds is None:
            return False
        return (time.monotonic() - ts) > self.config.ttl_seconds

    # =====================================================
    # MEMORY CACHE
    # =====================================================

    def _update_memory(self, key, entry):
        self.memory_cache[key] = entry
        self.memory_cache.move_to_end(key)

        if len(self.memory_cache) > self.config.max_memory_items:
            self.memory_cache.popitem(last=False)

    # =====================================================
    # GET
    # =====================================================

    def get(self, data: Any) -> Optional[Dict[str, Any]]:

        key = self._hash_input(data)

        # LAT-4: hold the lock only long enough to consult/mutate the
        # in-memory dict. Disk reads (especially gzip decompress) used to
        # block every other inference request because the entire body of
        # ``get`` ran under ``self._lock``.
        with self._lock:
            if self.config.enable_memory_cache:
                entry = self.memory_cache.get(key)
                if entry and not self._is_expired(entry["ts"]):
                    return entry["value"]

            disk_enabled = self.config.enable_disk_cache

        if disk_enabled:
            path = self._cache_path(key)
            if path.exists():
                try:
                    entry = self._safe_read(path)
                except Exception:
                    path.unlink(missing_ok=True)
                    return None

                if self._is_expired(entry["ts"]):
                    path.unlink(missing_ok=True)
                    return None

                # MEM-2: touch mtime so disk LRU eviction (driven by
                # ``_evict_disk_if_needed`` on ``set``) treats this entry
                # as recently used. Best-effort — failure here must not
                # break the read path.
                try:
                    os.utime(path, None)
                except OSError:
                    pass

                if self.config.enable_memory_cache:
                    with self._lock:
                        self._update_memory(key, entry)

                return entry["value"]

        return None

    # =====================================================
    # SET
    # =====================================================

    def set(self, data: Any, value: Dict[str, Any]):

        key = self._hash_input(data)

        # LAT-3: store the raw value in memory and serialise exactly once
        # when writing to disk. The previous code did
        # ``json.loads(json.dumps(value))`` here AND ``json.dumps(entry)``
        # below — a full round-trip through JSON on every set.
        entry = {
            "ts": time.monotonic(),
            "value": value,
        }

        with self._lock:
            if self.config.enable_memory_cache:
                self._update_memory(key, entry)

            disk_enabled = self.config.enable_disk_cache

        if disk_enabled:
            path = self._cache_path(key)
            try:
                payload = json.dumps(
                    entry, separators=(",", ":"), default=_serialize
                )
                self._safe_write(path, payload)
            except Exception as exc:
                logger.warning(f"Cache write failed: {exc}")
            # MEM-2: enforce the disk-side LRU bound (when configured).
            self._evict_disk_if_needed()

    # =====================================================
    # MEM-2: DISK LRU EVICTION
    # =====================================================

    def _evict_disk_if_needed(self) -> None:
        """Cap the disk cache to ``config.max_disk_items`` by mtime.

        ``None`` keeps the previous unbounded behaviour (suitable when
        the operator manages the volume out-of-band). With a bound set,
        the oldest-by-mtime files are unlinked until the count fits.
        """
        cap = self.config.max_disk_items
        if not cap or cap <= 0:
            return
        if not self.config.enable_disk_cache:
            return

        try:
            entries = [
                p for p in self.cache_dir.iterdir()
                if p.is_file() and p.suffix != ".tmp" and not p.name.endswith(".tmp")
            ]
        except OSError:
            return

        if len(entries) <= cap:
            return

        try:
            entries.sort(key=lambda p: p.stat().st_mtime)
        except OSError:
            return

        for p in entries[: len(entries) - cap]:
            try:
                p.unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("Cache eviction unlink failed for %s: %s", p, exc)

    # =====================================================
    # 🔥 LAT-5: SINGLE-FLIGHT
    # =====================================================

    def get_or_compute(self, data: Any, compute_fn) -> Dict[str, Any]:
        """Return the cached entry for ``data`` or compute it once.

        ``self._inflight`` was previously declared but never used, so two
        concurrent requests for the same uncached input would each run a
        full inference. Now the second caller blocks on a per-key lock
        until the first one writes the result, and then reads it from
        cache — only one forward pass per (input, version) tuple.
        """

        cached = self.get(data)
        if cached is not None:
            return cached

        key = self._hash_input(data)

        with self._lock:
            inflight = self._inflight.get(key)
            if inflight is None:
                inflight = Lock()
                inflight.acquire()
                self._inflight[key] = inflight
                owner = True
            else:
                owner = False

        if not owner:
            with inflight:
                pass
            cached = self.get(data)
            if cached is not None:
                return cached
            return self.get_or_compute(data, compute_fn)

        try:
            value = compute_fn()
            self.set(data, value)
            return value
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            try:
                inflight.release()
            except RuntimeError:
                pass

    # =====================================================
    # INVALIDATE
    # =====================================================

    def invalidate(self, data: Any):

        key = self._hash_input(data)

        with self._lock:
            self.memory_cache.pop(key, None)

            path = self._cache_path(key)
            path.unlink(missing_ok=True)

    # =====================================================
    # CLEAR
    # =====================================================

    def clear(self):

        with self._lock:
            self.memory_cache.clear()

            if self.config.enable_disk_cache:
                for f in self.cache_dir.glob("*"):
                    f.unlink(missing_ok=True)

        logger.info("Cache cleared")