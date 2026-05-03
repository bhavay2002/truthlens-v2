# src/features/cache/feature_cache.py

from __future__ import annotations

import hashlib
import logging
import pickle
import threading
import time
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG  (single source of truth — re-exported by cache_manager)
# =========================================================

# Audit fix §2.5 — the previous serializer was ``json.dumps`` wrapped in
# ``gzip.compress``, which on a 6k-feature payload spent 70%+ of the
# cache time in the JSON encoder + a second copy in gzip. Pickle (with
# the default binary protocol) is ~10x faster end-to-end on the same
# payload, supports numpy scalars natively, and avoids the lossy
# ``default=str`` coercion that the JSON path was relying on. We bump
# CACHE_VERSION + change the file suffix so any old gzip-JSON entries
# from a previous run are treated as a miss instead of decoded as
# garbage.
#
# Audit fix §7.1 — the pickled payload also carries an optional
# ``fingerprint`` (caller-supplied; in practice the SHA-16 of the
# combined feature-set + lexicon fingerprints from ``CacheManager``).
# A read whose ``expected_fingerprint`` does not match the stored one
# is treated as a miss instead of returning stale data. This protects
# any code path that bypasses ``CacheManager.context_key`` (which
# already bakes the fingerprints into the path digest) — e.g. a future
# caller that reuses ``FeatureCache`` directly with arbitrary keys, or
# a cache directory copied between processes.
CACHE_VERSION = "v4"
_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL

# Cap the in-process path -> Path memoization dict so long-running
# services do not accumulate one entry per unique cache key forever.
_PATH_CACHE_MAX = 50_000


# =========================================================
# CACHE
# =========================================================

class FeatureCache:

    def __init__(self, cache_dir: str | Path = "cache") -> None:
        self.cache_dir = Path(cache_dir)
        # Bounded LRU so the in-process key->Path memo never grows
        # without limit on long-running services.
        self._path_cache: "OrderedDict[str, Path]" = OrderedDict()
        self._lock = threading.Lock()

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------

    def _get_path(self, key: str) -> Path:

        with self._lock:
            if key in self._path_cache:
                self._path_cache.move_to_end(key)
                return self._path_cache[key]

            digest = hashlib.sha256(key.encode()).hexdigest()
            # ``.pkl`` suffix prevents accidental cross-decoding with
            # the old ``.json.gz`` files left over from CACHE_VERSION
            # ≤ v3; those files now simply look like an unrelated
            # namespace and are pruned by the disk sweeper.
            filename = f"{digest}.pkl"

            path = self.cache_dir / filename
            self._path_cache[key] = path

            if len(self._path_cache) > _PATH_CACHE_MAX:
                self._path_cache.popitem(last=False)

            return path

    # -----------------------------------------------------
    # SAFE SERIALIZATION
    # -----------------------------------------------------

    def _serialize(self, data: Any, fingerprint: Optional[str] = None) -> bytes:

        payload = {
            "version": CACHE_VERSION,
            "fingerprint": fingerprint,
            "data": data,
        }

        return pickle.dumps(payload, protocol=_PICKLE_PROTOCOL)

    def _deserialize(
        self,
        raw: bytes,
        expected_fingerprint: Optional[str] = None,
    ) -> Any:

        try:
            payload = pickle.loads(raw)
        except Exception:
            # Corrupt or wrong-format file (e.g. an old gzip-JSON entry
            # from a previous CACHE_VERSION). Treat as a miss; the
            # caller deletes the offending file.
            raise

        if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
            logger.warning("Cache version mismatch")
            return None

        # Audit fix §7.1 — schema-fingerprint header check. Only enforce
        # when the caller supplied an expected fingerprint; legacy
        # blobs written before this field existed have ``None`` and
        # match a ``None`` expectation.
        if expected_fingerprint is not None:
            stored_fp = payload.get("fingerprint")
            if stored_fp != expected_fingerprint:
                logger.debug(
                    "Cache fingerprint mismatch (have=%r expect=%r) — miss",
                    stored_fp, expected_fingerprint,
                )
                return None

        return payload.get("data")

    # -----------------------------------------------------
    # ATOMIC WRITE (CRITICAL)
    # -----------------------------------------------------

    def save(
        self,
        key: str,
        data: Any,
        *,
        fingerprint: Optional[str] = None,
    ) -> Path:

        path = self._get_path(key)
        temp_path: Optional[Path] = None
        replaced = False

        try:
            serialized = self._serialize(data, fingerprint=fingerprint)

            with tempfile.NamedTemporaryFile(
                delete=False,
                dir=self.cache_dir,
            ) as tmp:

                tmp.write(serialized)
                tmp.flush()

                temp_path = Path(tmp.name)

            temp_path.replace(path)
            replaced = True

            return path

        except Exception:
            logger.exception("Cache save failed")
            raise

        finally:
            # Audit fix #1.5 — if the process is killed (or any exception
            # is raised) between tmp.flush() and temp_path.replace(path),
            # the temp file was orphaned forever and the cache dir grew
            # monotonically.  The try/finally guarantees the temp file is
            # removed on every code path that does not reach replace().
            if temp_path is not None and not replaced:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning(
                        "Cache temp cleanup failed (%s): %s", temp_path, exc
                    )

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    def load(
        self,
        key: str,
        *,
        expected_fingerprint: Optional[str] = None,
    ) -> Optional[Any]:

        path = self._get_path(key)

        if not path.exists():
            return None

        try:
            raw = path.read_bytes()
            return self._deserialize(
                raw, expected_fingerprint=expected_fingerprint
            )

        except Exception:
            logger.exception("Cache load failed → deleting corrupted file")
            path.unlink(missing_ok=True)
            return None

    # -----------------------------------------------------
    # BATCH LOAD
    # -----------------------------------------------------

    def load_many(
        self,
        keys: List[str],
        *,
        expected_fingerprint: Optional[str] = None,
    ) -> List[Optional[Any]]:
        return [
            self.load(k, expected_fingerprint=expected_fingerprint) for k in keys
        ]

    # -----------------------------------------------------

    def exists(self, key: str) -> bool:
        return self._get_path(key).exists()

    # -----------------------------------------------------

    def clear(self) -> None:
        for file in self.cache_dir.glob("*"):
            file.unlink(missing_ok=True)

        self._path_cache.clear()
        logger.info("Cache cleared: %s", self.cache_dir)

    # -----------------------------------------------------
    # DISK PRUNER  (audit fix #1.5)
    #
    # Without an eviction policy the on-disk cache grew unbounded.  This
    # prune step is invoked from the inference startup hook and provides
    # both age-based (TTL) and size-based (LRU-ish) eviction in one pass:
    #
    #   * max_age_days   — delete any cache file older than this
    #                      (mtime-based; orphan tempfiles are also caught).
    #   * max_bytes      — after age eviction, if the total namespace
    #                      size still exceeds this budget, delete the
    #                      least-recently-modified files until it fits.
    # -----------------------------------------------------

    def prune(
        self,
        *,
        max_bytes: Optional[int] = None,
        max_age_days: Optional[float] = None,
    ) -> Dict[str, int]:

        stats = {"removed_age": 0, "removed_size": 0, "kept": 0, "bytes": 0}

        if not self.cache_dir.exists():
            return stats

        now = time.time()
        ttl_seconds = (
            float(max_age_days) * 86400.0 if max_age_days is not None else None
        )

        # Collect file metadata once; ignore directories and unreadable
        # entries so a single bad file cannot abort the whole sweep.
        entries: List[tuple[Path, int, float]] = []
        for entry in self.cache_dir.iterdir():
            if not entry.is_file():
                continue
            try:
                st = entry.stat()
            except OSError:
                continue
            entries.append((entry, st.st_size, st.st_mtime))

        # ---------- age eviction ----------
        survivors: List[tuple[Path, int, float]] = []
        for path, size, mtime in entries:
            if ttl_seconds is not None and (now - mtime) > ttl_seconds:
                try:
                    path.unlink(missing_ok=True)
                    stats["removed_age"] += 1
                except OSError as exc:
                    logger.warning("Cache prune (age) failed: %s", exc)
                    survivors.append((path, size, mtime))
                continue
            survivors.append((path, size, mtime))

        # ---------- size eviction ----------
        total = sum(s for _, s, _ in survivors)
        if max_bytes is not None and total > max_bytes:
            # Oldest first, drop until under budget.
            survivors.sort(key=lambda t: t[2])
            kept: List[tuple[Path, int, float]] = []
            for path, size, mtime in survivors:
                if total > max_bytes:
                    try:
                        path.unlink(missing_ok=True)
                        stats["removed_size"] += 1
                        total -= size
                        continue
                    except OSError as exc:
                        logger.warning("Cache prune (size) failed: %s", exc)
                kept.append((path, size, mtime))
            survivors = kept

        stats["kept"] = len(survivors)
        stats["bytes"] = sum(s for _, s, _ in survivors)

        # Drop any in-process path memo that points at a now-deleted file.
        if stats["removed_age"] or stats["removed_size"]:
            with self._lock:
                live = {p for p, _, _ in survivors}
                for k in list(self._path_cache.keys()):
                    if self._path_cache[k] not in live:
                        del self._path_cache[k]

        logger.info(
            "Cache pruned (%s) | removed_age=%d removed_size=%d kept=%d bytes=%d",
            self.cache_dir,
            stats["removed_age"],
            stats["removed_size"],
            stats["kept"],
            stats["bytes"],
        )

        return stats
