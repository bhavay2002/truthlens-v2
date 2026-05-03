from __future__ import annotations

import hashlib
import json
import logging
import time
import zlib
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CACHE_VERSION = "v2"
EPS = 1e-12


class ExplanationCache:

    def __init__(
        self,
        max_size: int = 128,
        cache_dir: Optional[str | Path] = None,
        ttl_seconds: Optional[int] = None,
        enable_compression: bool = True,
    ) -> None:

        self.max_size = max_size
        self.memory_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.ttl_seconds = ttl_seconds
        self.enable_compression = enable_compression

        self._lock = RLock()

        #  stats
        self.hits = 0
        self.misses = 0

        logger.info("[ExplanationCache] initialized")

    # =====================================================
    # KEY GENERATION ( FIX)
    # =====================================================

    def _make_key(
        self,
        text: str,
        *,
        model_version: str = "default",
        method: Optional[str] = None,
    ) -> str:

        raw = f"{text}|{model_version}|{method}"
        return hashlib.sha256(raw.encode()).hexdigest()

    # =====================================================
    # SERIALIZATION
    # =====================================================

    def _serialize(self, data: Dict) -> bytes:
        raw = json.dumps(data).encode("utf-8")
        return zlib.compress(raw) if self.enable_compression else raw

    def _deserialize(self, data: bytes) -> Dict:
        raw = zlib.decompress(data) if self.enable_compression else data
        return json.loads(raw.decode("utf-8"))

    # =====================================================
    # EVICTION
    # =====================================================

    def _evict(self):
        while len(self.memory_cache) > self.max_size:
            self.memory_cache.popitem(last=False)

    # =====================================================
    # TTL CHECK
    # =====================================================

    def _is_expired(self, item: Dict) -> bool:
        if not self.ttl_seconds:
            return False

        ts = item.get("__timestamp__", 0)
        return (time.time() - ts) > self.ttl_seconds

    # =====================================================
    # GET
    # =====================================================

    def get(
        self,
        text: str,
        *,
        model_version: str = "default",
        method: Optional[str] = None,
    ) -> Optional[Dict]:

        key = self._make_key(text, model_version=model_version, method=method)

        with self._lock:

            # memory
            if key in self.memory_cache:
                item = self.memory_cache[key]

                if self._is_expired(item):
                    del self.memory_cache[key]
                    self.misses += 1
                    return None

                self.memory_cache.move_to_end(key)
                self.hits += 1
                return item["data"]

            # disk
            if self.cache_dir:
                path = self.cache_dir / key

                if path.exists():
                    try:
                        raw = path.read_bytes()
                        item = self._deserialize(raw)

                        if item.get("__version__") != CACHE_VERSION:
                            return None

                        if self._is_expired(item):
                            path.unlink(missing_ok=True)
                            self.misses += 1
                            return None

                        self.memory_cache[key] = item
                        self._evict()

                        self.hits += 1
                        return item["data"]

                    except Exception:
                        pass

            self.misses += 1
            return None

    # =====================================================
    # SET
    # =====================================================

    def set(
        self,
        text: str,
        data: Dict,
        *,
        model_version: str = "default",
        method: Optional[str] = None,
    ):

        key = self._make_key(text, model_version=model_version, method=method)

        item = {
            "__version__": CACHE_VERSION,
            "__timestamp__": time.time(),
            "data": data,
        }

        with self._lock:

            self.memory_cache[key] = item
            self.memory_cache.move_to_end(key)
            self._evict()

            if self.cache_dir:
                try:
                    path = self.cache_dir / key
                    path.write_bytes(self._serialize(item))
                except Exception as exc:
                    logger.warning(
                        "Cache disk write failed for key %.16s…: %s", key, exc
                    )

    # =====================================================
    # STATS ( NEW)
    # =====================================================

    def stats(self) -> Dict[str, float]:
        total = self.hits + self.misses + EPS
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total,
        }

    # =====================================================
    # CLEAR
    # =====================================================

    def clear_memory(self):
        with self._lock:
            self.memory_cache.clear()

    def clear_disk(self):
        if not self.cache_dir:
            return

        for f in self.cache_dir.glob("*"):
            try:
                f.unlink()
            except Exception:
                pass