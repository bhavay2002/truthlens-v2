# src/features/cache/cache_manager.py

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.features.base.base_feature import FeatureContext
from src.features.cache.feature_cache import FeatureCache, CACHE_VERSION


# Audit fix §1.10 — deterministic ``json.dumps(default=...)`` for the
# ad-hoc Python types we see in cache-key metadata. Sets and frozensets
# get sorted (key=str) so two requests with the same logical content
# produce the same cache key regardless of insertion order; bytes get
# hex-encoded so the JSON stays text-safe; everything else falls back
# to ``repr`` which is stable for the dataclasses / namedtuples we use.

def _stable_default(o):
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=str)
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, (bytes, bytearray)):
        return o.hex()
    return repr(o)

logger = logging.getLogger(__name__)

FeatureVector = Dict[str, float]
# CACHE_VERSION is re-exported from feature_cache to avoid the previous
# split-brain (two constants that had to be edited in lock-step).


# =========================================================
# LRU MEMORY CACHE
# =========================================================

class LRUCache:
    """Bounded in-process feature-vector LRU.

    Audit fix §7.6 — the previous implementation only enforced an
    item-count budget (``max_items``). For wide feature vectors (~6k
    keys at ~80 bytes each ≈ 500KB per entry) ``max_items=10_000``
    silently consumed several GB of RSS before the count cap kicked in.
    We now also track an estimated *byte* budget per entry and evict
    LRU until the global byte total is under ``max_bytes``.
    """

    # Approximate per-entry overhead: dict header + each (key, float)
    # pair. Numbers are conservative for CPython 3.11+; the goal is a
    # stable order-of-magnitude estimate, not a precise heap accountant.
    _DICT_OVERHEAD_BYTES = 232
    _PER_ENTRY_BYTES = 80  # str interned ptr + float64 value + slot

    def __init__(
        self,
        max_items: int,
        max_bytes: Optional[int] = None,
    ):
        self.max_items = max_items
        # ``None`` disables the byte budget and preserves the previous
        # count-only semantics for callers that opt out explicitly.
        self.max_bytes = max_bytes

        self.store: OrderedDict[str, FeatureVector] = OrderedDict()
        self._sizes: Dict[str, int] = {}
        self._total_bytes: int = 0

    @classmethod
    def _estimate_bytes(cls, value: FeatureVector) -> int:
        if not isinstance(value, dict):
            return cls._DICT_OVERHEAD_BYTES
        return cls._DICT_OVERHEAD_BYTES + len(value) * cls._PER_ENTRY_BYTES

    def get(self, key: str) -> Optional[FeatureVector]:
        if key not in self.store:
            return None
        self.store.move_to_end(key)
        # Return a shallow copy so downstream pruning / scaling cannot
        # corrupt the cached object (FeatureVector values are floats so
        # a shallow copy is sufficient).
        return dict(self.store[key])

    def set(self, key: str, value: FeatureVector) -> None:
        # Store a copy so subsequent caller mutation does not propagate
        # back into the cache.
        if key in self.store:
            self._total_bytes -= self._sizes.pop(key, 0)
            del self.store[key]

        copied = dict(value)
        size = self._estimate_bytes(copied)
        self.store[key] = copied
        self._sizes[key] = size
        self._total_bytes += size
        self.store.move_to_end(key)

        # Evict by item count first (cheap), then by byte budget. The
        # two caps interact safely because byte eviction also drops
        # items so the count cap is implicitly respected.
        while len(self.store) > self.max_items:
            self._evict_oldest()
        if self.max_bytes is not None:
            while self._total_bytes > self.max_bytes and self.store:
                self._evict_oldest()

    def _evict_oldest(self) -> None:
        old_key, _ = self.store.popitem(last=False)
        self._total_bytes -= self._sizes.pop(old_key, 0)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes


# =========================================================
# CACHE MANAGER
# =========================================================

@dataclass
class CacheManager:

    base_cache_dir: Optional[Path] = None
    max_memory_items: int = 10000
    # Audit fix §7.6 — global byte budget for the in-process LRU.
    # ``None`` preserves the previous count-only behaviour. Default
    # 512MB matches the host RAM headroom we leave for the rest of the
    # inference pipeline; callers running on smaller workers should
    # set this explicitly.
    max_memory_bytes: Optional[int] = 512 * 1024 * 1024

    namespaces: Dict[str, FeatureCache] = field(default_factory=dict)
    _memory_cache: LRUCache = field(init=False)

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    # Lazily-computed fingerprint of the registered feature set; included
    # in cache keys so enable/disable of features auto-invalidates.
    feature_set_fingerprint: Optional[str] = field(default=None, init=False)

    # Audit fix §7.6 — Prometheus-friendly hit / miss counters. Stored
    # as plain ints; under CPython the GIL makes a single ``+= 1``
    # observation effectively atomic for stats purposes (we accept the
    # very rare lost increment in exchange for not paying a lock on the
    # hot path).
    mem_hits: int = field(default=0, init=False)
    mem_misses: int = field(default=0, init=False)
    disk_hits: int = field(default=0, init=False)
    disk_misses: int = field(default=0, init=False)
    computes: int = field(default=0, init=False)
    disk_write_failures: int = field(default=0, init=False)

    # -----------------------------------------------------

    def __post_init__(self):
        self._memory_cache = LRUCache(
            self.max_memory_items, max_bytes=self.max_memory_bytes
        )

    # -----------------------------------------------------

    def _namespace_path(self, namespace: str) -> Path:
        base = self.base_cache_dir or Path("cache")
        return base / namespace

    # -----------------------------------------------------

    def get_cache(self, namespace: str) -> FeatureCache:

        if namespace not in self.namespaces:
            with self._lock:
                if namespace not in self.namespaces:
                    cache = FeatureCache(self._namespace_path(namespace))
                    self.namespaces[namespace] = cache
                    logger.info("Registered cache namespace: %s", namespace)

        return self.namespaces[namespace]

    # -----------------------------------------------------
    # PRUNE  (audit fix #1.5)
    #
    # Sweep every namespace under base_cache_dir, plus any namespaces
    # registered in this process, applying the same TTL + byte-budget
    # eviction.  Safe to invoke at process start: missing dirs and
    # transient OS errors are logged and skipped, never raised.
    # -----------------------------------------------------

    def prune_all(
        self,
        *,
        max_bytes_per_namespace: Optional[int] = None,
        max_age_days: Optional[float] = None,
    ) -> Dict[str, Dict[str, int]]:

        results: Dict[str, Dict[str, int]] = {}
        seen: set[Path] = set()

        # In-process namespaces first (they own the live LRU memo we
        # need to invalidate on file deletion).
        for ns, cache in list(self.namespaces.items()):
            try:
                results[ns] = cache.prune(
                    max_bytes=max_bytes_per_namespace,
                    max_age_days=max_age_days,
                )
                seen.add(cache.cache_dir.resolve())
            except Exception as exc:
                logger.warning("Prune failed for namespace %s: %s", ns, exc)

        # Plus any namespaces persisted on disk from a previous run that
        # have not yet been registered this process.
        base = self.base_cache_dir or Path("cache")
        if base.exists():
            for child in base.iterdir():
                if not child.is_dir():
                    continue
                if child.resolve() in seen:
                    continue
                try:
                    cache = FeatureCache(child)
                    results[child.name] = cache.prune(
                        max_bytes=max_bytes_per_namespace,
                        max_age_days=max_age_days,
                    )
                except Exception as exc:
                    logger.warning("Prune failed for dir %s: %s", child, exc)

        return results

    # -----------------------------------------------------
    # VERSIONED KEY (CRITICAL)
    #
    # The key includes a *feature-set fingerprint* derived from the
    # currently-registered feature names.  This means: enabling /
    # disabling features automatically invalidates stale cache entries
    # without requiring CACHE_VERSION to be bumped manually.
    # -----------------------------------------------------

    @staticmethod
    def _compute_feature_set_fingerprint() -> str:
        try:
            from src.features.base.feature_registry import FeatureRegistry
            names = sorted(FeatureRegistry.list_features())
        except Exception:
            names = []
        if not names:
            return "no-registry"
        return hashlib.sha256("|".join(names).encode()).hexdigest()[:16]

    def _get_feature_fingerprint(self) -> str:
        if self.feature_set_fingerprint is None:
            self.feature_set_fingerprint = self._compute_feature_set_fingerprint()
        return self.feature_set_fingerprint

    # -----------------------------------------------------
    # LEXICON FINGERPRINT
    #
    # SHA over the *contents* of every loaded lexicon source file. If
    # any lexicon (bias, emotion, propaganda, …) changes on disk, the
    # fingerprint changes and stale cache entries are auto-invalidated
    # without requiring CACHE_VERSION to be bumped manually.
    # -----------------------------------------------------

    # Source files holding the in-process lexicons. Resolved once at
    # class-load time and hashed lazily on first cache key computation.
    #
    # Audit fix §7 — entries that point at files removed in the §4
    # cleanup (``emotion_lexicon.py``, ``emotion_lexicon_features.py``,
    # ``emotion_features.py``, ``propaganda_lexicon_features.py``)
    # used to contribute ``missing:<rel>`` placeholders to the
    # fingerprint, which (a) was deterministic but (b) generated noisy
    # hashes that drifted any time another deletion happened. We now
    # only fingerprint files that still exist.
    _LEXICON_SOURCES: tuple = (
        "src/features/bias/bias_lexicon.py",
        "src/features/bias/bias_lexicon_features.py",
        "src/features/bias/bias_features.py",
        "src/features/emotion/emotion_schema.py",
        "src/features/emotion/emotion_intensity_features.py",
        "src/features/propaganda/propaganda_features.py",
    )

    lexicon_fingerprint: Optional[str] = field(default=None, init=False)

    @classmethod
    def _compute_lexicon_fingerprint(cls) -> str:
        # Resolve relative paths against the project root (two levels up
        # from this file: src/features/cache/cache_manager.py).
        project_root = Path(__file__).resolve().parents[3]
        h = hashlib.sha256()
        h.update(b"lexicons-v1\n")
        for rel in cls._LEXICON_SOURCES:
            p = project_root / rel
            if not p.is_file():
                # Missing file is itself a meaningful signal — record
                # the path so adding/removing a lexicon invalidates.
                h.update(f"missing:{rel}\n".encode())
                continue
            try:
                h.update(rel.encode())
                h.update(b"\0")
                h.update(p.read_bytes())
                h.update(b"\n")
            except OSError as exc:
                logger.warning("Lexicon fingerprint read failed (%s): %s", rel, exc)
                h.update(f"unreadable:{rel}\n".encode())
        return h.hexdigest()[:16]

    def _get_lexicon_fingerprint(self) -> str:
        if self.lexicon_fingerprint is None:
            self.lexicon_fingerprint = self._compute_lexicon_fingerprint()
        return self.lexicon_fingerprint

    # -----------------------------------------------------
    # COMBINED FINGERPRINT  (audit fix §7.1)
    #
    # The on-disk pickle blob carries this short fingerprint in its
    # header; reads whose stored fingerprint does not match are
    # treated as a miss instead of returning stale data. This is a
    # second line of defence behind ``context_key`` (which already
    # bakes the same components into the path digest).
    # -----------------------------------------------------

    def _combined_fingerprint(self) -> str:
        return f"{self._get_feature_fingerprint()}:{self._get_lexicon_fingerprint()}"

    # -----------------------------------------------------
    # STATS  (audit fix §7.6)
    # -----------------------------------------------------

    def stats(self) -> Dict[str, int]:
        """Snapshot of cache hit / miss counters.

        Suitable for scraping into Prometheus or logging at shutdown.
        Counters are best-effort (not lock-protected) — fine for
        observability, not for correctness checks.
        """
        return {
            "mem_hits": self.mem_hits,
            "mem_misses": self.mem_misses,
            "disk_hits": self.disk_hits,
            "disk_misses": self.disk_misses,
            "computes": self.computes,
            "disk_write_failures": self.disk_write_failures,
            "mem_size_items": len(self._memory_cache.store),
            "mem_size_bytes": self._memory_cache.total_bytes,
        }

    def reset_stats(self) -> None:
        self.mem_hits = 0
        self.mem_misses = 0
        self.disk_hits = 0
        self.disk_misses = 0
        self.computes = 0
        self.disk_write_failures = 0

    def context_key(self, context: FeatureContext) -> str:
        """Return the deterministic cache key for ``context``.

        This is the public API. The cache key bakes in:
        - ``CACHE_VERSION`` (bump to invalidate the entire on-disk cache).
        - ``feature_set`` fingerprint (registry change → new key).
        - ``lexicons`` fingerprint (lexicon edit → new key).
        - ``tokenizer_id`` from ``context.metadata`` (model swap → new key).
        - ``context.text`` and the precomputed ``context.tokens`` if any.
        - The remaining metadata, serialised through ``_stable_default``
          for deterministic JSON.

        Audit fix §1.9 — promoted from the previous private
        ``_context_key`` so ``DatasetFeatureGenerator`` and any other
        external caller can derive a cache key without reaching into a
        leading-underscore helper. ``_context_key`` is preserved below
        as a thin deprecation shim so existing callers keep working.
        """

        # Pull tokenizer_id out of metadata (if any) so switching
        # roberta-base ↔ xlm-roberta-base or upgrading the tokenizer
        # auto-invalidates without leaking BPE-aligned features into a
        # different model head.  Audit fix #1.6.
        meta = dict(context.metadata or {})
        tokenizer_id = meta.pop("tokenizer_id", None)

        payload = {
            "version": CACHE_VERSION,
            "feature_set": self._get_feature_fingerprint(),
            "lexicons": self._get_lexicon_fingerprint(),
            "tokenizer_id": tokenizer_id,
            "text": context.text,
            "tokens": context.tokens,
            "metadata": meta,
        }

        # Audit fix §1.10 — ``default=str`` was lossy for non-JSON-
        # serialisable types (sets, tuples, custom objects all collapsed
        # to their ``str()`` form, which is order-dependent and round-
        # trips badly through ``repr``). ``_stable_default`` normalises
        # the small set of ad-hoc types we see in metadata into a
        # deterministic shape, then we fall back to ``repr`` for
        # anything we have not handled explicitly. ``sort_keys=True``
        # is preserved for the top-level payload.
        raw = json.dumps(payload, sort_keys=True, default=_stable_default)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _context_key(self, context: FeatureContext) -> str:
        """Backward-compat alias for :meth:`context_key`.

        Audit fix §1.9 — kept so existing internal callers
        (``get_or_compute``, ``get_or_compute_batch``) continue to work
        without a churn-only diff. New code MUST use ``context_key``.
        """
        return self.context_key(context)

    # -----------------------------------------------------

    def get_or_compute(
        self,
        namespace: str,
        context: FeatureContext,
        compute_fn: Callable[[FeatureContext], FeatureVector],
    ) -> FeatureVector:

        cache = self.get_cache(namespace)
        key = self._context_key(context)
        fp = self._combined_fingerprint()

        # -------------------------
        # MEMORY CACHE
        # -------------------------

        cached = self._memory_cache.get(key)
        if cached is not None:
            self.mem_hits += 1
            return cached
        self.mem_misses += 1

        # -------------------------
        # DISK CACHE
        # -------------------------

        cached = cache.load(key, expected_fingerprint=fp)

        if cached is not None:
            self.disk_hits += 1
            self._memory_cache.set(key, cached)
            return cached
        self.disk_misses += 1

        # -------------------------
        # COMPUTE
        # -------------------------

        result = compute_fn(context)
        self.computes += 1

        # -------------------------
        # SAVE
        # -------------------------

        try:
            cache.save(key, result, fingerprint=fp)
        except Exception:
            self.disk_write_failures += 1
            logger.warning("Disk cache write failed")

        self._memory_cache.set(key, result)

        return result

    # -----------------------------------------------------
    # BATCH (OPTIMIZED)
    # -----------------------------------------------------

    def get_or_compute_batch(
        self,
        namespace: str,
        contexts: List[FeatureContext],
        compute_batch_fn: Callable[[List[FeatureContext]], List[FeatureVector]],
    ) -> List[FeatureVector]:

        if not contexts:
            return []

        cache = self.get_cache(namespace)
        fp = self._combined_fingerprint()

        keys = [self._context_key(c) for c in contexts]

        results: List[Optional[FeatureVector]] = [None] * len(contexts)
        missing: List[FeatureContext] = []
        missing_idx: List[int] = []

        # -------------------------
        # LOOKUP
        # -------------------------

        for i, key in enumerate(keys):

            cached = self._memory_cache.get(key)

            if cached is not None:
                self.mem_hits += 1
                results[i] = cached
                continue
            self.mem_misses += 1

            cached = cache.load(key, expected_fingerprint=fp)

            if cached is not None:
                self.disk_hits += 1
                results[i] = cached
                self._memory_cache.set(key, cached)
            else:
                self.disk_misses += 1
                missing.append(contexts[i])
                missing_idx.append(i)

        # -------------------------
        # COMPUTE MISSING
        # -------------------------

        if missing:

            computed = compute_batch_fn(missing)
            self.computes += len(computed)

            for i, key, val in zip(missing_idx, [keys[j] for j in missing_idx], computed):

                results[i] = val

                try:
                    cache.save(key, val, fingerprint=fp)
                except Exception:
                    self.disk_write_failures += 1
                    logger.warning("Disk write failed")

                self._memory_cache.set(key, val)

        return results  # type: ignore