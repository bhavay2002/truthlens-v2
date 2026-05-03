# src/features/pipelines/feature_batch_pipeline.py

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import torch
from torch.utils.data import Dataset, DataLoader

from src.features.base.base_feature import FeatureContext
from src.features.pipelines.feature_pipeline import (
    FeaturePipeline,
    partition_feature_sections,
)

logger = logging.getLogger(__name__)

# Bound the per-pipeline graph cache so long-running services do not
# accumulate one entry per unique input text indefinitely (OOM risk).
_GRAPH_CACHE_MAX = 2048

# Bumped whenever the in-memory graph cache key payload schema changes
# (audit fix #1.2 — config fingerprint now part of the key).
GRAPH_CACHE_VERSION = "v2"


def _graph_cache_key(text: str, graph_cfg_fingerprint: str = "") -> str:
    """
    Hash text + graph-pipeline config fingerprint into a stable cache key.

    Audit fix #1.2: the previous key was sha256(text) only — switching the
    entity NER model or narrative lexicon silently returned yesterday's
    graph features.  Embedding the GraphPipeline config fingerprint makes
    that mutation auto-invalidate.
    """
    payload = {
        "v": GRAPH_CACHE_VERSION,
        "text": text,
        "cfg": graph_cfg_fingerprint or "",
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode(
        "utf-8", errors="ignore"
    )
    return hashlib.sha256(raw).hexdigest()


# =========================================================
# DATASET
# =========================================================

class FeatureDataset(Dataset):
    def __init__(self, contexts: List[FeatureContext]):
        self.contexts = contexts

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return idx, self.contexts[idx]


def collate_fn(batch):
    indices, contexts = zip(*batch)
    return list(indices), list(contexts)


# =========================================================
# PIPELINE
# =========================================================

@dataclass
class BatchFeaturePipeline:

    pipeline: FeaturePipeline

    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True
    use_amp: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    _initialized: bool = field(default=False, init=False)

    # 🔥 GLOBAL SHARED CACHE
    _shared_cache: Dict[str, Any] = field(default_factory=dict, init=False)

    # Bounded LRU graph cache.  Key = sha256(text), value = graph
    # output dict.  Replaces the previously-unbounded plain dict.
    _graph_cache: "OrderedDict[str, Any]" = field(
        default_factory=OrderedDict, init=False
    )

    # -----------------------------------------------------

    def initialize(self) -> None:

        if self._initialized:
            return

        self.pipeline.initialize()

        # 🔥 Move model
        if hasattr(self.pipeline, "model"):
            try:
                self.pipeline.model.to(self.device)
            except Exception:
                logger.warning("Model device move failed")

            # COMPILE-OFF: ``torch.compile`` removed project-wide (see
            # src/training/training_setup.py for rationale). The previous
            # call wrapped ``self.pipeline.model`` with Dynamo tracing,
            # which caused environment-dependent instability and spurious
            # AMP overflow warnings. Run in eager mode.

        self._initialized = True

        logger.info(
            "BatchFeaturePipeline initialized | batch_size=%d device=%s",
            self.batch_size,
            self.device,
        )

    # =====================================================
    # 🔥 EMBEDDING OPTIMIZATION (NEW)
    # =====================================================

    def _compute_embeddings(self, batch: List[FeatureContext]):

        if not hasattr(self.pipeline, "encoder") or not hasattr(self.pipeline, "tokenizer"):
            return

        texts = [ctx.text for ctx in batch]

        try:
            device = torch.device(self.device)

            with torch.no_grad():
                with torch.autocast(self.device, enabled=self.device == "cuda"):

                    inputs = self.pipeline.tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)

                    outputs = self.pipeline.encoder(**inputs)

            embeddings = outputs.last_hidden_state

            # Detach + move to CPU so the cached tensor does not pin
            # GPU memory for the lifetime of every FeatureContext in
            # the batch (was a steady VRAM leak).
            embeddings_cpu = embeddings.detach().to("cpu").contiguous()

            for i, ctx in enumerate(batch):
                ctx.cache["_shared_cache"]["embedding"] = embeddings_cpu[i]

        except Exception as e:
            logger.warning("Embedding computation failed: %s", e)

    # =====================================================
    # 🔥 GRAPH CACHE (NEW)
    # =====================================================

    def _attach_graph_cache(self, batch: List[FeatureContext]):

        gp = self.pipeline.graph_pipeline
        if not gp:
            return

        # Audit fix #1.2 — config fingerprint participates in the key so
        # toggling any GraphPipelineConfig field auto-invalidates.
        try:
            cfg_fp = gp.config_fingerprint()
        except Exception:
            cfg_fp = ""

        for ctx in batch:

            key = _graph_cache_key(ctx.text, cfg_fp)

            cached = self._graph_cache.get(key)
            if cached is None:
                try:
                    cached = gp.run(ctx.text)
                except Exception as e:
                    logger.warning("Graph failed: %s", e)
                    cached = {}
                self._graph_cache[key] = cached
                # LRU eviction
                if len(self._graph_cache) > _GRAPH_CACHE_MAX:
                    self._graph_cache.popitem(last=False)
            else:
                self._graph_cache.move_to_end(key)

            # Audit fix #1.3 — eliminate double graph extraction.
            # Populate the same slot _merge_graph_features looks at, so
            # the per-sample merge step finds the cached output and does
            # NOT re-run gp.run(text).  Previously the batch path called
            # gp.run() here AND again in _merge_graph_features → the
            # heavy NetworkX/spaCy build ran twice per request.
            graph_slot = ctx.cache.setdefault("_graph", {})
            graph_slot["output"] = cached

            # Backwards-compatible alias for any consumer still reading
            # the legacy key.
            ctx.cache["graph_pipeline_output"] = cached

    # =====================================================
    # CORE EXECUTION
    # =====================================================

    def _run_batch_extract(self, batch: List[FeatureContext]):

        if hasattr(self.pipeline, "batch_extract"):
            return self.pipeline.batch_extract(batch)

        if hasattr(self.pipeline, "extract_batch"):
            return self.pipeline.extract_batch(batch)

        raise AttributeError("Pipeline missing batch API")

    # -----------------------------------------------------

    def _process_batch(self, batch: List[FeatureContext]):

        # 🔥 attach shared cache
        for ctx in batch:
            if not isinstance(ctx.cache, dict):
                ctx.cache = {}
            ctx.cache["_shared_cache"] = self._shared_cache

        # 🔥 NEW: embedding optimization
        self._compute_embeddings(batch)

        # 🔥 NEW: graph cache
        self._attach_graph_cache(batch)

        try:
            with torch.no_grad():

                if self.device == "cuda" and self.use_amp:

                    with torch.autocast(
                        device_type="cuda",
                        dtype=(
                            torch.bfloat16
                            if torch.cuda.is_bf16_supported()
                            else torch.float16
                        ),
                    ):
                        return self._run_batch_extract(batch)

                return self._run_batch_extract(batch)

        except Exception as e:
            # Batch-level failure (e.g. AMP / OOM): fall back to per-sample
            # extraction so a single bad row cannot null out the whole
            # batch.  Failed samples are logged and re-raised so callers
            # see them — never silently substitute empty dicts (that
            # masks real bugs and produces silently-broken training data).
            logger.warning(
                "Batch extract failed (%s); falling back to per-sample mode.", e
            )
            results: list[dict] = []
            for i, ctx in enumerate(batch):
                try:
                    results.append(self._run_batch_extract([ctx])[0])
                except Exception as inner:
                    logger.exception(
                        "Per-sample extract failed for index %d: %s", i, inner
                    )
                    raise
            return results

    # =====================================================
    # DATALOADER
    # =====================================================

    def _dataloader_extract(self, contexts: List[FeatureContext]):

        dataset = FeatureDataset(contexts)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers if self.device == "cuda" else 0,
            pin_memory=self.pin_memory if self.device == "cuda" else False,
            collate_fn=collate_fn,
            shuffle=False,
        )

        results: List[Optional[Dict[str, float]]] = [None] * len(contexts)

        logger.info("Starting batch extraction | samples=%d", len(contexts))

        for indices, batch in loader:

            batch_features = self._process_batch(batch)

            for idx, feat in zip(indices, batch_features):
                results[idx] = feat

        if any(r is None for r in results):
            raise RuntimeError("Incomplete extraction")

        logger.info("Batch extraction completed")

        return results  # type: ignore

    # =====================================================
    # PUBLIC API
    # =====================================================

    def extract(self, contexts: List[FeatureContext]):

        if not contexts:
            raise ValueError("Empty input")

        if not self._initialized:
            self.initialize()

        return self._dataloader_extract(contexts)

    # -----------------------------------------------------

    def extract_by_section(self, contexts: List[FeatureContext]):

        flat = self.extract(contexts)

        return [partition_feature_sections(f) for f in flat]

    # -----------------------------------------------------

    def extract_with_labels(
        self,
        contexts: List[FeatureContext],
        labels: Optional[List[int]] = None,
        fit: bool = False,
    ):

        return self.pipeline.process(contexts, labels=labels, fit=fit)