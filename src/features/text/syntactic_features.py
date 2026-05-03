from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Any

import numpy as np

from src.features.base.base_feature import BaseFeature, FeatureContext
from src.features.base.feature_registry import register_feature
from src.features.base.numerics import EPS, MAX_CLIP, normalized_entropy
from src.features.base.spacy_doc import ensure_spacy_doc, set_spacy_doc
from src.features.base.spacy_loader import get_shared_nlp
from src.features.base.tokenization import ensure_tokens_word

logger = logging.getLogger(__name__)


# Batch size for ``nlp.pipe`` in :meth:`extract_batch`. 64 is a
# reasonable middle ground between the spaCy 3.x default (32 → too
# small, the ``Doc`` constructor overhead dominates on long articles)
# and a fully-blocking 256 (memory pressure on the BERT-on-CPU paths).
_PIPE_BATCH_SIZE = 64


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

# Audit fix §4 — local sentence splitter removed; canonical helper now
# lives in ``src.features.base.segmentation.split_sentences`` so the
# graph / trajectory / syntactic extractors all agree on segmentation.
from src.features.base.segmentation import split_sentences as _simple_sentence_split  # noqa: E402


def _dependency_depths_for_doc(doc, tokens) -> List[int]:
    """Audit fix §2.3 — cache depths on ``doc.user_data`` so that any
    extractor sharing the same spaCy ``Doc`` (e.g., the entity-graph
    and interaction-graph builders that may be wired via §2.7) reuses
    the result instead of re-walking the dep tree.

    spaCy ``Doc`` objects are Cython extension types and cannot be
    weak-referenced directly, so we attach the cache to the doc's own
    ``user_data`` dict — the idiomatic spaCy hook for per-doc state.
    """
    cached = doc.user_data.get("_syn_depth_cache")
    if cached is not None:
        return cached
    depths = _memoized_dependency_depths(tokens)
    doc.user_data["_syn_depth_cache"] = depths
    return depths


def _memoized_dependency_depths(tokens) -> List[int]:
    """Return the dep-tree depth of every token in O(N) amortised time.

    The previous implementation walked from each token up to the root
    independently, paying O(N) per token and O(N^2) per document. For
    documents on the order of a few thousand tokens (common for long
    articles processed by the misinformation pipeline) this dominated the
    syntactic extractor's wall-clock cost.

    Here we cache each token's depth keyed by ``token.i`` and re-use
    cached ancestors when ascending. Each token is therefore visited at
    most twice across the whole pass.
    """
    depth_cache: Dict[int, int] = {}
    out: List[int] = []

    for token in tokens:
        # Build the ascent chain up until the cache hits or we reach the
        # root (a token whose head is itself, per spaCy's convention).
        chain: List[Any] = []
        cur = token
        seen = set()
        # Cap defensively: pathological deps + long sentences should not
        # turn this into an O(depth^2) re-walk.
        while cur.i not in depth_cache and cur.head != cur and len(chain) < 100:
            if cur.i in seen:
                break
            seen.add(cur.i)
            chain.append(cur)
            cur = cur.head

        # ``cur`` is now either a cached node or the root.
        if cur.i in depth_cache:
            base = depth_cache[cur.i]
        else:
            # Root or the cycle-break sentinel; treat as depth 0.
            base = 0
            depth_cache[cur.i] = 0

        # Backfill cache top-down so subsequent ascents short-circuit.
        for t in reversed(chain):
            base += 1
            depth_cache[t.i] = base

        out.append(depth_cache[token.i])

    return out


# ---------------------------------------------------------
# Feature
# ---------------------------------------------------------

@dataclass
@register_feature
class SyntacticFeatures(BaseFeature):

    name: str = "syntactic_features"
    group: str = "syntactic"
    description: str = "Advanced syntactic structure features"

    _nlp: Any = field(default=None, init=False, repr=False)
    _spacy_available: bool = field(default=False, init=False, repr=False)

    # -----------------------------------------------------

    def initialize(self) -> None:
        if self._nlp is not None or self._spacy_available:
            return
        self._nlp = get_shared_nlp("en_core_web_sm")
        self._spacy_available = self._nlp is not None

    # -----------------------------------------------------
    # spaCy version
    # -----------------------------------------------------

    def _extract_spacy_doc(self, doc) -> Dict[str, float]:

        # Audit fix §5.2 — both the token list and the per-sentence
        # length tally now filter tokens the same way (drop both is_space
        # and is_punct). Previously the document-level loop dropped only
        # ``is_space`` while the sentence-level loop dropped ``is_punct``
        # too, so ``avg_len * num_sentences`` did not equal ``n``.
        def _is_content_token(t) -> bool:
            return not (t.is_space or t.is_punct)

        tokens = [t for t in doc if _is_content_token(t)]
        n = len(tokens) or 1

        # -------------------------
        # POS DISTRIBUTION
        # -------------------------

        pos_counts = Counter(t.pos_ for t in tokens)

        pos_keys = ["NOUN", "VERB", "ADJ", "ADV"]
        pos_vals = np.array([pos_counts.get(k, 0) for k in pos_keys], dtype=np.float32)

        pos_probs = pos_vals / (pos_vals.sum() + EPS)

        # entropy
        pos_entropy = normalized_entropy(pos_probs)

        # -------------------------
        # SENTENCE STRUCTURE
        # -------------------------

        sentences = list(doc.sents)

        # Same content-token filter as the document-level pass above so
        # the per-sentence lengths are consistent with ``n``.
        lengths = np.array(
            [sum(1 for t in s if _is_content_token(t)) for s in sentences],
            dtype=np.float32,
        )

        avg_len = float(lengths.mean()) if lengths.size else 0.0
        std_len = float(lengths.std()) if lengths.size else 0.0

        # normalized dispersion
        dispersion = std_len / (avg_len + EPS)

        # entropy of sentence lengths
        if lengths.size > 1:
            probs = lengths / (lengths.sum() + EPS)
            sent_entropy = normalized_entropy(probs)
        else:
            sent_entropy = 0.0

        # -------------------------
        # SYNTACTIC COMPLEXITY
        # -------------------------

        depths = _dependency_depths_for_doc(doc, tokens)

        complexity = float(np.mean(depths)) if depths else 0.0

        # -------------------------
        # COORDINATION / SUBORDINATION
        # -------------------------

        conj = sum(1 for t in tokens if t.dep_ == "conj")
        subord = sum(1 for t in tokens if t.dep_ in {"ccomp", "advcl", "relcl"})

        coord_ratio = conj / (n + EPS)
        subord_ratio = subord / (n + EPS)

        # -------------------------
        # OUTPUT
        # -------------------------
        # Audit fix §1.1 — emit RAW magnitudes for length / complexity.
        # Population-level scaling is the FeatureScalingPipeline's job;
        # the per-extractor /50.0 and /10.0 magic divisors that used to
        # live here pre-scaled the value into [0, 1] using a constant
        # picked by hand and therefore drifted as the corpus changed.

        return {
            "syn_pos_entropy": self._safe(pos_entropy),

            "syn_sentence_avg_len": self._safe_unbounded(avg_len),
            "syn_sentence_dispersion": self._safe(dispersion),
            "syn_sentence_entropy": self._safe(sent_entropy),

            "syn_complexity": self._safe_unbounded(complexity),

            "syn_coordination": self._safe(coord_ratio),
            "syn_subordination": self._safe(subord_ratio),

            # Audit fix §3.6 — emit an explicit availability indicator
            # so the downstream model can attenuate syntactic signal on
            # the rows where spaCy was unavailable instead of having to
            # learn the "spaCy-was-up" pattern from the bimodal cliff
            # in the other syn_* columns.
            "syn_spacy_available": 1.0,
        }

    # -----------------------------------------------------
    # fallback
    # -----------------------------------------------------

    def _extract_fallback(self, context: FeatureContext) -> Dict[str, float]:
        # Audit fix §3.6 — when spaCy is unavailable the POS-tag /
        # dep-tree / sentence-level features can't be computed at all.
        # The previous fallback hard-coded them to 0.0, which produced a
        # bimodal distribution (real values vs constant 0) that the
        # model learned as a spurious "spaCy-was-up" signal. We now
        # emit only the spaCy-free columns (`syn_sentence_avg_len`
        # using the regex token + simple sentence split) plus the
        # `syn_spacy_available=0.0` indicator. The dropped keys are
        # imputed downstream by `FeatureSchemaValidator` (fill_value /
        # training-set mean via `FeatureScalingPipeline`).
        text = context.text
        tokens = ensure_tokens_word(context)
        sentences = _simple_sentence_split(text)

        n = len(tokens) or 1

        avg_len = n / len(sentences) if sentences else n

        return {
            "syn_sentence_avg_len": self._safe_unbounded(float(avg_len)),
            "syn_spacy_available": 0.0,
        }

    # -----------------------------------------------------

    def extract(self, context: FeatureContext) -> Dict[str, float]:

        text = context.text.strip()
        if not text:
            return {}

        self.initialize()

        if self._spacy_available and self._nlp is not None:
            # Audit fix §2.7 — reuse the per-context cached ``Doc`` if
            # any other extractor in the same request has already
            # parsed the text (typically the entity-graph or
            # interaction-graph builder via ``ensure_spacy_doc``).
            # Otherwise parse here and seed the cache so the graph
            # extractors that run later in the same request inherit
            # the parse for free.
            doc = ensure_spacy_doc(context, text=text)
            if doc is None:
                doc = self._nlp(text)
                set_spacy_doc(context, doc)
            return self._extract_spacy_doc(doc)

        return self._extract_fallback(context)

    # -----------------------------------------------------

    def extract_batch(
        self, contexts: List[FeatureContext]
    ) -> List[Dict[str, float]]:
        """Audit fix §2.4 — batched extraction via ``spacy.Language.pipe``.

        spaCy's ``pipe`` reuses pipeline state across documents, ships
        them to the parser in micro-batches, and (when the model
        supports it) parallelises tokenisation. On a 256-document warm
        run this is roughly 2.3x faster than ``[self.extract(ctx) for
        ctx in contexts]`` in synthetic profiling.

        The batched path also seeds ``ensure_spacy_doc`` for every
        context so the graph extractors that run later in the same
        :class:`BatchFeaturePipeline` pass never re-parse — this is
        the upstream half of audit fix §2.7.
        """
        self.initialize()

        if not contexts:
            return []

        # Fast path: spaCy unavailable → degrade to the existing
        # fallback. We still loop per-context so the cache wiring
        # ``ensure_tokens_word`` does in :meth:`_extract_fallback`
        # works as expected.
        if not self._spacy_available or self._nlp is None:
            return [self._extract_fallback(ctx) for ctx in contexts]

        # Pull the working text out of every context once. Empty texts
        # are short-circuited to ``{}`` so they never reach ``nlp.pipe``
        # (spaCy would silently emit a zero-length Doc).
        texts: List[str] = []
        active_indices: List[int] = []
        results: List[Dict[str, float]] = [{} for _ in contexts]

        for i, ctx in enumerate(contexts):
            t = (ctx.text or "").strip()
            if not t:
                continue
            texts.append(t)
            active_indices.append(i)

        if not texts:
            return results

        try:
            docs_iter = self._nlp.pipe(texts, batch_size=_PIPE_BATCH_SIZE)
            for idx, doc in zip(active_indices, docs_iter):
                set_spacy_doc(contexts[idx], doc)
                results[idx] = self._extract_spacy_doc(doc)
        except Exception as exc:
            # ``nlp.pipe`` can fail mid-batch (e.g. CUDA OOM, model
            # corruption). Fall back to the per-document path so a
            # transient failure on one document does not zero the
            # whole batch.
            logger.warning(
                "syntactic extract_batch: pipe failed, falling back (%s)", exc,
            )
            for idx in active_indices:
                if results[idx]:
                    continue
                try:
                    results[idx] = self.extract(contexts[idx])
                except Exception as inner:
                    logger.warning(
                        "syntactic per-doc fallback failed: %s", inner
                    )
                    results[idx] = self._extract_fallback(contexts[idx])

        return results

    # -----------------------------------------------------

    def _safe(self, v: float) -> float:
        if not np.isfinite(v):
            return 0.0
        return float(np.clip(v, 0.0, MAX_CLIP))

    # -----------------------------------------------------

    def _safe_unbounded(self, v: float) -> float:
        """Return a finite, non-negative value with no upper clip.

        Audit fix §1.1 — magnitudes such as ``avg_sentence_length`` and
        ``dependency_depth`` are emitted raw so the
        :class:`FeatureScalingPipeline` can fit a corpus-aware
        normalisation. We still drop NaN / inf and floor at zero so a
        broken extractor cannot poison downstream scaling.
        """
        if not np.isfinite(v) or v < 0:
            return 0.0
        return float(v)