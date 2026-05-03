"""
TruthLens task-aware data augmentation.

Notes:
- No module-level ``random.seed`` (it polluted the global RNG).
- No module-level ``nltk.download`` (silent network call at import).
  Resources are downloaded lazily on first call.
- Uses a per-call ``random.Random`` instance for reproducibility without
  global side-effects.
"""

from __future__ import annotations

import hashlib
import logging
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd

from src.data_processing.data_contracts import CONTRACTS, DataContract, get_contract

logger = logging.getLogger(__name__)

# A label signature is either an int (single-label classification) or a
# tuple of 0/1 ints (multi-label). This is what label-coupled ops gate on.
LabelSig = Optional[Union[int, Tuple[int, ...]]]

# =========================================================
# CONFIG
# =========================================================

@dataclass
class AugmentationConfig:
    multiplier: float = 1.5
    enable_heavy_ops: bool = False  # MLM + embeddings
    similarity_threshold: float = 0.75
    random_seed: int = 42


# =========================================================
# LAZY RESOURCES
# =========================================================

_STOPWORDS: Optional[set] = None
_MLM = None
_EMBEDDER = None
_NLTK_READY = False


def _ensure_nltk():
    global _STOPWORDS, _NLTK_READY
    if _NLTK_READY:
        return
    try:
        import nltk
        from nltk.corpus import stopwords

        for pkg in ("wordnet", "stopwords", "omw-1.4"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        _STOPWORDS = set(stopwords.words("english"))
    except Exception:
        _STOPWORDS = set()
    _NLTK_READY = True


def _get_synonyms(word: str) -> List[str]:
    _ensure_nltk()
    try:
        from nltk.corpus import wordnet
    except Exception:
        return []
    syns = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            w = lemma.name().replace("_", " ")
            if w != word:
                syns.add(w)
    return list(syns)


def _get_mlm():
    global _MLM
    if _MLM is None:
        try:
            from transformers import pipeline
            _MLM = pipeline("fill-mask", model="roberta-base", top_k=3)
        except Exception:
            _MLM = False  # mark as failed
    return _MLM if _MLM is not False else None


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _EMBEDDER = False
    return _EMBEDDER if _EMBEDDER is not False else None


# =========================================================
# BASIC OPS  (each receives a Random instance)
#
# Every op accepts an optional ``label=`` kwarg so the call site is
# uniform. Ops that mutate the *meaning* of the row (propaganda /
# bias / emotion markers) MUST gate on that label so we never inject
# a propaganda marker into a propaganda_label=0 row — that would be
# label-corrupted training data and actively teach the model to
# ignore the marker. (CRIT-D6)
# =========================================================

def _is_positive(label: LabelSig) -> bool:
    """True if at least one label position is positive (multilabel) or
    the single label equals 1 (binary classification)."""
    if label is None:
        # No label info available → fall through to original (loud, label-
        # safe) behaviour: refuse to inject the marker.
        return False
    if isinstance(label, (tuple, list)):
        return any(int(x) == 1 for x in label)
    try:
        return int(label) == 1
    except (TypeError, ValueError):
        return False


def synonym_replacement(text: str, rng: random.Random, *, label: LabelSig = None) -> str:
    _ensure_nltk()
    stop = _STOPWORDS or set()
    words = text.split()
    indices = list(range(len(words)))
    rng.shuffle(indices)
    for i in indices:
        w = words[i]
        if w.lower() not in stop and len(w) > 3:
            syns = _get_synonyms(w)
            if syns:
                words[i] = rng.choice(syns)
                break
    return " ".join(words)


def random_deletion(text: str, rng: random.Random, p: float = 0.1, *, label: LabelSig = None) -> str:
    words = text.split()
    if len(words) < 5:
        return text
    return " ".join(w for w in words if rng.random() > p)


def random_swap(text: str, rng: random.Random, *, label: LabelSig = None) -> str:
    words = text.split()
    if len(words) < 3:
        return text
    i, j = rng.sample(range(len(words)), 2)
    words[i], words[j] = words[j], words[i]
    return " ".join(words)


def ideology_frame_shift(text: str, rng: random.Random, *, label: LabelSig = None) -> str:
    return f"In a broader ideological context, {text}"


def propaganda_injection(text: str, rng: random.Random, *, label: LabelSig = None) -> str:
    # Only augment positives (CRIT-D6) — otherwise we teach the model that
    # "Clearly, …" is irrelevant to the propaganda label.
    if not _is_positive(label):
        return text
    return f"Clearly, {text}"


def narrative_reframe(text: str, rng: random.Random, *, label: LabelSig = None) -> str:
    return f"From another perspective, {text}"


def emotion_amplify(text: str, rng: random.Random, *, label: LabelSig = None) -> str:
    # Only augment rows that already have at least one emotion positive
    # (CRIT-D6). For all-zero rows the amplifier would be label-corrupting.
    if not _is_positive(label):
        return text
    return f"{text} This is extremely emotional."


def bias_injection(text: str, rng: random.Random, *, label: LabelSig = None) -> str:
    # Only augment positives (CRIT-D6).
    if not _is_positive(label):
        return text
    return f"{text} Obviously biased."


# =========================================================
# HEAVY OPS (OPTIONAL)
# =========================================================

def contextual_replacement(text: str, rng: random.Random, *, label: LabelSig = None) -> str:
    mlm = _get_mlm()
    if mlm is None:
        return text
    words = text.split()
    if len(words) < 6:
        return text
    idx = rng.randint(0, len(words) - 1)
    words[idx] = "<mask>"
    try:
        preds = mlm(" ".join(words))
        words[idx] = preds[0]["token_str"]
    except Exception:
        return text
    return " ".join(words)


def semantic_valid(original: str, augmented: str, threshold: float) -> bool:
    embedder = _get_embedder()
    if embedder is None:
        return True  # cannot check; accept
    from sklearn.metrics.pairwise import cosine_similarity
    emb = embedder.encode([original, augmented])
    score = cosine_similarity([emb[0]], [emb[1]])[0][0]
    return score >= threshold


# =========================================================
# TASK ROUTING
# =========================================================

TASK_OPS: Dict[str, List[Callable]] = {
    "bias": [synonym_replacement, random_deletion],
    "ideology": [ideology_frame_shift],
    "propaganda": [propaganda_injection],
    "narrative_frame": [random_swap],
    "narrative": [narrative_reframe],
    "emotion": [emotion_amplify],
}


def select_operation(task: str, config: AugmentationConfig, rng: random.Random):
    ops = list(TASK_OPS.get(task, []))
    if config.enable_heavy_ops:
        ops.append(contextual_replacement)
    if not ops:
        raise ValueError(f"No augmentation ops for task: {task}")
    return rng.choice(ops)


# =========================================================
# CORE
# =========================================================

def augment_text(
    text: str,
    *,
    task: str,
    config: AugmentationConfig,
    rng: random.Random,
    label: LabelSig = None,
) -> str:
    text = str(text).strip()
    if not text:
        return text
    op = select_operation(task, config, rng)
    augmented = op(text, rng, label=label)
    if config.enable_heavy_ops:
        if not semantic_valid(text, augmented, config.similarity_threshold):
            return text
    return augmented


# =========================================================
# STRATIFIED SAMPLING + LEAK-AWARE AUGMENTATION (CRIT-D7 + CRIT-D5)
# =========================================================

def _label_signature(row: Dict[str, Any], contract: Optional[DataContract]) -> LabelSig:
    """Build the per-row label key that label-coupled ops gate on."""
    if contract is None:
        return None
    cols = contract.label_columns
    if contract.task_type == "classification":
        v = row.get(cols[0])
        try:
            return int(v) if v is not None and not (isinstance(v, float) and pd.isna(v)) else None
        except (TypeError, ValueError):
            return None
    # multilabel
    sig: List[int] = []
    for c in cols:
        v = row.get(c)
        try:
            sig.append(int(v))
        except (TypeError, ValueError):
            sig.append(0)
    return tuple(sig)


def _stratified_weights(
    records: Sequence[Dict[str, Any]],
    contract: Optional[DataContract],
) -> Optional[List[float]]:
    """Inverse-frequency weights so rare classes are oversampled (CRIT-D7).

    Without this, ``rng.choice`` is uniform over the input — a 95/5
    dataset stays 95/5 after augmentation and ``balancing.method:
    oversample`` is a no-op.
    """
    if contract is None or not records:
        return None
    sigs = [_label_signature(r, contract) for r in records]
    counts: Counter = Counter(sigs)
    # Inverse frequency: rarer signature ⇒ larger weight.
    return [1.0 / counts[s] for s in sigs]


def _leak_key(text: Any) -> str:
    """Mirror leakage_checker._normalize / _hash_text so the per-row
    pre-filter agrees with the post-augmentation leakage check (CRIT-D5)."""
    norm = "" if text is None else str(text).strip().lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest() if norm else ""


def _build_held_out_hashes(
    held_out_dfs: Optional[Iterable[pd.DataFrame]],
    text_column: str,
) -> Set[str]:
    if not held_out_dfs:
        return set()
    out: Set[str] = set()
    for d in held_out_dfs:
        if d is None or text_column not in d.columns:
            continue
        for t in d[text_column].tolist():
            k = _leak_key(t)
            if k:
                out.add(k)
    return out


def augment_dataset(
    df: pd.DataFrame,
    *,
    task: str,
    text_column: str = "text",
    config: Optional[AugmentationConfig] = None,
    held_out_dfs: Optional[Iterable[pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Augment a training dataframe.

    - Stratified by inverse class frequency so balancing actually balances. (CRIT-D7)
    - Label-coupled ops only fire on positive rows. (CRIT-D6)
    - If ``held_out_dfs`` (val + test) are provided, candidates whose
      cleaned text collides with any held-out row are rejected and
      resampled. Catches the leakage hole where augmentation could
      mutate a train row into a near-duplicate of a val/test row. (CRIT-D5)
    """
    config = config or AugmentationConfig()
    if config.multiplier <= 1:
        return df.copy()

    contract = CONTRACTS.get(task)  # None ⇒ unknown task, skip label-aware paths
    rng = random.Random(config.random_seed)
    records = df.to_dict("records")
    if not records:
        return df.copy()

    extra = int(len(records) * (config.multiplier - 1))
    weights = _stratified_weights(records, contract)
    held_out_hashes = _build_held_out_hashes(held_out_dfs, text_column)

    indices = list(range(len(records)))
    augmented: List[Dict] = []
    rejected = 0
    max_attempts_per_slot = 5

    for _ in range(extra):
        for _attempt in range(max_attempts_per_slot):
            idx = (
                rng.choices(indices, weights=weights, k=1)[0]
                if weights
                else rng.choice(indices)
            )
            row = records[idx].copy()
            label = _label_signature(row, contract)
            new_text = augment_text(
                row[text_column], task=task, config=config, rng=rng, label=label,
            )
            if held_out_hashes and _leak_key(new_text) in held_out_hashes:
                rejected += 1
                continue
            row[text_column] = new_text
            augmented.append(row)
            break
        # If every attempt collided with held-out, drop this slot rather than
        # writing a known-leaky row — better to under-augment than to poison
        # the training set.

    result = pd.concat([df, pd.DataFrame(augmented)], ignore_index=True)

    if rejected:
        logger.info(
            "Augmentation | task=%s | held-out collisions rejected=%d (pre-filter saved leakage)",
            task, rejected,
        )
    logger.info(
        "Augmented | task=%s | original=%d | added=%d | total=%d",
        task, len(df), len(augmented), len(result),
    )
    return result
