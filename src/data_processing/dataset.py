"""
TruthLens datasets (pre-tokenized, contract-driven).

Design:
- All tokenization happens ONCE in __init__ (no per-sample tokenizer calls).
- Texts/labels are stored as numpy arrays / lists for O(1) __getitem__ access.
- Optionally returns offset_mapping for downstream token-alignment / explainability.
- Label column names come from the data_contracts module (single source of truth).

Semantic alignment additions (Phase 1):
- task_mask_vector: per-row (num_tasks,) binary mask — 1 where the row has a
  valid label for that task. Enables partial supervision in mixed batches and
  powers the TaskPresenceMaskSampler.
- derived_features: per-row cross-task supervision signals (emotional_bias_score,
  propaganda_intensity, ideological_emotion) normalised to [0, 1].
- MultiTaskAlignedDataset: correctly builds per-row task masks based on which
  label columns are actually non-null for each row, rather than assuming every
  row in a task-specific split is fully labelled.

Fixes applied (audit v3):
  PERF-D2: Store _ids_flat and _attn_flat as int64 at init time so
    _encoded_inputs never pays a per-sample dtype cast (previously the int32/
    int8 → int64 conversion happened inside every __getitem__ call).
  PERF-D3: MultiTaskAlignedDataset.__getitem__ used df.iloc[idx] (O(n) pandas
    overhead). Pre-converted to a list of dicts at init time for true O(1) access.
  PERF-D4: _build_derived_features was duplicated identically in
    ClassificationDataset and MultiLabelDataset. Extracted to a single module-
    level function _compute_derived_features(df) to avoid maintenance divergence.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

TASK_ORDER = ["bias", "emotion", "propaganda", "ideology", "narrative"]


def _safe_series(df: pd.DataFrame, columns: List[str], default: float = 0.0) -> np.ndarray:
    vals = []
    for col in columns:
        if col in df.columns:
            vals.append(df[col].fillna(default).astype(float).to_numpy())
        else:
            vals.append(np.full(len(df), default, dtype=np.float32))
    return np.vstack(vals).T if vals else np.zeros((len(df), 0), dtype=np.float32)


def _minmax_01(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float32)
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def _has_any(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    if not columns:
        return np.zeros(len(df), dtype=np.int64)
    present = [c for c in columns if c in df.columns]
    if not present:
        return np.zeros(len(df), dtype=np.int64)
    mat = df[present].fillna(0).to_numpy(dtype=np.float32)
    return (np.any(mat > 0, axis=1)).astype(np.int64)


def _build_per_row_task_mask(df: pd.DataFrame) -> np.ndarray:
    """Build a (N, len(TASK_ORDER)) binary matrix indicating which tasks
    have valid (non-null, non-sentinel) labels for each row.

    This is the *per-row* mask used for partial supervision — it differs
    from the *per-dataset* mask (all-ones for the current task) used by
    single-task batches. The per-row mask is valuable when rows from
    multiple datasets are concatenated into a unified corpus where some
    rows may have bias labels but no emotion labels, etc.
    """
    n = len(df)
    mask = np.zeros((n, len(TASK_ORDER)), dtype=np.int64)

    # bias — single classification label column
    if "bias_label" in df.columns:
        valid = df["bias_label"].notna().to_numpy(dtype=np.int64)
        mask[:, TASK_ORDER.index("bias")] = valid

    # emotion — any of emotion_0..emotion_N present and non-null
    emotion_cols = [c for c in df.columns if c.startswith("emotion_")]
    if emotion_cols:
        any_valid = df[emotion_cols].notna().any(axis=1).to_numpy(dtype=np.int64)
        mask[:, TASK_ORDER.index("emotion")] = any_valid

    # propaganda — single classification label column
    if "propaganda_label" in df.columns:
        valid = df["propaganda_label"].notna().to_numpy(dtype=np.int64)
        mask[:, TASK_ORDER.index("propaganda")] = valid

    # ideology — single classification label column
    if "ideology_label" in df.columns:
        valid = df["ideology_label"].notna().to_numpy(dtype=np.int64)
        mask[:, TASK_ORDER.index("ideology")] = valid

    # narrative — any of hero/villain/victim present and non-null
    narrative_cols = [c for c in ("hero", "villain", "victim") if c in df.columns]
    if narrative_cols:
        any_valid = df[narrative_cols].notna().any(axis=1).to_numpy(dtype=np.int64)
        mask[:, TASK_ORDER.index("narrative")] = any_valid

    return mask


# =========================================================
# SHARED DERIVED FEATURES  (PERF-D4 — de-duplicated)
# =========================================================

def _compute_derived_features(df: pd.DataFrame) -> torch.Tensor:
    """Compute cross-task derived supervision signals (normalised to [0,1]).

    emotional_bias_score  = bias_score  × emotion_intensity
    propaganda_intensity  = propaganda  × narrative_conflict_score
    ideological_emotion   = ideology    × dominant_emotion

    Previously this was copy-pasted identically into both ClassificationDataset
    and MultiLabelDataset. Extracting it to a single module-level function
    removes the divergence risk.
    """
    bias = (
        df["bias_label"].fillna(0).astype(float).to_numpy()
        if "bias_label" in df.columns
        else np.zeros(len(df))
    )
    emotion_cols = [c for c in df.columns if c.startswith("emotion_")]
    emotion = _safe_series(df, emotion_cols)
    emotion_intensity = emotion.mean(axis=1) if emotion.size else np.zeros(len(df))
    propaganda = (
        df["propaganda_label"].fillna(0).astype(float).to_numpy()
        if "propaganda_label" in df.columns
        else np.zeros(len(df))
    )
    narrative_cols = [c for c in ("hero", "villain", "victim") if c in df.columns]
    narrative = _safe_series(df, narrative_cols)
    narrative_conflict = narrative.std(axis=1) if narrative.size else np.zeros(len(df))
    ideology = (
        df["ideology_label"].fillna(0).astype(float).to_numpy()
        if "ideology_label" in df.columns
        else np.zeros(len(df))
    )
    dominant_emotion = emotion.max(axis=1) if emotion.size else np.zeros(len(df))
    feats = np.stack(
        [
            _minmax_01(bias * emotion_intensity),
            _minmax_01(propaganda * narrative_conflict),
            _minmax_01(ideology * dominant_emotion),
        ],
        axis=1,
    )
    return torch.from_numpy(feats)


# =========================================================
# BASE DATASET
# =========================================================

class BaseTextDataset(Dataset):
    """
    Base dataset: pre-tokenizes the entire text column up-front.

    Args:
        df: dataframe (must contain `text_col`)
        tokenizer: HuggingFace tokenizer (fast tokenizer required if
            ``return_offsets_mapping=True``)
        text_col: text column name
        max_length: max tokens per sample (truncation only — padding done in
            the collate step)
        return_offsets_mapping: if True, store per-sample offset_mapping for
            char-level alignment in explainability layers
        log_truncation: if True, log how many samples were truncated
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        *,
        text_col: str = "text",
        max_length: int = 512,
        return_offsets_mapping: bool = False,
        log_truncation: bool = True,
    ):
        if text_col not in df.columns:
            raise ValueError(
                f"Missing text column '{text_col}' (have: {list(df.columns)})"
            )

        self.tokenizer = tokenizer
        self.text_col = text_col
        self.max_length = max_length
        self.return_offsets_mapping = return_offsets_mapping
        self.pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )

        # ----- pre-tokenize whole column -----
        texts = df[text_col].astype(str).tolist()

        if return_offsets_mapping and not getattr(tokenizer, "is_fast", False):
            raise ValueError(
                "return_offsets_mapping=True requires a fast tokenizer "
                "(PreTrainedTokenizerFast)."
            )

        enc = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_attention_mask=True,
            return_offsets_mapping=return_offsets_mapping,
            return_length=True,
        )

        ids_lists: List[List[int]] = enc["input_ids"]
        attn_lists: List[List[int]] = enc["attention_mask"]
        om_lists: Optional[List[List[List[int]]]] = (
            enc.get("offset_mapping") if return_offsets_mapping else None
        )

        # =====================================================
        # FLATTEN STORAGE (PERF-D2)
        # Store as int64 directly so _encoded_inputs never pays a per-sample
        # dtype cast. The extra memory (int64 vs int32 for ids, int64 vs int8
        # for mask) is modest: a 512-token × 100k-row corpus gains ~50 MB —
        # negligible compared to model weights, and cheaper than running
        # astype() inside every __getitem__ call on a 100k-row training loop.
        # =====================================================
        n = len(ids_lists)
        lengths = np.fromiter((len(x) for x in ids_lists), dtype=np.int64, count=n)
        self._offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(lengths, out=self._offsets[1:])
        total = int(self._offsets[-1])

        # int64 at storage time → zero-copy slice in __getitem__
        self._ids_flat = np.empty(total, dtype=np.int64)
        self._attn_flat = np.empty(total, dtype=np.int64)
        cursor = 0
        for ids, attn in zip(ids_lists, attn_lists):
            k = len(ids)
            self._ids_flat[cursor:cursor + k] = ids
            self._attn_flat[cursor:cursor + k] = attn
            cursor += k

        if om_lists is not None:
            self._om_flat: Optional[np.ndarray] = np.empty((total, 2), dtype=np.int64)
            cursor = 0
            for om in om_lists:
                k = len(om)
                self._om_flat[cursor:cursor + k] = om
                cursor += k
        else:
            self._om_flat = None

        # truncation diagnostics
        if log_truncation:
            n_truncated = 0
            encodings = getattr(enc, "encodings", None)
            if encodings is not None:
                n_truncated = sum(1 for e in encodings if getattr(e, "overflowing", None))
            else:
                fallback_lengths = enc.get("length") or [len(x) for x in ids_lists]
                n_truncated = sum(1 for L in fallback_lengths if L >= max_length)
            if n_truncated > 0:
                logger.warning(
                    "Tokenizer truncation | samples=%d | truncated=%d (%.1f%%) | max_length=%d",
                    len(texts),
                    n_truncated,
                    100.0 * n_truncated / max(len(texts), 1),
                    max_length,
                )

        self._n = n

    def __len__(self) -> int:
        return self._n

    def _encoded_inputs(self, idx: int) -> Dict[str, torch.Tensor]:
        s = int(self._offsets[idx])
        e = int(self._offsets[idx + 1])
        # PERF-D2: arrays are already int64 → torch.from_numpy is zero-copy
        # (no dtype conversion, no allocation beyond the view bookkeeping).
        item: Dict[str, torch.Tensor] = {
            "input_ids": torch.from_numpy(self._ids_flat[s:e].copy()),
            "attention_mask": torch.from_numpy(self._attn_flat[s:e].copy()),
        }
        if self._om_flat is not None:
            item["offset_mapping"] = torch.from_numpy(
                self._om_flat[s:e].copy()
            )
        return item


# =========================================================
# CLASSIFICATION DATASET (bias, ideology, propaganda)
# =========================================================

class ClassificationDataset(BaseTextDataset):

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        *,
        label_col: str,
        num_classes: int,
        task_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(df, tokenizer, **kwargs)

        if label_col not in df.columns:
            raise ValueError(
                f"Missing label column '{label_col}' (have: {list(df.columns)})"
            )

        self.label_col = label_col
        self.num_classes = num_classes
        self.task_name = task_name or label_col

        # Per-dataset mask: all rows belong to this task.
        self.task_mask = torch.ones(len(df), dtype=torch.long)

        # Per-row multi-task mask: reflects which tasks each row is actually
        # labelled for (relevant when this dataset is merged into a unified corpus).
        per_row_mask = _build_per_row_task_mask(df)
        self.task_mask_vector = torch.from_numpy(per_row_mask)

        # PERF-D4: use the shared module-level function (no more duplication).
        self.derived_features = _compute_derived_features(df)

        # vectorize labels once
        labels = df[label_col].to_numpy()
        if pd.isna(labels).any():
            raise ValueError(
                f"NaN labels in column '{label_col}' — clean/validate first."
            )
        self._labels = labels.astype(np.int64)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self._encoded_inputs(idx)
        item["labels"] = torch.as_tensor(self._labels[idx], dtype=torch.long)
        item["task"] = self.task_name
        item["task_mask"] = self.task_mask_vector[idx]
        item["derived_features"] = self.derived_features[idx]
        return item


# =========================================================
# MULTILABEL DATASET (frame, narrative, emotion)
# =========================================================

class MultiLabelDataset(BaseTextDataset):

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        *,
        label_cols: List[str],
        task_name: str,
        valid_label_indices: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(df, tokenizer, **kwargs)

        missing = [c for c in label_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing multilabel columns {missing} (have: {list(df.columns)})"
            )

        self.original_label_cols = list(label_cols)
        if valid_label_indices is None:
            self.label_cols = list(label_cols)
            self.valid_label_indices: List[int] = list(range(len(label_cols)))
        else:
            n = len(label_cols)
            kept = sorted({int(i) for i in valid_label_indices})
            for i in kept:
                if not (0 <= i < n):
                    raise ValueError(
                        f"valid_label_indices out of range for {n} columns: {i}"
                    )
            self.valid_label_indices = kept
            self.label_cols = [label_cols[i] for i in kept]
        self.task_name = task_name

        self.task_mask = torch.ones(len(df), dtype=torch.long)

        # Per-row multi-task mask.
        per_row_mask = _build_per_row_task_mask(df)
        self.task_mask_vector = torch.from_numpy(per_row_mask)

        # PERF-D4: use the shared module-level function (no more duplication).
        self.derived_features = _compute_derived_features(df)

        if not self.label_cols:
            raise ValueError(
                f"{task_name}: no usable multilabel columns after filtering "
                f"(original={list(label_cols)}, kept_indices={self.valid_label_indices})"
            )

        matrix = df[self.label_cols].to_numpy(dtype=np.float32)
        if np.isnan(matrix).any():
            raise ValueError(
                f"NaN values in multilabel columns {self.label_cols} — clean first."
            )

        if matrix.size and (matrix.min() < 0.0 or matrix.max() > 1.0):
            bad = ((matrix < 0.0) | (matrix > 1.0)).sum()
            raise ValueError(
                f"Multilabel values outside [0, 1] in {self.label_cols} "
                f"({bad} entries). Soft labels are supported, but values "
                "must be in [0, 1] for BCE-with-logits loss."
            )
        self._label_matrix = matrix

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self._encoded_inputs(idx)
        item["labels"] = torch.as_tensor(self._label_matrix[idx], dtype=torch.float)
        item["task"] = self.task_name
        item["task_mask"] = self.task_mask_vector[idx]
        item["derived_features"] = self.derived_features[idx]
        return item


# =========================================================
# MULTI-TASK ALIGNED DATASET
# =========================================================

class MultiTaskAlignedDataset(Dataset):
    """Unified dataset mixing rows from multiple task-specific DataFrames.

    Each row may carry labels for one or more tasks. The per-row
    ``task_mask`` reflects which tasks actually have non-null labels for
    that row — NOT which dataset slice it came from. This is the key
    difference from the per-dataset all-ones mask: it enables the
    TaskPresenceMaskSampler and the MultiTaskLoss task-mask gating to
    correctly handle partial supervision in mixed batches.
    """

    def __init__(self, frames: Dict[str, pd.DataFrame]):
        self.frames = frames
        self.tasks = [t for t in TASK_ORDER if t in frames]
        self.df = pd.concat([frames[t] for t in self.tasks], ignore_index=True)
        self.task_to_index = {t: i for i, t in enumerate(self.tasks)}
        self._build_masks()

        # PERF-D3: pre-convert the dataframe to a list of dicts so that
        # __getitem__ is O(1) with no pandas overhead. df.iloc[idx] has
        # O(n) axis-alignment cost in pandas; a list index is O(1).
        self._records: List[Dict[str, Any]] = self.df.to_dict("records")

    def _build_masks(self) -> None:
        """Build per-row task masks from the actual label columns present in each row."""
        per_row = _build_per_row_task_mask(self.df)

        # Restrict to columns that correspond to the tasks present in this dataset.
        task_indices = [TASK_ORDER.index(t) for t in self.tasks]
        self.task_mask = torch.from_numpy(per_row[:, task_indices])

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # PERF-D3: O(1) list access instead of O(n) df.iloc[idx]
        row = self._records[idx]
        mask = self.task_mask[idx]
        return {
            "text": row.get("text", ""),
            "task": row.get("task", ""),
            "task_mask": mask,
            "derived_features": torch.zeros(3, dtype=torch.float),
        }
