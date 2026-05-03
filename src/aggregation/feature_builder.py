"""
AggregatorFeatureBuilder — structured feature vector for the NeuralAggregator.

Spec §3 (Aggregation Engine v2).

Canonical feature vector layout (spec §3.1)
--------------------------------------------
Group                   Dim   Indices
─────────────────────── ───── ────────────────────
bias_prob               2     [0:2]    softmax probs (not-biased, biased)
emotion_probs           11    [2:13]   full emotion distribution
ideology_probs          3     [13:16]  ideology class probs
propaganda_prob         2     [16:18]  not-propaganda / propaganda probs
narrative_roles         3     [18:21]  hero / villain / victim
narrative_frames        5     [21:26]  RE / HI / CO / MO / EC
analyzer_slots          8     [26:34]  fixed external-analyzer feature slots
entropy                 5     [34:39]  per-task normalised entropy
confidence              5     [39:44]  per-task max-prob confidence
cross_task              3     [44:47]  bias×emotion, propaganda×narrative,
                                      ideology×emotion
─────────────────────── ───── ────────────────────
TOTAL                   47

Missing / unavailable values are filled with 0.0.
Feature engineering rules (spec §3.2):
  * All values normalised to [0, 1] before insertion.
  * Per-task probabilities re-normalised if they do not sum to 1.
  * Skewed per-task scalar features are log-scaled: log(1 + x).
  * Entropy values are already normalised in [0, 1] by FeatureMapper.

Usage
-----
    from src.aggregation.feature_builder import AggregatorFeatureBuilder

    builder = AggregatorFeatureBuilder()
    x = builder.build(model_outputs, task_signals, section_profile)
    # x: np.ndarray of shape (47,) dtype float32

    # Batch:
    X = builder.build_batch([dict1, dict2, ...])
    # X: np.ndarray of shape (N, 47) dtype float32
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-9


# =========================================================
# CANONICAL FEATURE LAYOUT  (spec §3.1)
# =========================================================

# Task order for entropy / confidence slots (CANONICAL — must match training)
_SIGNAL_TASKS: Tuple[str, ...] = (
    "bias", "emotion", "ideology", "propaganda", "narrative"
)

# Per-task expected probability dimensionality
_TASK_PROB_DIM: Dict[str, int] = {
    "bias":            2,
    "emotion":         11,
    "ideology":        3,
    "propaganda":      2,
    "narrative":       3,
    "narrative_frame": 5,
}

# Named feature groups and their slice widths (in canonical order)
_FEATURE_GROUPS: List[Tuple[str, int]] = [
    ("bias_prob",       2),
    ("emotion_probs",   11),
    ("ideology_probs",  3),
    ("propaganda_prob", 2),
    ("narrative_roles", 3),
    ("narrative_frames", 5),
    ("analyzer_slots",  8),
    ("entropy",         5),   # one slot per _SIGNAL_TASKS entry
    ("confidence",      5),   # one slot per _SIGNAL_TASKS entry
    ("cross_task",      3),   # bias×emotion, propaganda×narrative, ideology×emotion
]

FEATURE_DIM: int = sum(w for _, w in _FEATURE_GROUPS)  # 47


def _build_feature_slices() -> Dict[str, slice]:
    slices: Dict[str, slice] = {}
    offset = 0
    for name, width in _FEATURE_GROUPS:
        slices[name] = slice(offset, offset + width)
        offset += width
    return slices


_SLICES: Dict[str, slice] = _build_feature_slices()


def _build_feature_names() -> List[str]:
    names: List[str] = []
    # bias_prob
    names += ["bias_prob_neg", "bias_prob_pos"]
    # emotion_probs (11 classes)
    EMOTION_LABELS = [
        "neutral", "approval", "disapproval", "admiration", "gratitude",
        "disappointment", "annoyance", "anger", "fear", "joy", "sadness",
    ]
    names += [f"emotion_{e}" for e in EMOTION_LABELS[:11]]
    # ideology_probs
    names += ["ideology_left", "ideology_center", "ideology_right"]
    # propaganda_prob
    names += ["propaganda_neg", "propaganda_pos"]
    # narrative_roles
    names += ["narrative_hero", "narrative_villain", "narrative_victim"]
    # narrative_frames
    FRAME_LABELS = ["responsibility", "human_interest", "conflict", "morality", "economics"]
    names += [f"frame_{f}" for f in FRAME_LABELS[:5]]
    # analyzer slots
    names += [f"analyzer_{i}" for i in range(8)]
    # entropy per task
    names += [f"entropy_{t}" for t in _SIGNAL_TASKS]
    # confidence per task
    names += [f"confidence_{t}" for t in _SIGNAL_TASKS]
    # cross-task
    names += ["cross_bias_emotion", "cross_propaganda_narrative", "cross_ideology_emotion"]
    assert len(names) == FEATURE_DIM, f"Feature name count mismatch: {len(names)} ≠ {FEATURE_DIM}"
    return names


_FEATURE_NAMES: List[str] = _build_feature_names()


# =========================================================
# HELPERS
# =========================================================

def _safe_probs(probs_raw: Any, expected_dim: int) -> np.ndarray:
    """Convert raw probability input to a normalised (expected_dim,) array."""
    if probs_raw is None:
        return np.zeros(expected_dim, dtype=np.float32)
    try:
        arr = np.asarray(probs_raw, dtype=np.float64).ravel()
    except Exception:
        return np.zeros(expected_dim, dtype=np.float32)

    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)

    # Resize: truncate or zero-pad to expected_dim
    if arr.size > expected_dim:
        arr = arr[:expected_dim]
    elif arr.size < expected_dim:
        arr = np.pad(arr, (0, expected_dim - arr.size))

    total = arr.sum()
    if total > EPS:
        arr = arr / total

    return arr.astype(np.float32)


def _safe_scalar(v: Any) -> float:
    if v is None:
        return 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return float(np.clip(f if math.isfinite(f) else 0.0, 0.0, 1.0))


def _log_scale(v: float) -> float:
    """Log-scale a skewed feature to [0, 1] using log(1 + x) / log(2)."""
    return float(np.clip(math.log1p(v) / math.log(2.0), 0.0, 1.0))


# =========================================================
# FEATURE BUILDER
# =========================================================

class AggregatorFeatureBuilder:
    """Build fixed-dim feature vectors for the NeuralAggregator.

    Thread-safety: instances are read-only after construction — safe for
    concurrent use across inference threads.

    Parameters
    ----------
    include_cross_task : bool
        Whether to include the three cross-task product features
        (spec §3.3). Set to ``False`` only for ablation studies; the
        default (``True``) matches the spec recommendation.
    log_scale_analyzers : bool
        Apply ``log(1 + x)`` to analyzer slot features before
        inserting into the vector (spec §3.2 — log-scale skewed
        distributions). Default ``True``.
    """

    def __init__(
        self,
        include_cross_task: bool = True,
        log_scale_analyzers: bool = True,
    ) -> None:
        self.include_cross_task = include_cross_task
        self.log_scale_analyzers = log_scale_analyzers

        logger.info(
            "AggregatorFeatureBuilder | dim=%d | cross_task=%s | log_scale=%s",
            FEATURE_DIM, include_cross_task, log_scale_analyzers,
        )

    # -----------------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------------

    @property
    def feature_dim(self) -> int:
        """Canonical feature vector dimensionality (47)."""
        return FEATURE_DIM

    def feature_names(self) -> List[str]:
        """Ordered list of feature names corresponding to each vector index."""
        return list(_FEATURE_NAMES)

    def build(
        self,
        model_outputs: Dict[str, Any],
        task_signals: Optional[Dict[str, Any]] = None,
        section_profile: Optional[Dict[str, Any]] = None,
        analyzer_features: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Build a (47,) float32 feature vector.

        Parameters
        ----------
        model_outputs : Dict[str, Any]
            Raw model output dict. For each task, expects a dict with
            ``"probabilities"`` and/or ``"logits"``.
        task_signals : Dict[str, TaskSignal], optional
            Pre-computed ``{task: TaskSignal}`` from
            ``FeatureMapper.extract_task_signals``. When provided,
            ``confidence`` and ``entropy`` slots are filled from these
            (avoids re-computing softmax).
        section_profile : Dict[str, Dict[str, float]], optional
            Per-section feature profile from ``FeatureMapper``. Used
            for analyzer section fallback.
        analyzer_features : Dict[str, float], optional
            Up to 8 external analyzer signals. Keys are arbitrary; they
            are inserted in sorted-key order into the 8 analyzer slots.

        Returns
        -------
        np.ndarray of shape (47,) dtype float32
        """
        x = np.zeros(FEATURE_DIM, dtype=np.float32)
        task_signals = task_signals or {}
        section_profile = section_profile or {}

        # ── bias (2D) ──────────────────────────────────────────────────────
        x[_SLICES["bias_prob"]] = self._get_probs(
            model_outputs, task_signals, "bias", expected_dim=2
        )

        # ── emotion (11D) ──────────────────────────────────────────────────
        x[_SLICES["emotion_probs"]] = self._get_probs(
            model_outputs, task_signals, "emotion", expected_dim=11
        )

        # ── ideology (3D) ──────────────────────────────────────────────────
        x[_SLICES["ideology_probs"]] = self._get_probs(
            model_outputs, task_signals, "ideology", expected_dim=3
        )

        # ── propaganda (2D) ───────────────────────────────────────────────
        x[_SLICES["propaganda_prob"]] = self._get_probs(
            model_outputs, task_signals, "propaganda", expected_dim=2
        )

        # ── narrative roles (3D) ───────────────────────────────────────────
        x[_SLICES["narrative_roles"]] = self._get_probs(
            model_outputs, task_signals, "narrative", expected_dim=3
        )

        # ── narrative frames (5D) ─────────────────────────────────────────
        x[_SLICES["narrative_frames"]] = self._get_probs(
            model_outputs, task_signals, "narrative_frame", expected_dim=5
        )

        # ── analyzer slots (8D) ───────────────────────────────────────────
        x[_SLICES["analyzer_slots"]] = self._build_analyzer_slots(
            analyzer_features, section_profile
        )

        # ── entropy (5D) ──────────────────────────────────────────────────
        x[_SLICES["entropy"]] = self._signal_vec(
            task_signals, "entropy"
        )

        # ── confidence (5D) ───────────────────────────────────────────────
        x[_SLICES["confidence"]] = self._signal_vec(
            task_signals, "confidence"
        )

        # ── cross-task (3D) ───────────────────────────────────────────────
        if self.include_cross_task:
            x[_SLICES["cross_task"]] = self._cross_task_features(x)

        return x

    def build_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Build a (N, 47) float32 matrix.

        Each element of ``items`` is a dict with optional keys:
        ``model_outputs``, ``task_signals``, ``section_profile``,
        ``analyzer_features``.

        Parameters
        ----------
        items : list of dict
            Length N; each dict is forwarded as **kwargs to :meth:`build`.

        Returns
        -------
        np.ndarray of shape (N, 47) dtype float32
        """
        rows = [
            self.build(
                item.get("model_outputs", {}),
                item.get("task_signals"),
                item.get("section_profile"),
                item.get("analyzer_features"),
            )
            for item in items
        ]
        return np.stack(rows, axis=0)

    # -----------------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------------

    def _get_probs(
        self,
        model_outputs: Dict[str, Any],
        task_signals: Dict[str, Any],
        task: str,
        expected_dim: int,
    ) -> np.ndarray:
        """Return normalised probability vector for ``task``."""

        # Prefer pre-computed TaskSignal probabilities (avoids re-softmax)
        sig = task_signals.get(task)
        if sig is not None:
            raw = getattr(sig, "probability", None)
            if raw is not None:
                return _safe_probs(raw, expected_dim)

        # Fall back to model_outputs[task]["probabilities"] / ["logits"]
        task_out = model_outputs.get(task)
        if isinstance(task_out, dict):
            probs = task_out.get("probabilities")
            if probs is not None:
                return _safe_probs(probs, expected_dim)
            logits = task_out.get("logits")
            if logits is not None:
                arr = np.asarray(logits, dtype=np.float64).ravel()
                arr = np.nan_to_num(arr, nan=0.0)
                if arr.size == 1:
                    p = 1.0 / (1.0 + np.exp(-np.clip(arr[0], -88, 88)))
                    probs_arr = np.array([1.0 - p, p], dtype=np.float32)
                else:
                    e = np.exp(np.clip(arr - arr.max(), -88, 0))
                    probs_arr = (e / (e.sum() + EPS)).astype(np.float32)
                return _safe_probs(probs_arr, expected_dim)

        return np.zeros(expected_dim, dtype=np.float32)

    def _signal_vec(
        self,
        task_signals: Dict[str, Any],
        attr: str,
    ) -> np.ndarray:
        """Build a 5D vector of ``attr`` (entropy or confidence) values."""
        out = np.zeros(5, dtype=np.float32)
        for i, task in enumerate(_SIGNAL_TASKS):
            sig = task_signals.get(task)
            if sig is not None:
                v = getattr(sig, attr, 0.0)
                out[i] = float(np.clip(v if math.isfinite(float(v)) else 0.0, 0.0, 1.0))
        return out

    def _build_analyzer_slots(
        self,
        analyzer_features: Optional[Dict[str, float]],
        section_profile: Dict[str, Any],
    ) -> np.ndarray:
        """Fill 8 analyzer slots from external features + profile fallback."""
        slots = np.zeros(8, dtype=np.float32)
        values: List[float] = []

        # External analyzer features (sorted for determinism)
        if analyzer_features:
            for k in sorted(analyzer_features.keys())[:8]:
                v = analyzer_features[k]
                fv = _safe_scalar(v)
                if self.log_scale_analyzers:
                    fv = _log_scale(fv)
                values.append(fv)

        # Profile section fallback: pull scalar features not yet covered
        if len(values) < 8:
            for section in ("discourse", "argument", "graph", "analysis"):
                sec = section_profile.get(section, {})
                if not isinstance(sec, dict):
                    continue
                for feat_val in sec.values():
                    if len(values) >= 8:
                        break
                    fv = _safe_scalar(feat_val)
                    if self.log_scale_analyzers:
                        fv = _log_scale(fv)
                    values.append(fv)
                if len(values) >= 8:
                    break

        slots[:len(values)] = values[:8]
        return slots

    def _cross_task_features(self, x: np.ndarray) -> np.ndarray:
        """Compute spec §3.3 cross-task product features.

        Returns a (3,) vector:
            bias×emotion_max, propaganda×narrative_max, ideology_entropy×emotion_max
        """
        # bias score (max of positive-bias prob)
        bias_score = float(x[_SLICES["bias_prob"]][-1])

        # emotion max (dominant emotion signal)
        em = x[_SLICES["emotion_probs"]]
        emotion_max = float(em.max())

        # propaganda score
        prop_score = float(x[_SLICES["propaganda_prob"]][-1])

        # narrative max
        nar = x[_SLICES["narrative_roles"]]
        narrative_max = float(nar.max())

        # ideology entropy (spread across ideologies)
        ideo = x[_SLICES["ideology_probs"]]
        ideology_spread = float(
            -np.sum(np.where(ideo > EPS, ideo * np.log(ideo + EPS), 0.0))
        )
        ideology_spread = float(np.clip(ideology_spread / math.log(3.0 + EPS), 0.0, 1.0))

        return np.array(
            [
                np.clip(bias_score * emotion_max, 0.0, 1.0),
                np.clip(prop_score * narrative_max, 0.0, 1.0),
                np.clip(ideology_spread * emotion_max, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
