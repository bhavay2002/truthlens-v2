from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Mapping

import numpy as np

logger = logging.getLogger(__name__)

EPS = 1e-12

_LOGIT_CLIP = 88.0


# =========================================================
# REC-AG-1: cached per-task signal so confidence and entropy are
# computed exactly once (was being recomputed up to three times: in
# `map_from_model_outputs`, `extract_confidence`, and the pipeline's
# `_compute_entropy`).
# =========================================================

@dataclass
class TaskSignal:
    probability: np.ndarray
    confidence: float
    entropy: float
    max_class: int
    is_multilabel: bool = False
    had_nan: bool = False


# =========================================================
# DEFAULT FEATURE MAP
#
# Each section maps a display-name to the raw key produced by
# `map_from_model_outputs` (Branch A). Confidence and entropy are NOT
# registered here — they flow through dedicated channels (see audit
# tag NORM-AG-4) so they never get arithmetic-mean-aggregated together
# with semantic feature scores. Confidence is exposed via
# `extract_confidence`; entropy is computed by the pipeline.
# =========================================================

DEFAULT_FEATURE_MAP: Dict[str, Dict[str, str]] = {
    "bias": {
        "probability": "bias_probability",
    },
    "emotion": {
        "probability": "emotion_probability",
        "intensity":   "emotion_intensity",
    },
    "narrative": {
        "probability": "narrative_probability",
        "score":       "narrative_score",
    },
    "ideology": {
        "probability": "ideology_probability",
        "score":       "ideology_score",
    },
    "graph": {
        "consistency": "graph_consistency",
    },
    # CRIT-AG-3: discourse, argument and analysis are referenced by the
    # score calculator's composite formulas — they must be registered
    # here so the corresponding `{task}_probability` keys flow through
    # `map_features` instead of being silently dropped.
    "discourse": {
        "probability": "discourse_probability",
    },
    "argument": {
        "probability": "argument_probability",
    },
    "analysis": {
        "probability": "analysis_probability",
    },
}


# =========================================================
# UTILS
# =========================================================

def _safe_numeric(value: Any, strict: bool) -> Optional[float]:

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        if strict:
            raise TypeError(f"Invalid numeric: {value}")
        return None

    if not np.isfinite(value):
        if strict:
            raise ValueError(f"Non-finite: {value}")
        return None

    return float(value)


def _compute_entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, EPS, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def _normalize_probs(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)

    # NORM-AG-2: do not re-normalise a single value — division by its
    # own sum collapses any 1-element vector to 1.0 and would falsely
    # advertise "100 % confidence" for a single-class output.
    if arr.size < 2:
        return arr

    total = arr.sum()
    if not np.isfinite(total) or total <= EPS:
        return arr

    return arr / total


def _logits_to_probs(logits: np.ndarray) -> np.ndarray:
    """Convert raw logits to a probability vector (uncalibrated fallback)."""
    arr = np.asarray(logits, dtype=np.float64).ravel()
    clipped = np.clip(arr, -_LOGIT_CLIP, _LOGIT_CLIP)
    if clipped.size == 1:
        p = 1.0 / (1.0 + np.exp(-clipped[0]))
        return np.array([1.0 - p, p], dtype=np.float64)
    e = np.exp(clipped - np.max(clipped))
    return np.clip(e / (np.sum(e) + EPS), 0.0, 1.0)


# =========================================================
# FEATURE MAPPER
# =========================================================

class FeatureMapper:

    def __init__(
        self,
        feature_map: Optional[Dict[str, Dict[str, str]]] = None,
        *,
        strict: bool = False,
        normalize: bool = False,
        calibrator: Optional[Any] = None,
    ):
        # NORM-AG-1: per-section max-norm is OFF by default — values
        # produced by Branch A are already in [0, 1] (clipped on emit)
        # and the calculator does not need a second magnitude rescale.
        self.feature_map = feature_map or DEFAULT_FEATURE_MAP
        self.strict = strict
        self.normalize = normalize
        # CRIT-AG-6: calibration belongs at the logit boundary, not at
        # the feature boundary. The mapper accepts an optional
        # calibrator and applies it when consuming logits.
        self.calibrator = calibrator

        logger.info(
            "[FeatureMapper] init | strict=%s normalize=%s calibrated=%s",
            strict,
            normalize,
            bool(calibrator is not None and getattr(calibrator, "fitted", False)),
        )

    # =====================================================
    # MAIN
    # =====================================================

    def map_features(self, raw_outputs: Dict[str, Any]) -> Dict[str, Dict[str, float]]:

        if not isinstance(raw_outputs, dict):
            raise ValueError("raw_outputs must be dict")

        profile: Dict[str, Dict[str, float]] = {}

        for section, mapping in self.feature_map.items():

            section_data: Dict[str, float] = {}

            for feature_name, raw_key in mapping.items():

                if raw_key not in raw_outputs:
                    continue

                val = _safe_numeric(raw_outputs[raw_key], self.strict)

                if val is None:
                    continue

                val = float(np.clip(val, 0.0, 1.0))
                section_data[feature_name] = val

            if section_data:
                profile[section] = section_data

        if self.normalize:
            profile = self._normalize(profile)

        return profile

    # =====================================================
    # MULTI-TASK SUPPORT
    # =====================================================

    def map_from_model_outputs(
        self,
        model_outputs: Dict[str, Any],
    ) -> Dict[str, Dict[str, float]]:
        """Map a multi-task prediction dict to a per-section profile.

        Two input shapes are supported (CRIT-AG-2 in the audit):

        * **Branch A — model outputs.** ``model_outputs[task]`` is a
          dict containing ``probabilities`` and/or ``logits``. The
          winning-class probability is emitted as ``{task}_probability``
          and routed through :data:`DEFAULT_FEATURE_MAP`.

        * **Branch B — pre-built profile sections.** ``model_outputs``
          is a dict of section -> ``{feature_name: float}`` (the shape
          produced by :class:`BiasProfileBuilder`). Each section is
          passed through verbatim. This avoids silently dropping the
          entire input when the upstream caller already did the
          analysis-level feature work.
        """

        if not isinstance(model_outputs, dict):
            raise ValueError("model_outputs must be dict")

        flat: Dict[str, float] = {}
        direct_sections: Dict[str, Dict[str, float]] = {}

        for task, outputs in model_outputs.items():

            if not isinstance(outputs, dict):
                continue

            has_probs = "probabilities" in outputs
            has_logits = "logits" in outputs

            # ---------- Branch A: model logits/probabilities ----------
            if has_probs or has_logits:
                probs_arr = self._extract_probs(outputs, has_probs, has_logits)
                if probs_arr is None or probs_arr.size == 0:
                    continue

                conf = float(np.max(probs_arr))
                if not np.isfinite(conf):
                    conf = 0.0

                flat[f"{task}_probability"] = float(np.clip(conf, 0.0, 1.0))
                continue

            # ---------- Branch B: pre-built profile section ----------
            section_feats: Dict[str, float] = {}
            for feat_name, val in outputs.items():
                num = _safe_numeric(val, self.strict)
                if num is None:
                    continue
                section_feats[feat_name] = float(np.clip(num, 0.0, 1.0))

            if section_feats:
                direct_sections[task] = section_feats

        profile = self.map_features(flat)

        # Branch B sections bypass DEFAULT_FEATURE_MAP — they are
        # already canonical per-section feature dicts. If the same
        # section name was produced by both branches, prefer the
        # already-mapped Branch A keys but keep any non-overlapping
        # Branch B features.
        for sec, feats in direct_sections.items():
            existing = profile.get(sec, {})
            merged = dict(feats)
            merged.update(existing)
            profile[sec] = merged

        return profile

    # =====================================================
    # HELPERS
    # =====================================================

    def _extract_probs(
        self,
        outputs: Dict[str, Any],
        has_probs: bool,
        has_logits: bool,
    ) -> Optional[np.ndarray]:
        """Resolve a 1-D probability vector for a single task output."""

        if has_probs:
            arr = np.nan_to_num(
                np.asarray(outputs["probabilities"], dtype=np.float64),
                nan=0.0, posinf=1.0, neginf=0.0,
            )
            if arr.ndim > 1:
                arr = arr[0]
            return np.clip(arr, 0.0, 1.0)

        if not has_logits:
            return None

        arr = np.asarray(outputs["logits"], dtype=np.float64)
        if arr.ndim > 1:
            arr = arr[0]

        # CRIT-AG-6: apply calibration at the logit boundary when a
        # fitted calibrator is available; otherwise use the standard
        # softmax/sigmoid conversion.
        if (
            self.calibrator is not None
            and getattr(self.calibrator, "fitted", False)
        ):
            try:
                calibrated = self.calibrator.transform(arr.reshape(1, -1))
                probs = np.asarray(calibrated, dtype=np.float64).ravel()
                return np.clip(probs, 0.0, 1.0)
            except Exception as exc:
                logger.debug("[FeatureMapper] calibration failed: %s", exc)

        return _logits_to_probs(arr)

    # =====================================================
    # SIGNAL EXTRACTION (REC-AG-1)
    #
    # `extract_task_signals` is the new single-pass entry point: for
    # every task it cleans the probability vector exactly once and
    # returns probability + confidence + entropy + max_class + a
    # multilabel flag in a `TaskSignal`. The pipeline reuses this
    # cached object and no longer recomputes the same values three
    # times.
    # =====================================================

    def extract_task_signals(
        self,
        model_outputs: Dict[str, Any],
        task_types: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, TaskSignal]:

        signals: Dict[str, TaskSignal] = {}

        if not isinstance(model_outputs, dict):
            return signals

        task_types = task_types or {}

        for task, outputs in model_outputs.items():

            if not isinstance(outputs, dict):
                continue

            raw_probs = outputs.get("probabilities")
            raw_logits = outputs.get("logits")

            probs_arr: Optional[np.ndarray] = None
            had_nan = False

            if raw_probs is not None:
                orig = np.asarray(raw_probs, dtype=np.float64)
                had_nan = bool(np.isnan(orig).any())
                if had_nan:
                    # EDGE-AG: explicit warning rather than silently
                    # converting NaN -> 0 and emitting "max confidence".
                    logger.warning(
                        "[FeatureMapper] task=%s probabilities contain "
                        "NaN — coerced to zero", task,
                    )
                probs_arr = np.nan_to_num(
                    orig, nan=0.0, posinf=1.0, neginf=0.0,
                )
                if probs_arr.ndim > 1:
                    probs_arr = probs_arr[0]
                probs_arr = np.clip(probs_arr, 0.0, 1.0)

            elif raw_logits is not None:
                probs_arr = self._extract_probs(outputs, False, True)

            if probs_arr is None or probs_arr.size == 0:
                continue

            is_multilabel = (task_types.get(task) == "multilabel")

            if is_multilabel:
                # Each label is an independent Bernoulli; do not renorm.
                p = np.clip(probs_arr, EPS, 1.0 - EPS)
                h = float(-np.sum(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))
                denom = float(p.size * np.log(2.0))
                ent_val = h / denom if denom > 0.0 else 0.0
            else:
                p = np.clip(probs_arr, EPS, 1.0)
                total = float(p.sum())
                if not np.isfinite(total) or total <= 0.0:
                    ent_val = 0.0
                else:
                    p = p / total
                    h = float(-np.sum(p * np.log(p)))
                    denom = float(np.log(max(p.size, 2)))
                    ent_val = h / denom if denom > 0.0 else 0.0

            conf = float(np.clip(np.max(probs_arr), 0.0, 1.0))
            max_class = int(np.argmax(probs_arr))

            signals[task] = TaskSignal(
                probability=probs_arr,
                confidence=conf,
                entropy=float(np.clip(ent_val, 0.0, 1.0)),
                max_class=max_class,
                is_multilabel=is_multilabel,
                had_nan=had_nan,
            )

        return signals

    # -----------------------------------------------------
    # Legacy thin wrappers — kept for backwards compatibility.
    # New callers should prefer `extract_task_signals`.
    # -----------------------------------------------------

    def extract_confidence(
        self,
        model_outputs: Dict[str, Any],
        signals: Optional[Mapping[str, TaskSignal]] = None,
    ) -> Dict[str, float]:
        if signals is None:
            signals = self.extract_task_signals(model_outputs)
        return {task: sig.confidence for task, sig in signals.items()}

    def extract_entropy(
        self,
        model_outputs: Dict[str, Any],
        task_types: Optional[Mapping[str, str]] = None,
        signals: Optional[Mapping[str, TaskSignal]] = None,
    ) -> Dict[str, float]:
        if signals is None:
            signals = self.extract_task_signals(model_outputs, task_types)
        return {task: sig.entropy for task, sig in signals.items()}

    # =====================================================
    # NORMALIZATION (per-section max-norm — disabled by default;
    # see NORM-AG-1)
    # =====================================================

    def _normalize(self, profile: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:

        for section in profile:

            values = list(profile[section].values())

            if not values:
                continue

            max_val = max(values) + EPS

            for k in profile[section]:
                profile[section][k] = float(
                    np.clip(profile[section][k] / max_val, 0.0, 1.0)
                )

        return profile

    # =====================================================
    # BATCH (vectorized per-task)
    # =====================================================

    def map_batch(
        self,
        batch_outputs: List[Dict[str, Any]],
    ) -> List[Dict[str, Dict[str, float]]]:

        return [self.map_features(x) for x in batch_outputs]
