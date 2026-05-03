# src/analysis/feature_merger.py

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

class FeatureMergerConfig:
    """
    Controls merging behavior.
    """

    def __init__(
        self,
        *,
        prefix_sections: bool = True,
        separator: str = ".",
        sort_keys: bool = True,
        normalize: bool = False,
        fill_missing: float = 0.0,
        track_metadata: bool = True,
    ):
        self.prefix_sections = prefix_sections
        self.separator = separator
        self.sort_keys = sort_keys
        self.normalize = normalize
        self.fill_missing = fill_missing
        self.track_metadata = track_metadata


# =========================================================
# MERGER
# =========================================================

class FeatureMerger:

    def __init__(self, config: Optional[FeatureMergerConfig] = None):
        self.config = config or FeatureMergerConfig()

    # -----------------------------------------------------

    def merge(
        self,
        sections: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:

        merged: Dict[str, float] = {}

        for section_name, features in sections.items():

            if not isinstance(features, dict):
                logger.warning("Skipping invalid section: %s", section_name)
                continue

            for key, value in features.items():

                full_key = self._make_key(section_name, key)

                merged[full_key] = self._safe(value)

        if self.config.normalize:
            merged = self._normalize(merged)

        return merged

    # -----------------------------------------------------

    def to_vector(
        self,
        sections: Dict[str, Dict[str, float]],
    ) -> Tuple[np.ndarray, List[str]]:

        merged = self.merge(sections)

        keys = list(merged.keys())

        if self.config.sort_keys:
            keys.sort()

        vector = np.array(
            [merged.get(k, self.config.fill_missing) for k in keys],
            dtype=np.float32,
        )

        return vector, keys

    # -----------------------------------------------------

    def merge_with_metadata(
        self,
        sections: Dict[str, Dict[str, float]],
    ) -> Dict:

        merged = self.merge(sections)

        result = {
            "features": merged,
        }

        if not self.config.track_metadata:
            return result

        result["meta"] = {
            "num_features": len(merged),
            "sections": list(sections.keys()),
            "feature_groups": {
                section: len(features)
                for section, features in sections.items()
            },
            "completeness": self._completeness(sections),
        }

        return result

    # -----------------------------------------------------

    def _make_key(self, section: str, key: str) -> str:

        if not self.config.prefix_sections:
            return key

        return f"{section}{self.config.separator}{key}"

    # -----------------------------------------------------

    def _normalize(self, features: Dict[str, float]) -> Dict[str, float]:

        values = np.array(list(features.values()), dtype=np.float32)

        total = float(values.sum())

        if total == 0:
            return features

        norm_values = values / total

        return dict(zip(features.keys(), norm_values.astype(float)))

    # -----------------------------------------------------

    def _safe(self, value: float) -> float:

        if not isinstance(value, (int, float)):
            return 0.0

        if not np.isfinite(value):
            return 0.0

        return float(value)

    # -----------------------------------------------------

    def _completeness(
        self,
        sections: Dict[str, Dict[str, float]],
    ) -> float:
        """
        Fraction of features that are present and numerically valid.
        A feature counts as "complete" when its value is a finite
        numeric, regardless of whether it is exactly zero — a legitimate
        zero (e.g. no fear-appeal terms found) still represents a
        complete measurement.
        """

        total = 0
        complete = 0

        for section in sections.values():
            if not isinstance(section, dict):
                continue
            for v in section.values():
                total += 1
                if isinstance(v, (int, float)) and np.isfinite(v):
                    complete += 1

        if total == 0:
            return 0.0

        return complete / total