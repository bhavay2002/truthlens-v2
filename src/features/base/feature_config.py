
#File Name: feature_config.py


from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.features.base.feature_registry import FeatureRegistry

logger = logging.getLogger(__name__)


# =========================================================
# UTILS
# =========================================================

def merge_params(
    global_params: Dict[str, Any],
    group_params: Dict[str, Any],
    feature_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge parameters with priority:
    feature > group > global
    """
    merged = {}
    merged.update(global_params)
    merged.update(group_params)
    merged.update(feature_params)
    return merged


# =========================================================
# FEATURE DEFINITION
# =========================================================

@dataclass
class FeatureDefinition:
    """
    Configuration for a single feature.
    """

    name: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)

    # 🔥 NEW
    priority: int = 0
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None  # evaluated using global_params

    # -----------------------------------------------------

    def validate(self) -> None:

        if not self.name:
            raise ValueError("FeatureDefinition must include a feature name")

        if not isinstance(self.params, dict):
            raise ValueError(f"{self.name}: params must be dict")

        if not FeatureRegistry.has_feature(self.name):
            raise ValueError(f"{self.name} not registered in FeatureRegistry")


# =========================================================
# GROUP CONFIG
# =========================================================

@dataclass
class FeatureGroupConfig:
    """
    Logical group of features.
    """

    group_name: str
    enabled: bool = True

    features: List[FeatureDefinition] = field(default_factory=list)

    # 🔥 NEW
    group_params: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0

    # -----------------------------------------------------

    def validate(self) -> None:

        if not self.group_name:
            raise ValueError("FeatureGroupConfig must define group_name")

        for feature in self.features:
            feature.validate()


# =========================================================
# PIPELINE CONFIG
# =========================================================

@dataclass
class FeaturePipelineConfig:
    """
    Top-level pipeline configuration.
    """

    groups: List[FeatureGroupConfig] = field(default_factory=list)
    global_params: Dict[str, Any] = field(default_factory=dict)

    # =====================================================
    # VALIDATION
    # =====================================================

    def validate(self) -> None:

        if not isinstance(self.groups, list):
            raise ValueError("groups must be list")

        for group in self.groups:
            group.validate()

        self._validate_dependencies()

    # -----------------------------------------------------

    def _validate_dependencies(self) -> None:

        enabled = set(self.enabled_features())

        for group in self.groups:
            for feature in group.features:

                for dep in feature.depends_on:
                    if dep not in enabled:
                        raise ValueError(
                            f"{feature.name} depends on {dep}, but it is disabled"
                        )

    # =====================================================
    # ENABLED FEATURES
    # =====================================================

    def enabled_features(self) -> List[str]:

        enabled = []

        for group in self.groups:
            if not group.enabled:
                continue

            for feature in group.features:
                if feature.enabled:
                    enabled.append(feature.name)

        return enabled

    # =====================================================
    # PARAM RETRIEVAL
    # =====================================================

    def feature_parameters(self, feature_name: str) -> Dict[str, Any]:

        for group in self.groups:
            for feature in group.features:
                if feature.name == feature_name:
                    return merge_params(
                        self.global_params,
                        group.group_params,
                        feature.params,
                    )

        raise KeyError(f"No parameters defined for '{feature_name}'")

    # =====================================================
    # CONDITION CHECK
    # =====================================================

    def _check_condition(self, condition: Optional[str]) -> bool:

        if not condition:
            return True

        try:
            return bool(eval(condition, {}, self.global_params))
        except Exception as e:
            logger.warning("Condition failed: %s", e)
            return False

    # =====================================================
    # BUILD PIPELINE (CRITICAL)
    # =====================================================

    def build_features(self) -> List:

        features = []

        for group in sorted(self.groups, key=lambda g: g.priority):

            if not group.enabled:
                continue

            for feat in sorted(group.features, key=lambda f: f.priority):

                if not feat.enabled:
                    continue

                if not self._check_condition(feat.condition):
                    continue

                cls = FeatureRegistry.get_feature(feat.name)

                params = merge_params(
                    self.global_params,
                    group.group_params,
                    feat.params,
                )

                try:
                    instance = cls(**params)
                except TypeError:
                    # fallback if feature has no params
                    instance = cls()

                features.append(instance)

        logger.info("Built %d features", len(features))

        return features

    # =====================================================
    # EXPLAINABILITY SUPPORT
    # =====================================================

    def feature_to_group_map(self) -> Dict[str, str]:

        mapping = {}

        for group in self.groups:
            for feature in group.features:
                mapping[feature.name] = group.group_name

        return mapping


# =========================================================
# CONFIG LOADER
# =========================================================

class FeatureConfigLoader:
    """
    Convert dict (YAML) → FeaturePipelineConfig
    """

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> FeaturePipelineConfig:

        if not isinstance(config_dict, dict):
            raise TypeError("Feature config must be dict")

        groups_raw = config_dict.get("groups")

        if not isinstance(groups_raw, list):
            raise ValueError("config must contain 'groups' list")

        groups: List[FeatureGroupConfig] = []

        for group_data in groups_raw:

            features_raw = group_data.get("features", [])

            feature_defs = []

            for f in features_raw:

                feature_defs.append(
                    FeatureDefinition(
                        name=f["name"],
                        enabled=f.get("enabled", True),
                        params=f.get("params", {}),
                        priority=f.get("priority", 0),
                        depends_on=f.get("depends_on", []),
                        condition=f.get("condition"),
                    )
                )

            groups.append(
                FeatureGroupConfig(
                    group_name=group_data["group_name"],
                    enabled=group_data.get("enabled", True),
                    features=feature_defs,
                    group_params=group_data.get("group_params", {}),
                    priority=group_data.get("priority", 0),
                )
            )

        pipeline = FeaturePipelineConfig(
            groups=groups,
            global_params=config_dict.get("global_params", {}),
        )

        pipeline.validate()

        logger.info(
            "Loaded config | groups=%d | features=%d",
            len(groups),
            len(pipeline.enabled_features()),
        )

        return pipeline