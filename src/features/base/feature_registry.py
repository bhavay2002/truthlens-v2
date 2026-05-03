# src/features/base/feature_registry.py

from __future__ import annotations

import importlib
import logging
import pkgutil
import threading
from typing import Dict, List, Type, Any

from src.features.base.base_feature import BaseFeature

logger = logging.getLogger(__name__)


class FeatureRegistry:
    """
    Advanced Feature Registry with metadata, grouping, and auto-discovery.
    """

    _registry: Dict[str, Type[BaseFeature]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}

    _lock = threading.Lock()
    _frozen = False

    # =====================================================
    # REGISTRATION
    # =====================================================

    @classmethod
    def register(
        cls,
        feature_cls: Type[BaseFeature],
        *,
        override: bool = False,
    ) -> Type[BaseFeature]:

        if not issubclass(feature_cls, BaseFeature):
            raise ValueError(f"{feature_cls} must inherit BaseFeature")

        feature_name = getattr(feature_cls, "name", feature_cls.__name__)
        group = getattr(feature_cls, "group", "general")
        description = getattr(feature_cls, "description", "")
        version = getattr(feature_cls, "version", "1.0")

        with cls._lock:

            if cls._frozen:
                raise RuntimeError("Registry is frozen")

            if feature_name in cls._registry and not override:
                raise ValueError(f"Feature '{feature_name}' already registered")

            cls._registry[feature_name] = feature_cls

            cls._metadata[feature_name] = {
                "group": group,
                "description": description,
                "version": version,
                "module": feature_cls.__module__,
            }

        logger.debug("Registered feature: %s (%s)", feature_name, group)

        return feature_cls

    # =====================================================
    # RETRIEVAL
    # =====================================================

    @classmethod
    def get_feature(cls, name: str) -> Type[BaseFeature]:

        if name not in cls._registry:
            raise KeyError(f"Feature '{name}' not found")

        return cls._registry[name]

    @classmethod
    def create_feature(cls, name: str, **kwargs) -> BaseFeature:

        feature_cls = cls.get_feature(name)

        try:
            return feature_cls(**kwargs)
        except TypeError:
            return feature_cls()

    # =====================================================
    # METADATA
    # =====================================================

    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:

        if name not in cls._metadata:
            raise KeyError(f"No metadata for '{name}'")

        return cls._metadata[name]

    @classmethod
    def list_features(cls) -> List[str]:
        return sorted(cls._registry.keys())

    @classmethod
    def list_by_group(cls, group: str) -> List[str]:

        return [
            name for name, meta in cls._metadata.items()
            if meta["group"] == group
        ]

    @classmethod
    def groups(cls) -> List[str]:

        return sorted(set(meta["group"] for meta in cls._metadata.values()))

    @classmethod
    def has_feature(cls, name: str) -> bool:
        return name in cls._registry

    # =====================================================
    # AUTO DISCOVERY (CRITICAL)
    # =====================================================

    @classmethod
    def auto_discover(cls, package: str) -> None:
        """
        Automatically import all modules in a package.

        Example:
            FeatureRegistry.auto_discover("src.features")
        """

        logger.info("Auto-discovering features in %s", package)

        module = importlib.import_module(package)

        for _, name, _ in pkgutil.walk_packages(module.__path__, module.__name__ + "."):
            try:
                importlib.import_module(name)
            except Exception as e:
                logger.warning("Failed to import %s: %s", name, e)

    # =====================================================
    # BULK
    # =====================================================

    @classmethod
    def register_many(cls, features: List[Type[BaseFeature]]) -> None:
        for f in features:
            cls.register(f)

    # =====================================================
    # SAFETY
    # =====================================================

    @classmethod
    def freeze(cls) -> None:
        cls._frozen = True
        logger.info("Feature registry frozen")

    @classmethod
    def clear_registry(cls) -> None:

        with cls._lock:
            cls._registry.clear()
            cls._metadata.clear()

        logger.warning("Registry cleared")

    # =====================================================
    # DEBUG
    # =====================================================

    @classmethod
    def describe_registry(cls) -> Dict[str, Dict[str, Any]]:
        return dict(cls._metadata)


# =========================================================
# DECORATOR
# =========================================================

def register_feature(feature_cls: Type[BaseFeature]) -> Type[BaseFeature]:
    return FeatureRegistry.register(feature_cls)