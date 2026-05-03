from __future__ import annotations

import importlib
import logging
import os
from typing import List, Optional, Set

from src.features.base.feature_registry import FeatureRegistry
from src.features.feature_config import FeatureConfig

logger = logging.getLogger(__name__)

_BOOTSTRAPPED = False


# =========================================================
# CPU THREAD CAP — audit fix §6.2
# =========================================================
# On a Linux container with 8 visible CPUs PyTorch will spawn 8 threads
# by default for both intra- and inter-op parallelism. That competes
# with FastAPI's worker pool and produces severe context-switch noise on
# small inference batches. Cap to 4 threads (or fewer if the container
# is throttled). Wrapped in a try/except because torch is an optional
# dependency for some unit-test environments.

def _apply_torch_thread_cap(cap: int = 4) -> None:
    try:
        import torch  # noqa: WPS433 — local import is intentional
    except ImportError:
        return
    try:
        budget = max(1, min(cap, os.cpu_count() or cap))
        torch.set_num_threads(budget)
        # ``set_num_interop_threads`` can only be called before any
        # parallel region is created — guard so a re-bootstrap does not
        # raise.
        try:
            torch.set_num_interop_threads(budget)
        except RuntimeError:
            pass
        logger.info("torch threads capped to %d (cpu_count=%s)", budget, os.cpu_count())
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("torch thread cap skipped: %s", exc)


# Apply with the legacy default (4) at import time so bare
# ``import src.features`` still gets a sane CPU thread budget. The
# ``bootstrap_feature_registry`` path will re-apply with the
# config-driven value when it runs.
_apply_torch_thread_cap()


# =========================================================
# FEATURE MODULES (DEDUP SAFE)
# =========================================================
# Audit fix §4 — three redundant extractors deleted:
#   * narrative.narrative_features        → role keys absorbed by
#     narrative_role_features, conflict keys by conflict_features.
#   * propaganda.propaganda_lexicon_features → folded into
#     propaganda_features (same lexicon families).
#   * emotion.emotion_features + emotion.emotion_lexicon_features →
#     emotion_intensity_features now owns the lexicon + transformer
#     hybrid path, so the two stand-alone lexicon paths were duplicate
#     work that overwrote each other on collision.

FEATURE_MODULES: List[str] = list(dict.fromkeys([
    # -------------------------
    # TEXT
    # -------------------------
    "src.features.text.lexical_features",
    "src.features.text.semantic_features",
    "src.features.text.syntactic_features",
    "src.features.text.token_features",

    # -------------------------
    # BIAS
    # -------------------------
    "src.features.bias.bias_features",
    "src.features.bias.bias_lexicon_features",
    "src.features.bias.framing_features",
    "src.features.bias.ideological_features",

    # -------------------------
    # DISCOURSE
    # -------------------------
    "src.features.discourse.argument_structure_features",
    "src.features.discourse.discourse_features",

    # -------------------------
    # EMOTION (audit §4 — single hybrid extractor)
    # -------------------------
    "src.features.emotion.emotion_intensity_features",
    "src.features.emotion.emotion_target_features",
    "src.features.emotion.emotion_trajectory_features",

    # -------------------------
    # GRAPH
    # -------------------------
    "src.features.graph.entity_graph_features",
    "src.features.graph.interaction_graph_features",

    # -------------------------
    # NARRATIVE (audit §4 — narrative_features.py removed)
    # -------------------------
    "src.features.narrative.conflict_features",
    "src.features.narrative.narrative_frame_features",
    "src.features.narrative.narrative_role_features",

    # -------------------------
    # PROPAGANDA (audit §4 — propaganda_lexicon_features.py removed)
    # -------------------------
    "src.features.propaganda.manipulation_patterns",
    "src.features.propaganda.propaganda_features",

    # -------------------------
    # ANALYSIS ADAPTER
    # -------------------------
    "src.features.analysis.analysis_adapter_features",
]))


# =========================================================
# BOOTSTRAP
# =========================================================

def bootstrap_feature_registry(
    *,
    strict: bool = False,
    reload: bool = False,
    auto_discover: bool = False,
    auto_package: str = "src.features",
    config: Optional[FeatureConfig] = None,
) -> None:
    """
    Initialize and register all feature modules.

    Audit fix §9 — accepts an optional :class:`FeatureConfig`.  When
    ``None`` we instantiate the default (env-driven) config and apply
    it to :mod:`runtime_config` before any feature module is imported.
    The torch thread cap is then re-applied with the config value so
    operators have a single source of truth for runtime tuning.
    """

    global _BOOTSTRAPPED

    if _BOOTSTRAPPED and not reload:
        logger.debug("Feature registry already bootstrapped")
        return

    # Audit fix §9 — single source of truth for runtime config.
    if config is None:
        config = FeatureConfig()
    config.apply_to_runtime()
    _apply_torch_thread_cap(cap=config.torch_thread_cap)

    success = 0
    failed = []
    loaded_modules: Set[str] = set()

    # -----------------------------------------------------
    # Manual module loading
    # -----------------------------------------------------

    for module_path in FEATURE_MODULES:

        if module_path in loaded_modules:
            continue

        try:
            if reload:
                importlib.invalidate_caches()

            importlib.import_module(module_path)

            loaded_modules.add(module_path)
            success += 1

        except Exception as exc:
            failed.append((module_path, exc))
            logger.warning(
                "Feature module import failed: %s (%s)",
                module_path,
                exc,
            )

    # -----------------------------------------------------
    # Auto discovery (optional)
    # -----------------------------------------------------

    if auto_discover:
        try:
            FeatureRegistry.auto_discover(auto_package)
            logger.info("Auto-discovery completed: %s", auto_package)
        except Exception as exc:
            logger.warning("Auto-discovery failed: %s", exc)
            if strict:
                raise

    # -----------------------------------------------------
    # Validation check (NEW 🔥)
    # -----------------------------------------------------

    try:
        registered = FeatureRegistry.list_features()

        if not registered:
            raise RuntimeError("No features registered after bootstrap")

        logger.info("Registered features: %d", len(registered))

        # Audit fix §9 — startup diff log. Surface (a) modules that
        # were declared in FEATURE_MODULES but failed to register any
        # extractor and (b) extractors that registered themselves
        # without being in the manual list. Both are silent
        # configuration drift in the previous code path.
        expected_modules = set(loaded_modules)
        registered_modules = set()
        for name in registered:
            try:
                meta = FeatureRegistry.get_metadata(name)
                mod = meta.get("module")
                if mod:
                    registered_modules.add(mod)
            except Exception:
                continue

        missing_from_registry = sorted(
            m for m in expected_modules if m not in registered_modules
        )
        if missing_from_registry:
            logger.warning(
                "Feature bootstrap diff | imported but no extractor registered: %s",
                missing_from_registry,
            )

        unexpected_in_registry = sorted(
            m for m in registered_modules if m not in expected_modules
        )
        if unexpected_in_registry:
            logger.warning(
                "Feature bootstrap diff | extractor registered outside FEATURE_MODULES: %s",
                unexpected_in_registry,
            )

    except Exception as exc:
        logger.error("Feature registry validation failed: %s", exc)
        if strict:
            raise

    # -----------------------------------------------------
    # Final status
    # -----------------------------------------------------

    total = len(FEATURE_MODULES)

    logger.info(
        "Feature bootstrap complete | loaded=%d failed=%d total=%d",
        success,
        len(failed),
        total,
    )

    if failed:
        for mod, err in failed:
            logger.debug("FAILED → %s | %s", mod, err)

        if strict:
            raise RuntimeError(
                f"{len(failed)} feature modules failed to load"
            )

    # -----------------------------------------------------
    # Freeze registry (production safety)
    # -----------------------------------------------------

    try:
        FeatureRegistry.freeze()
        logger.debug("FeatureRegistry frozen")
    except Exception:
        logger.debug("FeatureRegistry freeze skipped")

    _BOOTSTRAPPED = True


# =========================================================
# DEBUG UTIL
# =========================================================

def list_loaded_features() -> List[str]:
    """
    Return all registered feature names.
    """
    return FeatureRegistry.list_features()