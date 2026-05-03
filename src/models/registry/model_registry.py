from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.utils.settings import load_settings
from src.models.registry.model_factory import ModelFactory
from src.models.metadata.model_metadata import ModelMetadata
from src.models.metadata.model_versioning import (
    ModelVersionInfo,
    ModelVersionRegistry,
)

logger = logging.getLogger(__name__)

RobertaTokenizer = AutoTokenizer
RobertaForSequenceClassification = AutoModelForSequenceClassification

MULTITASK_MODEL_TYPE = "multitask_truthlens"

_SETTINGS = None


# =========================================================
# SETTINGS
# =========================================================

def _get_settings():
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load_settings()
    return _SETTINGS


# =========================================================
# MULTITASK LOADER
# =========================================================

def _load_multitask_model(model_path: Path, device: torch.device):

    from src.models.multitask.multitask_truthlens_model import (
        MultiTaskTruthLensConfig,
        MultiTaskTruthLensModel,
    )

    config_path = model_path / "config.json"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_cfg = MultiTaskTruthLensConfig(
        model_name=cfg.get("model_name", "roberta-base"),
        dropout=cfg.get("dropout", 0.1),
        pooling=cfg.get("pooling", "cls"),
        init_from_config_only=True,
    )

    model = MultiTaskTruthLensModel(config=model_cfg)

    weights_path = model_path / "pytorch_model.bin"

    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)

        load_result = model.load_state_dict(state_dict, strict=False)

        if load_result.missing_keys:
            raise RuntimeError(f"Missing keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            raise RuntimeError(f"Unexpected keys: {load_result.unexpected_keys}")

        logger.info("Loaded multitask weights: %s", weights_path)
    else:
        logger.warning("No weights found: using random init")

    return model


# =========================================================
# REGISTRY
# =========================================================

class ModelRegistry:

    @staticmethod
    def load_model(
        model_name: Optional[str] = "truthlens_model",
        model_type: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:

        try:

            logger.info("[REGISTRY] Loading model: %s", model_name)

            settings = _get_settings()

            base_dir = Path(settings.model.path)
            vectorizer_path = Path(settings.paths.tfidf_vectorizer_path)

            # -------------------------------------------------
            # RESOLVE MODEL PATH
            # -------------------------------------------------

            model_path = base_dir

            if model_name:
                candidate = base_dir / model_name

                if candidate.exists():
                    model_path = candidate
                elif (base_dir / "config.json").exists():
                    model_path = base_dir
                else:
                    raise FileNotFoundError(f"Model not found: {model_name}")

            device_obj = torch.device(
                device or ("cuda" if torch.cuda.is_available() else "cpu")
            )

            # -------------------------------------------------
            # LOAD CONFIG
            # -------------------------------------------------

            saved_cfg = {}
            config_path = model_path / "config.json"

            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    saved_cfg = json.load(f)

            saved_model_type = saved_cfg.get("model_type")

            base_model_name = (
                saved_cfg.get("model_name")
                or getattr(getattr(settings.model, "encoder", None), "name", None)
                or settings.model.name
            )

            # -------------------------------------------------
            # TOKENIZER
            # -------------------------------------------------

            try:
                tokenizer = RobertaTokenizer.from_pretrained(str(model_path))
            except Exception:
                tokenizer = RobertaTokenizer.from_pretrained(base_model_name)

            # -------------------------------------------------
            # MODEL LOAD
            # -------------------------------------------------

            if model_type is None and saved_model_type == MULTITASK_MODEL_TYPE:

                model = _load_multitask_model(model_path, device_obj)

            elif model_type is None:

                model = RobertaForSequenceClassification.from_pretrained(model_path)

            else:

                config_file = model_path / "model_config.json"

                if not config_file.exists():
                    raise FileNotFoundError("Missing model_config.json")

                with open(config_file, "r", encoding="utf-8") as f:
                    cfg_dict = json.load(f)

                model = ModelFactory.create(model_type, cfg_dict)

                checkpoint_path = model_path / "model.pt"

                if checkpoint_path.exists():
                    state_dict = torch.load(
                        checkpoint_path,
                        map_location=device_obj,
                        weights_only=True,
                    )

                    load_result = model.load_state_dict(state_dict, strict=False)

                    if load_result.missing_keys:
                        logger.warning("Missing keys: %s", load_result.missing_keys)
                    if load_result.unexpected_keys:
                        logger.warning("Unexpected keys: %s", load_result.unexpected_keys)

            if hasattr(model, "to"):
                model.to(device_obj)

            if hasattr(model, "eval"):
                model.eval()

            # -------------------------------------------------
            # VECTORIZE
            # -------------------------------------------------

            vectorizer = None

            if vectorizer_path.exists():
                try:
                    vectorizer = joblib.load(vectorizer_path)
                except Exception:
                    logger.warning("Vectorizer load failed")

            # -------------------------------------------------
            # METADATA
            # -------------------------------------------------

            metadata = None
            metadata_path = model_path / "metadata.json"

            if metadata_path.exists():
                try:
                    metadata = ModelMetadata.load_json(metadata_path)
                except Exception:
                    logger.warning("Metadata load failed")

            return {
                "model": model,
                "tokenizer": tokenizer,
                "vectorizer": vectorizer,
                "device": device_obj,
                "metadata": metadata,
            }

        except Exception as e:
            logger.exception("Registry load failed")
            raise RuntimeError("Model loading failed") from e

    # =====================================================
    # VERSIONING
    # =====================================================

    @staticmethod
    def list_versions(
        model_name: str,
        registry_dir: Optional[str] = None,
    ):

        target_dir = Path(registry_dir) if registry_dir else Path.cwd()

        registry = ModelVersionRegistry(target_dir)

        return registry.list_versions(model_name)

    @staticmethod
    def get_latest_version(
        model_name: str,
        registry_dir: Optional[str] = None,
    ):

        target_dir = Path(registry_dir) if registry_dir else Path.cwd()

        registry = ModelVersionRegistry(target_dir)

        return registry.get_latest_version(model_name)

    @staticmethod
    def get_version(
        model_name: str,
        version: str,
        registry_dir: Optional[str] = None,
    ):

        target_dir = Path(registry_dir) if registry_dir else Path.cwd()

        registry = ModelVersionRegistry(target_dir)

        return registry.get_version(model_name, version)


# =========================================================
# HELPER
# =========================================================

def get_model() -> Dict[str, Any]:
    return ModelRegistry.load_model()