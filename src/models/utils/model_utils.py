from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import torch

logger = logging.getLogger(__name__)


# =========================================================
# SAVE
# =========================================================

def save_model(
    model: Any,
    path: str | Path,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if isinstance(model, torch.nn.Module):
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "metadata": metadata or {},
                },
                path,
            )
        else:
            joblib.dump(
                {
                    "model": model,
                    "metadata": metadata or {},
                },
                path,
            )

        logger.info("Model saved: %s", path)

        return path

    except Exception as e:
        logger.exception("Save failed")
        raise RuntimeError from e


# =========================================================
# LOAD
# =========================================================

def load_model(
    path: str | Path,
    *,
    model_class: Optional[type] = None,
    device: Optional[str] = None,
) -> Any:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    try:
        if path.suffix in {".pt", ".pth"}:

            # A5.1: centralised detection so CUDA / MPS / CPU fallback
            # is identical across the codebase.
            from src.models._device import detect_device

            device_obj = detect_device(device)

            # C1.3: ``weights_only=True`` forbids arbitrary pickle code
            # execution at load time. ``save_model`` above only writes a
            # dict of {"state_dict": <tensors>, "metadata": <plain
            # python>}, both of which round-trip safely through the
            # restricted unpickler.
            checkpoint = torch.load(
                path,
                map_location=device_obj,
                weights_only=True,
            )

            if model_class is None:
                return checkpoint

            model = model_class()
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device_obj)
            model.eval()

            return model

        else:

            obj = joblib.load(path)

            if isinstance(obj, dict) and "model" in obj:
                return obj["model"]

            return obj

    except Exception as e:
        logger.exception("Load failed")
        raise RuntimeError from e


# =========================================================
# TEXT PREPROCESS
# =========================================================

def preprocess_text(
    text: str,
    *,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_html: bool = True,
) -> str:

    if text is None:
        raise ValueError("text cannot be None")

    if not isinstance(text, str):
        raise TypeError("text must be string")

    text = text.strip()

    if not text:
        raise ValueError("text empty")

    # -------------------------
    # BASIC CLEANING
    # -------------------------

    text = text.replace("\n", " ").replace("\t", " ")

    if remove_html:
        text = re.sub(r"<.*?>", " ", text)

    if remove_urls:
        text = re.sub(r"http\S+|www\S+", " ", text)

    text = re.sub(r"\s+", " ", text)

    if lowercase:
        text = text.lower()

    return text.strip()