from __future__ import annotations

import logging
from typing import Callable, Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
import networkx as nx
from transformers import AutoTokenizer

from src.utils.device_utils import move_batch, autocast_context
from src.utils.metrics_utils import logits_to_predictions
from src.config.task_config import TASK_CONFIG

# 🔥 NEW
from src.inference.prediction_service import PredictionService

logger = logging.getLogger(__name__)


# =========================================================
# DEVICE
# =========================================================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# TOKENIZATION
# =========================================================

def tokenize_batch(tokenizer, texts: List[str], max_length: int = 512):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


# =========================================================
# BATCHED PREDICT
# =========================================================

def batched_predict(
    model,
    texts: List[str],
    task: str,
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
):
    device = device or get_device()

    model.to(device)
    model.eval()

    all_logits = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):

            batch_texts = texts[i: i + batch_size]
            encoded = tokenize_batch(tokenizer, batch_texts)
            encoded = move_batch(encoded, device)

            with autocast_context():
                outputs = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    task=task,
                )

            logits = outputs["logits"].detach().cpu().numpy()
            all_logits.append(logits)

    return np.vstack(all_logits)


# =========================================================
# 🔥 POSTPROCESS (UPGRADED)
# =========================================================

def postprocess_predictions(
    logits: np.ndarray,
    task: str,
    threshold: Optional[float] = None,
):

    task_type = TASK_CONFIG[task]["type"]

    logits_tensor = torch.tensor(logits)

    preds = logits_to_predictions(
        logits_tensor,
        task_type=task_type,
    ).numpy()

    if task_type == "multiclass":
        probs = torch.softmax(logits_tensor, dim=-1).numpy()
        confidence = np.max(probs, axis=1)

    else:
        probs = torch.sigmoid(logits_tensor).numpy()
        confidence = probs if probs.ndim == 1 else np.max(probs, axis=1)

        # CRIT E7: apply the threshold override to the per-label probability
        # tensor — not to ``confidence`` (which collapses ``np.max`` across
        # labels in the multilabel case and makes the threshold compare
        # against the wrong axis). Binary stays a 1-D vector; multilabel
        # remains (N, L).
        if threshold is not None:
            if task_type == "multilabel" and probs.ndim == 2:
                preds = (probs >= threshold).astype(int)
            else:
                preds = (confidence >= threshold).astype(int)

    return preds, probs, confidence


# =========================================================
# 🔥 HIGH-LEVEL PREDICT (SERVICE-AWARE)
# =========================================================

def predict_texts(
    model=None,
    texts: List[str] = None,
    task: str = None,
    tokenizer=None,
    batch_size: int = 32,
    prediction_service: Optional[PredictionService] = None,
):

    # 🔥 LAT-1: replace the per-sample loop with a single batched call so
    # tokenisation + the model forward pass run once for the whole batch
    # instead of once per text.
    if prediction_service:
        if hasattr(prediction_service, "predict_full_batch"):
            outputs = prediction_service.predict_full_batch(list(texts))
        else:  # pragma: no cover - legacy fallback
            outputs = [prediction_service.predict(t) for t in texts]

        return {
            "predictions": [o.get("label") for o in outputs],
            "probabilities": [o.get("fake_probability") for o in outputs],
            "raw": outputs,
        }

    # fallback
    logits = batched_predict(
        model=model,
        texts=texts,
        task=task,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    preds, probs, confidence = postprocess_predictions(logits, task)

    return {
        "predictions": preds,
        "probabilities": probs,
        "confidence": confidence,
        "logits": logits,
    }


# =========================================================
# 🔥 GRAPH METRICS (UPGRADED)
# =========================================================

def actor_graph_metrics(df: pd.DataFrame) -> Dict[str, float]:

    graph = nx.DiGraph()

    for h, v in zip(df.get("hero_entities", []), df.get("villain_entities", [])):
        if pd.notna(h) and pd.notna(v):
            graph.add_edge(str(h), str(v))

    if graph.number_of_nodes() == 0:
        return {}

    pagerank = nx.pagerank(graph)

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "avg_degree": float(np.mean([d for _, d in graph.degree()])),
        "pagerank_mean": float(np.mean(list(pagerank.values()))),
        "centrality_max": float(np.max(list(pagerank.values()))),  # 🔥 NEW
        "components": nx.number_weakly_connected_components(graph),
    }


# =========================================================
# FRAME COHERENCE
# =========================================================

def frame_coherence(pred, true) -> float:
    pred = np.asarray(pred)
    true = np.asarray(true)

    if pred.shape != true.shape:
        raise ValueError("Shape mismatch")

    return float(np.mean(pred == true))


# =========================================================
# 🔥 ABLATION (SERVICE-AWARE)
# =========================================================

def ablation_importance(
    model,
    texts,
    y,
    feature_names,
    task,
    tokenizer,
    metric: Callable,
    batch_size=32,
    prediction_service: Optional[PredictionService] = None,
):

    def predict_fn(text_batch):
        result = predict_texts(
            model=model,
            texts=text_batch,
            task=task,
            tokenizer=tokenizer,
            batch_size=batch_size,
            prediction_service=prediction_service,
        )
        return result["predictions"]

    from src.evaluation.importance.feature_ablation import FeatureAblation

    ablator = FeatureAblation(metric=metric)

    return ablator.single_feature_ablation(
        X=texts,
        y=y,
        feature_names=feature_names,
        predict_fn=predict_fn,
    )


# =========================================================
# 🔥 PERMUTATION (SERVICE-AWARE)
# =========================================================

def permutation_importance(
    model,
    texts,
    y,
    feature_names,
    task,
    tokenizer,
    metric,
    n_repeats=5,
    batch_size=32,
    prediction_service: Optional[PredictionService] = None,
):

    def predict_fn(text_batch):
        result = predict_texts(
            model=model,
            texts=text_batch,
            task=task,
            tokenizer=tokenizer,
            batch_size=batch_size,
            prediction_service=prediction_service,
        )
        return result["predictions"]

    from src.evaluation.importance.permutation_importance import PermutationImportance

    perm = PermutationImportance(metric=metric)

    return perm.compute(
        X=texts,
        y=y,
        feature_names=feature_names,
        predict_fn=predict_fn,
        n_repeats=n_repeats,
    )


# =========================================================
# 🔥 SHAP (SERVICE-AWARE)
# =========================================================

def shap_importance(
    model,
    texts,
    tokenizer,
    task,
    max_samples=200,
    batch_size=16,
    prediction_service: Optional[PredictionService] = None,
):

    texts_small = texts[:max_samples]

    def predict_fn(text_batch):
        result = predict_texts(
            model=model,
            texts=text_batch,
            task=task,
            tokenizer=tokenizer,
            batch_size=batch_size,
            prediction_service=prediction_service,
        )
        return result["probabilities"]

    from src.evaluation.importance.shap_importance import ShapImportance

    shap_calc = ShapImportance()

    return shap_calc.compute_with_function(
        predict_fn=predict_fn,
        X=texts_small,
        feature_names=None,
    )