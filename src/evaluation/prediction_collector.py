from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from scipy.special import expit, softmax
from transformers import AutoTokenizer

from src.config.task_config import TASK_CONFIG, get_task_type
from src.utils.device_utils import autocast_context, move_batch

logger = logging.getLogger(__name__)


# =========================================================
# DEVICE
# =========================================================

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# TOKENIZATION
# =========================================================

def _tokenize(tokenizer: AutoTokenizer, texts: List[str], max_length: int) -> Dict[str, Any]:
    # HuggingFace returns a BatchEncoding (UserDict subclass, not dict).
    # Explicitly convert to a plain dict so move_batch and any isinstance(x, dict)
    # checks downstream never see an unexpected type.
    encoding = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return dict(encoding)


# =========================================================
# POSTPROCESS
# =========================================================

def _postprocess(logits: np.ndarray, task_type: str, *, threshold: float = 0.5):
    """Convert raw logits into ``(preds, probs)`` for a given task type.

    Notably handles the binary case where the head emits 2 logits (softmax over
    {0, 1}) versus a single logit (sigmoid).

    HIGH E3: use ``scipy.special.softmax`` / ``expit`` directly on numpy to
    skip the numpy → torch → numpy round-trip that this hot path used to do
    on every batch.
    """
    arr = np.asarray(logits, dtype=float)

    if task_type == "multiclass":
        probs = softmax(arr, axis=-1)
        preds = np.argmax(probs, axis=1).astype(int)

    elif task_type == "binary":
        if arr.ndim == 2 and arr.shape[-1] == 2:
            probs = softmax(arr, axis=-1)[:, 1]
        else:
            probs = expit(arr).reshape(-1)
        preds = (probs >= threshold).astype(int)

    elif task_type == "multilabel":
        probs = expit(arr)
        preds = (probs >= threshold).astype(int)

    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    return preds, probs


# =========================================================
# COLLECTOR — class wrapper used by Evaluator
# =========================================================

class PredictionCollector:
    """Light wrapper that bundles raw model output into a uniform record.

    The class is intentionally stateless; downstream consumers (ErrorAnalyzer,
    ThresholdOptimizer) only need the dictionary it returns.
    """

    @staticmethod
    def collect(
        *,
        y_true: Optional[Iterable] = None,
        y_pred: Optional[Iterable] = None,
        y_proba: Optional[Iterable] = None,
        logits: Optional[Iterable] = None,
        task: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {"task": task, "task_type": task_type}

        if y_true is not None:
            record["y_true"] = np.asarray(y_true)
        if y_pred is not None:
            record["y_pred"] = np.asarray(y_pred)
        if y_proba is not None:
            record["y_proba"] = np.asarray(y_proba)
        if logits is not None:
            record["logits"] = np.asarray(logits)

        return record


# =========================================================
# SINGLE-TASK COLLECTION FROM TEXTS
# =========================================================

def collect_predictions(
    model,
    texts: List[str],
    task: str,
    tokenizer: AutoTokenizer,
    *,
    batch_size: int = 32,
    max_length: int = 512,
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    if task not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {task}")

    device = device or get_device()
    task_type = TASK_CONFIG[task]["type"]

    model.to(device)
    model.eval()

    all_logits: List[np.ndarray] = []

    logger.info("[COLLECT] task=%s samples=%d", task, len(texts))

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            encoded = _tokenize(tokenizer, batch_texts, max_length)
            encoded = move_batch(encoded, device)

            with autocast_context():
                out = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    task=task,
                )

            # The model can return logits in three different shapes depending
            # on whether outputs went through Predictor._format_outputs or not:
            #
            #   Shape A (flattened, via Predictor):
            #     out["bias_logits"] = Tensor
            #
            #   Shape B (raw multi-task forward, per-task dict):
            #     out["bias"] = {"logits": Tensor, ...}
            #
            #   Shape C (raw multi-task forward, parallel view):
            #     out["task_logits"] = {"bias": Tensor, ...}
            #
            #   Shape D (generic single-task / legacy):
            #     out["logits"] = Tensor
            #
            raw_logits: Optional[torch.Tensor] = None

            # Shape A
            if f"{task}_logits" in out and isinstance(out[f"{task}_logits"], torch.Tensor):
                raw_logits = out[f"{task}_logits"]

            # Shape B
            elif task in out and isinstance(out[task], dict) and "logits" in out[task]:
                raw_logits = out[task]["logits"]

            # Shape C
            elif (
                "task_logits" in out
                and isinstance(out["task_logits"], dict)
                and task in out["task_logits"]
                and isinstance(out["task_logits"][task], torch.Tensor)
            ):
                raw_logits = out["task_logits"][task]

            # Shape D
            elif "logits" in out and isinstance(out["logits"], torch.Tensor):
                raw_logits = out["logits"]

            else:
                raise KeyError(
                    f"collect_predictions: no logits found for task '{task}'. "
                    f"Tried keys: '{task}_logits', out['{task}']['logits'], "
                    f"out['task_logits']['{task}'], 'logits'. "
                    f"Model returned top-level keys: {list(out.keys())}"
                )

            logits = raw_logits.detach().cpu().float().numpy()
            all_logits.append(logits)

    logits_arr = np.vstack(all_logits) if all_logits else np.empty((0,))
    preds, probs = _postprocess(logits_arr, task_type, threshold=threshold)

    return {
        "task": task,
        "task_type": task_type,
        "logits": logits_arr,
        "probabilities": probs,
        "predictions": preds,
    }


# =========================================================
# MULTI-TASK COLLECTION FROM TEXTS
# =========================================================

def collect_all_tasks(
    model,
    texts: List[str],
    tokenizer: AutoTokenizer,
    *,
    tasks: Optional[List[str]] = None,
    batch_size: int = 32,
    max_length: int = 512,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run inference for several tasks while sharing the encoder pass.

    Section 7: the previous implementation called :func:`collect_predictions`
    once per task, which re-tokenized the texts and re-ran the RoBERTa
    encoder for every task — six tasks meant six full encoder passes per
    batch. We now tokenize + run the encoder a single time per batch and
    fan out across the per-task heads. Falls back to the per-task path if
    the model doesn't expose a head-only forward (``forward_heads`` /
    ``encode``), so older checkpoints keep working.
    """
    selected_tasks = tasks or list(TASK_CONFIG.keys())
    device = device or get_device()

    encode_fn = getattr(model, "encode", None)
    head_fn = getattr(model, "forward_heads", None)

    if not (callable(encode_fn) and callable(head_fn)):
        # Fall back to the per-task path on older models.
        results: Dict[str, Dict[str, Any]] = {}
        logger.info(
            "[COLLECT] multi-task start (%d tasks, fallback per-task)",
            len(selected_tasks),
        )
        for task in selected_tasks:
            results[task] = collect_predictions(
                model=model,
                texts=texts,
                task=task,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_length=max_length,
                device=device,
            )
        logger.info("[COLLECT] multi-task done")
        return results

    model.to(device)
    model.eval()

    n = len(texts)
    logits_buf: Dict[str, List[np.ndarray]] = {t: [] for t in selected_tasks}

    logger.info(
        "[COLLECT] multi-task start (%d tasks, shared encoder)",
        len(selected_tasks),
    )

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_texts = texts[i: i + batch_size]
            encoded = _tokenize(tokenizer, batch_texts, max_length)
            encoded = move_batch(encoded, device)

            with autocast_context():
                hidden = encode_fn(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )
                for task in selected_tasks:
                    head_out = head_fn(hidden, task=task)
                    if isinstance(head_out, dict):
                        if f"{task}_logits" in head_out and isinstance(head_out[f"{task}_logits"], torch.Tensor):
                            logits = head_out[f"{task}_logits"]           # Shape A
                        elif "logits" in head_out and isinstance(head_out["logits"], torch.Tensor):
                            logits = head_out["logits"]                   # Shape B/D
                        elif task in head_out and isinstance(head_out[task], torch.Tensor):
                            logits = head_out[task]                       # bare task key
                        else:
                            raise KeyError(
                                f"collect_all_tasks: no logits for task '{task}' "
                                f"in head output keys: {list(head_out.keys())}"
                            )
                    else:
                        logits = head_out
                    logits_buf[task].append(logits.detach().cpu().float().numpy())

    results = {}
    for task in selected_tasks:
        if not logits_buf[task]:
            continue
        task_type = TASK_CONFIG[task]["type"]
        logits_arr = np.vstack(logits_buf[task])
        preds, probs = _postprocess(logits_arr, task_type)
        results[task] = {
            "task": task,
            "task_type": task_type,
            "logits": logits_arr,
            "probabilities": probs,
            "predictions": preds,
        }

    logger.info("[COLLECT] multi-task done")
    return results


# =========================================================
# DATALOADER PATH (used by EvaluationEngine)
# =========================================================

def _normalize_label_dict(batch_labels: Any, tasks: List[str]) -> Dict[str, torch.Tensor]:
    """Pull a ``{task: tensor}`` dict out of a batch's label payload.

    HIGH E5: be tolerant of the common single-task tuple/tensor case. Many
    DataLoaders return ``(input_ids, attention_mask, labels_tensor)`` and the
    previous strict ``isinstance(dict)`` check raised TypeError on every batch.
    When ``labels`` is a single tensor and the caller declared exactly one
    task, treat it as that task's labels.
    """
    if isinstance(batch_labels, dict):
        return {t: batch_labels[t] for t in tasks if t in batch_labels}

    if isinstance(batch_labels, torch.Tensor) and len(tasks) == 1:
        return {tasks[0]: batch_labels}

    found_keys = (
        sorted(batch_labels.keys())
        if hasattr(batch_labels, "keys")
        else type(batch_labels).__name__
    )
    raise TypeError(
        "DataLoader batches must yield a dict-like ``labels`` field keyed by "
        f"task (or a single tensor when only one task is selected). Found: "
        f"{found_keys!r}"
    )


def collect_all_tasks_from_loader(
    model,
    dataloader,
    *,
    tasks: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, Any]]:
    """Run inference over a DataLoader, returning per-task arrays + ground truth."""
    selected_tasks = tasks or list(TASK_CONFIG.keys())
    device = device or get_device()

    model.to(device)
    model.eval()

    logits_buf: Dict[str, List[np.ndarray]] = {t: [] for t in selected_tasks}
    labels_buf: Dict[str, List[np.ndarray]] = {t: [] for t in selected_tasks}

    with torch.no_grad():
        for batch in dataloader:
            batch_on_device = move_batch(batch, device)
            inputs = {
                k: v
                for k, v in batch_on_device.items()
                if k in ("input_ids", "attention_mask", "token_type_ids")
            }

            label_dict = _normalize_label_dict(
                batch_on_device.get("labels", batch_on_device),
                selected_tasks,
            )

            for task in selected_tasks:
                with autocast_context():
                    out = model(task=task, **inputs)

                logits_buf[task].append(out["logits"].detach().cpu().numpy())

                if task in label_dict:
                    labels_buf[task].append(
                        label_dict[task].detach().cpu().numpy()
                    )

    results: Dict[str, Dict[str, Any]] = {}
    for task in selected_tasks:
        if not logits_buf[task]:
            continue

        task_type = get_task_type(task)
        logits_arr = np.vstack(logits_buf[task])
        preds, probs = _postprocess(logits_arr, task_type, threshold=threshold)

        record: Dict[str, Any] = {
            "task": task,
            "task_type": task_type,
            "logits": logits_arr,
            "probabilities": probs,
            "predictions": preds,
            "y_pred": preds,
            "y_proba": probs,
        }
        if labels_buf[task]:
            y_true = np.concatenate(labels_buf[task], axis=0)
            record["y_true"] = y_true
            record["labels"] = y_true

        results[task] = record

    return results


# =========================================================
# STREAMING (LARGE DATASETS)
# =========================================================

def stream_logits(
    model,
    texts: List[str],
    task: str,
    tokenizer: AutoTokenizer,
    *,
    batch_size: int = 32,
    max_length: int = 512,
):
    device = get_device()
    model.to(device)
    model.eval()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        encoded = _tokenize(tokenizer, batch_texts, max_length)
        encoded = move_batch(encoded, device)

        with torch.no_grad(), autocast_context():
            out = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                task=task,
            )

        yield out["logits"].detach().cpu().numpy()


__all__ = [
    "PredictionCollector",
    "collect_all_tasks",
    "collect_all_tasks_from_loader",
    "collect_predictions",
    "get_device",
    "stream_logits",
]
