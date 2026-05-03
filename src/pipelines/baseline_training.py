# src/pipelines/hf_baseline_training.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.models.multitask.multitask_model import TruthLensMultiTaskModel
from src.utils.seed_utils import set_seed

logger = logging.getLogger(__name__)


# =========================================================
# METRICS
# =========================================================

def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    acc = accuracy_score(labels, preds)

    try:
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        roc_auc = roc_auc_score(labels, probs[:, 1])
    except Exception:
        roc_auc = 0.0

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


# =========================================================
# DATA
# =========================================================

def _split(df: pd.DataFrame, seed: int):

    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=seed, stratify=df["label"]
    )

    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=seed, stratify=temp_df["label"]
    )

    return train_df, val_df, test_df


def _to_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df.reset_index(drop=True))


def _tokenize(dataset: Dataset, tokenizer, text_column: str, max_length: int):

    return dataset.map(
        lambda x: tokenizer(
            x[text_column],
            truncation=True,
            padding=False,
            max_length=max_length,
        ),
        batched=True,
    )


# =========================================================
# MAIN API
# =========================================================

def train_baseline_model(
    df: pd.DataFrame,
    *,
    model_name: str,
    output_dir: Path,
    text_column: str = "text",
    label_column: str = "label",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    seed: int = 42,
) -> Dict[str, Any]:

    set_seed(seed)

    # -------------------------
    # SPLIT
    # -------------------------

    train_df, val_df, test_df = _split(df, seed)

    # -------------------------
    # TOKENIZER
    # -------------------------

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = _to_dataset(train_df)
    val_ds = _to_dataset(val_df)

    train_ds = _tokenize(train_ds, tokenizer, text_column, 512)
    val_ds = _tokenize(val_ds, tokenizer, text_column, 512)

    # -------------------------
    # MODEL
    # -------------------------

    model = TruthLensMultiTaskModel(model_name)

    if torch.cuda.is_available():
        model = model.to("cuda")

    # -------------------------
    # TRAINING ARGS
    # -------------------------

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=100,
        report_to="none",
    )

    # -------------------------
    # TRAINER
    # -------------------------

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    metrics = trainer.evaluate()

    # -------------------------
    # RETURN STRUCTURED OUTPUT
    # -------------------------

    return {
        "trainer": trainer,
        "metrics": metrics,
        "val_dataset": val_ds,
        "test_dataframe": test_df,
    }