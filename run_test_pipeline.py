"""End-to-end test runner for TruthLens AI.

Loads all 6 test CSVs (bias, ideology, propaganda, narrative_frame,
narrative, emotion), wires up the full TruthLensPipeline
(Inference → Analysis → Aggregation → Evaluation → Explainability),
and prints a structured summary of every stage.

Usage:
    uv run python run_test_pipeline.py
    uv run python run_test_pipeline.py --samples 5
    uv run python run_test_pipeline.py --samples 5 --explainability
    uv run python run_test_pipeline.py --generate-data   # create synthetic CSVs first
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Lightweight imports at module level.
# torch / transformers / TruthLensPipeline are deferred to inside main()
# so that a "no data → exit" run finishes in ~1 s instead of 22 s.
import numpy as np
import pandas as pd

from src.config.config_loader import load_config
from src.utils.logging_utils import configure_logging
from src.utils.seed_utils import set_seed

CONFIG_PATH = Path("config/config.yaml")
DATA_DIR = Path("data/test")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_test_pipeline")


# =========================================================
# DATASET LOADERS
# =========================================================

def load_bias(n: int) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(DATA_DIR / "bias.csv").dropna(subset=["text", "bias_label"]).head(n)
    return df["text"].tolist(), df["bias_label"].astype(int).values


def load_ideology(n: int) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(DATA_DIR / "ideology.csv").dropna(subset=["text", "ideology_label"]).head(n)
    return df["text"].tolist(), df["ideology_label"].astype(int).values


def load_propaganda(n: int) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(DATA_DIR / "propaganda.csv").dropna(subset=["text", "propaganda_label"]).head(n)
    return df["text"].tolist(), df["propaganda_label"].astype(int).values


def load_narrative_frame(n: int) -> Tuple[List[str], np.ndarray]:
    """frame.csv: columns CO, EC, HI, MO, RE → multilabel (5 cols)."""
    df = pd.read_csv(DATA_DIR / "frame.csv").dropna(subset=["text"]).head(n)
    label_cols = ["CO", "EC", "HI", "MO", "RE"]
    for c in label_cols:
        if c not in df.columns:
            df[c] = 0
    labels = df[label_cols].fillna(0).astype(int).values
    return df["text"].tolist(), labels


def load_narrative(n: int) -> Tuple[List[str], np.ndarray]:
    """narrative.csv: columns hero, villain, victim → multilabel (3 cols)."""
    df = pd.read_csv(DATA_DIR / "narrative.csv").dropna(subset=["text"]).head(n)
    label_cols = ["hero", "villain", "victim"]
    for c in label_cols:
        if c not in df.columns:
            df[c] = 0
    labels = df[label_cols].fillna(0).astype(int).values
    return df["text"].tolist(), labels


def load_emotion(n: int) -> Tuple[List[str], np.ndarray]:
    """emotion.csv: columns emotion_0..emotion_10 → multilabel (11 cols)."""
    df = pd.read_csv(DATA_DIR / "emotion.csv").dropna(subset=["text"]).head(n)
    label_cols = [f"emotion_{i}" for i in range(11)]
    for c in label_cols:
        if c not in df.columns:
            df[c] = 0
    labels = df[label_cols].fillna(0).astype(int).values
    return df["text"].tolist(), labels


DATASET_LOADERS = {
    "bias": load_bias,
    "ideology": load_ideology,
    "propaganda": load_propaganda,
    "narrative_frame": load_narrative_frame,
    "narrative": load_narrative,
    "emotion": load_emotion,
}


# =========================================================
# LABEL FORMATTERS
# =========================================================

def labels_to_eval_fmt(task: str, labels: np.ndarray) -> Any:
    """Convert raw numpy label arrays into the format expected by
    run_evaluation_pipeline (lists of int for multiclass / lists of
    binary lists for multilabel)."""
    if labels.ndim == 1:
        return labels.tolist()
    return labels.tolist()


# =========================================================
# REPORTING HELPERS
# =========================================================

_SEP = "─" * 70


def _hdr(title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)


def _print_article(idx: int, text: str, result: Dict[str, Any]) -> None:
    print(f"\n  [Article {idx + 1}]  {text[:80]}{'...' if len(text) > 80 else ''}")
    errors = result.get("errors") or {}
    if errors:
        for stage, msg in errors.items():
            print(f"    ⚠  {stage}: {msg[:120]}")

    scores = result.get("scores") or {}
    if scores:
        score_str = "  ".join(f"{k}={v:.3f}" for k, v in sorted(scores.items()) if isinstance(v, float))
        print(f"    scores      : {score_str or '(none)'}")

    preds = result.get("predictions") or {}
    if preds:
        def _fmt_val(v: Any) -> str:
            if isinstance(v, (list, np.ndarray)):
                return str(v)
            if isinstance(v, float):
                return f"{v:.3f}"
            return str(v)
        pred_str = "  ".join(f"{k}={_fmt_val(v)}" for k, v in sorted(preds.items()))
        print(f"    predictions : {pred_str[:120]}")

    stages = (result.get("metadata") or {}).get("stages") or {}
    if stages:
        timing = "  ".join(f"{k}={v*1000:.0f}ms" for k, v in stages.items())
        print(f"    timing      : {timing}")

    agg = result.get("aggregation") or {}
    credibility = (agg.get("scores") or agg.get("raw_scores") or {}).get("credibility_score")
    if credibility is not None:
        print(f"    credibility : {credibility:.3f}")

    expl = result.get("explainability")
    if expl:
        top_tokens = (expl.get("tokens") or [])[:5]
        if top_tokens:
            print(f"    top tokens  : {top_tokens}")


def _print_eval_report(task: str, task_report: Dict[str, Any]) -> None:
    metrics = task_report.get("metrics") or {}
    if not metrics:
        print(f"    {task}: no metrics computed (model not yet trained)")
        return
    parts = []
    for k in ("accuracy", "f1", "precision", "recall", "roc_auc"):
        v = metrics.get(k)
        if v is not None:
            parts.append(f"{k}={v:.3f}")
    print(f"    {task}: {' | '.join(parts) if parts else str(metrics)[:120]}")


# =========================================================
# MAIN
# =========================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TruthLens end-to-end test pipeline")
    p.add_argument("--samples", type=int, default=3,
                   help="Number of samples to pull from each test dataset (default 3)")
    p.add_argument("--explainability", action="store_true",
                   help="Enable explainability stage — fast mode (attention + bias, no LIME)")
    p.add_argument("--full-explainability", dest="full_explainability",
                   action="store_true",
                   help="Enable explainability with LIME (slow, ~5 s/article) instead of attention-only")
    p.add_argument("--no-parallel", action="store_true",
                   help="Disable parallel analysis/graph stages")
    p.add_argument("--generate-data", action="store_true",
                   help="Generate synthetic test CSVs in data/test/ before running")
    return p.parse_args()


def _generate_synthetic_data(n: int = 100) -> None:
    """Write synthetic test CSVs to data/test/ so the pipeline can run without real data."""
    import random
    import csv

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    _TEMPLATES = [
        "The government announced new policies on {topic} that critics say will harm {group}.",
        "Scientists published research showing {claim} according to data from {source}.",
        "Protesters gathered outside {place} demanding change over {topic} amid growing tension.",
        "The report reveals that {group} faces serious challenges due to {topic} in recent years.",
        "Officials defended their stance on {topic} calling opposition claims misleading.",
        "New evidence suggests {claim}, contradicting previous statements by authorities.",
        "Analysts warn that rising {topic} could destabilise the economy over the next decade.",
        "Lawmakers approved a bill addressing {topic} despite fierce opposition from {group}.",
        "Experts say the recent surge in {topic} is unprecedented and demands urgent action.",
        "The administration blamed {group} for spreading misinformation about {topic}.",
        "Investigative journalists uncovered documents showing {claim} was known for years.",
        "Corporate profits soared while {group} struggled with the impact of {topic}.",
        "The media played a key role in shaping public perception of {topic} this quarter.",
        "Civil society groups called for accountability after {claim} became public knowledge.",
        "International observers expressed concern over {topic} and its effects on democracy.",
    ]
    _TOPICS = ["inflation", "immigration", "healthcare", "climate change", "election integrity",
               "trade policy", "social welfare", "national security", "housing costs", "education reform"]
    _GROUPS = ["working families", "minority communities", "small businesses", "the middle class",
               "rural voters", "urban populations", "young adults", "senior citizens"]
    _CLAIMS = ["data was manipulated", "funds were misused", "officials knew in advance",
                "the policy failed", "numbers were underreported", "public was misled"]
    _SOURCES = ["Nature", "Reuters", "WHO", "the UN", "a leaked report", "internal documents"]
    _PLACES = ["the Capitol", "city hall", "corporate headquarters", "the White House", "Parliament"]

    def _make_text() -> str:
        tmpl = random.choice(_TEMPLATES)
        return tmpl.format(
            topic=random.choice(_TOPICS),
            group=random.choice(_GROUPS),
            claim=random.choice(_CLAIMS),
            source=random.choice(_SOURCES),
            place=random.choice(_PLACES),
        ) + " " + random.choice(_TEMPLATES).format(
            topic=random.choice(_TOPICS),
            group=random.choice(_GROUPS),
            claim=random.choice(_CLAIMS),
            source=random.choice(_SOURCES),
            place=random.choice(_PLACES),
        )

    texts = [_make_text() for _ in range(n)]

    def _write(path: Path, rows: list, fieldnames: list) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    _write(DATA_DIR / "bias.csv",
           [{"text": t, "bias_label": random.randint(0, 2)} for t in texts],
           ["text", "bias_label"])

    _write(DATA_DIR / "ideology.csv",
           [{"text": t, "ideology_label": random.randint(0, 2)} for t in texts],
           ["text", "ideology_label"])

    _write(DATA_DIR / "propaganda.csv",
           [{"text": t, "propaganda_label": random.randint(0, 1)} for t in texts],
           ["text", "propaganda_label"])

    _write(DATA_DIR / "frame.csv",
           [{"text": t, "CO": random.randint(0, 1), "EC": random.randint(0, 1),
             "HI": random.randint(0, 1), "MO": random.randint(0, 1), "RE": random.randint(0, 1)}
            for t in texts],
           ["text", "CO", "EC", "HI", "MO", "RE"])

    _write(DATA_DIR / "narrative.csv",
           [{"text": t, "hero": random.randint(0, 1), "villain": random.randint(0, 1),
             "victim": random.randint(0, 1)} for t in texts],
           ["text", "hero", "villain", "victim"])

    emotion_cols = [f"emotion_{i}" for i in range(11)]
    _write(DATA_DIR / "emotion.csv",
           [{"text": t, **{f"emotion_{i}": random.randint(0, 1) for i in range(11)}}
            for t in texts],
           ["text"] + emotion_cols)

    logger.info("Synthetic test data written to %s  (%d articles × 6 datasets)", DATA_DIR, n)


def main() -> None:
    args = parse_args()
    configure_logging()

    # ----------------------------------------------------------
    # 0. OPTIONAL: GENERATE SYNTHETIC DATA  (fast, stdlib only)
    # ----------------------------------------------------------
    if args.generate_data:
        _generate_synthetic_data(n=max(args.samples * 2, 50))

    # ----------------------------------------------------------
    # 1. CHECK DATASETS EXIST  (fast CSV check, NO heavy imports yet)
    # ----------------------------------------------------------
    _hdr("1 / 6  LOADING DATASETS")

    all_texts: Dict[str, List[str]] = {}
    all_labels: Dict[str, Any] = {}
    missing: List[str] = []

    _csv_map = {
        "bias": DATA_DIR / "bias.csv",
        "ideology": DATA_DIR / "ideology.csv",
        "propaganda": DATA_DIR / "propaganda.csv",
        "narrative_frame": DATA_DIR / "frame.csv",
        "narrative": DATA_DIR / "narrative.csv",
        "emotion": DATA_DIR / "emotion.csv",
    }
    for task, loader in DATASET_LOADERS.items():
        csv_path = _csv_map[task]
        if not csv_path.exists():
            logger.warning("  ✗  %s — file not found: %s  (use --generate-data to create synthetic CSVs)", task, csv_path)
            missing.append(task)
            continue
        try:
            texts, labels = loader(args.samples)
            all_texts[task] = texts
            all_labels[task] = labels_to_eval_fmt(task, labels)
            shape = labels.shape if hasattr(labels, "shape") else len(labels)
            logger.info("  ✓  %-16s %d samples  labels shape=%s", task, len(texts), shape)
        except Exception as exc:
            logger.error("  ✗  %s — load failed: %s", task, exc, exc_info=True)
            missing.append(task)

    if missing:
        logger.warning("Skipped datasets: %s", missing)
    if not all_texts:
        logger.error("No datasets loaded — aborting.  Tip: run with --generate-data to create synthetic test CSVs.")
        sys.exit(1)

    # ----------------------------------------------------------
    # Heavy imports only reach here when there IS data to process.
    # This avoids 20+ seconds of PyTorch startup on empty runs.
    # ----------------------------------------------------------
    import torch
    from transformers import AutoTokenizer
    from src.pipelines.truthlens_pipeline import TruthLensPipeline
    from src.explainability.orchestrator import ExplainabilityConfig
    from src.evaluation.evaluation_pipeline import run_evaluation_pipeline

    config = load_config(CONFIG_PATH)
    set_seed(config.project.seed)

    logger.info("=" * 60)
    logger.info("TruthLens  End-to-End Test Pipeline")
    logger.info("datasets  : %s", list(DATASET_LOADERS))
    logger.info("samples   : %d per dataset", args.samples)
    _expl_mode = "full-LIME" if args.full_explainability else ("fast" if args.explainability else "off")
    logger.info("explainability: %s", _expl_mode)
    logger.info("=" * 60)

    # ----------------------------------------------------------
    # 2. BUILD PIPELINE
    # ----------------------------------------------------------
    _hdr("2 / 6  BUILDING PIPELINE")

    tokenizer = AutoTokenizer.from_pretrained(config.model.encoder)
    logger.info("Tokenizer loaded: %s", config.model.encoder)

    predictor: Optional[Any] = None
    ckpt_path = Path("saved_models/checkpoint.pt")
    if ckpt_path.is_file():
        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            from src.models.inference.predictor import Predictor
            model_obj = state.get("model") if isinstance(state, dict) else None
            if isinstance(model_obj, torch.nn.Module):
                predictor = Predictor(model=model_obj)
                logger.info("✓  Predictor loaded from %s", ckpt_path)
            else:
                logger.warning("Checkpoint found but contains no nn.Module — prediction disabled")
        except Exception as exc:
            logger.warning("Checkpoint load failed (%s) — prediction disabled", exc)
    else:
        logger.warning(
            "No checkpoint at %s — analysis/aggregation/graph will run; "
            "prediction/evaluation/explainability require a trained model. "
            "Run `uv run python main.py --mode train` first.", ckpt_path
        )

    _enable_expl = args.explainability or args.full_explainability
    if _enable_expl:
        if args.full_explainability:
            # Full mode: LIME (25 samples) + IG (8 steps) + attention rollout
            _expl_cfg = ExplainabilityConfig(
                use_lime=True,
                use_shap=False,
                use_attention_rollout=True,
                use_bias_emotion=True,
                use_graph_explainer=False,
                use_explanation_metrics=False,
                ig_steps=8,
            )
        else:
            # Fast mode: attention rollout only (1 forward pass) — no LIME, no IG
            _expl_cfg = ExplainabilityConfig(
                use_lime=False,
                use_shap=False,
                use_attention_rollout=True,
                use_bias_emotion=True,
                use_graph_explainer=False,
                use_explanation_metrics=False,
                ig_steps=0,
            )
    else:
        _expl_cfg = None

    pipeline = TruthLensPipeline(
        predictor=predictor,
        tokenizer=tokenizer,
        model_version=config.model.encoder,
        enable_explainability=_enable_expl,
        enable_evaluation=predictor is not None,
        explainability_config=_expl_cfg,
        parallel_stages=not args.no_parallel,
    )
    logger.info("Pipeline ready  (explainability=%s  evaluation=%s  parallel=%s)",
                _expl_mode, predictor is not None, not args.no_parallel)

    # ----------------------------------------------------------
    # 3. RUN INFERENCE + ANALYSIS + AGGREGATION per dataset
    # ----------------------------------------------------------
    _hdr("3 / 6  INFERENCE · ANALYSIS · AGGREGATION")

    dataset_results: Dict[str, Dict[str, Any]] = {}
    total_articles = 0
    t_pipeline_start = time.time()

    for task, texts in all_texts.items():
        logger.info("\n  ── Dataset: %s (%d texts) ──", task, len(texts))
        labels_for_eval = {task: all_labels[task]} if predictor is not None else None

        try:
            batch = pipeline.analyze_batch(texts, labels=labels_for_eval)
        except Exception as exc:
            logger.error("  analyze_batch failed for %s: %s", task, exc, exc_info=True)
            continue

        dataset_results[task] = batch
        total_articles += len(texts)

        for i, (text, result) in enumerate(zip(texts, batch["articles"])):
            _print_article(i, text, result)

        meta = batch.get("batch_metadata", {})
        logger.info("  ✓  %s complete | %d articles | %.2fs total",
                    task, meta.get("n_articles", 0), meta.get("total_time", 0))

    pipeline_elapsed = time.time() - t_pipeline_start
    logger.info("\n  Pipeline total: %d articles in %.2fs (%.0f ms/article)",
                total_articles, pipeline_elapsed,
                pipeline_elapsed / max(1, total_articles) * 1000)

    # ----------------------------------------------------------
    # 4. EVALUATION  (dataset-level, requires trained model)
    # ----------------------------------------------------------
    _hdr("4 / 6  EVALUATION")

    if predictor is None:
        logger.warning(
            "Evaluation skipped — no trained model loaded.\n"
            "  Train first:  uv run python main.py --mode train\n"
            "  Then re-run:  uv run python run_test_pipeline.py"
        )
    else:
        # Run evaluation once per dataset so each call gets its own texts +
        # labels dict.  This avoids the KeyError that arose when evaluation
        # defaulted to all TASK_CONFIG tasks but only one task had labels,
        # and the shape mismatch from mixing datasets with different lengths.
        t_eval_total = time.time()
        for task, texts in all_texts.items():
            if task not in all_labels:
                continue
            try:
                t_eval = time.time()
                eval_report = run_evaluation_pipeline(
                    model=getattr(predictor, "model", None),
                    tokenizer=tokenizer,
                    texts=texts,
                    labels={task: all_labels[task]},
                    tasks=[task],
                    enable_calibration=True,
                    enable_threshold_opt=True,
                    enable_uncertainty=True,
                    enable_error_analysis=True,
                    enable_correlation=False,
                )
                logger.info("  Eval [%s] complete in %.2fs", task, time.time() - t_eval)
                tasks_report = eval_report.get("tasks") or {}
                for task_name, task_report in tasks_report.items():
                    _print_eval_report(task_name, task_report)
            except Exception as exc:
                logger.error("  Eval [%s] failed: %s", task, exc, exc_info=True)
        logger.info("Evaluation total: %.2fs", time.time() - t_eval_total)

    # ----------------------------------------------------------
    # 5. EXPLAINABILITY SUMMARY
    # ----------------------------------------------------------
    _hdr("5 / 6  EXPLAINABILITY")

    _EXPL_METHOD_FIELDS = [
        ("shap", "shap_explanation"),
        ("lime", "lime_explanation"),
        ("attention", "attention_explanation"),
        ("propaganda", "propaganda_explanation"),
        ("bias", "bias_explanation"),
        ("emotion", "emotion_explanation"),
    ]

    expl_count = 0
    for task, batch in dataset_results.items():
        for article in batch.get("articles", []):
            expl = article.get("explainability")
            if expl:
                expl_count += 1
                methods_used = [
                    label for label, field in _EXPL_METHOD_FIELDS
                    if expl.get(field) is not None
                ]
                agg = expl.get("aggregated_explanation")
                top_tokens = []
                if isinstance(agg, dict):
                    top_tokens = (agg.get("tokens") or [])[:5]
                logger.info(
                    "  %s | methods=%s%s",
                    task,
                    methods_used,
                    f"  top_tokens={top_tokens}" if top_tokens else "",
                )

    if expl_count == 0:
        reason = "no trained model" if predictor is None else "explainability flag not set"
        logger.info("  No explanations generated (%s).", reason)
        if not _enable_expl:
            logger.info(
                "  Re-run with --explainability (fast: attention+bias) or "
                "--full-explainability (includes LIME, slower) to enable explanations."
            )

    # ----------------------------------------------------------
    # 6. AGGREGATION SCORE SUMMARY
    # ----------------------------------------------------------
    _hdr("6 / 6  AGGREGATION SCORE SUMMARY")

    score_rows: List[Dict[str, Any]] = []
    for task, batch in dataset_results.items():
        for i, article in enumerate(batch.get("articles", [])):
            agg = article.get("aggregation") or {}
            raw_scores = agg.get("scores") or agg.get("raw_scores") or {}
            scores_flat = article.get("scores") or {}
            merged = {**raw_scores, **scores_flat}
            credibility = merged.get("credibility_score") or merged.get("credibility")
            errors = list((article.get("errors") or {}).keys())
            score_rows.append({
                "dataset": task,
                "sample": i + 1,
                "credibility": round(float(credibility), 3) if credibility is not None else None,
                "n_errors": len(errors),
                "error_stages": errors if errors else [],
            })

    if score_rows:
        df = pd.DataFrame(score_rows)
        with pd.option_context("display.max_columns", None, "display.width", 120):
            print(df.to_string(index=False))

        by_dataset = df.groupby("dataset")["credibility"].agg(["mean", "min", "max"]).round(3)
        print("\n  Credibility by dataset:")
        print(by_dataset.to_string())

        total_errors = df["n_errors"].sum()
        print(f"\n  Total stage errors across all articles: {total_errors}")
        if total_errors == 0:
            print("  ✅ All pipeline stages completed without errors.")
        else:
            err_df = df[df["n_errors"] > 0][["dataset", "sample", "error_stages"]]
            print(err_df.to_string(index=False))

    # ----------------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------------
    _hdr("PIPELINE COMPLETE")
    logger.info("Datasets processed : %d / %d", len(dataset_results), len(DATASET_LOADERS))
    logger.info("Articles analysed  : %d", total_articles)
    logger.info("Wall time          : %.2fs", pipeline_elapsed)
    logger.info("Model checkpoint   : %s", "loaded" if predictor else "not found (train first)")
    logger.info("Evaluation         : %s", "ran" if predictor else "skipped")
    logger.info("Explainability     : %s", "ran" if (args.explainability and predictor) else "skipped")

    if predictor is None:
        logger.info("\n  Next step → train the model:")
        logger.info("    uv run python main.py --mode train")
        logger.info("  Then re-run with full evaluation + explainability:")
        logger.info("    uv run python run_test_pipeline.py --explainability")


if __name__ == "__main__":
    main()
