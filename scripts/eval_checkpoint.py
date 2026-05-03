"""
eval_checkpoint.py
------------------
Run the full TruthLens evaluation pipeline using real test data from data/test/
and a saved model checkpoint.

Usage
-----
    # auto-detect checkpoint, evaluate all 6 test datasets
    uv run python scripts/eval_checkpoint.py

    # explicit checkpoint
    uv run python scripts/eval_checkpoint.py --checkpoint saved_models/checkpoint.pt

    # custom test directory
    uv run python scripts/eval_checkpoint.py --test-dir data/test

    # save JSON report
    uv run python scripts/eval_checkpoint.py --report reports/eval_report.json

    # just show what test files are present (no model needed)
    uv run python scripts/eval_checkpoint.py --status
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.task_config import TASK_CONFIG
from src.data_processing.test_loader import TestDataLoader
from src.evaluation.evaluate_model import evaluate

logging.basicConfig(
    format="%(levelname)s %(name)s — %(message)s",
    level=logging.WARNING,
)
logger = logging.getLogger("eval_checkpoint")


# ─────────────────────────────────────────────────────────────
# CHECKPOINT DISCOVERY
# ─────────────────────────────────────────────────────────────

def find_checkpoint(hint: str | None) -> Path:
    if hint:
        p = Path(hint)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    search_root = ROOT / "saved_models"
    for name in ("checkpoint.pt", "model.pt", "best_model.pt"):
        p = search_root / name
        if p.exists():
            return p

    if search_root.exists():
        step_dirs = sorted(
            (d for d in search_root.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")),
            key=lambda d: int(d.name.split("-")[1]) if d.name.split("-")[1].isdigit() else 0,
            reverse=True,
        )
        for d in step_dirs:
            for name in ("checkpoint.pt", "model.pt"):
                p = d / name
                if p.exists():
                    return p

    raise FileNotFoundError(
        "No checkpoint found in saved_models/.\n"
        "  Copy your checkpoint.pt to saved_models/ and re-run,\n"
        "  or pass --checkpoint <path>."
    )


# ─────────────────────────────────────────────────────────────
# CHECKPOINT LOADING
# ─────────────────────────────────────────────────────────────

def load_checkpoint(path: Path) -> dict:
    print(f"\nLoading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    top_keys = list(ckpt.keys()) if isinstance(ckpt, dict) else ["<tensor>"]
    print(f"  keys: {top_keys}")
    return ckpt


# ─────────────────────────────────────────────────────────────
# MODEL + TOKENIZER
# ─────────────────────────────────────────────────────────────

def build_model(ckpt: dict):
    from src.models.architectures.hybrid_truthlens_model import HybridTruthLensModel
    from src.models.config.model_config import MultiTaskModelConfig, ModelConfigLoader

    cfg_path = ROOT / "config" / "model_config.yaml"
    cfg = ModelConfigLoader().load(cfg_path) if cfg_path.exists() else MultiTaskModelConfig()

    model = HybridTruthLensModel(cfg)

    state_key = next(
        (k for k in ("model_state_dict", "state_dict", "model") if k in ckpt),
        None,
    )
    state_dict = ckpt[state_key] if state_key else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys (%d): %s%s", len(missing), missing[:5],
                       "..." if len(missing) > 5 else "")
    if unexpected:
        logger.warning("Unexpected keys (%d): %s%s", len(unexpected), unexpected[:5],
                       "..." if len(unexpected) > 5 else "")
    model.eval()
    print(f"  Model loaded  (missing={len(missing)}  unexpected={len(unexpected)})")
    return model


def load_tokenizer(ckpt: dict):
    from transformers import AutoTokenizer
    name = ckpt.get("tokenizer_name") or ckpt.get("model_name") or "roberta-base"
    print(f"  Tokenizer: {name}")
    return AutoTokenizer.from_pretrained(name)


# ─────────────────────────────────────────────────────────────
# INFERENCE  (per-dataset, so texts vary per task)
# ─────────────────────────────────────────────────────────────

def run_inference(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 16,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """
    Returns dict {task: logits_array (N, C)}.
    Handles both dict-output and ModelOutput-with-.logits models.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    accumulated: dict[str, list] = {t: [] for t in TASK_CONFIG}

    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            outputs = model(**enc)

            if isinstance(outputs, dict):
                for task, lg in outputs.items():
                    if task in accumulated:
                        accumulated[task].append(lg.detach().cpu().float().numpy())
            elif hasattr(outputs, "logits"):
                lg = outputs.logits
                if isinstance(lg, dict):
                    for task, arr in lg.items():
                        if task in accumulated:
                            accumulated[task].append(arr.detach().cpu().float().numpy())
                else:
                    raise RuntimeError(
                        "Model returned a single logit tensor, not per-task dict. "
                        "Check that HybridTruthLensModel returns {'task': logits, ...}."
                    )
            else:
                raise RuntimeError(f"Unknown model output type: {type(outputs)}")

    return {t: np.concatenate(v) for t, v in accumulated.items() if v}


# ─────────────────────────────────────────────────────────────
# POSTPROCESS  logits → predictions + probabilities
# ─────────────────────────────────────────────────────────────

def postprocess(
    task: str,
    logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.special import softmax, expit

    cfg = TASK_CONFIG.get(task, {})
    ttype = cfg.get("type", "multiclass")

    if ttype == "multilabel":
        probs = expit(logits)
        preds = (probs >= 0.5).astype(int)
    else:
        probs = softmax(logits, axis=-1)
        preds = np.argmax(probs, axis=1)

    return preds, probs


# ─────────────────────────────────────────────────────────────
# EVALUATE ONE TASK
# ─────────────────────────────────────────────────────────────

def evaluate_task(
    task: str,
    texts: list[str],
    labels: dict[str, np.ndarray],
    model,
    tokenizer,
    batch_size: int,
) -> dict:
    logits_map = run_inference(model, tokenizer, texts, batch_size=batch_size)

    if task not in logits_map:
        return {"error": f"Model did not produce logits for task '{task}'"}

    preds, probs = postprocess(task, logits_map[task])
    y_true = labels[task]

    try:
        result = evaluate(y_true=y_true, y_pred=preds, y_proba=probs, task=task)
        return result.get("metrics", result)
    except Exception as exc:
        logger.warning("Evaluation failed for %s: %s", task, exc)
        return {"error": str(exc)}


# ─────────────────────────────────────────────────────────────
# REPORT PRINTER
# ─────────────────────────────────────────────────────────────

def print_report(results: dict[str, dict]) -> None:
    print("\n" + "=" * 65)
    print("  TruthLens Evaluation Report")
    print("=" * 65)

    for task, m in results.items():
        cfg = TASK_CONFIG.get(task, {})
        ttype = cfg.get("type", "?")
        n = m.get("n_samples", "?")

        if "error" in m:
            print(f"\n[{task:<20}]  ERROR: {m['error']}")
            continue

        acc  = m.get("accuracy", m.get("subset_accuracy", "n/a"))
        f1   = m.get("f1_macro", m.get("f1", "n/a"))
        prec = m.get("precision", "n/a")
        rec  = m.get("recall", "n/a")
        fmt  = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)

        print(
            f"\n[{task:<20}]  type={ttype:<11}  n={n}\n"
            f"    acc={fmt(acc)}   f1={fmt(f1)}"
            f"   prec={fmt(prec)}   rec={fmt(rec)}"
        )

    print("\n" + "=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate TruthLens checkpoint on real test data")
    ap.add_argument("--checkpoint", default=None,
                    help="Path to checkpoint.pt (auto-detected from saved_models/ if omitted)")
    ap.add_argument("--test-dir", default="data/test",
                    help="Directory containing the 6 test dataset files (default: data/test)")
    ap.add_argument("--report", default=None,
                    help="Optional path to write JSON report (e.g. reports/eval_report.json)")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Inference batch size (default: 16; reduce to 8 if OOM)")
    ap.add_argument("--status", action="store_true",
                    help="Show which test files are present/missing and exit (no model needed)")
    args = ap.parse_args()

    # ── status check only ─────────────────────────────────────
    loader = TestDataLoader(args.test_dir)
    if args.status:
        loader.summary()
        return

    # ── load test datasets ────────────────────────────────────
    print(f"\nLoading test datasets from: {args.test_dir}")
    datasets = loader.load_all()

    if not datasets:
        print("\nNo test datasets could be loaded. "
              "Upload your CSV/JSON files to data/test/ and re-run.")
        sys.exit(1)

    # ── load checkpoint + model ───────────────────────────────
    ckpt_path = find_checkpoint(args.checkpoint)
    ckpt = load_checkpoint(ckpt_path)
    model = build_model(ckpt)
    tokenizer = load_tokenizer(ckpt)

    # ── evaluate each task independently ─────────────────────
    all_results: dict[str, dict] = {}

    for task, (texts, labels) in datasets.items():
        print(f"\n── {task} ({len(texts)} samples) ──")
        all_results[task] = evaluate_task(
            task=task,
            texts=texts,
            labels=labels,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
        )
        m = all_results[task]
        if "error" not in m:
            acc = m.get("accuracy", m.get("subset_accuracy", "n/a"))
            f1  = m.get("f1_macro", m.get("f1", "n/a"))
            fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"   acc={fmt(acc)}   f1={fmt(f1)}")

    # ── print full report ─────────────────────────────────────
    print_report(all_results)

    # ── save JSON ─────────────────────────────────────────────
    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_results, indent=2, default=str))
        print(f"Report saved → {out}")


if __name__ == "__main__":
    main()
