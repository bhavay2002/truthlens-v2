from __future__ import annotations

# TOKENIZERS-FORK-FIX: HuggingFace's fast tokenizers spawn a Rust thread
# pool eagerly; when the DataLoader later forks worker processes the
# child inherits a poisoned tokenizer and prints
# "huggingface/tokenizers: process just got forked..." on every step.
# In the worst case the child deadlocks on the parent's lock. Setting
# this BEFORE ``transformers`` is imported is the only reliable fix —
# setting it after import is silently ignored. Override with
# ``TOKENIZERS_PARALLELISM=true`` in the environment if a caller knows
# the dataloader is single-process.
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoTokenizer

from src.config.settings_loader import load_settings
from src.config.config_loader import load_config

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "config.yaml"

from src.data_processing.data_pipeline import run_data_pipeline, DataPipelineConfig
from src.data_processing.dataloader_factory import DataLoaderConfig
from src.training.create_trainer_fn import create_trainer_fn
from src.training.create_multitask_trainer_fn import create_multitask_trainer_fn
from src.pipelines.truthlens_pipeline import TruthLensPipeline
from src.utils.logging_utils import configure_logging
from src.utils.seed_utils import set_seed

logger = logging.getLogger(__name__)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="truthlens",
        description=(
            "TruthLens entry point. Use --mode infer to run the "
            "analysis pipeline on a few sample texts without loading "
            "any training data; --mode train runs the data + training "
            "stages only; --mode both does both."
        ),
    )
    parser.add_argument("--mode", choices=("train", "infer", "both"), default="infer")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--enable-explainability", action="store_true")
    parser.add_argument("--enable-evaluation", action="store_true")
    parser.add_argument("--no-parallel-stages", action="store_true")
    parser.add_argument(
        "--labels-file",
        type=str,
        default=None,
        help=(
            "Path to a JSON file containing ground-truth labels for the "
            "sample texts. Required for --enable-evaluation to produce real "
            "metrics. Format: {\"bias\": [0,1,...], \"emotion\": [2,3,...], "
            "\"propaganda\": [0,1,...], \"narrative\": [0,1,...], "
            "\"narrative_frame\": [0,1,...], \"ideology\": [0,1,...]}. "
            "The list length must match --num-samples."
        ),
    )
    return parser.parse_args()

def main():
    args = _parse_args()
    try:
        settings = load_settings(validate_data=args.mode in ("train", "both"))
        config = load_config(CONFIG_PATH)
        configure_logging()
        set_seed(config.project.seed)
        logger.info(" TruthLens System Started | mode=%s", args.mode)
        tokenizer = AutoTokenizer.from_pretrained(config.model.encoder)
        if args.mode in ("train", "both"):
            data_config: Dict = {
                "bias": {"train": settings.data.get("bias", "train"), "val": settings.data.get("bias", "val"), "test": settings.data.get("bias", "test")},
                "ideology": {"train": settings.data.get("ideology", "train"), "val": settings.data.get("ideology", "val"), "test": settings.data.get("ideology", "test")},
                "propaganda": {"train": settings.data.get("propaganda", "train"), "val": settings.data.get("propaganda", "val"), "test": settings.data.get("propaganda", "test")},
                "narrative": {"train": settings.data.get("narrative", "train"), "val": settings.data.get("narrative", "val"), "test": settings.data.get("narrative", "test")},
                "narrative_frame": {"train": settings.data.get("narrative_frame", "train"), "val": settings.data.get("narrative_frame", "val"), "test": settings.data.get("narrative_frame", "test")},
                "emotion": {"train": settings.data.get("emotion", "train"), "val": settings.data.get("emotion", "val"), "test": settings.data.get("emotion", "test")},
            }
            loader_cfg = DataLoaderConfig.from_yaml_data(config.data)
            datasets = run_data_pipeline(
                data_config=data_config,
                tokenizer=tokenizer,
                build_dataloaders=False,
                config=DataPipelineConfig(enable_cache=True, dataloader_config=loader_cfg),
            )
            logger.info("✅ Data pipeline completed")

            save_dir = Path("saved_models")
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = save_dir / "checkpoint.pt"

            # MULTITASK-DEFAULT: train ALL heads in a single unified
            # MultiTaskTruthLensModel run (one shared roberta-base
            # encoder + per-task heads + true multi-task LossEngine
            # over every task at once). This replaces the legacy
            # per-task loop which:
            #   * recreated the encoder N times (5x roberta-base
            #     instantiated and trained sequentially → 5x compute,
            #     5x memory, zero shared representation)
            #   * built a LossEngine with len(task_configs)==1 every
            #     time, which the engine itself force-disables the
            #     EMA normalizer + coverage tracker for and prints
            #     "MT-1: LossEngine instantiated with 1 task(s)"
            #   * trained each task for ``TRUTHLENS_TRAIN_EPOCHS``
            #     epochs *independently*, defaulting to 1, so the
            #     "Epoch 1/1" log line was real — config.training.epochs
            #     was never read by the per-task path.
            # The single-task path is preserved behind
            # ``TRUTHLENS_USE_SINGLE_TASK=1`` strictly as an escape
            # hatch for the rare case someone needs to debug one head
            # in isolation. Everyone else gets the right architecture
            # by default.
            use_single_task = os.environ.get(
                "TRUTHLENS_USE_SINGLE_TASK", "0"
            ) == "1"

            # Honour env-var epoch override on top of YAML
            # ``training.epochs`` (configured to 4). The env var is the
            # canonical knob the trainer factories already accept and
            # is preserved here so existing CI / smoke-run scripts
            # continue to work. Without an override, YAML wins.
            epochs_override = os.environ.get("TRUTHLENS_TRAIN_EPOCHS")
            if epochs_override is not None:
                try:
                    config.training.epochs = int(epochs_override)
                    logger.info(
                        "Epochs overridden by TRUTHLENS_TRAIN_EPOCHS=%s",
                        epochs_override,
                    )
                except (TypeError, ValueError):
                    logger.warning(
                        "Ignoring non-integer TRUTHLENS_TRAIN_EPOCHS=%r",
                        epochs_override,
                    )

            if not use_single_task:
                # Pin enabled tasks to whatever the data pipeline
                # actually materialised so a missing dataset doesn't
                # blow up the trainer. ``MultiTaskTruthLensModel``'s
                # default head set covers every task in
                # ``config.yaml::tasks`` including ``narrative_frame``
                # (5-label) — which the legacy single-task factory
                # had no mapping for and silently skipped.
                enabled_tasks = list(datasets.keys())
                logger.info(
                    "🧠 Creating UNIFIED multi-task trainer | tasks=%s | "
                    "epochs=%d | shared_encoder=%s",
                    enabled_tasks,
                    int(config.training.epochs),
                    config.model.encoder,
                )
                trainer = create_multitask_trainer_fn(
                    settings=config,
                    data_bundle=datasets,
                    tokenizer=tokenizer,
                    enabled_tasks=enabled_tasks,
                    config_path=str(CONFIG_PATH),
                )
                logger.info("🔥 Training | unified multi-task")
                trainer.train()
                torch.save(
                    {
                        "model": trainer.model.cpu().eval(),
                        "task": "multitask",
                        "tasks": enabled_tasks,
                        "encoder": config.model.encoder,
                    },
                    ckpt_path,
                )
                logger.info(
                    "📦 Saved checkpoint → %s (multi-task, %d heads)",
                    ckpt_path, len(enabled_tasks),
                )
            else:
                # ───────── LEGACY SINGLE-TASK PATH (escape hatch) ─────────
                logger.warning(
                    "TRUTHLENS_USE_SINGLE_TASK=1 → falling back to legacy "
                    "per-task training. Encoder will be recreated for each "
                    "task and no shared representation will be learned."
                )
                # ``narrative_frame`` only exists inside the multitask spec
                # (a 5-label head); the single-task model factory has no
                # mapping for it, so skip it here with a clear note.
                SINGLE_TASK_SUPPORTED = {
                    "bias", "ideology", "propaganda", "narrative", "emotion",
                }
                unsupported = [t for t in datasets if t not in SINGLE_TASK_SUPPORTED]
                if unsupported:
                    logger.info(
                        "Skipping tasks not wired into single-task training: %s "
                        "(only the multitask path covers these heads)",
                        unsupported,
                    )

                trainers = {}
                for task in datasets:
                    if task not in SINGLE_TASK_SUPPORTED:
                        continue
                    logger.info("🧠 Creating trainer | task=%s", task)
                    trainer = create_trainer_fn(
                        task=task,
                        train_df=datasets[task]["train"],
                        val_df=datasets[task]["val"],
                        params={
                            "lr": float(config.optimizer.lr),
                            "batch_size": int(loader_cfg.batch_size),
                            "weight_decay": float(config.optimizer.weight_decay),
                            # CONFIG-PLUMBING-FIX: the legacy path used
                            # to default to 1 epoch unconditionally and
                            # ignored config.training.epochs entirely.
                            # Honour YAML now; env var still wins.
                            "epochs": int(os.environ.get(
                                "TRUTHLENS_TRAIN_EPOCHS",
                                str(int(config.training.epochs)),
                            )),
                            "num_workers": (
                                0
                                if os.environ.get(
                                    "TRUTHLENS_FORCE_SINGLE_WORKER"
                                ) == "1"
                                else int(loader_cfg.num_workers)
                            ),
                            "pin_memory": bool(loader_cfg.pin_memory),
                            "tokenizer": tokenizer,
                            "model_name": config.model.encoder,
                            "dropout": float(config.model.dropout),
                            "gradient_checkpointing": bool(config.model.gradient_checkpointing),
                            "use_compile": bool(config.model.torch_compile),
                            "compile_mode": str(config.model.compile_mode),
                            "amp": bool(config.precision.use_amp),
                            "amp_dtype": str(config.precision.amp_dtype),
                            "allow_tf32": bool(config.precision.allow_tf32),
                            "device": "cuda" if torch.cuda.is_available() else "cpu",
                        },
                    )
                    trainers[task] = trainer

                import gc
                for task, trainer in trainers.items():
                    logger.info("🔥 Training | task=%s", task)
                    trainer.train()
                    torch.save(
                        {
                            "model": trainer.model.cpu().eval(),
                            "task": task,
                            "encoder": config.model.encoder,
                        },
                        ckpt_path,
                    )
                    logger.info(
                        "📦 Saved checkpoint → %s (task=%s)", ckpt_path, task,
                    )
                    trainer.model = None
                    trainers[task] = None
                    del trainer
                    gc.collect()
        if args.mode in ("infer", "both"):
            logger.info("🧪 Running FULL TruthLens pipeline")
            model_version = getattr(getattr(config, "model", object()), "version", config.model.encoder)
            predictor = None
            try:
                from src.models.inference.predictor import Predictor
                from src.utils import settings as _runtime_settings
                runtime = _runtime_settings.load_settings()
                model_dir = Path(getattr(runtime.model, "path", "saved_models")).resolve()
                checkpoint_file = model_dir / "checkpoint.pt"
                if checkpoint_file.is_file():
                    logger.info(" Found checkpoint at %s — loading Predictor", checkpoint_file)
                    # ``weights_only=False`` is required because the
                    # checkpoint contains a *pickled* ``nn.Module``
                    # (``state["model"]``), not just a state-dict. Newer
                    # PyTorch defaults to ``weights_only=True`` which
                    # rejects arbitrary pickled objects. We control both
                    # the writer (``main.py --mode train``) and the
                    # reader, so opting out of the safe-load is fine.
                    state = torch.load(
                        checkpoint_file,
                        map_location="cpu",
                        weights_only=False,
                    )
                    multitask_model = state.get("model") if isinstance(state, dict) else None
                    if not isinstance(multitask_model, torch.nn.Module):
                        logger.warning(" Checkpoint at %s does not contain a serialised nn.Module under key 'model' — running without a predictor. Re-export the checkpoint with the training pipeline to enable prediction.", checkpoint_file)
                    else:
                        predictor = Predictor(model=multitask_model)
                        logger.info(" Predictor attached")
                else:
                    logger.warning(" No checkpoint at %s — running the analysis / features / aggregation stack only. Run `python main.py --mode train` (after placing the dataset CSVs under data/{train,val,test}/) to produce a checkpoint and unlock prediction.", checkpoint_file)
            except Exception:
                logger.exception(" Predictor load failed — continuing without prediction")
                predictor = None
            pipeline = TruthLensPipeline(predictor=predictor, tokenizer=tokenizer, model_version=model_version, enable_explainability=args.enable_explainability, enable_evaluation=args.enable_evaluation, parallel_stages=not args.no_parallel_stages)
            sample_texts = ["The government clearly failed the people.", "This is a neutral statement.", "The heroic leader saved the nation."][: max(1, args.num_samples)]

            # Load ground-truth labels for real evaluation metrics.
            labels = None
            if args.labels_file:
                import json as _json
                with open(args.labels_file) as _f:
                    labels = _json.load(_f)
                logger.info(" Labels loaded from %s", args.labels_file)
            elif args.enable_evaluation:
                logger.warning(
                    " --enable-evaluation is set but no --labels-file was provided. "
                    "Evaluation will be skipped and metrics will stay at 0.0. "
                    "Pass --labels-file path/to/labels.json to get real metrics. "
                    "Expected format: {\"bias\": [0,1,...], \"emotion\": [2,...], ...} "
                    "with one integer label per task per sample."
                )

            batch_result = pipeline.analyze_batch(sample_texts, labels=labels)
            logger.info(" BATCH SUMMARY: n=%d total_time=%.3fs model_version=%s", batch_result["batch_metadata"]["n_articles"], batch_result["batch_metadata"]["total_time"], batch_result["batch_metadata"]["model_version"])
            if batch_result.get("evaluation"):
                logger.info(" EVALUATION: %s", batch_result["evaluation"])
            for i, result in enumerate(batch_result["articles"]):
                logger.info(" RESULT %d:", i + 1)
                logger.info("  scores: %s", result.get("scores"))
                logger.info("  predictions keys: %s", list(result.get("predictions", {}).keys()))
                if result.get("errors"):
                    logger.warning("  stage errors: %s", result["errors"])
            pipeline.close()
        logger.info(" SYSTEM COMPLETED SUCCESSFULLY")
    except Exception as e:
        logger.error(" SYSTEM FAILED: %s", str(e), exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
