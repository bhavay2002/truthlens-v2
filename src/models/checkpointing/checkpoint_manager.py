from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.distributed as dist

from .validator import validate_checkpoint

# NEW MODULAR IMPORTS
from .io_utils import atomic_save, safe_load, ensure_dir, fsync_dir
from .integrity import attach_integrity_metadata, verify_from_metadata
from .metadata import save_metadata, load_metadata
from .resolver import resolve
from .selection import update_best_checkpoint
from .schema import attach_schema, prepare_checkpoint
from .loader_utils import move_optimizer_to_device

logger = logging.getLogger(__name__)

CHECKPOINT_FILE = "checkpoint.pt"


# =====================================================
# DISTRIBUTED SAFE
# =====================================================

def _is_primary():
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


# =====================================================
# MANAGER (V2)
# =====================================================

class CheckpointManager:

    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        ensure_dir(self.checkpoint_dir)

    # =====================================================
    # SAVE
    # =====================================================

    def save(
        self,
        *,
        step: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        save_best: bool = False,
        metric_name: str = "val_loss",
        mode: str = "min",
        compress: bool = False,
    ) -> Optional[Path]:

        if not _is_primary():
            return None

        ckpt_dir = self.checkpoint_dir / f"checkpoint-{step}"
        ensure_dir(ckpt_dir)

        path = ckpt_dir / CHECKPOINT_FILE

        # -------------------------
        # BUILD STATE
        # -------------------------

        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
            "config": config,
            "loss": (metrics or {}).get("loss"),
            "pytorch_version": torch.__version__,
        }

        # -------------------------
        # VALIDATE
        # -------------------------

        validate_checkpoint(state["model_state_dict"])

        # -------------------------
        # SCHEMA ATTACH
        # -------------------------

        state = attach_schema(state)

        # -------------------------
        # ATOMIC SAVE
        # -------------------------

        atomic_save(state, path, compress=compress)

        # -------------------------
        # FSYNC DIRECTORY (durability)
        # -------------------------

        fsync_dir(ckpt_dir)

        # -------------------------
        # INTEGRITY METADATA
        # -------------------------

        integrity_meta = attach_integrity_metadata(path)

        # -------------------------
        # SAVE METADATA.JSON
        # -------------------------

        meta = {
            "step": step,
            "epoch": epoch,
            "metrics": metrics or {},
            "config": config,
            **integrity_meta,
        }

        save_metadata(ckpt_dir, meta)

        # -------------------------
        # BEST MODEL TRACKING
        # -------------------------

        if save_best and metrics and metric_name in metrics:
            update_best_checkpoint(
                checkpoint_path=path,
                metric=float(metrics[metric_name]),
                metric_name=metric_name,
                mode=mode,
            )

        logger.info("Checkpoint saved: %s", ckpt_dir)

        return ckpt_dir

    # =====================================================
    # LOAD
    # =====================================================

    def load(
        self,
        model: torch.nn.Module,
        *,
        path: str | Path,
        optimizer=None,
        scheduler=None,
        scaler=None,
        strict: bool = True,
        map_location="cpu",
    ) -> Dict[str, Any]:

        # -------------------------
        # RESOLVE PATH
        # -------------------------

        path = resolve(path)

        logger.info("Loading checkpoint: %s", path)

        # -------------------------
        # LOAD FILE
        # -------------------------

        checkpoint = safe_load(path, map_location=map_location)

        # -------------------------
        # SCHEMA VALIDATION + MIGRATION
        # -------------------------

        checkpoint = prepare_checkpoint(checkpoint)

        # -------------------------
        # VERIFY INTEGRITY
        # -------------------------

        meta_path = path.parent / "metadata.json"

        if meta_path.exists():
            meta = load_metadata(meta_path)
            verify_from_metadata(path, meta)

        # -------------------------
        # LOAD MODEL  (C1.8: backup-and-rollback)
        # -------------------------
        #
        # Previously we always called ``load_state_dict(strict=False)``
        # and then post-checked the missing/unexpected key lists. Two
        # problems with that ordering:
        #
        #   • ``strict=False`` already mutates the model in place — by
        #     the time we discover the schema mismatch we have already
        #     overwritten any keys that *did* match. Raising at that
        #     point leaves the caller with a half-loaded model whose
        #     parameters are an unstructured mix of old and new.
        #
        #   • The caller's explicit choice of ``strict=True`` was
        #     silently downgraded — schema validation happened *after*
        #     the unsafe load, not before it.
        #
        # The fix is to honour ``strict`` at the ``load_state_dict``
        # call site, take a defensive snapshot of the current
        # parameters and buffers first, and roll the snapshot back if
        # the load raises.

        state_dict = checkpoint["model_state_dict"]

        backup_state = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }

        try:
            result = model.load_state_dict(state_dict, strict=strict)

            # In permissive mode we still log mismatches loudly so
            # silent partial loads remain auditable.
            if not strict:
                if result.missing_keys:
                    logger.warning(
                        "Partial load — %d missing key(s): %s",
                        len(result.missing_keys),
                        result.missing_keys,
                    )
                if result.unexpected_keys:
                    logger.warning(
                        "Partial load — %d unexpected key(s): %s",
                        len(result.unexpected_keys),
                        result.unexpected_keys,
                    )

        except Exception:
            # Restore the pre-load state so the caller's model object
            # is never left in a half-overwritten configuration.
            logger.exception(
                "load_state_dict raised; rolling model back to pre-load state."
            )
            model.load_state_dict(backup_state, strict=True)
            raise

        # -------------------------
        # OPTIMIZER
        # -------------------------

        if optimizer and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            move_optimizer_to_device(optimizer, next(model.parameters()).device)

        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if scaler and checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        return checkpoint

    # =====================================================
    # RESUME
    # =====================================================

    def resume_latest(
        self,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
    ):

        latest = self.latest()

        if latest is None:
            return None

        ckpt = self.load(
            model,
            path=latest,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

        return {
            "epoch": ckpt.get("epoch", 0),
            "step": ckpt.get("step", 0),
        }

    # =====================================================
    # LIST / RESOLVE
    # =====================================================

    def list_checkpoints(self) -> List[Path]:
        return sorted(
            [p for p in self.checkpoint_dir.iterdir() if p.is_dir()],
            key=lambda x: int(x.name.split("-")[-1]),
        )

    def latest(self) -> Optional[Path]:
        ckpts = self.list_checkpoints()
        return ckpts[-1] if ckpts else None

    # =====================================================
    # CLEANUP
    # =====================================================

    def cleanup(self, keep: int = 3):

        ckpts = self.list_checkpoints()

        for p in ckpts[:-keep]:
            logger.info("Removing old checkpoint: %s", p)
            import shutil
            shutil.rmtree(p, ignore_errors=True)

# Backward-compat alias used by src.training.trainer.
CheckpointEngine = CheckpointManager

