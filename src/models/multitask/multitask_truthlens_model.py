"""Canonical multi-task TruthLens model.

A single shared transformer encoder feeds a ``ModuleDict`` of task heads
(``ClassificationHead`` for multi-class targets, ``MultiLabelHead`` for
multi-label targets). The model can be constructed three ways:

  1. ``MultiTaskTruthLensModel(encoder, task_heads)`` — raw modules.
  2. ``MultiTaskTruthLensModel(config=MultiTaskTruthLensConfig(...))``
     — convenience path that builds both the encoder and a default set
     of TruthLens task heads from a small dataclass config. Used by the
     audit-issue tests and by `TruthLensMultiTaskModel`.
  3. ``MultiTaskTruthLensModel.from_model_config(MultiTaskModelConfig)``
     — full-fidelity path driven by the YAML-backed
     :class:`~src.models.config.MultiTaskModelConfig`. Used by the
     model registry and the inference engine.

The forward pass returns a dict that contains BOTH:

  * a per-task entry, e.g. ``outputs["bias"] = {"logits": ..., ...}``
    (the contract the test-evaluation pipeline relies on); and
  * an ``outputs["task_logits"]`` mapping ``{task_name: tensor}``
    (the contract the training step / loss / evaluation engines rely on).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional

import torch
import torch.nn as nn

from src.models.base.base_model import BaseModel


# =========================================================
# INFERENCE-CONTRACT-FIX V7 — explainability-facing heads view
#
# The explainability layer (``src.explainability.bias_explainer``,
# ``src.explainability.emotion_explainer``,
# ``src.aggregation.score_explainer``) calls into
# ``model.heads[task](cls)`` and expects the result to be a *tensor of
# logits* — see e.g. ``score_explainer._integrated_gradients`` which
# does ``logits = self.model.heads[task](cls)`` followed by
# ``logits[:, target_idx].sum()``.
#
# Internally, however, every TruthLens task head obeys the strict
# ``BaseHead`` contract (asserted in ``MultiTaskTruthLensModel.forward``)
# that requires returning a ``dict`` with at least a ``"logits"`` key.
# Calling ``self.task_heads[name](cls)`` therefore yields a dict, not a
# tensor, and the explainer crashes inside the very next tensor op.
#
# Pre-V7 the wrapper also only registered the heads under
# ``self.task_heads`` — there was no ``self.heads`` attribute at all,
# so ``bias_explainer._is_multitask`` (which keys on
# ``hasattr(model, "heads")``) returned False and the code fell through
# to the non-multitask branch, which then died on
# ``model.get_input_embeddings()`` (also missing).
#
# ``_HeadsLogitsView`` is a read-only mapping that wraps the
# ``task_heads`` ModuleDict and adapts each head into a callable that
# returns *just* the logits tensor. It does NOT register new modules
# (the wrappers are stored in a plain ``dict``), so it does not
# perturb ``model.parameters()`` or any optimizer state — it is a pure
# inference-side adapter that exists only to make the explainability
# contract line up with the heads' internal dict-output contract.
# =========================================================


class _HeadsLogitsView(Mapping):
    """Read-only mapping exposing each task head as a logits-returning callable.

    Parameters
    ----------
    task_heads:
        The live ``nn.ModuleDict`` of task heads. Held by reference so
        the view stays consistent if heads are added or replaced after
        construction. The view does not own the heads — it must not
        register them as submodules (that would cause double-counting
        in ``parameters()``).
    """

    def __init__(self, task_heads: nn.ModuleDict) -> None:
        self._task_heads = task_heads

    def __getitem__(self, key: str) -> Callable[[torch.Tensor], torch.Tensor]:
        head = self._task_heads[key]

        def _call(x: torch.Tensor) -> torch.Tensor:
            out = head(x)
            # The model-side contract guarantees this is a dict with
            # ``"logits"`` (see ``MultiTaskTruthLensModel.forward``),
            # but we still defend in case an explainer hands us a head
            # that has been monkey-patched in tests.
            if isinstance(out, torch.Tensor):
                return out
            if isinstance(out, dict) and "logits" in out:
                return out["logits"]
            raise RuntimeError(
                f"Head '{key}' returned an unsupported type for the "
                f"explainability contract: {type(out).__name__}"
            )

        return _call

    def __iter__(self) -> Iterator[str]:
        return iter(self._task_heads)

    def __len__(self) -> int:
        return len(self._task_heads)

    def __contains__(self, key: object) -> bool:
        return key in self._task_heads

logger = logging.getLogger(__name__)


# =========================================================
# CANONICAL TASK METADATA
# =========================================================

# A single source of truth for the default TruthLens head sizes and
# label vocabularies. ``MultiTaskTruthLensConfig.task_num_labels`` may
# override these per-task; everything below is just the fallback.

_DEFAULT_TASK_SPEC: Dict[str, Dict[str, Any]] = {
    "bias": {
        "task_type": "multi_class",
        "labels": ["non_bias", "bias"],
    },
    "ideology": {
        "task_type": "multi_class",
        "labels": ["left", "center", "right"],
    },
    "propaganda": {
        "task_type": "multi_class",
        "labels": ["non_propaganda", "propaganda"],
    },
    "narrative": {
        "task_type": "multi_label",
        "labels": ["hero", "villain", "victim"],
    },
    "narrative_frame": {
        "task_type": "multi_label",
        "labels": ["RE", "HI", "CO", "MO", "EC"],
    },
    "emotion": {
        "task_type": "multi_label",
        # EMOTION-11: reduced from 20 → 11 to match the canonical
        # schema in src/features/emotion/emotion_schema.py. Hardcoded
        # rather than imported here to avoid a heavy import in this
        # registry module; keep in sync with NUM_EMOTION_LABELS.
        "labels": [f"emotion_{i}" for i in range(11)],
    },
}


# =========================================================
# CONFIG
# =========================================================

@dataclass
class MultiTaskTruthLensConfig:
    """Lightweight config for the convenience-construction path.

    .. note::
       **A6 — naming caveat.** The TruthLens codebase has *two*
       multi-task configuration objects and they are NOT interchangeable:

         * :class:`MultiTaskTruthLensConfig` (this class) — a
           narrow, hand-tuned dataclass used by the convenience
           constructor ``MultiTaskTruthLensModel(config=...)``. Builds
           the canonical TruthLens task set with default head sizes;
           cannot express per-task type / loss / regression overrides.
         * :class:`~src.models.config.MultiTaskModelConfig` — the
           YAML-backed, fully-structured config used by the model
           registry, the inference engine and the training pipeline
           via :meth:`MultiTaskTruthLensModel.from_model_config`.

       Mixing them (e.g. passing a ``MultiTaskModelConfig`` as
       ``config=`` here) raises a ``TypeError`` rather than silently
       constructing the wrong model. Unknown keyword arguments are
       rejected by the underlying dataclass init.

    .. note::
       **CFG2 — strict validation.** Because this is a typed
       ``@dataclass`` with no ``**kwargs`` catch-all, any unknown
       attribute supplied at construction time fails immediately with
       ``TypeError``. There is no silent-drop / silent-coerce path:
       a typo in a YAML field surfaces as a load-time error rather
       than a model that quietly ignores the override. Update the
       dataclass schema (and the docs above) before adding a new
       field.
    """

    model_name: str = "roberta-base"
    pooling: str = "cls"
    dropout: float = 0.1
    device: Optional[str] = None

    # When True we build the underlying transformer from a HF *config*
    # rather than downloading pretrained weights — useful for fast unit
    # tests that just want a forward-pass-compatible model.
    init_from_config_only: bool = False

    # Per-task loss weights (consumed by downstream training code).
    bias_weight: float = 1.0
    ideology_weight: float = 1.0
    propaganda_weight: float = 1.0
    narrative_weight: float = 1.0
    emotion_weight: float = 1.0

    # Optional overrides for the default per-task head sizes.
    task_num_labels: Optional[Dict[str, int]] = None

    # Optional restriction to a subset of the canonical task list.
    enabled_tasks: Optional[List[str]] = None

    # Reserved for future per-task knobs (kept for forward-compat with
    # the old free-form `**kwargs` shim).
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


# =========================================================
# MULTI-TASK TRUTHLENS MODEL
# =========================================================

class MultiTaskTruthLensModel(BaseModel):
    """Shared-encoder, multi-headed TruthLens model.

    A3.5: inherits :class:`BaseModel` (not bare ``nn.Module``) so the
    multi-task model gets the same calibration-aware checkpoint helpers,
    cached ``device`` property and lazy ``attach_module`` discipline as
    every other production model in this codebase. Without that, the G4
    calibration-vs-base-weights split was silently bypassed for the
    flagship multi-task model.
    """

    # -----------------------------------------------------
    # Class-level metadata used by downstream label-helper code
    # and by the test suite (``tests/test_multitask_label_helpers.py``).
    # These mirror ``_DEFAULT_TASK_SPEC`` above.
    # -----------------------------------------------------

    BIAS_LABELS: List[str] = list(_DEFAULT_TASK_SPEC["bias"]["labels"])
    IDEOLOGY_LABELS: List[str] = list(_DEFAULT_TASK_SPEC["ideology"]["labels"])
    PROPAGANDA_LABELS: List[str] = list(_DEFAULT_TASK_SPEC["propaganda"]["labels"])
    NARRATIVE_LABELS: List[str] = list(_DEFAULT_TASK_SPEC["narrative"]["labels"])
    FRAME_LABELS: List[str] = list(_DEFAULT_TASK_SPEC["narrative_frame"]["labels"])
    EMOTION_LABELS: List[str] = list(_DEFAULT_TASK_SPEC["emotion"]["labels"])

    NUM_BIAS: int = len(BIAS_LABELS)
    NUM_IDEOLOGY: int = len(IDEOLOGY_LABELS)
    NUM_PROPAGANDA: int = len(PROPAGANDA_LABELS)
    NUM_NARRATIVE: int = len(NARRATIVE_LABELS)
    NUM_NARRATIVE_FRAMES: int = len(FRAME_LABELS)
    NUM_EMOTIONS: int = len(EMOTION_LABELS)

    # =====================================================
    # CONSTRUCTION
    # =====================================================

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        task_heads: Optional[Dict[str, nn.Module]] = None,
        *,
        config: Optional[MultiTaskTruthLensConfig] = None,
    ) -> None:
        super().__init__()

        # -------------------------------------------------
        # Convenience-config construction path
        # -------------------------------------------------
        if config is not None:
            if encoder is not None or task_heads is not None:
                raise ValueError(
                    "Pass either (encoder, task_heads) or `config=`, "
                    "not both."
                )

            if not isinstance(config, MultiTaskTruthLensConfig):
                raise TypeError(
                    "config must be a MultiTaskTruthLensConfig instance "
                    f"(got {type(config).__name__})"
                )

            encoder, task_heads = self._build_from_truthlens_config(config)
            self.config = config

        else:
            self.config = None

        # -------------------------------------------------
        # Validation (shared by both paths)
        # -------------------------------------------------
        if encoder is None:
            raise ValueError("encoder is required")

        if not isinstance(task_heads, dict) or not task_heads:
            raise ValueError("task_heads must be a non-empty dict")

        self.encoder = encoder
        self.task_heads = nn.ModuleDict(task_heads)

        logger.info(
            "MultiTaskTruthLensModel initialized | tasks=%s",
            list(self.task_heads.keys()),
        )

    # =====================================================
    # INFERENCE-CONTRACT-FIX V7 — explainability adapters
    #
    # See the comment block above ``_HeadsLogitsView`` for the full
    # rationale. In short:
    #
    #   * ``heads`` exposes a logits-returning view over
    #     ``self.task_heads`` so ``bias_explainer._is_multitask`` /
    #     ``score_explainer._integrated_gradients`` recognise this as
    #     the multi-task wrapper AND get a callable that returns a
    #     bare tensor (matching their ``logits = heads[task](cls)``
    #     contract).
    #
    #   * ``get_input_embeddings`` mirrors HuggingFace's
    #     ``PreTrainedModel.get_input_embeddings`` so the
    #     non-multitask explainer fallback path AND
    #     ``adversarial_training._embedding_weight`` (which keys on
    #     this method) work uniformly.
    #
    # Both are defined as a property / method (rather than set in
    # ``__init__``) so they always reflect the *current* state of
    # ``self.task_heads`` / ``self.encoder`` even after head
    # additions, swaps, or encoder hot-reloads.
    # =====================================================

    @property
    def heads(self) -> _HeadsLogitsView:
        """Logits-returning view of ``task_heads`` for explainers."""
        cached = getattr(self, "_heads_view", None)
        if cached is None or cached._task_heads is not self.task_heads:
            cached = _HeadsLogitsView(self.task_heads)
            self._heads_view = cached
        return cached

    def get_input_embeddings(self) -> nn.Module:
        """Return the underlying token-embedding module.

        Walks through the ``TransformerEncoder`` wrapper to reach the
        HuggingFace ``AutoModel.embeddings`` (or its
        ``word_embeddings`` sub-module if the wrapper exposes only
        the full embedding stack). Raises a clear error if neither
        path is available so the failure mode is greppable rather
        than the generic ``AttributeError`` the explainer used to
        surface.
        """
        encoder = self.encoder

        # Preferred path: ``TransformerEncoder.embeddings`` (also
        # added in INFERENCE-CONTRACT-FIX V7) → HF embedding module.
        emb = getattr(encoder, "embeddings", None)
        if emb is not None:
            # HF embedding modules expose ``word_embeddings`` as the
            # raw token-id → vector lookup; that is what
            # ``adversarial_training._embedding_weight`` and most
            # gradient-based attribution helpers actually want.
            we = getattr(emb, "word_embeddings", None)
            return we if we is not None else emb

        # Fallback: the wrapper is the HF model itself.
        getter = getattr(encoder, "get_input_embeddings", None)
        if callable(getter):
            return getter()

        raise AttributeError(
            "MultiTaskTruthLensModel.get_input_embeddings: encoder "
            f"({type(encoder).__name__}) does not expose an "
            "`embeddings` attribute or `get_input_embeddings` method."
        )

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, **inputs: Any) -> Dict[str, Any]:

        # -------------------------------------------------
        # ENCODER
        #
        # MT-MODEL-ENC-KWARG-FIX: previously this splat the *entire*
        # batch into the encoder — but the batch includes
        # ``labels`` (always, set by ``src.data_processing.collate``),
        # optional ``offset_mapping``, and any per-task auxiliary
        # tensors. ``TransformerEncoder.forward`` has a strict
        # ``(input_ids, attention_mask)`` signature and rejected the
        # extra kwargs with::
        #
        #     TypeError: TransformerEncoder.forward() got an
        #     unexpected keyword argument 'labels'
        #
        # The single-task model classes used to hide this because
        # they happened to declare ``forward(input_ids, attention_mask,
        # labels)``; the multi-task model is the new code path and
        # has to filter encoder inputs itself. ``training_step``'s
        # pre-filter still only strips ``task`` (its comment at line
        # 411 explicitly notes the single-task ``labels``-eating
        # behavior), so the encoder boundary is the right place.
        # -------------------------------------------------
        encoder_inputs = {
            k: inputs[k]
            for k in ("input_ids", "attention_mask")
            if k in inputs
        }
        encoder_outputs = self.encoder(**encoder_inputs)

        pooled = self._extract_pooled(encoder_outputs)

        # -------------------------------------------------
        # TASK HEADS
        # -------------------------------------------------
        outputs: Dict[str, Any] = {}

        for task_name, head in self.task_heads.items():
            try:
                head_output = head(pooled)
            except Exception as e:
                raise RuntimeError(
                    f"Head '{task_name}' forward failed: {e}"
                ) from e

            # A3.4: dict-only contract. Task heads must return a dict
            # with at least ``logits`` — see :class:`BaseHead`. The
            # previous tensor-fallback branch silently accepted broken
            # heads and only crashed deep inside calibration. Kept
            # behind two explicit checks so the failure mode names the
            # offending task.
            if not isinstance(head_output, dict):
                raise TypeError(
                    f"Head '{task_name}' must return a dict (got "
                    f"{type(head_output).__name__}); see "
                    f"src.models.heads.base_head.BaseHead."
                )
            if "logits" not in head_output:
                raise RuntimeError(
                    f"Head '{task_name}' dict missing 'logits' "
                    f"(keys={list(head_output)})"
                )

            outputs[task_name] = head_output

        # -------------------------------------------------
        # OUTPUT — A3.6: per-task entries are the source of truth.
        # ``task_logits`` was previously a parallel dict that was
        # populated alongside the per-task entries; downstream code
        # could mutate one and silently de-sync the other. We now
        # expose ``task_logits`` as a *thin view* computed from the
        # per-task entries on demand. The training pipeline still
        # reads ``outputs["task_logits"]`` (back-compat) but the data
        # has a single owner.
        # -------------------------------------------------
        outputs["task_logits"] = {
            name: outputs[name]["logits"] for name in self.task_heads.keys()
        }
        return outputs

    # =====================================================
    # ENCODER POOL HELPER
    # =====================================================

    @staticmethod
    def _extract_pooled(encoder_outputs: Any) -> torch.Tensor:
        """Strict pooled-embedding extraction (A6.1).

        Supports both raw ``dict`` outputs (our ``TransformerEncoder``
        wrapper returns ``{"pooled_output": ..., "sequence_output":
        ...}``) and HuggingFace-style ``ModelOutput`` objects.

        A6.1 — the previous implementation silently fell back to
        ``hidden[:, 0]`` (CLS-token slice) whenever the encoder did not
        publish a pooled output. That hid two real bugs:

          * a wrapper that *should* have produced a pooled output but
            forgot to do so (a regression in the encoder), and
          * a wrapper whose pooling strategy is something other than
            CLS (e.g. mean / attention) — falling back to CLS produced
            silently wrong embeddings.

        We now require an explicit pooled output. The callers that
        previously relied on the implicit CLS fallback should
        construct their encoder with ``pooling="cls"`` so the
        pooled-output channel is populated by the encoder itself.
        """

        # NB: ``a or b`` evaluates ``bool(a)``, which on a multi-element
        # tensor raises ``RuntimeError("Boolean value of Tensor with
        # more than one value is ambiguous")``. The keys must be probed
        # one at a time via explicit ``is None`` checks.
        if isinstance(encoder_outputs, dict):
            pooled = encoder_outputs.get("pooled_output")
            if pooled is None:
                pooled = encoder_outputs.get("pooler_output")
        else:
            pooled = getattr(encoder_outputs, "pooled_output", None)
            if pooled is None:
                pooled = getattr(encoder_outputs, "pooler_output", None)

        if pooled is None:
            raise RuntimeError(
                "Encoder did not return a pooled embedding "
                "(`pooled_output` / `pooler_output`); cannot feed task "
                "heads. Construct the encoder with an explicit pooling "
                "strategy (e.g. ``pooling=\"cls\"``) so the pooled "
                "channel is populated."
            )

        return pooled

    # =====================================================
    # CONSTRUCTION HELPERS
    # =====================================================

    @classmethod
    def _build_from_truthlens_config(
        cls,
        config: MultiTaskTruthLensConfig,
    ) -> "tuple[nn.Module, Dict[str, nn.Module]]":
        """Build (encoder, task_heads) from a ``MultiTaskTruthLensConfig``."""

        from src.models.encoder.encoder_config import EncoderConfig
        from src.models.encoder.encoder_factory import EncoderFactory

        encoder = EncoderFactory.create_transformer_encoder(
            EncoderConfig(
                model_type="transformer",
                model_name=config.model_name,
                pooling=config.pooling,
                device=config.device,
                init_from_config_only=config.init_from_config_only,
            )
        )

        hidden_size = int(getattr(encoder, "hidden_size"))

        task_specs = cls._resolve_task_specs(
            num_labels_overrides=config.task_num_labels,
            enabled_tasks=config.enabled_tasks,
        )

        task_heads = cls._build_default_heads(
            task_specs=task_specs,
            hidden_size=hidden_size,
            dropout=config.dropout,
        )

        return encoder, task_heads

    @classmethod
    def _resolve_task_specs(
        cls,
        num_labels_overrides: Optional[Dict[str, int]] = None,
        enabled_tasks: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Merge default task spec with optional overrides."""

        names = enabled_tasks or list(_DEFAULT_TASK_SPEC.keys())

        unknown = [n for n in names if n not in _DEFAULT_TASK_SPEC]
        if unknown:
            raise ValueError(
                f"Unknown TruthLens task name(s): {unknown}. "
                f"Known: {list(_DEFAULT_TASK_SPEC.keys())}"
            )

        resolved: Dict[str, Dict[str, Any]] = {}

        for name in names:
            base = _DEFAULT_TASK_SPEC[name]
            num_labels = (
                (num_labels_overrides or {}).get(name, len(base["labels"]))
            )
            resolved[name] = {
                "task_type": base["task_type"],
                "num_labels": int(num_labels),
            }

        return resolved

    @staticmethod
    def _build_default_heads(
        task_specs: Dict[str, Dict[str, Any]],
        hidden_size: int,
        dropout: float,
    ) -> Dict[str, nn.Module]:
        """Instantiate one head per task spec."""

        from src.models.heads.classification_head import (
            ClassificationHead,
            ClassificationHeadConfig,
        )
        from src.models.heads.multilabel_head import (
            MultiLabelHead,
            MultiLabelHeadConfig,
        )

        heads: Dict[str, nn.Module] = {}

        for name, spec in task_specs.items():
            task_type = spec["task_type"]
            num_labels = spec["num_labels"]

            if task_type == "multi_class":
                heads[name] = ClassificationHead(
                    ClassificationHeadConfig(
                        input_dim=hidden_size,
                        num_classes=num_labels,
                        dropout=dropout,
                    )
                )

            elif task_type == "multi_label":
                heads[name] = MultiLabelHead(
                    MultiLabelHeadConfig(
                        input_dim=hidden_size,
                        num_labels=num_labels,
                        dropout=dropout,
                    )
                )

            else:
                raise ValueError(
                    f"Unsupported task_type {task_type!r} for task {name!r}"
                )

        return heads

    # =====================================================
    # PUBLIC FACTORY: from MultiTaskModelConfig (YAML path)
    # =====================================================

    @classmethod
    def from_model_config(
        cls,
        model_config: Any,
    ) -> "MultiTaskTruthLensModel":
        """Build a model from a :class:`MultiTaskModelConfig`.

        This is the high-fidelity construction path used by the model
        registry, the inference engine and the YAML-driven training
        pipeline. Each entry in ``model_config.tasks`` is materialised
        as a head whose width comes from ``task_cfg.num_labels`` and
        whose head type comes from ``task_cfg.task_type``.
        """

        from src.models.config import MultiTaskModelConfig
        from src.models.encoder.encoder_factory import EncoderFactory
        from src.models.heads.classification_head import (
            ClassificationHead,
            ClassificationHeadConfig,
        )
        from src.models.heads.multilabel_head import (
            MultiLabelHead,
            MultiLabelHeadConfig,
        )

        if not isinstance(model_config, MultiTaskModelConfig):
            raise TypeError(
                "model_config must be a MultiTaskModelConfig "
                f"(got {type(model_config).__name__})"
            )

        if not model_config.tasks:
            raise ValueError("model_config.tasks must be non-empty")

        encoder = EncoderFactory.create_from_model_config(model_config)
        hidden_size = int(getattr(encoder, "hidden_size"))

        task_heads: Dict[str, nn.Module] = {}

        for task_name, task_cfg in model_config.tasks.items():
            num_labels = int(task_cfg.num_labels)

            if num_labels <= 0:
                raise ValueError(
                    f"Task {task_name!r}: num_labels must be positive "
                    f"(got {num_labels})"
                )

            if task_cfg.task_type == "multi_label":
                task_heads[task_name] = MultiLabelHead(
                    MultiLabelHeadConfig(
                        input_dim=hidden_size,
                        num_labels=num_labels,
                        dropout=model_config.dropout,
                    )
                )

            else:  # default: multi_class
                task_heads[task_name] = ClassificationHead(
                    ClassificationHeadConfig(
                        input_dim=hidden_size,
                        num_classes=num_labels,
                        dropout=model_config.dropout,
                    )
                )

        return cls(encoder=encoder, task_heads=task_heads)

    # =====================================================
    # UTILITIES
    # =====================================================

    def get_task_names(self) -> List[str]:
        return list(self.task_heads.keys())

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False
        logger.info("Encoder frozen")

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True
        logger.info("Encoder unfrozen")

    def freeze_heads(self) -> None:
        for head in self.task_heads.values():
            for p in head.parameters():
                p.requires_grad = False
        logger.info("All task heads frozen")

    def unfreeze_heads(self) -> None:
        for head in self.task_heads.values():
            for p in head.parameters():
                p.requires_grad = True
        logger.info("All task heads unfrozen")

    def freeze_task(self, task_name: str) -> None:
        if task_name not in self.task_heads:
            raise ValueError(f"Unknown task: {task_name}")

        for p in self.task_heads[task_name].parameters():
            p.requires_grad = False

        logger.info("Task '%s' frozen", task_name)

    def unfreeze_task(self, task_name: str) -> None:
        if task_name not in self.task_heads:
            raise ValueError(f"Unknown task: {task_name}")

        for p in self.task_heads[task_name].parameters():
            p.requires_grad = True

        logger.info("Task '%s' unfrozen", task_name)

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def extra_repr(self) -> str:
        return f"tasks={list(self.task_heads.keys())}"
