from __future__ import annotations

from typing import List, Sequence, Tuple, Literal, Dict, Any
import numpy as np


AggregationMethod = Literal["mean", "sum", "max"]

SPECIAL_TOKENS = {
    "[CLS]", "[SEP]", "<s>", "</s>", "[PAD]", "<pad>"
}

EPS = 1e-12

# RoBERTa / GPT-2 BPE uses U+0120 (Ġ) as a word-boundary prefix.
# Tokens that start with Ġ begin a new word; all others are continuations.
_BPE_BOUNDARY = "\u0120"  # Ġ


def align_tokens(
    tokens: Sequence[str],
    scores: Sequence[float],
    tokenizer_type: str = "wordpiece",
    aggregation: AggregationMethod = "mean",

    # 🔥 NEW FEATURES
    normalize: bool = False,
    clip: bool = False,
    max_tokens: int | None = None,
    return_structured: bool = False,
    return_variance: bool = False,
) -> Tuple[List[str], List[float]] | Dict[str, Any]:

    # ==================================================
    # VALIDATION
    # ==================================================

    if len(tokens) != len(scores):
        raise ValueError("tokens and scores must match in length")

    if tokenizer_type not in {"wordpiece", "sentencepiece", "bpe"}:
        raise ValueError("invalid tokenizer_type")

    if aggregation not in {"mean", "sum", "max"}:
        raise ValueError("invalid aggregation")

    if len(tokens) == 0:
        if return_structured:
            return {"tokens": [], "importance": []}
        if return_variance:
            return [], [], []
        return [], []

    # ==================================================
    # AGG FUNCTION
    # ==================================================

    def agg(values: List[float]) -> Tuple[float, float]:
        arr = np.array(values, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if aggregation == "mean":
            val = float(np.mean(arr))
        elif aggregation == "sum":
            val = float(np.sum(arr))
        else:
            val = float(arr[np.argmax(np.abs(arr))])

        variance = float(np.var(arr)) if len(arr) > 1 else 0.0
        return val, variance

    # ==================================================
    # MERGE
    # ==================================================

    merged_tokens: List[str] = []
    merged_scores: List[float] = []
    merged_variance: List[float] = []

    parts: List[str] = []
    vals: List[float] = []

    for token, score in zip(tokens, scores):

        if token is None:
            continue

        token = str(token).strip()

        if not token or token in SPECIAL_TOKENS:
            continue

        try:
            score = float(score)
        except Exception:
            score = 0.0

        if not np.isfinite(score):
            score = 0.0

        # ---------------- WORDPIECE ----------------
        if tokenizer_type == "wordpiece":

            if token.startswith("##"):
                parts.append(token[2:])
                vals.append(score)
                continue

            if parts:
                val, var = agg(vals)
                merged_tokens.append("".join(parts))
                merged_scores.append(val)
                merged_variance.append(var)

            parts = [token]
            vals = [score]

        # ---------------- SENTENCEPIECE ----------------
        elif tokenizer_type == "sentencepiece":

            if token.startswith("▁"):
                if parts:
                    val, var = agg(vals)
                    merged_tokens.append("".join(parts))
                    merged_scores.append(val)
                    merged_variance.append(var)

                parts = [token[1:]]
                vals = [score]

            else:
                parts.append(token)
                vals.append(score)

        # ---------------- RoBERTa / GPT-2 BPE (Ġ) ----------------
        else:  # tokenizer_type == "bpe"

            if token.startswith(_BPE_BOUNDARY):
                # Ġ prefix → this token starts a new word.
                if parts:
                    val, var = agg(vals)
                    merged_tokens.append("".join(parts))
                    merged_scores.append(val)
                    merged_variance.append(var)

                parts = [token[len(_BPE_BOUNDARY):]]  # strip the Ġ
                vals = [score]

            else:
                # No prefix → continuation of the current word.
                parts.append(token)
                vals.append(score)

    # final flush
    if parts:
        val, var = agg(vals)
        merged_tokens.append("".join(parts))
        merged_scores.append(val)
        merged_variance.append(var)

    # ==================================================
    # NORMALIZATION 🔥
    # ==================================================

    scores_arr = np.array(merged_scores, dtype=float)

    if normalize:
        scores_arr = np.abs(scores_arr)
        scores_arr = scores_arr / (np.sum(scores_arr) + EPS)

    if clip:
        scores_arr = np.clip(scores_arr, 0.0, 1.0)

    # ==================================================
    # TOKEN LIMIT 🔥
    # ==================================================

    if max_tokens is not None and len(scores_arr) > max_tokens:
        idx = np.argsort(np.abs(scores_arr))[::-1][:max_tokens]
        merged_tokens = [merged_tokens[i] for i in idx]
        scores_arr = scores_arr[idx]
        merged_variance = [merged_variance[i] for i in idx]

    scores_list = scores_arr.tolist()

    # ==================================================
    # OUTPUT MODES 🔥
    # ==================================================

    if return_structured:

        structured = [
            {"token": t, "importance": float(s)}
            for t, s in zip(merged_tokens, scores_list)
        ]

        result = {
            "tokens": merged_tokens,
            "importance": scores_list,
            "structured": structured,
        }

        if return_variance:
            result["variance"] = merged_variance

        return result

    if return_variance:
        return merged_tokens, scores_list, merged_variance

    return merged_tokens, scores_list