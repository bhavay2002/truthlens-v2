from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.base.tokenization import tokenize_words

logger = logging.getLogger(__name__)


# =========================================================
# CORE ENGINE
# =========================================================
# Audit fix §5.3 — sklearn's default ``token_pattern=r"(?u)\b\w\w+\b"``
# silently drops single-character tokens, contractions (``don't`` ->
# ``don``, ``t``) and accented characters. Every other extractor in
# this codebase reads ``ensure_tokens_word`` from the per-context
# cache, which uses the canonical Unicode-aware ``tokenize_words``
# helper. Aligning the two avoids the (otherwise silent) divergence
# between the TF-IDF vocabulary and the rest of the lexical/semantic
# feature universe.

@dataclass
class TfidfEngineering:

    max_features: int = 5000
    top_terms_per_doc: int = 5
    lowercase: bool = True

    vectorizer: Optional[TfidfVectorizer] = None

    # -----------------------------------------------------

    def _build_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            max_features=self.max_features,
            lowercase=self.lowercase,
            # Audit §5.3 — drive the vocabulary off ``tokenize_words``
            # so TF-IDF matches the lexical / bias / propaganda
            # extractors. ``token_pattern=None`` disables sklearn's
            # default regex (required when a tokenizer callable is
            # supplied, otherwise sklearn warns and ignores the
            # tokenizer).
            tokenizer=tokenize_words,
            token_pattern=None,
        )

    def fit(self, texts: List[str]) -> None:
        """
        Fit TF-IDF vectorizer.
        """

        if not texts:
            raise ValueError("Texts cannot be empty")

        self.vectorizer = self._build_vectorizer()

        self.vectorizer.fit(texts)

        logger.info(
            "TF-IDF fitted | vocab_size=%d",
            len(self.vectorizer.get_feature_names_out()),
        )

    # -----------------------------------------------------

    def transform(self, texts: List[str]) -> List[str]:
        """
        Transform texts into engineered text using top TF-IDF terms.
        """

        if self.vectorizer is None:
            raise RuntimeError("Vectorizer must be fitted first")

        matrix = self.vectorizer.transform(texts)
        feature_names = self.vectorizer.get_feature_names_out()

        engineered: List[str] = []

        for i in range(matrix.shape[0]):

            row = matrix.getrow(i)

            if row.nnz == 0:
                engineered.append("")
                continue

            # top-k indices
            order = row.data.argsort()[::-1][: self.top_terms_per_doc]
            indices = row.indices[order]

            terms = [str(feature_names[idx]) for idx in indices]

            engineered.append(" ".join(terms))

        return engineered

    # -----------------------------------------------------

    def fit_transform(self, texts: List[str]) -> List[str]:
        """
        Fit + transform in one step.
        """

        self.fit(texts)
        return self.transform(texts)


# =========================================================
# DATAFRAME WRAPPERS
# =========================================================

def apply_tfidf_engineering(
    df: pd.DataFrame,
    *,
    text_column: str = "text",
    max_features: int = 5000,
    top_terms_per_doc: int = 5,
) -> Tuple[pd.DataFrame, TfidfEngineering]:
    """
    Fit TF-IDF and create engineered_text column.
    """

    if text_column not in df.columns:
        raise ValueError(f"Missing column: {text_column}")

    texts = df[text_column].fillna("").astype(str).tolist()

    engine = TfidfEngineering(
        max_features=max_features,
        top_terms_per_doc=top_terms_per_doc,
    )

    engineered = engine.fit_transform(texts)

    df_out = df.copy()
    df_out["engineered_text"] = engineered

    return df_out, engine


# ---------------------------------------------------------

def transform_tfidf_engineering(
    df: pd.DataFrame,
    *,
    engine: TfidfEngineering,
    text_column: str = "text",
) -> pd.DataFrame:
    """
    Apply pre-fitted TF-IDF transformation.
    """

    if engine.vectorizer is None:
        raise RuntimeError("TF-IDF engine is not fitted")

    texts = df[text_column].fillna("").astype(str).tolist()

    engineered = engine.transform(texts)

    df_out = df.copy()
    df_out["engineered_text"] = engineered

    return df_out


# =========================================================
# ADVANCED: DIRECT FEATURE MATRIX (OPTIONAL)
# =========================================================

def tfidf_matrix(
    texts: List[str],
    *,
    max_features: int = 5000,
) -> Tuple:
    """
    Return raw TF-IDF matrix (for ML models).

    Uses the same ``tokenize_words`` helper as :class:`TfidfEngineering`
    (audit §5.3) so the matrix vocabulary is consistent with every
    other extractor in ``src/features/``.
    """

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        tokenizer=tokenize_words,
        token_pattern=None,
    )

    matrix = vectorizer.fit_transform(texts)

    return matrix, vectorizer