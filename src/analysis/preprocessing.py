"""
File Name: preprocessing_pipeline.py 
Module: Data Processing - Text Preprocessing Pipeline
Description:
    Implements an advanced preprocessing pipeline for textual inputs used in the
    TruthLens AI system. The pipeline performs text normalization, cleaning,
    sentence segmentation, tokenization, lemma extraction, and language detection.
    It also supports scalable batch preprocessing and parallel processing to
    efficiently handle large volumes of textual data.
    
Dependencies:
    logging
    re
    typing
    dataclasses
    concurrent.futures
    spacy
    langdetect

Inputs:
    Raw text string or list of text strings

Outputs:
    Structured preprocessing results including normalized text, tokens, lemmas,
    sentences, and detected language
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

try:
    from langdetect import detect, LangDetectException
except ImportError:  # pragma: no cover - optional dependency
    class LangDetectException(Exception):
        """Fallback exception when langdetect is unavailable."""

    def detect(_text: str) -> str:
        return "unknown"


from src.analysis.analysis_config import ANALYSIS_CONFIG
from src.analysis.spacy_loader import get_nlp


logger = logging.getLogger(__name__)


def _get_nlp():
    """
    Backwards-compatible accessor for the worker-side spaCy model.

    Delegates to the shared loader so that all preprocessing — main
    process or worker — uses the same cached pipeline configuration.
    """
    return get_nlp()


@dataclass
class PreprocessingResult:
    """
    Dataclass representing structured preprocessing output.
    """

    normalized_text: str
    tokens: List[str]
    lemmas: List[str]
    sentences: List[str]
    language: str


class PreprocessingPipeline:
    """
    Production-grade preprocessing pipeline supporting scalable text processing.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        max_workers: Optional[int] = None,
    ) -> None:
        """
        Initialize NLP pipeline used for preprocessing.

        Parameters
        ----------
        spacy_model : str
            spaCy model name.
        max_workers : Optional[int]
            Maximum number of parallel workers for batch preprocessing.
        """

        try:
            # Use the shared loader so we hit a process-local cache and
            # honour `analysis_config.SpacyConfig` (model, GPU, etc.).
            self.nlp = get_nlp(spacy_model)
        except Exception as exc:
            logger.exception("spaCy model loading failed")
            raise RuntimeError("Failed to load spaCy model") from exc

        self.max_workers = max_workers

        logger.info(
            "PreprocessingPipeline initialized",
            extra={"spacy_model": spacy_model, "max_workers": max_workers},
        )

    def preprocess(self, text: str) -> PreprocessingResult:
        """
        Run preprocessing pipeline on a single text input.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        PreprocessingResult
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        try:
            language = self._detect_language(text)

            normalized_text = self._normalize_text(text)

            doc = self.nlp(normalized_text)

            tokens = self._extract_tokens(doc)

            lemmas = self._extract_lemmas(doc)

            sentences = self._extract_sentences(doc)

        except Exception as exc:
            logger.exception("Text preprocessing failed")
            raise RuntimeError("Preprocessing pipeline failed") from exc

        return PreprocessingResult(
            normalized_text=normalized_text,
            tokens=tokens,
            lemmas=lemmas,
            sentences=sentences,
            language=language,
        )

    def preprocess_batch(
        self,
        texts: List[str],
        parallel: bool = True,
    ) -> List[PreprocessingResult]:
        """
        Preprocess a batch of texts.

        Parameters
        ----------
        texts : List[str]
            List of raw text inputs.
        parallel : bool
            Whether to use parallel processing.

        Returns
        -------
        List[PreprocessingResult]
        """

        if not isinstance(texts, list):
            raise ValueError("texts must be a list of strings")

        if not texts:
            return []

        if parallel:
            return self._parallel_preprocess(texts)

        return [self.preprocess(text) for text in texts]

    def _parallel_preprocess(self, texts: List[str]) -> List[PreprocessingResult]:
        """
        Batched preprocessing via spaCy's tuned multi-process pipeline.

        PERF-A4: the previous implementation forked a ``ProcessPoolExecutor``
        and re-loaded spaCy + transformers in every worker, which is
        slower than a single-process ``nlp.pipe`` for any meaningful
        batch size. Use ``nlp.pipe`` with the configured
        ``ANALYSIS_CONFIG.spacy`` knobs so we get spaCy's optimized
        pickle path (or a single-process pass when ``n_process == 1``)
        and avoid double-loading the model.
        """

        if not texts:
            return []

        # Pre-normalize once per text — keeps the per-doc loop tight and
        # produces the same cleaned strings the static worker used.
        normalized = [self._normalize_text(t) for t in texts]

        languages: List[str] = []
        for raw in texts:
            try:
                languages.append(detect(raw))
            except LangDetectException:
                languages.append("unknown")

        spacy_cfg = getattr(ANALYSIS_CONFIG, "spacy", None)
        try:
            batch_size = int(getattr(spacy_cfg, "batch_size", 32) or 32)
        except (TypeError, ValueError):
            batch_size = 32
        try:
            n_process = int(getattr(spacy_cfg, "n_process", 1) or 1)
        except (TypeError, ValueError):
            n_process = 1
        if n_process < 1:
            n_process = 1

        # Honor an explicit max_workers override (kept for backward
        # compat) — but cap at the configured n_process so we don't
        # oversubscribe cores beyond what the rest of the system expects.
        if self.max_workers is not None:
            try:
                n_process = max(1, min(int(self.max_workers), n_process))
            except (TypeError, ValueError):
                pass

        results: List[PreprocessingResult] = []

        try:
            for normalized_text, language, doc in zip(
                normalized,
                languages,
                self.nlp.pipe(
                    normalized,
                    batch_size=batch_size,
                    n_process=n_process,
                ),
            ):
                tokens = self._extract_tokens(doc)
                lemmas = self._extract_lemmas(doc)
                sentences = self._extract_sentences(doc)

                results.append(
                    PreprocessingResult(
                        normalized_text=normalized_text,
                        tokens=tokens,
                        lemmas=lemmas,
                        sentences=sentences,
                        language=language,
                    )
                )
        except Exception as exc:
            logger.exception("Batch preprocessing failed")
            raise RuntimeError("Parallel preprocessing failed") from exc

        return results

    def _detect_language(self, text: str) -> str:
        """
        Detect language of input text.

        Parameters
        ----------
        text : str

        Returns
        -------
        str
        """

        try:
            return detect(text)
        except LangDetectException:
            logger.warning("Language detection failed")
            return "unknown"

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by removing excessive whitespace and unwanted characters.
        """

        text = text.strip()

        text = re.sub(r"\s+", " ", text)

        text = re.sub(r"[^\w\s\.\,\!\?\-']", "", text)

        return text

    def _extract_tokens(self, doc) -> List[str]:
        """
        Extract tokens from spaCy document.
        """

        return [token.text.lower() for token in doc if not token.is_space]

    def _extract_lemmas(self, doc) -> List[str]:
        """
        Extract lemmas from spaCy document.
        """

        return [token.lemma_.lower() for token in doc if not token.is_space]

    def _extract_sentences(self, doc) -> List[str]:
        """
        Extract sentence list from spaCy document.
        """

        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
