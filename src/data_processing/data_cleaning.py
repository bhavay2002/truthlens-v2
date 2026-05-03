from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from src.data_processing.data_contracts import CONTRACTS, get_contract

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================

@dataclass
class DataCleaningConfig:
    drop_duplicates: bool = True
    drop_empty_text: bool = True
    normalize_whitespace: bool = True
    lowercase: bool = False              # keep False for most NLP tasks
    strip_urls: bool = False             # optional (can remove signal)
    strip_html: bool = True
    min_text_len: int = 3
    max_text_len: int = 20000

    # CFG-D3: previously the YAML claimed these three were "on" but no
    # corresponding fields existed on the dataclass — every flag was
    # silently ignored. Wired up as real, opt-in fields. Defaults are
    # off so existing experiments stay byte-for-byte reproducible; the
    # YAML now controls them once read into DataCleaningConfig.
    normalize_unicode: bool = False      # NFKC normalisation
    remove_emojis: bool = False
    expand_contractions: bool = False

    # label handling
    fill_missing_labels: bool = False    # usually False (let validator fail)
    label_fill_value: int = 0

    # reporting
    log_stats: bool = True


# =========================================================
# REGEX
# =========================================================

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<.*?>")
WS_RE = re.compile(r"\s+")

# CFG-D3: emoji + symbol ranges. The character classes cover the core
# Unicode emoji blocks (Misc Symbols & Pictographs, Emoticons, Transport
# & Map, Supplemental Symbols & Pictographs, Dingbats, Variation
# Selectors, Regional Indicators) plus skin-tone modifiers and the ZWJ
# joiner used to splice family/profession sequences. Keeping the regex
# scoped (no general "remove all symbols" sweep) avoids stripping
# meaningful punctuation like math operators or currency.
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # misc symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F700-\U0001F77F"   # alchemical
    "\U0001F780-\U0001F7FF"   # geometric shapes ext
    "\U0001F800-\U0001F8FF"   # supplemental arrows-c
    "\U0001F900-\U0001F9FF"   # supplemental symbols & pictographs
    "\U0001FA00-\U0001FA6F"   # chess symbols
    "\U0001FA70-\U0001FAFF"   # symbols & pictographs ext-a
    "\U00002700-\U000027BF"   # dingbats
    "\U00002600-\U000026FF"   # misc symbols
    "\U0001F1E6-\U0001F1FF"   # regional indicators (flags)
    "\U0001F3FB-\U0001F3FF"   # skin tone modifiers
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U0000200D"              # zero-width joiner
    "]+",
    flags=re.UNICODE,
)

# CFG-D3: minimal, ASCII-safe contractions table. Deliberately small —
# this is enough to handle the common cases without dragging in
# ``contractions`` as a hard dep, and stays deterministic across runs
# (no model-driven expansion that could shift cache keys silently).
_CONTRACTIONS = {
    r"\bcan't\b":     "cannot",
    r"\bcant\b":      "cannot",
    r"\bwon't\b":     "will not",
    r"\bshan't\b":    "shall not",
    r"\bain't\b":     "is not",
    r"\bn't\b":       " not",
    r"\b'm\b":        " am",
    r"\b're\b":       " are",
    r"\b's\b":        " is",
    r"\b'd\b":        " would",
    r"\b'll\b":       " will",
    r"\b've\b":       " have",
    r"\blet's\b":     "let us",
    r"\by'all\b":     "you all",
    r"\bgonna\b":     "going to",
    r"\bwanna\b":     "want to",
    r"\bgotta\b":     "got to",
}
_CONTRACTIONS_RE = [
    (re.compile(pat, flags=re.IGNORECASE), repl) for pat, repl in _CONTRACTIONS.items()
]


def _expand_contractions(s: pd.Series) -> pd.Series:
    for pattern, repl in _CONTRACTIONS_RE:
        s = s.str.replace(pattern, repl, regex=True)
    return s


def _normalize_unicode_series(s: pd.Series) -> pd.Series:
    # ``str.normalize`` is a vectorized accessor in modern pandas —
    # equivalent to ``s.map(lambda x: unicodedata.normalize("NFKC", x))``
    # but stays in C-land. Fall back to ``map`` for older pandas without
    # the accessor.
    if hasattr(s.str, "normalize"):
        return s.str.normalize("NFKC")
    return s.map(lambda x: unicodedata.normalize("NFKC", x) if isinstance(x, str) else x)


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def _clean_text(text: str, cfg: DataCleaningConfig) -> str:
    if not isinstance(text, str):
        return ""

    t = text

    # CFG-D3: NFKC first so the downstream regexes (URL/HTML/whitespace)
    # work against canonical code points (e.g. fullwidth ASCII collapses
    # to ASCII before URL_RE runs).
    if cfg.normalize_unicode:
        t = unicodedata.normalize("NFKC", t)

    if cfg.strip_html:
        t = HTML_RE.sub(" ", t)

    if cfg.strip_urls:
        t = URL_RE.sub(" ", t)

    if cfg.remove_emojis:
        t = EMOJI_RE.sub(" ", t)

    if cfg.expand_contractions:
        for pattern, repl in _CONTRACTIONS_RE:
            t = pattern.sub(repl, t)

    if cfg.normalize_whitespace:
        t = WS_RE.sub(" ", t)

    if cfg.lowercase:
        t = t.lower()

    return t.strip()


# =========================================================
# CORE CLEANING
# =========================================================

def clean_dataframe(
    df: pd.DataFrame,
    *,
    config: Optional[DataCleaningConfig] = None,
    label_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Clean a dataframe in a controlled, reproducible way.

    Args:
        df: input dataframe (must contain 'text')
        config: cleaning behavior
        label_cols: optional label columns to process

    Returns:
        cleaned dataframe
    """

    if "text" not in df.columns:
        raise ValueError("Missing 'text' column")

    cfg = config or DataCleaningConfig()

    orig_rows = len(df)

    # -----------------------------------------------------
    # COPY (avoid side-effects)
    # -----------------------------------------------------
    df = df.copy()

    # -----------------------------------------------------
    # TEXT CLEANING — vectorized via pandas .str ops (PERF-D1).
    # ~5-10x faster than a Python .map(_clean_text) loop on 100k rows
    # because every step stays in the C-level pandas/regex engine.
    #
    # CFG-D3: the order intentionally mirrors ``_clean_text`` —
    # NFKC → HTML → URLs → emojis → contractions → whitespace → lowercase
    # — so both code paths produce byte-identical output for any given
    # config.
    # -----------------------------------------------------
    # Preserve the original NaN mask BEFORE coercing to ``str``.
    # ``astype(str)`` turns NaN into the literal "nan" and ``None`` into
    # the literal "None", which would silently survive the
    # ``drop_empty_text`` filter below (because ``.isna()`` returns
    # False on a string). Captured here, this mask drives both the
    # drop count and the keep filter so genuinely-missing rows are
    # logged as "(NaN)" and removed.
    nan_mask = df["text"].isna()
    s = df["text"].astype(str)
    if cfg.normalize_unicode:
        s = _normalize_unicode_series(s)
    if cfg.strip_html:
        s = s.str.replace(HTML_RE, " ", regex=True)
    if cfg.strip_urls:
        s = s.str.replace(URL_RE, " ", regex=True)
    if cfg.remove_emojis:
        s = s.str.replace(EMOJI_RE, " ", regex=True)
    if cfg.expand_contractions:
        s = _expand_contractions(s)
    if cfg.normalize_whitespace:
        s = s.str.replace(WS_RE, " ", regex=True)
    if cfg.lowercase:
        s = s.str.lower()
    df["text"] = s.str.strip()

    # -----------------------------------------------------
    # DROP EMPTY / SHORT / LONG TEXT
    #
    # EDGE-CASE (audit §9 — "extremely long text > 20k chars dropped
    # silently"): split the mask so we can log how many rows were lost
    # to each cap separately. A run that quietly halves its training
    # corpus because of a too-aggressive ``max_text_len`` is exactly
    # the silent failure that audit flagged.
    # -----------------------------------------------------
    if cfg.drop_empty_text:
        text_len = df["text"].str.len()
        # ``df["text"]`` is now an all-string column (post-astype), so
        # we use ``nan_mask`` captured pre-astype to identify genuinely
        # missing rows. ``~nan_mask`` is the "originally non-null" set.
        valid = ~nan_mask
        too_short_mask = valid & (text_len < cfg.min_text_len)
        too_long_mask = valid & (text_len > cfg.max_text_len)
        keep_mask = (
            valid
            & (text_len >= cfg.min_text_len)
            & (text_len <= cfg.max_text_len)
        )
        n_short = int(too_short_mask.sum())
        n_long = int(too_long_mask.sum())
        n_nan = int(nan_mask.sum())
        if cfg.log_stats and (n_short or n_long or n_nan):
            logger.info(
                "Length filter | dropped %d (NaN) + %d (<%d chars) + %d (>%d chars)",
                n_nan, n_short, cfg.min_text_len, n_long, cfg.max_text_len,
            )
        df = df[keep_mask]

    # -----------------------------------------------------
    # DROP DUPLICATES (TEXT-LEVEL) — case-insensitive to stay
    # consistent with leakage_checker._normalize, which lowercases
    # before hashing. Otherwise "Foo" and "foo" both survive dedup
    # but collide in the leakage check, raising a false positive
    # under strict=True. (LEAK-D3)
    # -----------------------------------------------------
    if cfg.drop_duplicates:
        before = len(df)
        norm = df["text"].str.lower()
        df = df.loc[~norm.duplicated()]
        removed = before - len(df)
        if cfg.log_stats and removed > 0:
            logger.info("Removed %d duplicate rows", removed)

    # -----------------------------------------------------
    # LABEL HANDLING (OPTIONAL)
    # -----------------------------------------------------
    if label_cols and cfg.fill_missing_labels:
        for col in label_cols:
            if col in df.columns:
                df[col] = df[col].fillna(cfg.label_fill_value)

    # -----------------------------------------------------
    # FINAL RESET
    # -----------------------------------------------------
    df = df.reset_index(drop=True)

    if cfg.log_stats:
        logger.info(
            "Data cleaned | rows: %d → %d",
            orig_rows,
            len(df),
        )

    return df


# =========================================================
# TASK-AWARE CLEANING (OPTIONAL)
# =========================================================

def clean_for_task(
    df: pd.DataFrame,
    task: str,
    *,
    config: Optional[DataCleaningConfig] = None,
) -> pd.DataFrame:
    """
    Apply task-specific cleaning rules. (CRIT-D1)

    Label columns are pulled from the canonical contracts table
    (``data_contracts.CONTRACTS``) instead of a duplicated lookup —
    so adding/renaming a label in one place keeps cleaning, validation,
    factory, and sampler perfectly in sync.
    """

    cfg = config or DataCleaningConfig()

    if task in CONTRACTS:
        label_cols: List[str] = list(get_contract(task).label_columns)
    else:
        # Unknown task → behave as before (no label-aware cleaning),
        # but warn loudly so a typo doesn't silently disable label fill.
        logger.warning("clean_for_task called with unknown task=%s — skipping label-aware cleaning", task)
        label_cols = []

    return clean_dataframe(
        df,
        config=cfg,
        label_cols=label_cols,
    )