"""
Generate small smoke-test datasets for the 6 TruthLens tasks.

Writes 10 rows per (task, split) CSV under ``data/{train,val,test}/{task}.csv``.

Layout:
    data/train/bias.csv          text, bias_label
    data/train/ideology.csv      text, ideology_label
    data/train/propaganda.csv    text, propaganda_label
    data/train/frame.csv         text, CO, EC, HI, MO, RE
    data/train/narrative.csv     text, hero, villain, victim, hero_entities, villain_entities, victim_entities
    data/train/emotion.csv       text, emotion_0..emotion_10  (EMOTION-11)
    (same for val/test)

The texts are unique across train/val/test to satisfy the leakage check,
and contain >= ~30 words to satisfy text-length checks.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SPLITS = ("train", "val", "test")
N_ROWS = 10


# -------------------------------------------------------------------
# TEXT POOL — 200 distinct ~35-word sentences. Enough to give every
# (split, task, row) a unique text without collisions across splits.
# -------------------------------------------------------------------

TOPICS = [
    "the federal budget proposal", "rising inflation rates",
    "the international climate accord", "the disputed election outcome",
    "the new immigration policy", "the supreme court ruling",
    "the technology antitrust case", "the public health response",
    "the recent diplomatic summit", "the economic stimulus package",
    "the controversial energy bill", "the educational reform plan",
    "the privacy regulation debate", "the foreign aid allocation",
    "the housing affordability crisis", "the tax overhaul legislation",
    "the trade tariff dispute", "the criminal justice reform",
    "the labor union negotiations", "the artificial intelligence policy",
]

ACTORS = [
    "Senator Brooks", "Governor Patel", "the administration", "opposition leaders",
    "the central bank", "industry analysts", "civic groups", "the regulator",
    "the investigative committee", "the trade alliance",
]

VERBS = [
    "praised", "criticized", "celebrated", "condemned", "examined",
    "questioned", "endorsed", "denounced", "supported", "challenged",
]

QUALIFIERS = [
    "as a balanced and forward-looking decision",
    "as a reckless and dangerous overreach",
    "with measured optimism and analytical caution",
    "with sweeping rhetoric and emotional appeals",
    "while urging swift bipartisan cooperation",
    "while warning of severe long-term consequences",
    "amid widespread protests across major cities",
    "amid quiet diplomatic backchannel negotiations",
    "after months of careful expert deliberation",
    "after intense partisan exchanges in the chamber",
]

ENTITIES_HERO = ["Dr. Lopez", "Captain Harris", "the volunteer corps", "the rescue team", "Engineer Wong"]
ENTITIES_VILLAIN = ["the cartel network", "the rogue state", "the corrupt official", "the militia group", "the shadow lobby"]
ENTITIES_VICTIM = ["displaced families", "small business owners", "rural communities", "the elderly residents", "the school children"]


def make_text(seed_idx: int) -> str:
    """Build a deterministic ~35-word sentence indexed by seed_idx."""
    rng = random.Random(seed_idx * 9973 + 17)
    topic = TOPICS[seed_idx % len(TOPICS)]
    actor = rng.choice(ACTORS)
    verb = rng.choice(VERBS)
    qualifier = rng.choice(QUALIFIERS)
    return (
        f"In a closely watched session, {actor} {verb} {topic} {qualifier}, "
        f"highlighting key concerns about transparency, accountability, and "
        f"the long-term impact on ordinary citizens, businesses, and "
        f"international partners observing the debate from abroad."
    )


def text_pool(split_idx: int, task_idx: int) -> list[str]:
    """Return N_ROWS unique texts — namespace by (split, task) so the leakage
    check passes (no exact-text overlap across splits)."""
    base = (split_idx * 100) + (task_idx * 10)
    return [make_text(base + i) for i in range(N_ROWS)]


# -------------------------------------------------------------------
# WRITERS
# -------------------------------------------------------------------

def _write(rows: list[dict], header: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def gen_bias(texts: list[str]) -> tuple[list[dict], list[str]]:
    rows = [{"text": t, "bias_label": i % 2} for i, t in enumerate(texts)]
    return rows, ["text", "bias_label"]


def gen_ideology(texts: list[str]) -> tuple[list[dict], list[str]]:
    rows = [{"text": t, "ideology_label": i % 3} for i, t in enumerate(texts)]
    return rows, ["text", "ideology_label"]


def gen_propaganda(texts: list[str]) -> tuple[list[dict], list[str]]:
    rows = [{"text": t, "propaganda_label": (i + 1) % 2} for i, t in enumerate(texts)]
    return rows, ["text", "propaganda_label"]


def gen_frame(texts: list[str]) -> tuple[list[dict], list[str]]:
    cols = ["CO", "EC", "HI", "MO", "RE"]
    rows = []
    for i, t in enumerate(texts):
        row = {"text": t}
        for j, c in enumerate(cols):
            row[c] = (i + j) % 2
        rows.append(row)
    return rows, ["text", *cols]


def gen_narrative(texts: list[str]) -> tuple[list[dict], list[str]]:
    rows = []
    for i, t in enumerate(texts):
        rows.append({
            "text": t,
            "hero": i % 2,
            "villain": (i + 1) % 2,
            "victim": (i + 2) % 2,
            "hero_entities": ENTITIES_HERO[i % len(ENTITIES_HERO)],
            "villain_entities": ENTITIES_VILLAIN[i % len(ENTITIES_VILLAIN)],
            "victim_entities": ENTITIES_VICTIM[i % len(ENTITIES_VICTIM)],
        })
    header = [
        "text", "hero", "villain", "victim",
        "hero_entities", "villain_entities", "victim_entities",
    ]
    return rows, header


def gen_emotion(texts: list[str]) -> tuple[list[dict], list[str]]:
    # EMOTION-11: reduced from 20 → 11 columns to match the live schema.
    cols = [f"emotion_{i}" for i in range(11)]
    rows = []
    for i, t in enumerate(texts):
        row = {"text": t}
        for j, c in enumerate(cols):
            # alternate pattern so every column has both 0s and 1s across 10 rows
            row[c] = (i + j) % 2
        rows.append(row)
    return rows, ["text", *cols]


GENERATORS = {
    "bias": gen_bias,
    "ideology": gen_ideology,
    "propaganda": gen_propaganda,
    "frame": gen_frame,
    "narrative": gen_narrative,
    "emotion": gen_emotion,
}


def main() -> None:
    for split_idx, split in enumerate(SPLITS):
        for task_idx, (task, gen) in enumerate(GENERATORS.items()):
            texts = text_pool(split_idx, task_idx)
            rows, header = gen(texts)
            out = DATA_DIR / split / f"{task}.csv"
            _write(rows, header, out)
            print(f"  wrote {out}  rows={len(rows)} cols={len(header)}")


if __name__ == "__main__":
    main()
