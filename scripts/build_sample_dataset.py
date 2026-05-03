"""Build a 10-row TruthLens sample dataset for end-to-end testing.

Writes ``data/truthlens_sample_dataset.csv`` with the schema expected by
``tests/test_e2e_dataset.py``:

  title, text,
  bias_label, ideology_label, propaganda_label,
  frame, CO, EC, HI, MO, RE,
  hero, villain, victim,
  hero_entities, villain_entities, victim_entities,
  emotion_0..emotion_19,
  dataset

Constraints honored:
  * 10 rows
  * >=4 biased and >=4 non-biased rows
  * All biased rows have propaganda_label = "yes"
  * >=3 distinct frame codes from {CO, EC, HI, MO, RE}
  * >=2 distinct source datasets
  * Each row has >=1 emotion flag set
  * Entity columns are JSON lists
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

# EMOTION-11: reduced from 20 → 11 names to match the canonical
# schema (src/features/emotion/emotion_schema.py).
EMOTION_NAMES = [
    "neutral", "admiration", "approval", "gratitude", "annoyance",
    "amusement", "curiosity", "disapproval", "love", "optimism",
    "anger",
]


def emo(*active: str) -> dict[str, int]:
    return {
        f"emotion_{i}": (1 if EMOTION_NAMES[i] in active else 0)
        for i in range(len(EMOTION_NAMES))
    }


def frame_flags(code: str) -> dict[str, int]:
    return {c: (1 if c == code else 0) for c in ("CO", "EC", "HI", "MO", "RE")}


def role_flags(*active: str) -> dict[str, int]:
    return {role: (1 if role in active else 0) for role in ("hero", "villain", "victim")}


ROWS = [
    {
        "title": "City Council Approves New Public Library Funding",
        "text": (
            "The city council voted on Tuesday to approve a $4.2 million "
            "budget for a new public library in the downtown district. "
            "Construction is expected to begin in March and finish within "
            "eighteen months, according to officials."
        ),
        "bias_label": "non-biased", "ideology_label": "center", "propaganda_label": "no",
        "frame": "EC",
        "hero_entities": ["City Council"],
        "villain_entities": [], "victim_entities": [],
        "active_roles": ("hero",),
        "emotions": ("approval", "optimism"),
        "dataset": "liar",
    },
    {
        "title": "Researchers Publish Peer-Reviewed Study on Coastal Erosion",
        "text": (
            "A team of marine geologists at the University of Maine published "
            "a peer-reviewed study this week documenting coastal erosion "
            "rates along the Atlantic seaboard. The data, gathered over twelve "
            "years, was released through the Journal of Coastal Research."
        ),
        "bias_label": "non-biased", "ideology_label": "neutral", "propaganda_label": "no",
        "frame": "HI",
        "hero_entities": ["University of Maine"],
        "villain_entities": [], "victim_entities": ["coastal communities"],
        "active_roles": ("hero", "victim"),
        "emotions": ("curiosity", "realization"),
        "dataset": "liar",
    },
    {
        "title": "Federal Reserve Holds Interest Rates Steady at March Meeting",
        "text": (
            "The Federal Reserve announced on Wednesday that it would hold "
            "the benchmark interest rate steady at its current range, citing "
            "stable inflation data and continued labor market resilience as "
            "the main reasons for the unanimous decision."
        ),
        "bias_label": "non-biased", "ideology_label": "center", "propaganda_label": "no",
        "frame": "EC",
        "hero_entities": [], "villain_entities": [], "victim_entities": [],
        "active_roles": (),
        "emotions": ("neutral",),
        "dataset": "liar",
    },
    {
        "title": "Local Volunteers Open Community Kitchen for Winter Months",
        "text": (
            "More than forty volunteers gathered Saturday morning to open a "
            "community kitchen serving free hot meals through the winter. "
            "Organizers said donations from local restaurants and grocers had "
            "already covered the first month of supplies."
        ),
        "bias_label": "non-biased", "ideology_label": "neutral", "propaganda_label": "no",
        "frame": "HI",
        "hero_entities": ["volunteers", "local restaurants"],
        "villain_entities": [], "victim_entities": ["unhoused residents"],
        "active_roles": ("hero", "victim"),
        "emotions": ("caring", "gratitude", "admiration"),
        "dataset": "fakenewsnet",
    },
    {
        "title": "Health Department Confirms Rise in Seasonal Flu Cases",
        "text": (
            "The state health department confirmed on Monday that seasonal "
            "flu cases had risen by twelve percent compared to the same "
            "period last year. Officials urged residents to consider "
            "vaccination and to follow standard hygiene precautions."
        ),
        "bias_label": "non-biased", "ideology_label": "neutral", "propaganda_label": "no",
        "frame": "HI",
        "hero_entities": ["state health department"],
        "villain_entities": [], "victim_entities": ["residents"],
        "active_roles": ("hero", "victim"),
        "emotions": ("caring", "concern" if False else "neutral"),
        "dataset": "fakenewsnet",
    },
    {
        "title": "SHOCKING Truth EXPOSED: Politicians Are LYING About Everything",
        "text": (
            "WAKE UP! The corrupt establishment politicians are HIDING the "
            "truth from you. Every single one of them is lying through their "
            "teeth while ordinary patriots suffer. They will NEVER tell you "
            "what is really going on — only WE can."
        ),
        "bias_label": "biased", "ideology_label": "right", "propaganda_label": "yes",
        "frame": "CO",
        "hero_entities": ["patriots"],
        "villain_entities": ["establishment politicians"],
        "victim_entities": ["ordinary citizens"],
        "active_roles": ("hero", "villain", "victim"),
        "emotions": ("anger", "disgust", "disapproval"),
        "dataset": "fakenewsnet",
    },
    {
        "title": "Greedy Corporations Are DESTROYING Our Planet for Profit",
        "text": (
            "Massive corporations are knowingly poisoning our rivers and "
            "burning our forests just to line the pockets of their billionaire "
            "executives. They feel ZERO remorse. The brave activists trying "
            "to stop them are silenced again and again."
        ),
        "bias_label": "biased", "ideology_label": "left", "propaganda_label": "yes",
        "frame": "MO",
        "hero_entities": ["activists"],
        "villain_entities": ["corporations", "billionaire executives"],
        "victim_entities": ["the planet", "communities"],
        "active_roles": ("hero", "villain", "victim"),
        "emotions": ("anger", "sadness", "disgust"),
        "dataset": "fakenewsnet",
    },
    {
        "title": "BREAKING: Mainstream Media Caught in MASSIVE Cover-Up Again",
        "text": (
            "Once AGAIN the dishonest mainstream media has been caught red "
            "handed pushing a fake narrative on the public. Real journalists "
            "have been silenced while propagandists are promoted. The American "
            "people deserve the TRUTH, not these lies."
        ),
        "bias_label": "biased", "ideology_label": "right", "propaganda_label": "yes",
        "frame": "CO",
        "hero_entities": ["independent journalists"],
        "villain_entities": ["mainstream media"],
        "victim_entities": ["the American people"],
        "active_roles": ("hero", "villain", "victim"),
        "emotions": ("anger", "disapproval", "annoyance"),
        "dataset": "liar",
    },
    {
        "title": "The Elites Want You POOR — Here's How They Are Doing It",
        "text": (
            "A cabal of unelected globalist elites is deliberately impoverishing "
            "working families through engineered inflation and rigged taxes. "
            "They laugh at us while we struggle to fill our gas tanks. Wake up "
            "before it is too late!"
        ),
        "bias_label": "biased", "ideology_label": "right", "propaganda_label": "yes",
        "frame": "EC",
        "hero_entities": [],
        "villain_entities": ["globalist elites"],
        "victim_entities": ["working families"],
        "active_roles": ("villain", "victim"),
        "emotions": ("anger", "disgust", "disappointment"),
        "dataset": "fakenewsnet",
    },
    {
        "title": "Supreme Court Hears Arguments in Free Speech Case",
        "text": (
            "The Supreme Court heard oral arguments on Wednesday in a closely "
            "watched free speech case involving public university campus "
            "policies. Justices from both wings of the bench questioned "
            "attorneys for nearly two hours before adjourning."
        ),
        "bias_label": "non-biased", "ideology_label": "center", "propaganda_label": "no",
        "frame": "RE",
        "hero_entities": [], "villain_entities": [], "victim_entities": [],
        "active_roles": (),
        "emotions": ("curiosity", "neutral"),
        "dataset": "liar",
    },
]


def build_row(spec: dict) -> dict:
    row = {
        "title": spec["title"],
        "text": spec["text"],
        "bias_label": spec["bias_label"],
        "ideology_label": spec["ideology_label"],
        "propaganda_label": spec["propaganda_label"],
        "frame": spec["frame"],
    }
    row.update(frame_flags(spec["frame"]))
    row.update(role_flags(*spec["active_roles"]))
    row["hero_entities"] = json.dumps(spec["hero_entities"])
    row["villain_entities"] = json.dumps(spec["villain_entities"])
    row["victim_entities"] = json.dumps(spec["victim_entities"])
    row.update(emo(*spec["emotions"]))
    row["dataset"] = spec["dataset"]
    return row


def main() -> Path:
    out_path = Path("data/truthlens_sample_dataset.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [build_row(s) for s in ROWS]

    fieldnames = [
        "title", "text",
        "bias_label", "ideology_label", "propaganda_label",
        "frame", "CO", "EC", "HI", "MO", "RE",
        "hero", "villain", "victim",
        "hero_entities", "villain_entities", "victim_entities",
        *[f"emotion_{i}" for i in range(len(EMOTION_NAMES))],
        "dataset",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path


if __name__ == "__main__":
    path = main()
    print(f"Wrote {path}")
