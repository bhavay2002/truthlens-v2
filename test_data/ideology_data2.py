import pandas as pd
import numpy as np
from pathlib import Path

# ======================================================
# CONFIG
# ======================================================

INPUT_PATH = Path("data/data")
OUTPUT_PATH = Path("data/data_new")

TEXT_COL = "text"
LABEL_COL = "ideology_label"

HEAD_CHARS = 800
TAIL_CHARS = 800

SEED = 42
rng = np.random.RandomState(SEED)  # reproducible randomness

# ======================================================
# FUNCTION: SMART TRIM WITH RANDOM SHIFT
# ======================================================

def smart_trim(text):
    text = str(text).strip()

    if len(text) <= (HEAD_CHARS + TAIL_CHARS):
        return text

    # Safe dynamic shift
    max_shift = min(600, len(text) - (HEAD_CHARS + TAIL_CHARS))
    shift = rng.randint(0, max_shift + 1) if max_shift > 0 else 0

    start = shift

    return (
        text[start:start + HEAD_CHARS]
        + " [SEP] "
        + text[-TAIL_CHARS:]
    )

# ======================================================
# PROCESS SPLIT FILES
# ======================================================

def process_file(input_file, output_file):
    df = pd.read_csv(input_file)

    # Keep only required columns
    df = df[[TEXT_COL, LABEL_COL]]

    # Clean text
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()

    # Apply trimming
    df[TEXT_COL] = df[TEXT_COL].apply(smart_trim)

    # Remove empty text if any
    df = df[df[TEXT_COL].str.len() > 0]

    # Reset index
    df = df.reset_index(drop=True)

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"Processed: {input_file} → {output_file} | Rows: {len(df)}")

# ======================================================
# RUN FOR TRAIN / TEST / VAL
# ======================================================

splits = ["train", "test", "val"]

for split in splits:
    input_file = INPUT_PATH / split / "ideology.csv"
    output_file = OUTPUT_PATH / split / "ideology_new.csv"

    process_file(input_file, output_file)

print("\n✅ All files processed successfully!")

# ======================================================
# VERIFY LENGTH DISTRIBUTION
# ======================================================

for split in splits:
    file_path = OUTPUT_PATH / split / "ideology_new.csv"
    df = pd.read_csv(file_path)

    print(f"\n{split.upper()} LENGTH STATS:")
    print(df["text"].str.len().describe())