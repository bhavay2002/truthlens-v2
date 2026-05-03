import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_PATH = Path("data/data")
OUTPUT_PATH = Path("data/data_new")

SEED = 42

TEXT_COL = "text"
FRAME_COLS = ["RE", "HI", "CO", "MO", "EC"]

MIN_TEXT_LEN = 5

# LOAD
train_df = pd.read_csv(BASE_PATH / "train/frame.csv")
test_df = pd.read_csv(BASE_PATH / "test/frame.csv")
val_df = pd.read_csv(BASE_PATH / "val/frame.csv")

df = pd.concat([train_df, test_df, val_df], ignore_index=True)

print(f"\nOriginal size: {len(df)}")

# SCHEMA
required_cols = {TEXT_COL} | set(FRAME_COLS)
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

# CLEAN TEXT
df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()

empty_text = df[TEXT_COL].str.len() == 0
print(f"Empty text removed: {empty_text.sum()}")
df = df[~empty_text]

# LABEL VALIDATION
for col in FRAME_COLS:
    df[col] = df[col].fillna(0).astype(int)
    invalid = ~df[col].isin([0, 1])
    print(f"Invalid labels removed ({col}): {invalid.sum()}")
    df = df[~invalid]

# TEXT LENGTH
df["text_len"] = df[TEXT_COL].str.len()

too_short = df["text_len"] < MIN_TEXT_LEN
print(f"Too short removed: {too_short.sum()}")
df = df[~too_short]

print(f"Long samples (kept): {(df['text_len'] > 2000).sum()}")

# NOISE
def is_noisy(text):
    return any(char * 5 in text for char in set(text))

noise_mask = df[TEXT_COL].apply(is_noisy)
print(f"Noisy rows removed: {noise_mask.sum()}")
df = df[~noise_mask]

# LEAKAGE FIX
df["label_signature"] = df[FRAME_COLS].astype(str).agg("_".join, axis=1)

dup_conflict = df.groupby(TEXT_COL)["label_signature"].nunique() > 1
conflict_texts = dup_conflict[dup_conflict].index

print(f"Conflicting label texts removed: {len(conflict_texts)}")
df = df[~df[TEXT_COL].isin(conflict_texts)]

df = df.drop(columns=["label_signature"])

# DEDUP
before = len(df)
df = df.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)
print(f"Duplicates removed: {before - len(df)}")

# STATS
print("\n=== DATASET STATS ===")
print(f"Final size: {len(df)}")

for col in FRAME_COLS:
    print(f"\n{col} distribution:")
    print(df[col].value_counts(normalize=True))

# SHUFFLE
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# SPLIT
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED)

# SAVE
for split in ["train", "test", "val"]:
    (OUTPUT_PATH / split).mkdir(parents=True, exist_ok=True)

train_df.to_csv(OUTPUT_PATH / "train/frame.csv", index=False)
test_df.to_csv(OUTPUT_PATH / "test/frame.csv", index=False)
val_df.to_csv(OUTPUT_PATH / "val/frame.csv", index=False)

print("\nSaved frame dataset successfully!")