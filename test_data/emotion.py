import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ======================================================
# CONFIG
# ======================================================

BASE_PATH = Path("data/data")
OUTPUT_PATH = Path("data/data_new")

SEED = 42

TEXT_COL = "text"

# EMOTION-11: emotion_0 → emotion_10 (reduced from 20)
EMOTION_COLS = [f"emotion_{i}" for i in range(11)]

MIN_TEXT_LEN = 5

# ======================================================
# LOAD DATA
# ======================================================

train_df = pd.read_csv(BASE_PATH / "train/emotion.csv")
test_df = pd.read_csv(BASE_PATH / "test/emotion.csv")
val_df = pd.read_csv(BASE_PATH / "val/emotion.csv")

df = pd.concat([train_df, test_df, val_df], ignore_index=True)

print(f"\nOriginal size: {len(df)}")

# ======================================================
# 1. SCHEMA VALIDATION
# ======================================================

required_cols = {TEXT_COL} | set(EMOTION_COLS)
missing_cols = required_cols - set(df.columns)

if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

# ======================================================
# 2. CLEAN TEXT
# ======================================================

df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()

empty_text = df[TEXT_COL].str.len() == 0
print(f"Empty text removed: {empty_text.sum()}")
df = df[~empty_text]

# ======================================================
# 3. LABEL VALIDATION (MULTI-LABEL 11)
# ======================================================

for col in EMOTION_COLS:
    df[col] = df[col].fillna(0).astype(int)
    invalid = ~df[col].isin([0, 1])
    print(f"Invalid labels removed ({col}): {invalid.sum()}")
    df = df[~invalid]

# ======================================================
# 4. TEXT LENGTH CHECK (NO TRIM)
# ======================================================

df["text_len"] = df[TEXT_COL].str.len()

too_short = df["text_len"] < MIN_TEXT_LEN
print(f"Too short removed: {too_short.sum()}")
df = df[~too_short]

print(f"Long samples (kept): {(df['text_len'] > 2000).sum()}")

# ======================================================
# 5. NOISE FILTER
# ======================================================

def is_noisy(text):
    return any(char * 5 in text for char in set(text))

noise_mask = df[TEXT_COL].apply(is_noisy)
print(f"Noisy rows removed: {noise_mask.sum()}")
df = df[~noise_mask]

# ======================================================
# 6. LEAKAGE CHECK (MULTI-LABEL SAFE)
# ======================================================

df["label_signature"] = df[EMOTION_COLS].astype(str).agg("_".join, axis=1)

dup_conflict = df.groupby(TEXT_COL)["label_signature"].nunique() > 1
conflict_texts = dup_conflict[dup_conflict].index

print(f"Conflicting label texts removed: {len(conflict_texts)}")

df = df[~df[TEXT_COL].isin(conflict_texts)]
df = df.drop(columns=["label_signature"])

# ======================================================
# 7. DEDUPLICATION
# ======================================================

before = len(df)
df = df.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)
print(f"Duplicates removed: {before - len(df)}")

# ======================================================
# 8. DATASET STATS
# ======================================================

print("\n=== DATASET STATS ===")
print(f"Final size: {len(df)}")

for col in EMOTION_COLS:
    print(f"\n{col} distribution:")
    print(df[col].value_counts(normalize=True))

# ======================================================
# 9. SHUFFLE
# ======================================================

df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# ======================================================
# 10. SPLIT (NO STRATIFY)
# ======================================================

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED)

# ======================================================
# 11. SAVE
# ======================================================

for split in ["train", "test", "val"]:
    (OUTPUT_PATH / split).mkdir(parents=True, exist_ok=True)

train_df.to_csv(OUTPUT_PATH / "train/emotion.csv", index=False)
test_df.to_csv(OUTPUT_PATH / "test/emotion.csv", index=False)
val_df.to_csv(OUTPUT_PATH / "val/emotion.csv", index=False)

print("\nSaved emotion dataset successfully!")