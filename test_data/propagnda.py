import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ======================================================
# CONFIG
# ======================================================

BASE_PATH = Path("data/data")
OUTPUT_PATH = Path("data/data_new")

SEED = 42

MIN_TEXT_LEN = 5

HEAD_CHARS = 800
TAIL_CHARS = 800

# ======================================================
# LOAD DATA
# ======================================================

train_df = pd.read_csv(BASE_PATH / "train/propaganda.csv")
test_df = pd.read_csv(BASE_PATH / "test/propaganda.csv")
val_df = pd.read_csv(BASE_PATH / "val/propaganda.csv")

df = pd.concat([train_df, test_df, val_df], ignore_index=True)

print(f"\nOriginal size: {len(df)}")

# ======================================================
# 1. SCHEMA VALIDATION
# ======================================================

required_cols = {"text", "propaganda_label"}
missing_cols = required_cols - set(df.columns)

if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

# ======================================================
# 2. BASIC CLEANING
# ======================================================

df["text"] = df["text"].astype(str).str.strip()

empty_text = df["text"].str.len() == 0
print(f"Empty text rows removed: {empty_text.sum()}")
df = df[~empty_text]

null_labels = df["propaganda_label"].isna()
print(f"Null label rows removed: {null_labels.sum()}")
df = df[~null_labels]

df["propaganda_label"] = df["propaganda_label"].astype(int)

# ======================================================
# 3. LABEL VALIDATION
# ======================================================

invalid_labels = ~df["propaganda_label"].isin([0, 1])
print(f"Invalid labels removed: {invalid_labels.sum()}")
df = df[~invalid_labels]

# ======================================================
# 4. TEXT QUALITY + SMART TRIM
# ======================================================

df["text_len"] = df["text"].str.len()

too_short = df["text_len"] < MIN_TEXT_LEN
print(f"Too short removed: {too_short.sum()}")
df = df[~too_short]

# -------- SMART TRIM WITH RANDOM SHIFT --------
def smart_trim(text):
    text = str(text).strip()

    if len(text) <= (HEAD_CHARS + TAIL_CHARS):
        return text

    max_shift = min(400, len(text) - (HEAD_CHARS + TAIL_CHARS))
    shift = np.random.randint(0, max_shift + 1) if max_shift > 0 else 0

    start = shift

    return (
        text[start:start + HEAD_CHARS]
        + " [SEP] "
        + text[-TAIL_CHARS:]
    )

# Apply trimming
too_long = df["text_len"] > (HEAD_CHARS + TAIL_CHARS)
print(f"Too long (will be trimmed): {too_long.sum()}")

df.loc[too_long, "text"] = df.loc[too_long, "text"].apply(smart_trim)

# Recompute length
df["text_len"] = df["text"].str.len()

# ======================================================
# 5. NOISE FILTER
# ======================================================

def is_noisy(text):
    return any(char * 5 in text for char in set(text))

noise_mask = df["text"].apply(is_noisy)
print(f"Noisy rows removed: {noise_mask.sum()}")
df = df[~noise_mask]

# ======================================================
# 6. LEAKAGE CHECK
# ======================================================

dup_conflict = df.groupby("text")["propaganda_label"].nunique() > 1
conflict_texts = dup_conflict[dup_conflict].index

print(f"Conflicting label texts removed: {len(conflict_texts)}")
df = df[~df["text"].isin(conflict_texts)]

# ======================================================
# 7. DEDUPLICATION
# ======================================================

before_dedup = len(df)
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
print(f"Duplicates removed: {before_dedup - len(df)}")

# ======================================================
# 8. CLASS BALANCING (UNDERSAMPLE)
# ======================================================

df_0 = df[df["propaganda_label"] == 0]
df_1 = df[df["propaganda_label"] == 1]

print("\nBefore balancing:")
print(df["propaganda_label"].value_counts())

# Downsample class 1 to 1.5x class 0
df_1_down = df_1.sample(int(len(df_0) * 1.5), random_state=SEED)

df = pd.concat([df_0, df_1_down]).sample(frac=1, random_state=SEED).reset_index(drop=True)

print("\nAfter balancing:")
print(df["propaganda_label"].value_counts(normalize=True))

# ======================================================
# 9. DATASET STATS
# ======================================================

print("\n=== DATASET STATS ===")
print(f"Final size: {len(df)}")

print("\nLabel Distribution:")
print(df["propaganda_label"].value_counts(normalize=True))

print("\nText Length Stats:")
print(df["text_len"].describe())

# ======================================================
# 10. SHUFFLE
# ======================================================

df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# ======================================================
# 11. STRATIFIED SPLIT
# ======================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["propaganda_label"],
    random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["propaganda_label"],
    random_state=SEED
)

# ======================================================
# 12. SAVE
# ======================================================

for split in ["train", "test", "val"]:
    (OUTPUT_PATH / split).mkdir(parents=True, exist_ok=True)

train_df.to_csv(OUTPUT_PATH / "train/propaganda.csv", index=False)
test_df.to_csv(OUTPUT_PATH / "test/propaganda.csv", index=False)
val_df.to_csv(OUTPUT_PATH / "val/propaganda.csv", index=False)

print("\nSaved new dataset successfully!")

# ======================================================
# 13. BALANCE CHECK
# ======================================================

def check_balance(df, name):
    print(f"\n{name} Distribution:")
    print(df["propaganda_label"].value_counts(normalize=True))
    print(df["propaganda_label"].value_counts())

check_balance(train_df, "TRAIN")
check_balance(val_df, "VAL")
check_balance(test_df, "TEST")