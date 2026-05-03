

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

# ======================================================
# CONFIG
# ======================================================

BASE_PATH = Path("data/data")
OUTPUT_PATH = Path("data/data_new")

SEED = 42
VALID_LABELS = {0, 1, 2}

MIN_TEXT_LEN = 5

# ======================================================
# LOAD DATA
# ======================================================

train_df = pd.read_csv(BASE_PATH / "train/ideology.csv")
test_df = pd.read_csv(BASE_PATH / "test/ideology.csv")
val_df = pd.read_csv(BASE_PATH / "val/ideology.csv")

df = pd.concat([train_df, test_df, val_df], ignore_index=True)

original_size = len(df)
print(f"\nOriginal size: {original_size}")

# ======================================================
# 1. SCHEMA VALIDATION
# ======================================================

required_cols = {"text", "ideology_label"}
missing_cols = required_cols - set(df.columns)

if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

# ======================================================
# 2. BASIC CLEANING
# ======================================================

df["text"] = df["text"].astype(str).str.strip()

empty_text = df["text"].str.len() == 0
print(f"Empty text removed: {empty_text.sum()}")
df = df[~empty_text]

null_labels = df["ideology_label"].isna()
print(f"Null labels removed: {null_labels.sum()}")
df = df[~null_labels]

df["ideology_label"] = df["ideology_label"].astype(int)

# ======================================================
# 3. LABEL VALIDATION
# ======================================================

invalid_labels = ~df["ideology_label"].isin(VALID_LABELS)
print(f"Invalid labels removed: {invalid_labels.sum()}")
df = df[~invalid_labels]

# ======================================================
# 4. TEXT LENGTH ANALYSIS (NO HARD FILTER)
# ======================================================

df["text_len"] = df["text"].str.len()

print("\n=== RAW LENGTH STATS ===")
print(df["text_len"].describe())

p95 = int(np.percentile(df["text_len"], 95))
p99 = int(np.percentile(df["text_len"], 99))

print(f"P95 length: {p95}")
print(f"P99 length: {p99}")

# Only remove extremely short text
too_short = df["text_len"] < MIN_TEXT_LEN
print(f"Too short removed: {too_short.sum()}")
df = df[~too_short]

# DO NOT REMOVE LONG TEXT
too_long = df["text_len"] > p99
print(f"Very long samples (kept, will truncate later): {too_long.sum()}")

# ======================================================
# 5. NOISE DETECTION
# ======================================================

def is_noisy(text):
    return any(char * 5 in text for char in set(text))

noise_mask = df["text"].apply(is_noisy)
print(f"Noisy rows removed: {noise_mask.sum()}")
df = df[~noise_mask]

# ======================================================
# 6. LEAKAGE CHECK (CRITICAL)
# ======================================================

dup_conflict = df.groupby("text")["ideology_label"].nunique() > 1
conflict_texts = dup_conflict[dup_conflict].index

print(f"Conflicting-label texts removed: {len(conflict_texts)}")
df = df[~df["text"].isin(conflict_texts)]

# ======================================================
# 7. DEDUPLICATION
# ======================================================

before = len(df)
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
print(f"Duplicates removed: {before - len(df)}")

# ======================================================
# 8. SAFETY CHECK (CRITICAL)
# ======================================================

final_size = len(df)
print(f"\nFinal size after cleaning: {final_size}")

if final_size < 0.5 * original_size:
    raise ValueError(
        "🚨 More than 50% data removed — check cleaning logic!"
    )

# ======================================================
# 9. DATASET STATS
# ======================================================

print("\n=== FINAL DATASET STATS ===")

label_dist = df["ideology_label"].value_counts(normalize=True)
print("\nLabel Distribution:")
print(label_dist)

print("\nText Length Stats:")
print(df["text_len"].describe())

# ======================================================
# 10. SHUFFLE
# ======================================================

df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# ======================================================
# 11. SAFE STRATIFIED SPLIT
# ======================================================

label_counts = Counter(df["ideology_label"])
print("\nLabel counts:", label_counts)

min_count = min(label_counts.values())

if min_count < 3:
    print("⚠️ Too few samples for stratified split → using random split")

    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=SEED
    )

    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED
    )

else:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["ideology_label"],
        random_state=SEED
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["ideology_label"],
        random_state=SEED
    )

# ======================================================
# 12. SAVE
# ======================================================

for split in ["train", "test", "val"]:
    (OUTPUT_PATH / split).mkdir(parents=True, exist_ok=True)

train_df.to_csv(OUTPUT_PATH / "train/ideology.csv", index=False)
test_df.to_csv(OUTPUT_PATH / "test/ideology.csv", index=False)
val_df.to_csv(OUTPUT_PATH / "val/ideology.csv", index=False)

print("\nSaved new ideology dataset!")

# ======================================================
# 13. BALANCE CHECK
# ======================================================

def check_balance(df, name):
    print(f"\n{name} Distribution:")
    print(df["ideology_label"].value_counts(normalize=True))
    print(df["ideology_label"].value_counts())

check_balance(train_df, "TRAIN")
check_balance(val_df, "VAL")
check_balance(test_df, "TEST")

# ======================================================
# 14. SAVE DATA STATS (OPTIONAL BUT RECOMMENDED)
# ======================================================

stats = {
    "original_size": int(original_size),
    "final_size": int(final_size),
    "label_distribution": label_dist.to_dict(),
    "p95_length": int(p95),
    "p99_length": int(p99),
}

Path("data_stats_ideology.json").write_text(
    pd.Series(stats).to_json()
)

print("\nSaved dataset stats → data_stats_ideology.json")