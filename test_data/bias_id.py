import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ======================================================
# CONFIG
# ======================================================

BASE_PATH = Path("data/data_new")
OUTPUT_PATH = Path("data/data_new")

SEED = 42

TEXT_COL = "text"
LABEL_COL = "bias_label"

INPUT_FILES = [
    BASE_PATH / "bias_1.csv",
    BASE_PATH / "bias_2.csv",
    BASE_PATH / "bias_3.csv",
    BASE_PATH / "bias_4.csv",
]

MIN_TEXT_LEN = 5

# ======================================================
# LOAD + MERGE (HANDLE MIXED LABEL NAMES)
# ======================================================

dfs = []

for file in INPUT_FILES:
    if not file.exists():
        raise FileNotFoundError(f"Missing file: {file}")
    
    df = pd.read_csv(file)

    # ---- Handle label column ----
    if "bias_label" in df.columns:
        pass
    elif "label" in df.columns:
        df = df.rename(columns={"label": "bias_label"})
    else:
        raise ValueError(f"No valid label column in {file}")

    # ---- Keep only required columns ----
    if TEXT_COL not in df.columns:
        raise ValueError(f"Missing text column in {file}")

    df = df[[TEXT_COL, "bias_label"]]

    dfs.append(df)

# Merge all
df = pd.concat(dfs, ignore_index=True)

print(f"\nMerged size: {len(df)}")

# ======================================================
# CLEAN TEXT
# ======================================================

df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()

empty = df[TEXT_COL].str.len() == 0
print(f"Empty text removed: {empty.sum()}")
df = df[~empty]

# ======================================================
# LABEL CLEANING
# ======================================================

df[LABEL_COL] = df[LABEL_COL].astype(int)

invalid = ~df[LABEL_COL].isin([0, 1])
print(f"Invalid labels removed: {invalid.sum()}")
df = df[~invalid]

# ======================================================
# TEXT LENGTH FILTER
# ======================================================

df["text_len"] = df[TEXT_COL].str.len()

too_short = df["text_len"] < MIN_TEXT_LEN
print(f"Too short removed: {too_short.sum()}")
df = df[~too_short]

# ======================================================
# NOISE FILTER
# ======================================================

def is_noisy(text):
    return any(char * 5 in text for char in set(text))

noise_mask = df[TEXT_COL].apply(is_noisy)
print(f"Noisy rows removed: {noise_mask.sum()}")
df = df[~noise_mask]

# ======================================================
# LEAKAGE CHECK
# ======================================================

dup_conflict = df.groupby(TEXT_COL)[LABEL_COL].nunique() > 1
conflict_texts = dup_conflict[dup_conflict].index

print(f"Conflicting label texts removed: {len(conflict_texts)}")
df = df[~df[TEXT_COL].isin(conflict_texts)]

# ======================================================
# DEDUPLICATION
# ======================================================

before = len(df)
df = df.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)
print(f"Duplicates removed: {before - len(df)}")

# ======================================================
# DATASET STATS
# ======================================================

print("\n=== DATASET STATS ===")
print(f"Final size: {len(df)}")

print("\nLabel Distribution:")
print(df[LABEL_COL].value_counts(normalize=True))
print(df[LABEL_COL].value_counts())

# ======================================================
# SHUFFLE
# ======================================================

df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# ======================================================
# STRATIFIED SPLIT
# ======================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df[LABEL_COL],
    random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df[LABEL_COL],
    random_state=SEED
)

# ======================================================
# SAVE
# ======================================================

for split in ["train", "test", "val"]:
    (OUTPUT_PATH / split).mkdir(parents=True, exist_ok=True)

train_df.to_csv(OUTPUT_PATH / "train/bias.csv", index=False)
test_df.to_csv(OUTPUT_PATH / "test/bias.csv", index=False)
val_df.to_csv(OUTPUT_PATH / "val/bias.csv", index=False)

print("\nSaved bias dataset successfully!")

# ======================================================
# BALANCE CHECK
# ======================================================

def check_balance(df, name):
    print(f"\n{name} Distribution:")
    print(df[LABEL_COL].value_counts(normalize=True))
    print(df[LABEL_COL].value_counts())

check_balance(train_df, "TRAIN")
check_balance(val_df, "VAL")
check_balance(test_df, "TEST")