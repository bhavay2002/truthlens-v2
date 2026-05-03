from pathlib import Path
import pandas as pd

# =========================
# CONFIG
# =========================
SRC_DIR = Path("data/data")
DST_DIR = Path("datasmall")
N_ROWS = 10

# =========================
# CREATE SMALL DATASET
# =========================
def create_small_dataset(src_dir: Path, dst_dir: Path, n_rows: int = 10):
    if not src_dir.exists():
        raise ValueError(f"Source directory not found: {src_dir}")

    for subfolder in src_dir.iterdir():
        if not subfolder.is_dir():
            continue

        # Create matching subfolder
        target_subfolder = dst_dir / subfolder.name
        target_subfolder.mkdir(parents=True, exist_ok=True)

        for file in subfolder.iterdir():
            if file.is_file():
                try:
                    if file.suffix == ".csv":
                        df = pd.read_csv(file)
                        df_small = df.head(n_rows)
                        df_small.to_csv(target_subfolder / file.name, index=False)

                    elif file.suffix in [".parquet"]:
                        df = pd.read_parquet(file)
                        df_small = df.head(n_rows)
                        df_small.to_parquet(target_subfolder / file.name, index=False)

                    else:
                        print(f"Skipping unsupported file: {file.name}")

                except Exception as e:
                    print(f"Error processing {file}: {e}")

    print("✅ datasmall created successfully")

# =========================
# RUN
# =========================
create_small_dataset(SRC_DIR, DST_DIR, N_ROWS)