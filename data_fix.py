import pandas as pd
import logging
from typing import List, Dict

# EMOTION-11: emotion column count reduced from 20 → 11.
# EXPECTED_COLS = 17 non-emotion columns + N emotion columns + 1 dataset.
_NUM_EMOTION_COLS = 11
EXPECTED_COLS = 18 + _NUM_EMOTION_COLS  # was 38; now 29

# -----------------------------
# 🧠 SCHEMA
# -----------------------------
SCHEMA = {
    "title": "str",
    "text": "str",
    "bias_label": "bin",
    "ideology_label": "bin",
    "propaganda_label": "bin",
    "frame": "str",
    "CO": "bin", "EC": "bin", "HI": "bin", "MO": "bin", "RE": "bin",
    "hero": "bin", "villain": "bin", "victim": "bin",
    "hero_entities": "str",
    "villain_entities": "str",
    "victim_entities": "str",
    **{f"emotion_{i}": "bin" for i in range(_NUM_EMOTION_COLS)},
    "dataset": "str"
}

COLUMNS = list(SCHEMA.keys())

# -----------------------------
# 🧾 Logging
# -----------------------------
logging.basicConfig(
    filename="csv_repair.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


class AdvancedCSVPipeline:
    def __init__(self, path: str):
        self.path = path

        self.total_rows = 0
        self.recovered_rows = 0
        self.bad_rows: List[str] = []

        self.invalid_row_indices = set()
        self.dropped_rows = 0

        self.column_errors: Dict[str, int] = {col: 0 for col in COLUMNS}
        self.missing_counts: Dict[str, int] = {col: 0 for col in COLUMNS}

    # -----------------------------
    # 🧠 Helpers
    # -----------------------------
    def is_missing(self, val):
        return val is None or str(val).strip() == ""

    def normalize_binary(self, val, col, row_idx):
        if self.is_missing(val):
            self.missing_counts[col] += 1
            return None

        val = str(val).strip().lower()

        mapping = {
            "1": 1, "true": 1, "yes": 1,
            "0": 0, "false": 0, "no": 0
        }

        if val in mapping:
            return mapping[val]

        try:
            num = float(val)
            return 1 if num > 0 else 0
        except:
            pass

        # ❌ INVALID → mark row for deletion
        self.column_errors[col] += 1
        self.invalid_row_indices.add(row_idx)

        logging.warning(f"[INVALID] row={row_idx} col={col} val={val}")

        return None

    # -----------------------------
    # 🧠 Parser
    # -----------------------------
    def parse_line(self, line: str):
        row, current = [], []
        in_quotes = False

        i = 0
        while i < len(line):
            char = line[i]

            if char == '"':
                if i + 1 < len(line) and line[i + 1] == '"':
                    current.append('"')
                    i += 1
                else:
                    in_quotes = not in_quotes

            elif char == "," and not in_quotes:
                row.append("".join(current))
                current = []
            else:
                current.append(char)

            i += 1

        row.append("".join(current).rstrip("\n"))
        return row, in_quotes

    # -----------------------------
    # 🔧 Structure Fix
    # -----------------------------
    def fix_structure(self, parsed):
        if len(parsed) == EXPECTED_COLS:
            return parsed

        self.recovered_rows += 1

        if len(parsed) > EXPECTED_COLS:
            return parsed[:EXPECTED_COLS - 1] + [
                ",".join(parsed[EXPECTED_COLS - 1:])
            ]

        return parsed + [""] * (EXPECTED_COLS - len(parsed))

    # -----------------------------
    # 🧬 Repair
    # -----------------------------
    def repair_by_schema(self, row, row_idx):
        repaired = []

        for col, val in zip(COLUMNS, row):
            dtype = SCHEMA[col]

            if dtype == "bin":
                fixed = self.normalize_binary(val, col, row_idx)
                repaired.append(fixed)
            else:
                if self.is_missing(val):
                    self.missing_counts[col] += 1
                    repaired.append(None)
                else:
                    val = str(val).replace('""', '"').strip()
                    repaired.append(val)

        return repaired

    # -----------------------------
    # 📥 Load + Clean
    # -----------------------------
    def load(self):
        rows = []
        buffer = ""

        with open(self.path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                self.total_rows += 1
                buffer += line

                parsed, in_quotes = self.parse_line(buffer)

                if in_quotes:
                    continue

                try:
                    structured = self.fix_structure(parsed)
                    repaired = self.repair_by_schema(structured, idx)

                    rows.append((idx, repaired))
                    buffer = ""

                except Exception as e:
                    self.bad_rows.append(buffer)
                    logging.error(str(e))
                    buffer = ""

        # 🔥 DROP INVALID ROWS
        valid_rows = [
            r for idx, r in rows
            if idx not in self.invalid_row_indices
        ]

        self.dropped_rows = len(rows) - len(valid_rows)

        df = pd.DataFrame(valid_rows[1:], columns=COLUMNS)
        return df

    # -----------------------------
    # 📊 Report
    # -----------------------------
    def report(self, df):
        print("\n=== DATA QUALITY REPORT ===")

        print(f"Total rows: {self.total_rows}")
        print(f"Dropped invalid rows: {self.dropped_rows}")
        print(f"Hard failures: {len(self.bad_rows)}")

        print("\n--- Column Errors ---")
        for col, count in self.column_errors.items():
            if count > 0:
                print(f"{col}: {count}")

        print("\n--- Missing Values ---")
        for col, count in self.missing_counts.items():
            if count > 0:
                print(f"{col}: {count}")

    # -----------------------------
    # 💾 Save (OVERWRITE SAME FILE)
    # -----------------------------
    def save(self, df):
        df.to_csv(self.path, index=False)


# -----------------------------
# 🚀 RUN
# -----------------------------
if __name__ == "__main__":
    path = r"data\unified_dataset_validation.csv"
    

    pipe = AdvancedCSVPipeline(path)

    df = pipe.load()

    pipe.report(df)

    pipe.save(df)

    print("\n✅ Dataset cleaned and overwritten (invalid rows removed)")