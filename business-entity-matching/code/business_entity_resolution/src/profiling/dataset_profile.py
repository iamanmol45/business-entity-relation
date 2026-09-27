import sys
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"

SOURCE_FILES = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
]

CHUNK_SIZE = 100_000


def profile_source(file_path):
    print("\n" + "=" * 80)
    print(f"PROFILING: {file_path.name}")
    print("=" * 80)

    total_rows = 0
    missing_counts = None
    unique_ids = set()
    duplicate_ids = 0

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        chunksize=CHUNK_SIZE,
        dtype=str
    ):
        total_rows += len(chunk)

        # Missing values
        chunk_missing = chunk.isna().sum()

        if missing_counts is None:
            missing_counts = chunk_missing
        else:
            missing_counts += chunk_missing

        # Entity IDs
        ids = chunk["entity_id"].dropna()

        duplicate_ids += ids.duplicated().sum()
        unique_ids.update(ids)

    print(f"\nRows: {total_rows:,}")

    print("\nColumns:")
    for column in missing_counts.index:
        print(f"  {column}")

    print("\nMissing values:")
    for column, count in missing_counts.items():
        percentage = (count / total_rows) * 100
        print(f"  {column}: {count:,} ({percentage:.2f}%)")

    print("\nEntity ID statistics:")
    print(f"  Unique IDs: {len(unique_ids):,}")
    print(f"  Duplicate IDs: {duplicate_ids:,}")


def profile_ground_truth(file_path):
    print("\n" + "=" * 80)
    print("PROFILING: train_ground_truth.tsv")
    print("=" * 80)

    total_rows = 0

    zero_matches = 0
    one_match = 0
    multiple_matches = 0

    match_distribution = {}

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        chunksize=CHUNK_SIZE,
        dtype=str
    ):
        total_rows += len(chunk)

        for value in chunk["matched_entity_ids"].fillna(""):
            value = value.strip()

            if not value:
                count = 0
            else:
                count = len(value.split(","))

            match_distribution[count] = (
                match_distribution.get(count, 0) + 1
            )

            if count == 0:
                zero_matches += 1
            elif count == 1:
                one_match += 1
            else:
                multiple_matches += 1

    print(f"\nGround-truth rows: {total_rows:,}")

    print("\nMatch distribution:")
    print(f"  Zero matches:     {zero_matches:,}")
    print(f"  One match:       {one_match:,}")
    print(f"  Multiple matches: {multiple_matches:,}")

    print("\nDetailed distribution:")

    for count in sorted(match_distribution):
        rows = match_distribution[count]
        percentage = (rows / total_rows) * 100

        print(
            f"  {count:>3} matches : "
            f"{rows:>10,} rows "
            f"({percentage:>6.2f}%)"
        )


def main():
    for filename in SOURCE_FILES:
        profile_source(TRAIN_DIR / filename)

    profile_ground_truth(
        TRAIN_DIR / "train_ground_truth.tsv"
    )


if __name__ == "__main__":
    main()