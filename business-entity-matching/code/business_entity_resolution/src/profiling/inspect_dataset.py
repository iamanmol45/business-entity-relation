import sys
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Change this to your actual dataset/train path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"

FILES = [
    "train_source1.tsv",
    "train_source2.tsv",
    "train_source3.tsv",
    "train_ground_truth.tsv",
]


def inspect_file(file_path):
    print("\n" + "=" * 80)
    print(f"FILE: {file_path.name}")
    print("=" * 80)

    if not file_path.exists():
        print(f"FILE NOT FOUND: {file_path}")
        return

    size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"Size: {size_mb:.2f} MB")

    # Only read a tiny sample.
    df = pd.read_csv(
        file_path,
        sep="\t",
        nrows=5
    )

    print("\nColumns:")
    for col in df.columns:
        print(f"  {col}")

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.to_string(index=False))


def main():
    for filename in FILES:
        inspect_file(TRAIN_DIR / filename)


if __name__ == "__main__":
    main()