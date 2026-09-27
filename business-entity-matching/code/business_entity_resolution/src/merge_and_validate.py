"""
High-Speed Streaming Merge & 19-Point Automated Submission Validation Suite.

Features:
1. Streaming Constant-Memory Merge: Merges outputs/shards/shard_XX.tsv into outputs/final_submission.tsv (<50 MB RAM).
2. 19-Point Comprehensive Validation: Validates exact header, row count, unique IDs, order consistency, candidate ID formats, and absence of duplicate IDs.
3. Automated Report Generation: Writes submission_validation_report.txt and mirrors to output/.
"""

import sys
import os
import re
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Set, Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC_DIR = Path(__file__).resolve().parent
PKG_DIR = SRC_DIR.parent
CODE_DIR = PKG_DIR.parent
PROJECT_ROOT = CODE_DIR.parent

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.src.test_inference import (
    S1_TEST_FILE,
    OUTPUTS_DIR,
    OUTPUT_DIR,
    DEFAULT_SUBMISSION_PATH
)

SHARDS_DIR = OUTPUTS_DIR / "shards"
VALID_CANDIDATE_REGEX = re.compile(r"^(?:S2|S3)-\d+$")


def merge_shards_streaming(
    shards_dir: Path = SHARDS_DIR,
    out_file: Path = DEFAULT_SUBMISSION_PATH,
    expected_shards: Optional[int] = None
) -> Path:
    """Streams all shard files in numerical order into the final submission file with constant <50 MB RAM."""
    print("=" * 80, flush=True)
    print("STREAMING MERGE OF SHARD FILES TO FINAL SUBMISSION", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()

    shard_files = sorted(shards_dir.glob("shard_*.tsv"), key=lambda p: int(re.search(r"shard_(\d+)", p.name).group(1)))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in: {shards_dir}")

    if expected_shards is not None and len(shard_files) != expected_shards:
        print(f"WARNING: Found {len(shard_files)} shards, expected {expected_shards}!", flush=True)

    print(f"Found {len(shard_files)} shard files to merge: {[p.name for p in shard_files]}", flush=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    with open(out_file, "w", encoding="utf-8", newline="") as f_out:
        # Write canonical header
        f_out.write("source1_entity_id\tmatched_entity_ids\n")

        for shard_idx, s_path in enumerate(shard_files):
            print(f"  Merging {s_path.name} ({s_path.stat().st_size / (1024**2):.2f} MB)...", flush=True)
            shard_rows = 0
            with open(s_path, "r", encoding="utf-8") as f_in:
                header = f_in.readline()  # Skip shard header if present
                for line in f_in:
                    f_out.write(line)
                    shard_rows += 1
            total_rows += shard_rows
            print(f"    -> Added {shard_rows:,} rows (Cumulative: {total_rows:,})", flush=True)

    total_sec = time.time() - t0
    final_mb = out_file.stat().st_size / (1024**2)
    print(f"\nSuccessfully merged {total_rows:,} rows in {total_sec:.2f}s ({total_rows/total_sec:,.0f} rows/s)!")
    print(f"Final Submission File: {out_file.resolve()} ({final_mb:.2f} MB)")
    print("=" * 80 + "\n", flush=True)
    return out_file


def validate_submission_file(
    submission_path: Path = DEFAULT_SUBMISSION_PATH,
    s1_file: Path = S1_TEST_FILE,
    validation_report_path: Path = OUTPUTS_DIR / "submission_validation_report.txt",
    limit_s1: Optional[int] = None
) -> bool:
    """Comprehensive 19-point automated validation of the submission file."""
    print("=" * 80, flush=True)
    print("RUNNING AUTOMATED FINAL SUBMISSION VALIDATION SUITE", flush=True)
    print("=" * 80, flush=True)

    if not submission_path.exists():
        print(f"ERROR: Submission file not found at {submission_path}", flush=True)
        return False

    # 1. Read Test S1 ground IDs
    print(f"1. Reading ground Source-1 IDs from {s1_file.name}...", flush=True)
    t0 = time.time()
    s1_ground_ids: List[str] = []
    with open(s1_file, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        s1_idx = header.index("entity_id")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) > s1_idx:
                s1_ground_ids.append(parts[s1_idx])
                if limit_s1 is not None and len(s1_ground_ids) >= limit_s1:
                    break

    ground_s1_count = len(s1_ground_ids)
    ground_s1_set = set(s1_ground_ids)
    print(f"   Loaded {ground_s1_count:,} test Source-1 IDs in {time.time()-t0:.2f}s.", flush=True)

    # 2. Inspect Submission File
    print(f"2. Inspecting and streaming {submission_path.name}...", flush=True)
    t0 = time.time()
    seen_s1_ids: Set[str] = set()
    submission_s1_ids: List[str] = []
    rows_missing = 0
    unexpected_rows = 0
    duplicate_s1_ids = 0
    rows_with_matches = 0
    rows_without_matches = 0
    total_predicted_matches = 0
    invalid_candidate_ids = 0
    duplicate_target_ids_in_row = 0
    malformed_rows = 0
    max_matches_row = 0
    min_matches_row = float("inf")
    match_counts = []

    with open(submission_path, "r", encoding="utf-8") as f:
        first_line = f.readline().rstrip("\r\n")
        correct_header = (first_line == "source1_entity_id\tmatched_entity_ids")

        for line_num, line in enumerate(f, start=2):
            raw = line.rstrip("\r\n")
            if not raw:
                continue

            parts = raw.split("\t")
            if len(parts) < 2:
                malformed_rows += 1
                continue

            s1_id = parts[0]
            matched_str = parts[1]

            if s1_id in seen_s1_ids:
                duplicate_s1_ids += 1
            seen_s1_ids.add(s1_id)
            submission_s1_ids.append(s1_id)

            if s1_id not in ground_s1_set:
                unexpected_rows += 1

            if not matched_str.strip():
                rows_without_matches += 1
                m_count = 0
            else:
                targets = matched_str.split(",")
                m_count = len(targets)
                rows_with_matches += 1
                total_predicted_matches += m_count

                if len(set(targets)) < m_count:
                    duplicate_target_ids_in_row += 1

                for tid in targets:
                    if not VALID_CANDIDATE_REGEX.match(tid.strip()):
                        invalid_candidate_ids += 1

            match_counts.append(m_count)
            if m_count > max_matches_row:
                max_matches_row = m_count
            if m_count < min_matches_row:
                min_matches_row = m_count

    submission_row_count = len(submission_s1_ids)
    rows_missing = ground_s1_count - len(seen_s1_ids.intersection(ground_s1_set))
    exact_order_match = (submission_s1_ids == s1_ground_ids)
    median_matches = float(np.median(match_counts)) if match_counts else 0.0
    file_size_mb = submission_path.stat().st_size / (1024**2)

    is_valid = (
        correct_header
        and rows_missing == 0
        and unexpected_rows == 0
        and duplicate_s1_ids == 0
        and submission_row_count == ground_s1_count
        and exact_order_match
        and malformed_rows == 0
        and invalid_candidate_ids == 0
        and duplicate_target_ids_in_row == 0
    )

    report = f"""==================================================
FINAL SUBMISSION VALIDATION REPORT
==================================================
Target Test S1 Entities     : {ground_s1_count:,}
Submission Output Rows      : {submission_row_count:,}
Header Format Correct       : {correct_header}
Missing S1 IDs              : {rows_missing}
Extra / Unexpected S1 IDs   : {unexpected_rows}
Duplicate S1 IDs            : {duplicate_s1_ids}
Malformed Rows              : {malformed_rows}
Exact Order Match with S1   : {exact_order_match}
Total Predicted Match Links : {total_predicted_matches:,}
Average Matches / S1        : {total_predicted_matches/ground_s1_count if ground_s1_count > 0 else 0.0:.4f}
Median Matches / S1         : {median_matches:.1f}
Minimum Matches / S1        : {min_matches_row if min_matches_row != float('inf') else 0}
Maximum Matches / S1        : {max_matches_row}
Rows with 0 Matches         : {rows_without_matches:,} ({(rows_without_matches/ground_s1_count)*100 if ground_s1_count > 0 else 0.0:.2f}%)
Rows with >= 1 Matches      : {rows_with_matches:,} ({(rows_with_matches/ground_s1_count)*100 if ground_s1_count > 0 else 0.0:.2f}%)
Invalid Target Candidate IDs: {invalid_candidate_ids}
Duplicate Targets in Row    : {duplicate_target_ids_in_row}
Submission File Size        : {file_size_mb:.2f} MB
Validation Status           : {"PASS (READY FOR SUBMISSION)" if is_valid else "FAIL"}
==================================================
"""
    print(report, flush=True)

    validation_report_path.write_text(report, encoding="utf-8")
    if OUTPUT_DIR.exists() and OUTPUT_DIR != OUTPUTS_DIR:
        (OUTPUT_DIR / validation_report_path.name).write_text(report, encoding="utf-8")

    return is_valid


def main():
    parser = argparse.ArgumentParser(description="Merge Shards & Validate Final Submission")
    parser.add_argument("--shards_dir", type=str, default=str(SHARDS_DIR), help="Directory containing shard TSVs")
    parser.add_argument("--output_file", type=str, default=str(DEFAULT_SUBMISSION_PATH), help="Merged final submission file")
    parser.add_argument("--s1_file", type=str, default=str(S1_TEST_FILE), help="Source-1 test TSV")
    parser.add_argument("--validate_only", action="store_true", help="Only validate existing submission file")
    parser.add_argument("--limit_s1", type=int, default=None, help="Limit S1 ground count for benchmark validation")
    args = parser.parse_args()

    out_file = Path(args.output_file)

    if not args.validate_only:
        merge_shards_streaming(
            shards_dir=Path(args.shards_dir),
            out_file=out_file
        )

    is_valid = validate_submission_file(
        submission_path=out_file,
        s1_file=Path(args.s1_file),
        limit_s1=args.limit_s1
    )

    if not is_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
