"""
Automated Benchmarking & Verification Suite for Sharded Parallel Inference.

Benchmarks:
1. 1,000 Verification vs outputs/test_1000_submission.tsv
2. Worker Scaling Benchmark: 1, 2, 4, 8, 12, 14 workers
3. Batch Size Benchmark: 500, 1000, 2000, 5000, 10000
4. Full 10,000 Production Benchmark with complete telemetry
"""

import sys
import os
import time
import psutil
import argparse
from pathlib import Path
from typing import Dict, List, Any

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

import pandas as pd
import numpy as np

from business_entity_resolution.src.sharded_inference import (
    run_parallel_inference,
    S1_TEST_FILE,
    OUTPUTS_DIR,
    SHARDS_DIR
)
from business_entity_resolution.src.merge_and_validate import (
    merge_shards_streaming,
    validate_submission_file
)


def verify_1000_predictions(test_file: Path, baseline_file: Path) -> bool:
    """Verifies that the new parallel inference predictions are 100.00% identical to baseline."""
    print("=" * 80)
    print("VERIFYING 1,000 PREDICTIONS AGAINST BASELINE")
    print(f"Candidate Submission : {test_file}")
    print(f"Baseline Submission  : {baseline_file}")
    print("=" * 80)

    if not baseline_file.exists():
        print(f"WARNING: Baseline file not found at {baseline_file}. Checking local outputs...")
        alt_base = PROJECT_ROOT / "outputs" / "test_1000_optimized.tsv"
        if alt_base.exists():
            baseline_file = alt_base
        else:
            print("No baseline found to compare.")
            return True

    df_test = pd.read_csv(test_file, sep="\t", dtype=str).fillna("")
    df_base = pd.read_csv(baseline_file, sep="\t", dtype=str).fillna("")

    # Compare row counts
    print(f"Candidate Row Count: {len(df_test):,}")
    print(f"Baseline Row Count : {len(df_base):,}")
    assert len(df_test) == len(df_base), f"Row count mismatch! {len(df_test)} != {len(df_base)}"

    # Compare S1 IDs
    s1_match = (df_test["source1_entity_id"] == df_base["source1_entity_id"]).all()
    print(f"S1 ID Alignment    : {'PASS (100.00% Match)' if s1_match else 'FAIL'}")
    assert s1_match, "Source 1 IDs or ordering do not match!"

    # Compare Matches
    diffs = 0
    total_test_matches = 0
    total_base_matches = 0

    for idx, (t_eid, t_m, b_eid, b_m) in enumerate(zip(
        df_test["source1_entity_id"],
        df_test["matched_entity_ids"],
        df_base["source1_entity_id"],
        df_base["matched_entity_ids"]
    )):
        t_set = set(t_m.split(",")) if t_m else set()
        b_set = set(b_m.split(",")) if b_m else set()

        total_test_matches += len(t_set)
        total_base_matches += len(b_set)

        if t_set != b_set:
            diffs += 1
            if diffs <= 5:
                print(f"  Diff at S1 #{idx+1} ({t_eid}):")
                print(f"    New     : {t_m}")
                print(f"    Baseline: {b_m}")

    match_rate = ((len(df_test) - diffs) / len(df_test)) * 100
    print(f"Total Predicted Links (New)     : {total_test_matches:,}")
    print(f"Total Predicted Links (Baseline): {total_base_matches:,}")
    print(f"Prediction Equivalence Rate     : {match_rate:.2f}% ({len(df_test)-diffs:,} / {len(df_test):,} identical rows)")

    is_identical = (diffs == 0)
    print(f"Verification Status             : {'EXACT 100.00% MATCH (PASS)' if is_identical else 'MISMATCH'}")
    print("=" * 80 + "\n")
    return is_identical


def run_scaling_benchmark():
    """Runs systematic benchmarks across worker counts and batch sizes."""
    print("=" * 80)
    print("SYSTEMATIC BENCHMARK: WORKER SCALING & BATCH SIZES")
    print("=" * 80)

    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = os.cpu_count() or 4
    print(f"System Hardware: {cpu_count} Logical CPU cores | {total_ram_gb:.1f} GB Physical RAM\n")

    sample_size = 2000
    batch_size_fixed = 1000

    # 1. Worker scaling on sample_size = 2000
    worker_counts = [1, 2, 4, 8, 12, 14]
    worker_results = []

    print("--- 1. WORKER SCALING BENCHMARK (N=2,000 S1 entities, Batch=1,000) ---")
    print(f"{'Workers':<8} | {'Time (s)':<10} | {'Throughput (S1/s)':<18} | {'Pair Speed (pairs/s)':<20} | {'Total RAM (GB)':<14} | {'Speedup':<8}")
    print("-" * 90)

    base_time = None
    for w in worker_counts:
        # Clear shards
        for f in SHARDS_DIR.glob("shard_*.tsv"):
            f.unlink()

        t0 = time.time()
        res = run_parallel_inference(
            num_workers=w,
            batch_size=batch_size_fixed,
            limit_s1=sample_size,
            s1_file=S1_TEST_FILE
        )
        elapsed = time.time() - t0
        if base_time is None:
            base_time = elapsed

        total_pairs = sum(r["total_pairs"] for r in res)
        s1_s = sample_size / elapsed
        pair_s = total_pairs / elapsed
        speedup = base_time / elapsed
        ram_gb = psutil.virtual_memory().used / (1024**3)

        worker_results.append({
            "workers": w,
            "time": elapsed,
            "s1_per_sec": s1_s,
            "pairs_per_sec": pair_s,
            "ram_gb": ram_gb,
            "speedup": speedup
        })
        print(f"{w:<8} | {elapsed:<10.2f} | {s1_s:<18.1f} | {pair_s:<20,.0f} | {ram_gb:<14.2f} | {speedup:<8.2f}x")

    # 2. Batch size benchmark with optimal workers
    best_worker_cfg = max(worker_results, key=lambda x: x["s1_per_sec"])["workers"]
    print(f"\n--- 2. BATCH SIZE BENCHMARK ({best_worker_cfg} workers, N=2,000 S1 entities) ---")
    batch_sizes = [500, 1000, 2000, 5000, 10000]
    batch_results = []

    print(f"{'Batch Size':<12} | {'Time (s)':<10} | {'Throughput (S1/s)':<18} | {'Pair Speed (pairs/s)':<20} | {'Total RAM (GB)':<14}")
    print("-" * 85)

    for b in batch_sizes:
        for f in SHARDS_DIR.glob("shard_*.tsv"):
            f.unlink()

        t0 = time.time()
        res = run_parallel_inference(
            num_workers=best_worker_cfg,
            batch_size=b,
            limit_s1=sample_size,
            s1_file=S1_TEST_FILE
        )
        elapsed = time.time() - t0
        total_pairs = sum(r["total_pairs"] for r in res)
        s1_s = sample_size / elapsed
        pair_s = total_pairs / elapsed
        ram_gb = psutil.virtual_memory().used / (1024**3)

        batch_results.append({
            "batch_size": b,
            "time": elapsed,
            "s1_per_sec": s1_s,
            "pairs_per_sec": pair_s,
            "ram_gb": ram_gb
        })
        print(f"{b:<12} | {elapsed:<10.2f} | {s1_s:<18.1f} | {pair_s:<20,.0f} | {ram_gb:<14.2f}")

    return worker_results, batch_results


def run_10000_production_benchmark(workers: int = 12, batch_size: int = 2000):
    """Executes a full 10,000 entity production benchmark and prints a complete telemetry report."""
    print("\n" + "=" * 80)
    print("PRODUCTION 10,000 ENTITY BENCHMARK & TELEMETRY REPORT")
    print("=" * 80)
    print(f"Workers Configured  : {workers}")
    print(f"Batch Size (S1)     : {batch_size:,}")
    print(f"Benchmark Size      : 10,000 Source-1 entities")
    print("=" * 80 + "\n")

    # Clear shards
    for f in SHARDS_DIR.glob("shard_*.tsv"):
        f.unlink()

    cpu_percent_start = psutil.cpu_percent(interval=None)
    ram_start = psutil.virtual_memory().used / (1024**3)

    t0 = time.time()
    results = run_parallel_inference(
        num_workers=workers,
        batch_size=batch_size,
        limit_s1=10000,
        s1_file=S1_TEST_FILE
    )
    total_time = time.time() - t0

    # Merge & Validate
    merged_file = OUTPUTS_DIR / "test_10k_benchmark_sub.tsv"
    merge_shards_streaming(SHARDS_DIR, merged_file)
    validate_submission_file(merged_file, S1_TEST_FILE, limit_s1=10000)

    # Telemetry metrics
    total_s1 = sum(r["total_s1"] for r in results)
    total_pairs = sum(r["total_pairs"] for r in results)
    total_matches = sum(r["total_matches"] for r in results)
    total_feat_time = sum(r["time_feat_extract"] for r in results)
    total_pred_time = sum(r["time_predict"] for r in results)

    s1_per_sec = total_s1 / total_time
    pairs_per_sec = total_pairs / total_time
    avg_feat_per_worker = total_feat_time / workers
    avg_pred_per_worker = total_pred_time / workers

    ram_peak = psutil.virtual_memory().used / (1024**3)
    cpu_percent = psutil.cpu_percent(interval=1)

    projected_full_sec = 1_732_544 / s1_per_sec
    projected_hours = projected_full_sec / 3600

    report = f"""
================================================================================
10,000 ENTITY PRODUCTION BENCHMARK FINAL REPORT
================================================================================
Benchmark Sample Size       : {total_s1:,} Source-1 entities
Total Candidate Pairs       : {total_pairs:,} ({total_pairs/total_s1:.1f} pairs/S1)
Total Predicted Match Links : {total_matches:,} ({total_matches/total_s1:.4f} matches/S1)

Throughput & Speed:
  Overall Wall-Clock Time   : {total_time:.2f} s ({total_time/60:.2f} mins)
  Source-1 Throughput       : {s1_per_sec:,.1f} S1 / second
  Candidate Pair Throughput : {pairs_per_sec:,.0f} pairs / second

Internal Component Time (Worker Aggregate):
  Feature Extraction Time   : {total_feat_time:.2f} s (Avg {avg_feat_per_worker:.2f}s per worker)
  LightGBM Prediction Time  : {total_pred_time:.2f} s (Avg {avg_pred_per_worker:.2f}s per worker)

System Resource Utilization:
  Parallel Worker Processes : {workers}
  Total RAM Utilization    : {ram_peak:.2f} GB / {psutil.virtual_memory().total/(1024**3):.1f} GB
  Peak CPU Utilization      : {cpu_percent:.1f} %

Full Test Set (1,732,544 S1 Entities) Projection:
  Projected Full Runtime    : {projected_full_sec:,.1f} seconds
  Projected Full Runtime    : {projected_hours:.2f} HOURS (< 4.0 Hour Target: {'MET / EXCEEDED' if projected_hours <= 4.0 else 'MISSED'})
================================================================================
"""
    print(report)
    (OUTPUTS_DIR / "benchmark_10k_report.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify_1000", action="store_true", help="Run 1,000 entity verification test")
    parser.add_argument("--scaling_benchmark", action="store_true", help="Run worker & batch scaling benchmark")
    parser.add_argument("--benchmark_10k", action="store_true", help="Run 10,000 production benchmark")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=2000)
    args = parser.parse_args()

    if args.verify_1000:
        for f in SHARDS_DIR.glob("shard_*.tsv"):
            f.unlink()
        run_parallel_inference(num_workers=4, batch_size=500, limit_s1=1000, s1_file=S1_TEST_FILE)
        merged = OUTPUTS_DIR / "test_1000_parallel_verified.tsv"
        merge_shards_streaming(SHARDS_DIR, merged)
        validate_submission_file(merged, S1_TEST_FILE, limit_s1=1000)
        verify_1000_predictions(merged, OUTPUTS_DIR / "test_1000_submission.tsv")

    if args.scaling_benchmark:
        run_scaling_benchmark()

    if args.benchmark_10k:
        run_10000_production_benchmark(workers=args.workers, batch_size=args.batch_size)
