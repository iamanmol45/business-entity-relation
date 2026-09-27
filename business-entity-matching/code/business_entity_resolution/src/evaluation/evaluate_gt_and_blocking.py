"""
Analysis of Ground Truth matching constraints and blocking recall on training data.
"""

import sys
import os
import re
import time
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = SRC_DIR.parents[1]
PROJECT_ROOT = SRC_DIR.parents[2]

TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"

def analyze_ground_truth_constraints():
    print("=" * 80)
    print("ANALYZING GROUND TRUTH MATCHING CONSTRAINTS (1-TO-1 VS MANY-TO-MANY)")
    print("=" * 80)
    
    t0 = time.time()
    s2_to_s1 = defaultdict(list)
    s3_to_s1 = defaultdict(list)
    s1_to_matches = {}
    
    total_s1 = 0
    total_links = 0
    empty_s1 = 0
    
    with open(GT_FILE, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 2:
                continue
            s1_id = parts[0]
            matches_str = parts[1]
            total_s1 += 1
            if not matches_str.strip():
                empty_s1 += 1
                s1_to_matches[s1_id] = []
                continue
            cands = matches_str.split(",")
            s1_to_matches[s1_id] = cands
            total_links += len(cands)
            for c in cands:
                if c.startswith("S2-"):
                    s2_to_s1[c].append(s1_id)
                elif c.startswith("S3-"):
                    s3_to_s1[c].append(s1_id)
                    
    print(f"Total Ground S1 Entities : {total_s1:,}")
    print(f"Total True Links         : {total_links:,}")
    print(f"Empty S1 Entities        : {empty_s1:,} ({empty_s1/total_s1*100:.2f}%)")
    print(f"Unique S2 Matched        : {len(s2_to_s1):,}")
    print(f"Unique S3 Matched        : {len(s3_to_s1):,}")
    
    # Check multi-mapping
    s2_multi = {k: v for k, v in s2_to_s1.items() if len(v) > 1}
    s3_multi = {k: v for k, v in s3_to_s1.items() if len(v) > 1}
    
    print(f"S2 mapping to >1 S1      : {len(s2_multi):,} / {len(s2_to_s1):,} ({len(s2_multi)/max(1, len(s2_to_s1))*100:.4f}%)")
    print(f"S3 mapping to >1 S1      : {len(s3_multi):,} / {len(s3_to_s1):,} ({len(s3_multi)/max(1, len(s3_to_s1))*100:.4f}%)")
    
    if s2_multi:
        print("  Sample S2 multi-mappings:")
        for k in list(s2_multi.keys())[:5]:
            print(f"    {k} -> {s2_multi[k]}")
            
    if s3_multi:
        print("  Sample S3 multi-mappings:")
        for k in list(s3_multi.keys())[:5]:
            print(f"    {k} -> {s3_multi[k]}")
            
    print(f"Elapsed: {time.time()-t0:.2f}s")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    analyze_ground_truth_constraints()
