"""
Exact Component Profiler for 1,000 Source-1 entities.
Measures time spent in:
1. mmap candidate lookup
2. SQLite retrieval & object creation
3. feature extraction
4. LightGBM prediction
5. output writing
"""

import sys
import os
import time
import gc
from pathlib import Path
from typing import Dict, List, Set, Tuple

SRC_DIR = Path(__file__).resolve().parents[1]
PKG_DIR = SRC_DIR.parent
CODE_DIR = PKG_DIR.parent
PROJECT_ROOT = CODE_DIR.parent

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import joblib
import psutil

from business_entity_resolution.src.test_inference import (
    ProcessedEntity,
    build_or_connect_sqlite_entities,
    load_mmap_inverted_indexes,
    query_candidates_for_s1_batch_mmap,
    extract_pair_features_fast,
    S1_TEST_FILE,
    S2_TEST_FILE,
    S3_TEST_FILE,
    DEFAULT_MODEL_PATH,
    TEST_CACHE_DIR,
    DB_PATH,
    FEATURE_NAMES
)

def run_profile_1000():
    print("=" * 80)
    print("PROFILING 1,000 S1 RUN ACROSS 5 CORE COMPONENTS")
    print("=" * 80)
    
    # 1. Load Model
    model_data = joblib.load(DEFAULT_MODEL_PATH)
    model = model_data["model"]
    optimal_threshold = float(model_data.get("optimal_threshold", 0.88))
    
    # 2. Connect SQLite & Mmap
    conn = build_or_connect_sqlite_entities(DB_PATH, S2_TEST_FILE, S3_TEST_FILE)
    keys_mmap, offsets_mmap, counts_mmap, postings_mmap = load_mmap_inverted_indexes(TEST_CACHE_DIR)
    
    # Read 1,000 S1
    s1_df = pd.read_csv(S1_TEST_FILE, sep="\t", dtype=str, nrows=1000)
    
    time_s1_prep = 0.0
    time_mmap_lookup = 0.0
    time_sqlite_retrieval = 0.0
    time_feat_extract = 0.0
    time_prediction = 0.0
    time_output_write = 0.0
    
    cand_entity_cache: Dict[str, ProcessedEntity] = {}
    batch_size = 500
    
    out_file = PROJECT_ROOT / "outputs" / "profile_1000_temp.tsv"
    f_out = open(out_file, "w", encoding="utf-8")
    f_out.write("source1_entity_id\tmatched_entity_ids\n")
    
    total_pairs = 0
    total_matches = 0
    
    t_start_total = time.time()
    
    for b_idx in range(0, 1000, batch_size):
        chunk_df = s1_df.iloc[b_idx : b_idx + batch_size]
        
        # S1 entity prep
        t0 = time.time()
        s1_entities = [
            ProcessedEntity(eid, c, n, a, is_s1=True)
            for eid, c, n, a in zip(
                chunk_df["entity_id"],
                chunk_df["country"].fillna(""),
                chunk_df["business_name"].fillna(""),
                chunk_df["business_address"].fillna("")
            )
        ]
        s1_map = {e.eid: e for e in s1_entities}
        time_s1_prep += (time.time() - t0)
        
        # 1. Mmap Candidate Lookup
        t0 = time.time()
        pairs, unique_cids = query_candidates_for_s1_batch_mmap(
            s1_entities, keys_mmap, offsets_mmap, counts_mmap, postings_mmap
        )
        time_mmap_lookup += (time.time() - t0)
        total_pairs += len(pairs)
        
        # 2. SQLite Retrieval & Candidate Normalization
        t0 = time.time()
        missing_cids = [cid for cid in unique_cids if cid not in cand_entity_cache]
        if missing_cids:
            cur = conn.cursor()
            CHUNK_SIZE = 900
            for i in range(0, len(missing_cids), CHUNK_SIZE):
                sub = missing_cids[i : i + CHUNK_SIZE]
                placeholders = ",".join("?" * len(sub))
                query = f"SELECT eid, country, name, addr FROM entities WHERE eid IN ({placeholders})"
                for eid, country, name, addr in cur.execute(query, sub):
                    cand_entity_cache[eid] = ProcessedEntity(eid, country, name, addr, is_s1=False)
        time_sqlite_retrieval += (time.time() - t0)
        
        # 3. Feature Extraction
        t0 = time.time()
        n_chunk = len(pairs)
        X = np.zeros((n_chunk, len(FEATURE_NAMES)), dtype=np.float32)
        for idx, (s1_id, cand_id, mask, c_cnt) in enumerate(pairs):
            s1_obj = s1_map[s1_id]
            cand_obj = cand_entity_cache.get(cand_id)
            if cand_obj is not None:
                extract_pair_features_fast(s1_obj, cand_obj, mask, c_cnt, X, idx)
        time_feat_extract += (time.time() - t0)
        
        # 4. LightGBM Prediction
        t0 = time.time()
        if n_chunk > 0:
            probas = model.predict_proba(X)[:, 1].astype(np.float32)
        else:
            probas = np.array([], dtype=np.float32)
        time_prediction += (time.time() - t0)
        
        # 5. Output Writing
        t0 = time.time()
        s1_matched_cands = {e.eid: [] for e in s1_entities}
        for (s1_id, cand_id, _, _), proba in zip(pairs, probas):
            if proba >= optimal_threshold:
                s1_matched_cands[s1_id].append(cand_id)
                total_matches += 1
                
        lines = []
        for e in s1_entities:
            matches = s1_matched_cands[e.eid]
            if not matches:
                lines.append(f"{e.eid}\t\n")
            else:
                lines.append(f"{e.eid}\t{','.join(matches)}\n")
        f_out.writelines(lines)
        f_out.flush()
        time_output_write += (time.time() - t0)
        
    f_out.close()
    conn.close()
    if out_file.exists():
        os.remove(out_file)
        
    total_time = time.time() - t_start_total
    
    print("\n" + "=" * 80)
    print("PROFILING BREAKDOWN FOR 1,000 S1 ENTITIES")
    print("=" * 80)
    print(f"Total S1 Entities Processed: 1,000")
    print(f"Total Candidate Pairs      : {total_pairs:,} ({total_pairs/1000:.1f} pairs/S1)")
    print(f"Total Matches Found        : {total_matches:,}")
    print(f"Total Measured Time        : {total_time:.3f} s ({1000/total_time:.2f} S1/s)\n")
    
    components = [
        ("1. mmap candidate lookup", time_mmap_lookup),
        ("2. SQLite retrieval & normalization", time_sqlite_retrieval),
        ("3. Feature extraction (69 features)", time_feat_extract),
        ("4. LightGBM prediction", time_prediction),
        ("5. Output writing & formatting", time_output_write),
        ("6. S1 entity preparation", time_s1_prep),
    ]
    
    print(f"{'Component':<42} | {'Time (s)':<10} | {'Percentage':<10} | {'Throughput':<20}")
    print("-" * 90)
    for name, t_comp in components:
        pct = (t_comp / total_time) * 100
        if "Feature" in name:
            tp = f"{total_pairs/t_comp:,.0f} pairs/s" if t_comp > 0 else "N/A"
        elif "prediction" in name:
            tp = f"{total_pairs/t_comp:,.0f} pairs/s" if t_comp > 0 else "N/A"
        elif "lookup" in name or "retrieval" in name:
            tp = f"{1000/t_comp:,.1f} S1/s" if t_comp > 0 else "N/A"
        else:
            tp = "-"
        print(f"{name:<42} | {t_comp:<10.3f} | {pct:>8.2f} % | {tp:<20}")
    print("=" * 80)

if __name__ == "__main__":
    run_profile_1000()
