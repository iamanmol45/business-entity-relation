"""
Evaluate Top-K Candidate Selection on the 1,000 Test Set against test_1000_submission.tsv
"""

import sys
import os
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import joblib

SRC_DIR = Path(__file__).resolve().parents[1]
PKG_DIR = SRC_DIR.parent
CODE_DIR = PKG_DIR.parent
PROJECT_ROOT = CODE_DIR.parent

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(CODE_DIR.parent) not in sys.path:
    sys.path.insert(0, str(CODE_DIR.parent))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.test_inference import (
    load_mmap_inverted_indexes,
    build_or_connect_sqlite_entities,
    S1_TEST_FILE,
    DEFAULT_MODEL_PATH,
    TEST_CACHE_DIR,
    DB_PATH,
    FEATURE_NAMES
)
from src.test_1k_compare import (
    OptimizedEntity,
    query_candidates_vectorized,
    extract_pair_features_fast
)

def test_topk_sweep():
    print("=" * 80)
    print("SWEEPING TOP-K CANDIDATES ON 1,000 TEST S1 vs test_1000_submission.tsv")
    print("=" * 80)

    # 1. Load Baseline Matches
    base_file = PROJECT_ROOT / "outputs" / "test_1000_submission.tsv"
    if not base_file.exists():
        base_file = PROJECT_ROOT / "outputs" / "test_1000_optimized.tsv"
    
    df_base = pd.read_csv(base_file, sep="\t", dtype=str).fillna("")
    base_matches = {}
    total_base_matches = 0
    for sid, m in zip(df_base["source1_entity_id"], df_base["matched_entity_ids"]):
        s = set(m.split(",")) if m else set()
        base_matches[sid] = s
        total_base_matches += len(s)
        
    print(f"Loaded Baseline: {len(base_matches):,} S1 entities, {total_base_matches:,} true predicted matches.")

    # 2. Load Model & Indexes
    model_data = joblib.load(DEFAULT_MODEL_PATH)
    model = model_data["model"]
    optimal_threshold = float(model_data.get("optimal_threshold", 0.88))
    
    conn = build_or_connect_sqlite_entities(DB_PATH)
    keys_mmap, offsets_mmap, counts_mmap, postings_mmap = load_mmap_inverted_indexes(TEST_CACHE_DIR)
    
    # 3. Read 1,000 S1
    df_s1 = pd.read_csv(S1_TEST_FILE, sep="\t", dtype=str, nrows=1000)
    s1_entities = [
        OptimizedEntity(eid, c, n, a)
        for eid, c, n, a in zip(df_s1["entity_id"], df_s1["country"].fillna(""), df_s1["business_name"].fillna(""), df_s1["business_address"].fillna(""))
    ]
    s1_map = {e.eid: e for e in s1_entities}

    # 4. Lookup all candidates
    t0 = time.time()
    pairs, unique_cids = query_candidates_vectorized(s1_entities, keys_mmap, offsets_mmap, counts_mmap, postings_mmap)
    print(f"Full Candidate Pairs: {len(pairs):,} ({len(pairs)/1000:.1f} per S1) generated in {time.time()-t0:.2f}s")
    print(f"Unique Candidate Entities: {len(unique_cids):,}")

    # Fetch all candidate entities from SQLite
    cand_entity_cache = {}
    cur = conn.cursor()
    CHUNK_SIZE = 900
    cid_list = list(unique_cids)
    t0_sql = time.time()
    for i in range(0, len(cid_list), CHUNK_SIZE):
        chunk = cid_list[i : i + CHUNK_SIZE]
        placeholders = ",".join("?" * len(chunk))
        query = f"SELECT eid, country, name, addr FROM entities WHERE eid IN ({placeholders})"
        for eid, country, name, addr in cur.execute(query, chunk):
            cand_entity_cache[eid] = OptimizedEntity(eid, country, name, addr)
    print(f"Fetched {len(cand_entity_cache):,} candidates from SQLite in {time.time()-t0_sql:.2f}s.")

    # Group pairs by S1
    cands_by_s1 = defaultdict(list)
    for s1_id, cid, mask, c_cnt in pairs:
        cands_by_s1[s1_id].append((cid, mask, c_cnt))

    # Evaluate different Top-K values
    top_k_values = [30, 50, 75, 100, 150, 200, 300, 500, None] # None means all candidates

    print("\n" + "=" * 115)
    print(f"{'Top-K':<10} | {'Pair Count':<12} | {'Feat Time (s)':<14} | {'Pred Time (s)':<14} | {'Matches Found':<14} | {'Match Recall vs Base':<22} | {'Mismatched S1':<14}")
    print("-" * 115)

    for k in top_k_values:
        t_feat_start = time.time()
        
        # Build filtered pair list
        filtered_pairs = []
        for s1 in s1_entities:
            sid = s1.eid
            c_list = cands_by_s1[sid]
            if not c_list:
                continue
                
            if k is not None and len(c_list) > k:
                # Rank by cheap score
                scored = []
                s1_nt = s1.name_tokens
                s1_an = s1.addr_nums
                s1_nc = s1.name_clean
                s1_nns = s1.name_no_suf
                s1_sp = s1.spaceless_name
                
                for cid, mask, c_cnt in c_list:
                    c_obj = cand_entity_cache.get(cid)
                    if c_obj is None:
                        continue
                    c_nc = c_obj.name_clean
                    c_nns = c_obj.name_no_suf
                    
                    # Exact name or suffix match gets top tier
                    if (s1_nc and s1_nc == c_nc) or (s1_nns and s1_nns == c_nns):
                        score = 100.0 + (10.0 if s1_nc == c_nc else 0.0)
                    else:
                        # Strategy mask presence
                        strat_score = bin(mask).count('1') * 2.0
                        # Token overlap
                        inter = len(s1_nt.intersection(c_obj.name_tokens))
                        min_tok = min(len(s1_nt), len(c_obj.name_tokens))
                        tok_ov = (inter / min_tok) if min_tok > 0 else 0.0
                        # House number overlap
                        hn_ov = 2.0 if (s1_an and c_obj.addr_nums and (s1_an.intersection(c_obj.addr_nums))) else 0.0
                        # 3-char prefix
                        pref = 1.0 if (s1_nc and c_nc and s1_nc[:3] == c_nc[:3]) else 0.0
                        
                        score = strat_score + 5.0 * tok_ov + hn_ov + pref
                        
                    scored.append((score, cid, mask, c_cnt))
                    
                scored.sort(key=lambda x: x[0], reverse=True)
                top_items = scored[:k]
                for sc, cid, mask, c_cnt in top_items:
                    filtered_pairs.append((sid, cid, mask, c_cnt))
            else:
                for cid, mask, c_cnt in c_list:
                    filtered_pairs.append((sid, cid, mask, c_cnt))
                    
        n_pairs = len(filtered_pairs)
        
        # Extract features
        X = np.zeros((n_pairs, len(FEATURE_NAMES)), dtype=np.float32)
        for idx, (s1_id, cand_id, mask, c_cnt) in enumerate(filtered_pairs):
            s1_obj = s1_map[s1_id]
            cand_obj = cand_entity_cache.get(cand_id)
            if cand_obj is not None:
                extract_pair_features_fast(s1_obj, cand_obj, mask, c_cnt, X, idx)
        t_feat = time.time() - t_feat_start
        
        # Predict
        t_pred_start = time.time()
        probas = model.predict_proba(X)[:, 1].astype(np.float32) if n_pairs > 0 else np.array([], dtype=np.float32)
        t_pred = time.time() - t_pred_start
        
        # Check matches
        s1_pred_matches = defaultdict(set)
        for (s1_id, cand_id, _, _), proba in zip(filtered_pairs, probas):
            if proba >= optimal_threshold:
                s1_pred_matches[s1_id].add(cand_id)
                
        total_found = sum(len(s) for s in s1_pred_matches.values())
        
        # Compare with baseline
        recovered_base = 0
        mismatches = 0
        for sid in s1_map:
            b_set = base_matches[sid]
            p_set = s1_pred_matches[sid]
            recovered_base += len(b_set.intersection(p_set))
            if b_set != p_set:
                mismatches += 1
                
        rec_pct = (recovered_base / total_base_matches) * 100 if total_base_matches > 0 else 0.0
        k_str = f"K = {k}" if k is not None else "ALL (No top-k)"
        print(f"{k_str:<10} | {n_pairs:>10,} | {t_feat:>12.2f}s | {t_pred:>12.3f}s | {total_found:>12,} | {rec_pct:>20.4f} % | {mismatches:>12,}")

    print("=" * 115)

if __name__ == "__main__":
    test_topk_sweep()
