"""
Benchmark Smart Candidate Filter on 1,000 S1 test entities vs test_1000_submission.tsv
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
CODE_DIR = SRC_DIR.parents[1]
PROJECT_ROOT = SRC_DIR.parents[2]

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_entity_resolution.src.test_inference import (
    load_mmap_inverted_indexes,
    build_or_connect_sqlite_entities,
    S1_TEST_FILE,
    DEFAULT_MODEL_PATH,
    TEST_CACHE_DIR,
    DB_PATH,
    FEATURE_NAMES
)
from business_entity_resolution.src.test_1k_compare import (
    OptimizedEntity,
    query_candidates_vectorized,
    extract_pair_features_fast
)

def test_smart_filter():
    print("=" * 90)
    print("TESTING CANDIDATE PRE-FILTER ON 1,000 S1 vs BASELINE (5,332 matches)")
    print("=" * 90)

    # 1. Load Baseline
    base_file = PROJECT_ROOT / "outputs" / "test_1000_optimized.tsv"
    if not base_file.exists():
        base_file = CODE_DIR / "outputs" / "test_1000_submission.tsv"
    if not base_file.exists():
        base_file = PROJECT_ROOT / "outputs" / "test_1000_submission.tsv"
    df_base = pd.read_csv(base_file, sep="\t", dtype=str).fillna("")
    base_matches = {sid: set(m.split(",")) if m else set() for sid, m in zip(df_base["source1_entity_id"], df_base["matched_entity_ids"])}
    total_base_matches = sum(len(s) for s in base_matches.values())
    print(f"Loaded Baseline: {len(base_matches):,} S1, {total_base_matches:,} true predicted matches.")

    # 2. Load Model & Indexes
    model_data = joblib.load(DEFAULT_MODEL_PATH)
    model = model_data["model"]
    optimal_threshold = float(model_data.get("optimal_threshold", 0.88))
    
    conn = build_or_connect_sqlite_entities(DB_PATH)
    keys_mmap, offsets_mmap, counts_mmap, postings_mmap = load_mmap_inverted_indexes(TEST_CACHE_DIR)
    
    df_s1 = pd.read_csv(S1_TEST_FILE, sep="\t", dtype=str, nrows=1000)
    s1_entities = [
        OptimizedEntity(eid, c, n, a)
        for eid, c, n, a in zip(df_s1["entity_id"], df_s1["country"].fillna(""), df_s1["business_name"].fillna(""), df_s1["business_address"].fillna(""))
    ]
    s1_map = {e.eid: e for e in s1_entities}

    t0_lookup = time.time()
    pairs, unique_cids = query_candidates_vectorized(s1_entities, keys_mmap, offsets_mmap, counts_mmap, postings_mmap)
    print(f"Raw Candidates: {len(pairs):,} ({len(pairs)/1000:.1f}/S1) in {time.time()-t0_lookup:.2f}s")

    # Fetch unique candidates from SQLite
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

    # Define Candidate Filter Modes
    filter_modes = [
        ("No Filter (Raw Baseline)", lambda s1, c, mask, cnt: True),
        ("Safe Filter (Strat>=2 OR NameOverlap OR HNMatch)", lambda s1, c, mask, cnt: (
            bin(mask).count('1') >= 2 or
            (s1.name_clean and s1.name_clean == c.name_clean) or
            (s1.name_no_suf and s1.name_no_suf == c.name_no_suf) or
            bool(s1.name_tokens.intersection(c.name_tokens)) or
            bool(s1.addr_nums and c.addr_nums and (s1.addr_nums.intersection(c.addr_nums)))
        )),
        ("Top-100 per S1", "top_100"),
        ("Top-150 per S1", "top_150"),
        ("Top-200 per S1", "top_200"),
    ]

    print("\n" + "=" * 115)
    print(f"{'Mode':<32} | {'Pairs':<10} | {'Feat (s)':<10} | {'Pred (s)':<10} | {'Total (s)':<10} | {'Matches':<10} | {'Recall vs Base':<18} | {'Throughput':<14}")
    print("-" * 115)

    cands_by_s1 = defaultdict(list)
    for s1_id, cid, mask, c_cnt in pairs:
        cands_by_s1[s1_id].append((cid, mask, c_cnt))

    for mode_name, filter_fn in filter_modes:
        t_start_mode = time.time()
        filtered_pairs = []

        if isinstance(filter_fn, str) and filter_fn.startswith("top_"):
            k_val = int(filter_fn.split("_")[1])
            for s1 in s1_entities:
                c_list = cands_by_s1[s1.eid]
                if len(c_list) <= k_val:
                    for cid, mask, c_cnt in c_list:
                        filtered_pairs.append((s1.eid, cid, mask, c_cnt))
                else:
                    # Cheap ranking
                    scored = []
                    s1_nt = s1.name_tokens
                    s1_an = s1.addr_nums
                    s1_nc = s1.name_clean
                    s1_nns = s1.name_no_suf
                    for cid, mask, c_cnt in c_list:
                        c_obj = cand_entity_cache.get(cid)
                        if c_obj is None:
                            continue
                        c_nc = c_obj.name_clean
                        c_nns = c_obj.name_no_suf
                        if (s1_nc and s1_nc == c_nc) or (s1_nns and s1_nns == c_nns):
                            score = 100.0 + (10.0 if s1_nc == c_nc else 0.0)
                        else:
                            strat_score = bin(mask).count('1') * 2.0
                            inter = len(s1_nt.intersection(c_obj.name_tokens))
                            min_tok = min(len(s1_nt), len(c_obj.name_tokens))
                            tok_ov = (inter / min_tok) if min_tok > 0 else 0.0
                            hn_ov = 2.0 if (s1_an and c_obj.addr_nums and (s1_an.intersection(c_obj.addr_nums))) else 0.0
                            pref = 1.0 if (s1_nc and c_nc and s1_nc[:3] == c_nc[:3]) else 0.0
                            score = strat_score + 5.0 * tok_ov + hn_ov + pref
                        scored.append((score, cid, mask, c_cnt))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    for _, cid, mask, c_cnt in scored[:k_val]:
                        filtered_pairs.append((s1.eid, cid, mask, c_cnt))
        else:
            for s1_id, cid, mask, c_cnt in pairs:
                s1_obj = s1_map[s1_id]
                c_obj = cand_entity_cache.get(cid)
                if c_obj is not None and filter_fn(s1_obj, c_obj, mask, c_cnt):
                    filtered_pairs.append((s1_id, cid, mask, c_cnt))

        n_p = len(filtered_pairs)
        t_f0 = time.time()
        X = np.zeros((n_p, len(FEATURE_NAMES)), dtype=np.float32)
        for idx, (s1_id, cid, mask, c_cnt) in enumerate(filtered_pairs):
            s1_obj = s1_map[s1_id]
            c_obj = cand_entity_cache[cid]
            extract_pair_features_fast(s1_obj, c_obj, mask, c_cnt, X, idx)
        t_feat = time.time() - t_f0

        t_p0 = time.time()
        probas = model.predict_proba(X)[:, 1].astype(np.float32) if n_p > 0 else np.array([], dtype=np.float32)
        t_pred = time.time() - t_p0

        pred_matches = defaultdict(set)
        for (s1_id, cid, _, _), proba in zip(filtered_pairs, probas):
            if proba >= optimal_threshold:
                pred_matches[s1_id].add(cid)

        total_found = sum(len(s) for s in pred_matches.values())
        rec_count = sum(len(base_matches[sid].intersection(pred_matches[sid])) for sid in s1_map)
        rec_pct = (rec_count / total_base_matches) * 100
        t_tot = time.time() - t_start_mode
        s1_s = 1000 / t_tot

        print(f"{mode_name:<32} | {n_p:>10,} | {t_feat:>8.2f}s | {t_pred:>8.3f}s | {t_tot:>8.2f}s | {total_found:>10,} | {rec_pct:>16.4f} % | {s1_s:>10.1f} S1/s")

    print("=" * 115)

if __name__ == "__main__":
    test_smart_filter()
