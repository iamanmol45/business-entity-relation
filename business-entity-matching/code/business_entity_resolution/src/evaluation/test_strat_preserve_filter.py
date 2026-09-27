"""
Test Strategy-Preserving Candidate Filter on 1,000 S1 test entities vs Baseline (5,332 matches)
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

def test_preserve():
    base_file = PROJECT_ROOT / "outputs" / "final_submission.tsv"
    df_base = pd.read_csv(base_file, sep="\t", dtype=str).fillna("")
    base_matches = {sid: set(m.split(",")) if m else set() for sid, m in zip(df_base["source1_entity_id"], df_base["matched_entity_ids"])}
    total_base_matches = sum(len(s) for s in base_matches.values())

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

    pairs, unique_cids = query_candidates_vectorized(s1_entities, keys_mmap, offsets_mmap, counts_mmap, postings_mmap)

    # Fetch unique candidates from SQLite
    cand_entity_cache = {}
    cur = conn.cursor()
    CHUNK_SIZE = 900
    cid_list = list(unique_cids)
    for i in range(0, len(cid_list), CHUNK_SIZE):
        chunk = cid_list[i : i + CHUNK_SIZE]
        placeholders = ",".join("?" * len(chunk))
        query = f"SELECT eid, country, name, addr FROM entities WHERE eid IN ({placeholders})"
        for eid, country, name, addr in cur.execute(query, chunk):
            cand_entity_cache[eid] = OptimizedEntity(eid, country, name, addr)

    # Strategy-preserving Filter Rules:
    # 1. Strat 0, 1, 2, 3, 6, 7 (Exact name, no suffix, spaceless, exact address, 2-token name, dist first token) -> ALWAYS KEEP 100%
    # 2. If mask has bit 0, 1, 2, 3, 6, 7 (mask & 0b11001111 != 0) -> ALWAYS KEEP!
    # 3. Only if mask has ONLY Strategy 4 or 5 (mask & 0b00110000 != 0 and mask & 0b11001111 == 0):
    #    Keep if: shared name token OR translit exact OR city match OR hn exact OR Top-N per S1

    print("=" * 115)
    print("STRATEGY-PRESERVING FILTER EVALUATION ON 1,000 S1")
    print("=" * 115)

    filter_configs = [
        ("Raw Baseline (100% pairs)", 0, None),
        ("Preserve High-Value Strats + Strat 4/5 Token/HN Filter", 1, None),
        ("Preserve High-Value Strats + Strat 4/5 Top-30", 2, 30),
        ("Preserve High-Value Strats + Strat 4/5 Top-50", 2, 50),
        ("Preserve High-Value Strats + Strat 4/5 Top-100", 2, 100),
    ]

    for name, mode, top_n in filter_configs:
        t0 = time.time()
        filtered_pairs = []

        if mode == 0:
            filtered_pairs = pairs
        elif mode == 1:
            for s1_id, cid, mask, c_cnt in pairs:
                # High-value strategies: 0, 1, 2, 3, 6, 7
                if mask & 0b11001111:
                    filtered_pairs.append((s1_id, cid, mask, c_cnt))
                else:
                    s1_obj = s1_map[s1_id]
                    c_obj = cand_entity_cache.get(cid)
                    if c_obj is None:
                        continue
                    # For strat 4 & 5 alone:
                    if (s1_obj.name_tokens.intersection(c_obj.name_tokens) or
                        (s1_obj.addr_nums and c_obj.addr_nums and (s1_obj.addr_nums == c_obj.addr_nums)) or
                        (s1_obj.city and c_obj.city and s1_obj.city == c_obj.city) or
                        (s1_obj.postal and c_obj.postal and s1_obj.postal == c_obj.postal)):
                        filtered_pairs.append((s1_id, cid, mask, c_cnt))
        elif mode == 2:
            s45_by_s1 = defaultdict(list)
            for s1_id, cid, mask, c_cnt in pairs:
                if mask & 0b11001111:
                    filtered_pairs.append((s1_id, cid, mask, c_cnt))
                else:
                    s45_by_s1[s1_id].append((cid, mask, c_cnt))
            
            for sid, c_list in s45_by_s1.items():
                if len(c_list) <= top_n:
                    for cid, mask, c_cnt in c_list:
                        filtered_pairs.append((sid, cid, mask, c_cnt))
                else:
                    s1_obj = s1_map[sid]
                    scored = []
                    s1_nt = s1_obj.name_tokens
                    s1_an = s1_obj.addr_nums
                    for cid, mask, c_cnt in c_list:
                        c_obj = cand_entity_cache.get(cid)
                        if c_obj is None:
                            continue
                        inter = len(s1_nt.intersection(c_obj.name_tokens))
                        hn_match = 2.0 if (s1_an and c_obj.addr_nums and s1_an == c_obj.addr_nums) else 0.0
                        city_match = 1.0 if (s1_obj.city and c_obj.city and s1_obj.city == c_obj.city) else 0.0
                        score = inter * 3.0 + hn_match + city_match
                        scored.append((score, cid, mask, c_cnt))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    for _, cid, mask, c_cnt in scored[:top_n]:
                        filtered_pairs.append((sid, cid, mask, c_cnt))

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
        t_tot = time.time() - t0
        s1_s = 1000 / t_tot

        print(f"{name:<55} | {n_p:>9,} pairs | {t_feat:>6.2f}s feat | {total_found:>6,} matches | {rec_pct:>10.4f} % | {s1_s:>7.1f} S1/s")

    print("=" * 115)

if __name__ == "__main__":
    test_preserve()
