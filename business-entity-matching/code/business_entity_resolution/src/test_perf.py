"""
Performance benchmark & optimizer for Business Entity Resolution Inference.
"""

import sys
import os
import time
import zlib
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict

SRC_DIR = Path(__file__).resolve().parent
CODE_DIR = SRC_DIR.parents[1]
PROJECT_ROOT = SRC_DIR.parents[2]

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import joblib
import rapidfuzz
from rapidfuzz import fuzz, distance
import anyascii
import numba as nb

from business_entity_resolution.src.features.feature_extractor import (
    ProcessedEntity,
    extract_pair_feature_vector,
    extract_name_core_tokens,
    extract_distinctive_name_tokens,
    extract_address_numbers,
    extract_street_tokens,
    normalize_name_clean,
    normalize_address,
    FEATURE_NAMES,
    LEGAL_SUFFIXES,
    GENERIC_WORDS,
    STREET_STOPWORDS,
    ADDRESS_REPLACEMENTS,
    ZIP_REGEX
)

from business_entity_resolution.src.test_inference import (
    load_mmap_inverted_indexes,
    build_or_connect_sqlite_entities,
    hash64,
    DB_PATH,
    S1_TEST_FILE,
    TEST_CACHE_DIR
)


def profile_current(s1_entities, keys_mmap, offsets_mmap, counts_mmap, postings_mmap):
    t0 = time.time()
    pairs = []
    unique_cids = set()
    for s1 in s1_entities:
        country = s1.country
        name_clean = s1.name_clean
        name_no_suf = s1.name_no_suf
        spaceless_name = s1.spaceless_name
        addr_norm = s1.addr_norm
        addr_nums = s1.addr_nums
        street_tokens = s1.street_tokens
        core_tokens = extract_name_core_tokens(name_clean)
        dist_tokens = extract_distinctive_name_tokens(name_clean)

        cand_masks = defaultdict(int)

        if name_clean:
            h = hash64(f"{country}\t{name_clean}".encode("utf-8"))
            k_arr = keys_mmap[0]
            idx = int(np.searchsorted(k_arr, h))
            if idx < len(k_arr) and k_arr[idx] == h:
                off = offsets_mmap[0][idx]
                cnt = counts_mmap[0][idx]
                for p in postings_mmap[0][off : off + cnt]:
                    cand_masks[int(p)] |= 1

        if name_no_suf and name_no_suf != name_clean:
            h = hash64(f"{country}\t{name_no_suf}".encode("utf-8"))
            k_arr = keys_mmap[1]
            idx = int(np.searchsorted(k_arr, h))
            if idx < len(k_arr) and k_arr[idx] == h:
                off = offsets_mmap[1][idx]
                cnt = counts_mmap[1][idx]
                for p in postings_mmap[1][off : off + cnt]:
                    cand_masks[int(p)] |= 2

        if spaceless_name and len(spaceless_name) >= 5:
            h = hash64(f"{country}\t{spaceless_name}".encode("utf-8"))
            k_arr = keys_mmap[2]
            idx = int(np.searchsorted(k_arr, h))
            if idx < len(k_arr) and k_arr[idx] == h:
                off = offsets_mmap[2][idx]
                cnt = counts_mmap[2][idx]
                for p in postings_mmap[2][off : off + cnt]:
                    cand_masks[int(p)] |= 4

        if addr_norm and len(addr_norm) >= 5:
            h = hash64(f"{country}\t{addr_norm}".encode("utf-8"))
            k_arr = keys_mmap[3]
            idx = int(np.searchsorted(k_arr, h))
            if idx < len(k_arr) and k_arr[idx] == h:
                off = offsets_mmap[3][idx]
                cnt = counts_mmap[3][idx]
                for p in postings_mmap[3][off : off + cnt]:
                    cand_masks[int(p)] |= 8

        if dist_tokens and addr_nums:
            k_arr = keys_mmap[4]
            o_arr = offsets_mmap[4]
            c_arr = counts_mmap[4]
            p_arr = postings_mmap[4]
            k_len = len(k_arr)
            for num in addr_nums:
                for tok in dist_tokens:
                    h = hash64(f"{country}\t{num}\t{tok}".encode("utf-8"))
                    idx = int(np.searchsorted(k_arr, h))
                    if idx < k_len and k_arr[idx] == h:
                        off = o_arr[idx]
                        cnt = c_arr[idx]
                        for p in p_arr[off : off + cnt]:
                            cand_masks[int(p)] |= 16

        if street_tokens and addr_nums:
            k_arr = keys_mmap[5]
            o_arr = offsets_mmap[5]
            c_arr = counts_mmap[5]
            p_arr = postings_mmap[5]
            k_len = len(k_arr)
            for num in addr_nums:
                for s_tok in street_tokens:
                    h = hash64(f"{country}\t{num}\t{s_tok}".encode("utf-8"))
                    idx = int(np.searchsorted(k_arr, h))
                    if idx < k_len and k_arr[idx] == h:
                        off = o_arr[idx]
                        cnt = c_arr[idx]
                        for p in p_arr[off : off + cnt]:
                            cand_masks[int(p)] |= 32

        if len(core_tokens) >= 2:
            h = hash64(f"{country}\t{core_tokens[0]}\t{core_tokens[1]}".encode("utf-8"))
            k_arr = keys_mmap[6]
            idx = int(np.searchsorted(k_arr, h))
            if idx < len(k_arr) and k_arr[idx] == h:
                off = offsets_mmap[6][idx]
                cnt = counts_mmap[6][idx]
                for p in postings_mmap[6][off : off + cnt]:
                    cand_masks[int(p)] |= 64

        if dist_tokens and len(dist_tokens[0]) >= 5:
            h = hash64(f"{country}\t{dist_tokens[0]}".encode("utf-8"))
            k_arr = keys_mmap[7]
            idx = int(np.searchsorted(k_arr, h))
            if idx < len(k_arr) and k_arr[idx] == h:
                off = offsets_mmap[7][idx]
                cnt = counts_mmap[7][idx]
                for p in postings_mmap[7][off : off + cnt]:
                    cand_masks[int(p)] |= 128

        cand_count = len(cand_masks)
        s1_eid = s1.eid
        for val, mask in cand_masks.items():
            cid_str = f"S3-{val & 0x7FFFFFFF}" if (val & 0x80000000) else f"S2-{val}"
            pairs.append((s1_eid, cid_str, mask, cand_count))
            unique_cids.add(cid_str)

    t1 = time.time()
    print(f"Current Lookup: {t1-t0:.3f}s, {len(pairs)} pairs, {len(unique_cids)} unique cids")
    return pairs, unique_cids


def query_candidates_vectorized(s1_entities, keys_mmap, offsets_mmap, counts_mmap, postings_mmap):
    t0 = time.time()
    n_s1 = len(s1_entities)
    cand_masks_by_s1 = [defaultdict(int) for _ in range(n_s1)]
    
    # Pre-extract S1 tokens once
    s1_precomputed = []
    for s1 in s1_entities:
        name_clean = s1.name_clean
        s1_precomputed.append((
            s1.country,
            name_clean,
            s1.name_no_suf,
            s1.spaceless_name,
            s1.addr_norm,
            s1.addr_nums,
            s1.street_tokens,
            extract_name_core_tokens(name_clean),
            extract_distinctive_name_tokens(name_clean)
        ))

    # Process each strategy in batch
    for s_idx in range(8):
        k_arr = keys_mmap[s_idx]
        o_arr = offsets_mmap[s_idx]
        c_arr = counts_mmap[s_idx]
        p_arr = postings_mmap[s_idx]
        k_len = len(k_arr)
        bit_val = 1 << s_idx
        
        query_hashes = []
        query_s1_idx = []

        for i, (country, name_clean, name_no_suf, spaceless_name, addr_norm, addr_nums, street_tokens, core_tokens, dist_tokens) in enumerate(s1_precomputed):
            if s_idx == 0:
                if name_clean:
                    query_hashes.append(hash64(f"{country}\t{name_clean}".encode("utf-8")))
                    query_s1_idx.append(i)
            elif s_idx == 1:
                if name_no_suf and name_no_suf != name_clean:
                    query_hashes.append(hash64(f"{country}\t{name_no_suf}".encode("utf-8")))
                    query_s1_idx.append(i)
            elif s_idx == 2:
                if spaceless_name and len(spaceless_name) >= 5:
                    query_hashes.append(hash64(f"{country}\t{spaceless_name}".encode("utf-8")))
                    query_s1_idx.append(i)
            elif s_idx == 3:
                if addr_norm and len(addr_norm) >= 5:
                    query_hashes.append(hash64(f"{country}\t{addr_norm}".encode("utf-8")))
                    query_s1_idx.append(i)
            elif s_idx == 4:
                if dist_tokens and addr_nums:
                    for num in addr_nums:
                        for tok in dist_tokens:
                            query_hashes.append(hash64(f"{country}\t{num}\t{tok}".encode("utf-8")))
                            query_s1_idx.append(i)
            elif s_idx == 5:
                if street_tokens and addr_nums:
                    for num in addr_nums:
                        for s_tok in street_tokens:
                            query_hashes.append(hash64(f"{country}\t{num}\t{s_tok}".encode("utf-8")))
                            query_s1_idx.append(i)
            elif s_idx == 6:
                if len(core_tokens) >= 2:
                    query_hashes.append(hash64(f"{country}\t{core_tokens[0]}\t{core_tokens[1]}".encode("utf-8")))
                    query_s1_idx.append(i)
            elif s_idx == 7:
                if dist_tokens and len(dist_tokens[0]) >= 5:
                    query_hashes.append(hash64(f"{country}\t{dist_tokens[0]}".encode("utf-8")))
                    query_s1_idx.append(i)

        if not query_hashes:
            continue

        q_hashes = np.array(query_hashes, dtype=np.uint64)
        q_s1 = np.array(query_s1_idx, dtype=np.int32)

        # Single vectorized searchsorted in C
        idx_arr = np.searchsorted(k_arr, q_hashes)
        valid = (idx_arr < k_len) & (k_arr[idx_arr] == q_hashes)
        
        if not np.any(valid):
            continue

        matched_idx = idx_arr[valid]
        matched_s1 = q_s1[valid]
        offs = o_arr[matched_idx]
        cnts = c_arr[matched_idx]

        for s1_i, off, cnt in zip(matched_s1, offs, cnts):
            m_dict = cand_masks_by_s1[s1_i]
            for p in p_arr[off : off + cnt]:
                m_dict[int(p)] |= bit_val

    # Convert to pairs and unique_cids
    pairs = []
    unique_cids = set()
    for i, s1 in enumerate(s1_entities):
        s1_eid = s1.eid
        cand_masks = cand_masks_by_s1[i]
        cand_count = len(cand_masks)
        for val, mask in cand_masks.items():
            cid_str = f"S3-{val & 0x7FFFFFFF}" if (val & 0x80000000) else f"S2-{val}"
            pairs.append((s1_eid, cid_str, mask, cand_count))
            unique_cids.add(cid_str)

    t1 = time.time()
    print(f"Vectorized Lookup: {t1-t0:.3f}s, {len(pairs)} pairs, {len(unique_cids)} unique cids")
    return pairs, unique_cids


def main():
    print("Loading test data...")
    keys_mmap, offsets_mmap, counts_mmap, postings_mmap = load_mmap_inverted_indexes(TEST_CACHE_DIR)
    df = pd.read_csv(S1_TEST_FILE, sep="\t", dtype=str, nrows=500)
    s1_entities = [
        ProcessedEntity(eid, c, n, a)
        for eid, c, n, a in zip(df["entity_id"], df["country"].fillna(""), df["business_name"].fillna(""), df["business_address"].fillna(""))
    ]
    p1, u1 = profile_current(s1_entities, keys_mmap, offsets_mmap, counts_mmap, postings_mmap)
    p2, u2 = query_candidates_vectorized(s1_entities, keys_mmap, offsets_mmap, counts_mmap, postings_mmap)
    
    # Verify exact equivalence
    assert u1 == u2, f"Unique cids mismatch! {len(u1)} vs {len(u2)}"
    assert len(p1) == len(p2), f"Pair count mismatch! {len(p1)} vs {len(p2)}"
    
    set1 = set((p[0], p[1], p[2], p[3]) for p in p1)
    set2 = set((p[0], p[1], p[2], p[3]) for p in p2)
    assert set1 == set2, "Pair tuples mismatch between baseline and vectorized!"
    print("EXACT VERIFICATION PASSED: Vectorized candidate lookup produces 100.00% identical pairs!")

if __name__ == "__main__":
    main()

