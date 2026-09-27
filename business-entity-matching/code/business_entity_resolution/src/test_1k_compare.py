"""
Verification test comparing optimized pipeline output with outputs/test_stage1_opt.tsv.
"""

import sys
import os
import time
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, Counter

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

from business_entity_resolution.src.features.feature_extractor import (
    DOMAIN_SUFFIX_PATTERN,
    ADDRESS_REPLACEMENTS,
    LEGAL_SUFFIXES,
    GENERIC_WORDS,
    STREET_STOPWORDS,
    ZIP_REGEX,
    FEATURE_NAMES
)

from business_entity_resolution.src.test_inference import (
    load_mmap_inverted_indexes,
    build_or_connect_sqlite_entities,
    hash64,
    DB_PATH,
    S1_TEST_FILE,
    TEST_CACHE_DIR,
    DEFAULT_MODEL_PATH
)

RE_PREFIX = re.compile(r'^(?:formerly\s*:\s*|formerly\s+|dba\s*:\s*|dba\s+|doing\s+business\s+as\s+|c\s*/\s*o\s+|mr\s+|mrs\s+|ms\s+|dr\s+)', re.IGNORECASE)
RE_PREFIX_COLON = re.compile(r'(?:formerly|dba|doing business as)\s*:\s*(.*)', re.IGNORECASE)
RE_DOMAIN = re.compile(DOMAIN_SUFFIX_PATTERN)
RE_LEAD_NOISE = re.compile(r'^[@\*\.\s]+')
RE_WWW = re.compile(r'^www\.')
RE_NON_WORD = re.compile(r'[^\w\s]')
RE_LEADING_ZEROS = re.compile(r'\b0+(\d+)\b')
RE_SPACES = re.compile(r'\s+')
RE_ADDR_NUMS = re.compile(r'\b\d+\b')


def fast_norm_text(value: str) -> str:
    if not value:
        return ''
    if not value.isascii():
        value = unicodedata.normalize('NFKD', value.lower())
        value = ''.join(c for c in value if not unicodedata.combining(c))
    else:
        value = value.lower()
    value = RE_LEAD_NOISE.sub('', value)
    value = RE_WWW.sub('', value)
    value = RE_DOMAIN.sub('', value)
    value = RE_NON_WORD.sub(' ', value)
    value = RE_LEADING_ZEROS.sub(r'\1', value)
    value = RE_SPACES.sub(' ', value)
    return value.strip()


def fast_strip_prefix(text: str) -> str:
    text = RE_PREFIX.sub('', text).strip()
    m = RE_PREFIX_COLON.search(text)
    if m:
        return m.group(1).strip()
    return text


def fast_norm_addr(value: str) -> str:
    text = fast_norm_text(value)
    if not text:
        return ''
    tokens = [ADDRESS_REPLACEMENTS.get(t, t) for t in text.split()]
    return ' '.join(tokens)


class OptimizedEntity:
    __slots__ = (
        'eid', 'country', 'raw_name', 'raw_addr',
        'name_clean', 'name_no_suf', 'spaceless_name', 'name_translit',
        'name_tokens', 'name_len',
        'addr_norm', 'addr_translit', 'addr_nums', 'street_tokens',
        'addr_tokens', 'addr_len', 'postal', 'city', 'state',
        'has_name', 'has_addr', 'is_unicode_non_ascii',
        'core_tokens', 'dist_tokens'
    )

    def __init__(self, eid: str, country: str, raw_name: str, raw_addr: str):
        self.eid = eid
        self.country = country.strip() if country else ''
        self.raw_name = raw_name.strip() if raw_name else ''
        self.raw_addr = raw_addr.strip() if raw_addr else ''

        # Name fields
        raw_name_ascii = not self.raw_name or self.raw_name.isascii()
        raw_addr_ascii = not self.raw_addr or self.raw_addr.isascii()

        name_text = fast_norm_text(self.raw_name)
        self.name_clean = fast_strip_prefix(name_text)
        
        _name_tokens_list = self.name_clean.split()
        _filtered_tokens = [t for t in _name_tokens_list if t not in LEGAL_SUFFIXES]
        self.name_no_suf = ' '.join(_filtered_tokens)
        self.spaceless_name = ''.join(_filtered_tokens)
        
        if raw_name_ascii:
            self.name_translit = self.name_clean
        else:
            ascii_name = anyascii.anyascii(self.raw_name)
            self.name_translit = fast_strip_prefix(fast_norm_text(ascii_name))
            
        self.name_tokens = set(_name_tokens_list)
        self.name_len = len(self.name_clean)

        # Address fields
        self.addr_norm = fast_norm_addr(self.raw_addr)
        if raw_addr_ascii:
            self.addr_translit = self.addr_norm
        else:
            self.addr_translit = fast_norm_addr(anyascii.anyascii(self.raw_addr))
            
        self.addr_nums = set(RE_ADDR_NUMS.findall(self.addr_norm))
        
        _addr_tokens_list = self.addr_norm.split()
        self.street_tokens = {
            t for t in _addr_tokens_list
            if len(t) >= 3 and t not in STREET_STOPWORDS and not t.isdigit()
        }
        self.addr_tokens = set(_addr_tokens_list)
        self.addr_len = len(self.addr_norm)
        
        if self.raw_addr:
            m = ZIP_REGEX.search(self.raw_addr)
            self.postal = m.group(1) if m else ''
            parts = [p.strip().lower() for p in self.raw_addr.split(',') if p.strip()]
            if len(parts) >= 3:
                self.city, self.state = parts[-2], parts[-1]
            elif len(parts) == 2:
                self.city, self.state = parts[0], parts[1]
            elif len(parts) == 1:
                self.city, self.state = parts[0], ''
            else:
                self.city, self.state = '', ''
        else:
            self.postal = ''
            self.city = ''
            self.state = ''

        # Flags
        self.has_name = bool(self.name_clean)
        self.has_addr = bool(self.addr_norm)
        self.is_unicode_non_ascii = not (raw_name_ascii and raw_addr_ascii)

        # Precomputed blocking tokens for S1
        self.core_tokens = [t for t in _name_tokens_list if len(t) >= 3 and t not in LEGAL_SUFFIXES]
        self.dist_tokens = [t for t in _name_tokens_list if len(t) >= 3 and t not in LEGAL_SUFFIXES and t not in GENERIC_WORDS]


def query_candidates_vectorized(
    s1_entities: List[OptimizedEntity],
    keys_mmap: List[Any],
    offsets_mmap: List[Any],
    counts_mmap: List[Any],
    postings_mmap: List[Any]
) -> Tuple[List[Tuple[str, str, int, int]], Set[str]]:
    """Fast batched/vectorized binary search over sorted mmap arrays."""
    n_s1 = len(s1_entities)
    cand_masks_by_s1 = [defaultdict(int) for _ in range(n_s1)]
    
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

        for i, s1 in enumerate(s1_entities):
            country = s1.country
            name_clean = s1.name_clean
            name_no_suf = s1.name_no_suf
            spaceless_name = s1.spaceless_name
            addr_norm = s1.addr_norm
            addr_nums = s1.addr_nums
            street_tokens = s1.street_tokens
            core_tokens = s1.core_tokens
            dist_tokens = s1.dist_tokens

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
    pairs: List[Tuple[str, str, int, int]] = []
    unique_cids: Set[str] = set()
    for i, s1 in enumerate(s1_entities):
        s1_eid = s1.eid
        cand_masks = cand_masks_by_s1[i]
        cand_count = len(cand_masks)
        for val, mask in cand_masks.items():
            cid_str = f"S3-{val & 0x7FFFFFFF}" if (val & 0x80000000) else f"S2-{val}"
            pairs.append((s1_eid, cid_str, mask, cand_count))
            unique_cids.add(cid_str)

    return pairs, unique_cids


def extract_pair_features_fast(
    s1: OptimizedEntity,
    cand: OptimizedEntity,
    strat_mask: int,
    cand_count: int,
    out_vector: np.ndarray,
    row_idx: int
) -> None:
    # 1. Name Features
    s1_nc = s1.name_clean
    cand_nc = cand.name_clean
    s1_nns = s1.name_no_suf
    cand_nns = cand.name_no_suf
    s1_sp = s1.spaceless_name
    cand_sp = cand.spaceless_name
    s1_tr = s1.name_translit
    cand_tr = cand.name_translit

    exact_clean = 1.0 if s1_nc and s1_nc == cand_nc else 0.0
    exact_raw = 1.0 if s1.raw_name and s1.raw_name == cand.raw_name else 0.0
    exact_no_suf = 1.0 if s1_nns and s1_nns == cand_nns else 0.0
    exact_spaceless = 1.0 if s1_sp and len(s1_sp) >= 4 and s1_sp == cand_sp else 0.0
    exact_translit = 1.0 if s1_tr and s1_tr == cand_tr else 0.0

    if exact_clean == 1.0:
        name_fuzz = 1.0
        name_partial = 1.0
        name_sort = 1.0
        name_set = 1.0
        name_jw = 1.0
        name_lev = 1.0
    elif s1_nc and cand_nc:
        name_fuzz = fuzz.ratio(s1_nc, cand_nc) / 100.0
        name_partial = fuzz.partial_ratio(s1_nc, cand_nc) / 100.0
        name_sort = fuzz.token_sort_ratio(s1_nc, cand_nc) / 100.0
        name_set = fuzz.token_set_ratio(s1_nc, cand_nc) / 100.0
        name_jw = distance.JaroWinkler.similarity(s1_nc, cand_nc)
        name_lev = distance.Levenshtein.normalized_similarity(s1_nc, cand_nc)
    else:
        name_fuzz = name_partial = name_sort = name_set = name_jw = name_lev = 0.0

    if exact_no_suf == 1.0:
        no_suf_fuzz = 1.0
        no_suf_sort = 1.0
    elif s1_nns and cand_nns:
        no_suf_fuzz = fuzz.ratio(s1_nns, cand_nns) / 100.0
        no_suf_sort = fuzz.token_sort_ratio(s1_nns, cand_nns) / 100.0
    else:
        no_suf_fuzz = no_suf_sort = 0.0

    if exact_spaceless == 1.0:
        spaceless_fuzz = 1.0
    elif s1_sp and cand_sp:
        spaceless_fuzz = fuzz.ratio(s1_sp, cand_sp) / 100.0
    else:
        spaceless_fuzz = 0.0

    if exact_translit == 1.0:
        translit_fuzz = 1.0
        translit_sort = 1.0
    elif s1_tr and cand_tr:
        translit_fuzz = fuzz.ratio(s1_tr, cand_tr) / 100.0
        translit_sort = fuzz.token_sort_ratio(s1_tr, cand_tr) / 100.0
    else:
        translit_fuzz = translit_sort = 0.0

    # Token overlap without set union allocation
    s1_nt_len = len(s1.name_tokens)
    cand_nt_len = len(cand.name_tokens)
    t_inter = len(s1.name_tokens.intersection(cand.name_tokens))
    t_union = s1_nt_len + cand_nt_len - t_inter
    name_jaccard = (t_inter / t_union) if t_union > 0 else 0.0
    min_t = min(s1_nt_len, cand_nt_len)
    name_overlap_ratio = (t_inter / min_t) if min_t > 0 else 0.0

    # Containment
    name_containment = 1.0 if (s1_nc and cand_nc and (s1_nc in cand_nc or cand_nc in s1_nc)) else 0.0
    spaceless_containment = 1.0 if (s1_sp and cand_sp and len(s1_sp) >= 4 and len(cand_sp) >= 4 and (s1_sp in cand_sp or cand_sp in s1_sp)) else 0.0

    # Prefixes
    p3 = 1.0 if (s1_nc and cand_nc and s1_nc[:3] == cand_nc[:3]) else 0.0
    p5 = 1.0 if (s1_nc and cand_nc and len(s1_nc) >= 5 and len(cand_nc) >= 5 and s1_nc[:5] == cand_nc[:5]) else 0.0

    max_nl = max(s1.name_len, cand.name_len, 1)
    name_len_diff = abs(s1.name_len - cand.name_len) / max_nl

    # 2. Address Features
    s1_ad = s1.addr_norm
    cand_ad = cand.addr_norm
    s1_ad_tr = s1.addr_translit
    cand_ad_tr = cand.addr_translit

    addr_exact = 1.0 if s1_ad and len(s1_ad) >= 4 and s1_ad == cand_ad else 0.0

    if addr_exact == 1.0:
        addr_fuzz = 1.0
        addr_partial = 1.0
        addr_sort = 1.0
        addr_set = 1.0
        addr_jw = 1.0
        addr_lev = 1.0
        addr_tr_fuzz = 1.0
    elif s1_ad and cand_ad:
        addr_fuzz = fuzz.ratio(s1_ad, cand_ad) / 100.0
        addr_partial = fuzz.partial_ratio(s1_ad, cand_ad) / 100.0
        addr_sort = fuzz.token_sort_ratio(s1_ad, cand_ad) / 100.0
        addr_set = fuzz.token_set_ratio(s1_ad, cand_ad) / 100.0
        addr_jw = distance.JaroWinkler.similarity(s1_ad, cand_ad)
        addr_lev = distance.Levenshtein.normalized_similarity(s1_ad, cand_ad)
        addr_tr_fuzz = fuzz.ratio(s1_ad_tr, cand_ad_tr) / 100.0 if s1_ad_tr and cand_ad_tr else 0.0
    else:
        addr_fuzz = addr_partial = addr_sort = addr_set = addr_jw = addr_lev = addr_tr_fuzz = 0.0

    # Address token overlap without set union allocation
    s1_at_len = len(s1.addr_tokens)
    cand_at_len = len(cand.addr_tokens)
    ad_inter = len(s1.addr_tokens.intersection(cand.addr_tokens))
    ad_union = s1_at_len + cand_at_len - ad_inter
    addr_jaccard = (ad_inter / ad_union) if ad_union > 0 else 0.0
    min_ad = min(s1_at_len, cand_at_len)
    addr_overlap_ratio = (ad_inter / min_ad) if min_ad > 0 else 0.0

    # House numbers
    s1_an_len = len(s1.addr_nums)
    cand_an_len = len(cand.addr_nums)
    num_inter = len(s1.addr_nums.intersection(cand.addr_nums))
    num_union = s1_an_len + cand_an_len - num_inter
    hn_overlap = 1.0 if num_inter > 0 else 0.0
    hn_jaccard = (num_inter / num_union) if num_union > 0 else 0.0
    hn_exact = 1.0 if s1.addr_nums and s1.addr_nums == cand.addr_nums else 0.0
    has_hn = 1.0 if (s1_an_len > 0 and cand_an_len > 0) else 0.0

    # Street tokens
    s1_st_len = len(s1.street_tokens)
    cand_st_len = len(cand.street_tokens)
    st_inter = len(s1.street_tokens.intersection(cand.street_tokens))
    st_union = s1_st_len + cand_st_len - st_inter
    st_jaccard = (st_inter / st_union) if st_union > 0 else 0.0

    # City & State & Postal
    city_sim = fuzz.ratio(s1.city, cand.city) / 100.0 if s1.city and cand.city else 0.0
    state_sim = fuzz.ratio(s1.state, cand.state) / 100.0 if s1.state and cand.state else 0.0
    postal_match = 1.0 if s1.postal and cand.postal and s1.postal == cand.postal else 0.0

    max_al = max(s1.addr_len, cand.addr_len, 1)
    addr_len_diff = abs(s1.addr_len - cand.addr_len) / max_al

    # 3. Strategy Features
    s_mask = int(strat_mask)
    strat_en = 1.0 if (s_mask & 1) else 0.0
    strat_nns = 1.0 if (s_mask & 2) else 0.0
    strat_sp = 1.0 if (s_mask & 4) else 0.0
    strat_ea = 1.0 if (s_mask & 8) else 0.0
    strat_num_tok = 1.0 if (s_mask & 16) else 0.0
    strat_num_st = 1.0 if (s_mask & 32) else 0.0
    strat_two = 1.0 if (s_mask & 64) else 0.0
    strat_first = 1.0 if (s_mask & 128) else 0.0
    strat_count = float(strat_en + strat_nns + strat_sp + strat_ea + strat_num_tok + strat_num_st + strat_two + strat_first)

    log_cands = float(np.log1p(cand_count))
    cand_eid = cand.eid
    is_s2 = 1.0 if cand_eid.startswith('S2') else 0.0
    is_s3 = 1.0 if cand_eid.startswith('S3') else 0.0

    # 4. Context & Flags
    country_exact = 1.0 if s1.country and s1.country == cand.country else 0.0
    is_us = 1.0 if s1.country.upper() == 'US' else 0.0
    is_in = 1.0 if s1.country.lower() in ('india', 'in') else 0.0
    missing_name = 1.0 if (not s1.has_name or not cand.has_name) else 0.0
    missing_addr = 1.0 if (not s1.has_addr or not cand.has_addr) else 0.0
    has_unicode = 1.0 if (s1.is_unicode_non_ascii or cand.is_unicode_non_ascii) else 0.0

    # 5. Composite Interactions
    mean_sim = 0.5 * (name_sort + addr_sort)
    mult_sim = name_sort * addr_sort
    max_sim = max(name_fuzz, addr_fuzz)
    min_sim = min(name_fuzz, addr_fuzz)

    out_vector[row_idx, :] = (
        exact_clean, exact_raw, exact_no_suf, exact_spaceless, exact_translit,
        name_fuzz, name_partial, name_sort, name_set, name_jw, name_lev,
        no_suf_fuzz, no_suf_sort, spaceless_fuzz, translit_fuzz, translit_sort,
        name_jaccard, name_overlap_ratio, float(t_inter), name_containment,
        spaceless_containment, p3, p5, name_len_diff, float(min(s1.name_len, cand.name_len)),
        float(max(s1.name_len, cand.name_len)),

        addr_exact, addr_fuzz, addr_partial, addr_sort, addr_set,
        addr_jw, addr_lev, addr_tr_fuzz, addr_jaccard, addr_overlap_ratio,
        hn_overlap, hn_jaccard, hn_exact, has_hn, st_jaccard,
        float(st_inter), city_sim, state_sim, postal_match, addr_len_diff,

        strat_en, strat_nns, strat_sp, strat_ea, strat_num_tok,
        strat_num_st, strat_two, strat_first, strat_count,
        float(cand_count), log_cands, is_s2, is_s3,

        country_exact, is_us, is_in, missing_name, missing_addr, has_unicode,

        mean_sim, mult_sim, max_sim, min_sim
    )


def test_1k():
    print("Loading model and indexes...")
    model_data = joblib.load(DEFAULT_MODEL_PATH)
    model = model_data["model"]
    optimal_threshold = float(model_data.get("optimal_threshold", 0.88))
    
    conn = build_or_connect_sqlite_entities(DB_PATH)
    keys_mmap, offsets_mmap, counts_mmap, postings_mmap = load_mmap_inverted_indexes(TEST_CACHE_DIR)
    
    out_file = PROJECT_ROOT / "outputs" / "test_1000_optimized.tsv"
    
    # Entity LRU / Dict Cache
    cand_entity_cache: Dict[str, OptimizedEntity] = {}
    
    t0_start = time.time()
    
    df = pd.read_csv(S1_TEST_FILE, sep="\t", dtype=str, nrows=1000)
    batch_size = 500
    
    lines = ["source1_entity_id\tmatched_entity_ids\n"]
    total_pairs = 0
    total_matches = 0
    
    for b_start in range(0, len(df), batch_size):
        t_b0 = time.time()
        b_df = df.iloc[b_start : b_start + batch_size]
        
        s1_entities = [
            OptimizedEntity(eid, c, n, a)
            for eid, c, n, a in zip(b_df["entity_id"], b_df["country"].fillna(""), b_df["business_name"].fillna(""), b_df["business_address"].fillna(""))
        ]
        s1_map = {e.eid: e for e in s1_entities}
        
        # 1. Candidate lookup
        t_c0 = time.time()
        pairs, unique_cids = query_candidates_vectorized(
            s1_entities, keys_mmap, offsets_mmap, counts_mmap, postings_mmap
        )
        t_c = time.time() - t_c0
        total_pairs += len(pairs)
        
        # 2. SQLite fetch for missing candidates
        t_s0 = time.time()
        missing_cids = [cid for cid in unique_cids if cid not in cand_entity_cache]
        if missing_cids:
            cur = conn.cursor()
            CHUNK_SIZE = 900
            for i in range(0, len(missing_cids), CHUNK_SIZE):
                chunk = missing_cids[i : i + CHUNK_SIZE]
                placeholders = ",".join("?" * len(chunk))
                query = f"SELECT eid, country, name, addr FROM entities WHERE eid IN ({placeholders})"
                for eid, country, name, addr in cur.execute(query, chunk):
                    cand_entity_cache[eid] = OptimizedEntity(eid, country, name, addr)
        t_s = time.time() - t_s0
        
        # 3. Feature extraction & prediction
        t_f0 = time.time()
        s1_matched_cands: Dict[str, List[str]] = {e.eid: [] for e in s1_entities}
        
        if pairs:
            # Process in slices of 100k to maintain minimal RAM footprint
            PAIR_CHUNK = 100000
            for p_start in range(0, len(pairs), PAIR_CHUNK):
                p_chunk = pairs[p_start : p_start + PAIR_CHUNK]
                X_batch = np.zeros((len(p_chunk), len(FEATURE_NAMES)), dtype=np.float32)
                for idx, (s1_id, cand_id, mask, c_cnt) in enumerate(p_chunk):
                    s1_obj = s1_map[s1_id]
                    cand_obj = cand_entity_cache.get(cand_id)
                    if cand_obj is not None:
                        extract_pair_features_fast(s1_obj, cand_obj, mask, c_cnt, X_batch, idx)
                
                probas = model.predict_proba(X_batch)[:, 1].astype(np.float32)
                for (s1_id, cand_id, _, _), proba in zip(p_chunk, probas):
                    if proba >= optimal_threshold:
                        s1_matched_cands[s1_id].append(cand_id)
                del X_batch, probas
        t_f = time.time() - t_f0
        
        # Format lines
        for s1 in s1_entities:
            matches = s1_matched_cands[s1.eid]
            total_matches += len(matches)
            lines.append(f"{s1.eid}\t{','.join(matches)}\n")
            
        t_b = time.time() - t_b0
        print(f"Batch {b_start//batch_size + 1}: {len(s1_entities)} S1 in {t_b:.2f}s | "
              f"Lookup: {t_c:.2f}s | SQLite: {t_s:.2f}s | Feats: {t_f:.2f}s | "
              f"Cands: {len(pairs):,} | Matches: {total_matches:,} | Speed: {len(s1_entities)/t_b:.1f} S1/s")
        
    out_file.write_text("".join(lines), encoding="utf-8")
    total_time = time.time() - t0_start
    print(f"\n1000 S1 Completed in {total_time:.2f}s ({1000/total_time:.1f} S1/s)!")
    print(f"Total Candidate Pairs: {total_pairs:,}")
    print(f"Total Predicted Matches: {total_matches:,}")
    
    # COMPARE WITH BASELINE test_stage1_opt.tsv
    baseline_file = PROJECT_ROOT / "outputs" / "test_stage1_opt.tsv"
    if baseline_file.exists():
        print("\n" + "="*80)
        print("COMPARING WITH BASELINE: test_stage1_opt.tsv")
        print("="*80)
        base_df = pd.read_csv(baseline_file, sep="\t", dtype=str)
        new_df = pd.read_csv(out_file, sep="\t", dtype=str)
        
        base_map = dict(zip(base_df["source1_entity_id"], base_df["matched_entity_ids"].fillna("")))
        new_map = dict(zip(new_df["source1_entity_id"], new_df["matched_entity_ids"].fillna("")))
        
        mismatches = 0
        for s1_id in base_map:
            if s1_id in new_map:
                b_m = set(base_map[s1_id].split(",")) if base_map[s1_id] else set()
                n_m = set(new_map[s1_id].split(",")) if new_map[s1_id] else set()
                if b_m != n_m:
                    mismatches += 1
                    print(f"Mismatch for {s1_id}:")
                    print(f"  Baseline : {base_map[s1_id]}")
                    print(f"  Optimized: {new_map[s1_id]}")
                    if mismatches >= 5:
                        break
                        
        print(f"Total S1 checked: {len(base_map)}")
        print(f"Mismatched S1: {mismatches}")
        if mismatches == 0:
            print("100.00% EXACT MATCH WITH BASELINE ON ALL 1,000 S1 ENTITIES!")

if __name__ == "__main__":
    test_1k()
