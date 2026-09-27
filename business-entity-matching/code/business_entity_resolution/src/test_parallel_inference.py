"""
Parallel feature extraction benchmark using multiprocessing on Windows.
"""

import sys
import os
import time
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

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


class CompactEntity:
    __slots__ = (
        'eid', 'country', 'raw_name', 'raw_addr',
        'name_clean', 'name_no_suf', 'spaceless_name', 'name_translit',
        'name_tokens', 'name_len',
        'addr_norm', 'addr_translit', 'addr_nums', 'street_tokens',
        'addr_tokens', 'addr_len', 'postal', 'city', 'state',
        'has_name', 'has_addr', 'is_unicode_non_ascii',
        'core_tokens', 'dist_tokens'
    )

    def __init__(self, eid: str, country: str, raw_name: str, raw_addr: str, is_s1: bool = False):
        self.eid = eid
        self.country = country.strip() if country else ''
        self.raw_name = raw_name.strip() if raw_name else ''
        self.raw_addr = raw_addr.strip() if raw_addr else ''

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

        self.has_name = bool(self.name_clean)
        self.has_addr = bool(self.addr_norm)
        self.is_unicode_non_ascii = not (raw_name_ascii and raw_addr_ascii)

        if is_s1:
            self.core_tokens = [t for t in _name_tokens_list if len(t) >= 3 and t not in LEGAL_SUFFIXES]
            self.dist_tokens = [t for t in _name_tokens_list if len(t) >= 3 and t not in LEGAL_SUFFIXES and t not in GENERIC_WORDS]
        else:
            self.core_tokens = None
            self.dist_tokens = None


def extract_pair_features_fast(
    s1: CompactEntity,
    cand: CompactEntity,
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


def extract_features_worker(
    s1_dict: Dict[str, CompactEntity],
    cand_dict: Dict[str, CompactEntity],
    pair_chunk: List[Tuple[str, str, int, int]]
) -> np.ndarray:
    """Worker process task for computing 69 features for a slice of pairs."""
    n_pairs = len(pair_chunk)
    X = np.zeros((n_pairs, len(FEATURE_NAMES)), dtype=np.float32)
    for idx, (s1_id, cand_id, mask, c_cnt) in enumerate(pair_chunk):
        s1_obj = s1_dict.get(s1_id)
        cand_obj = cand_dict.get(cand_id)
        if s1_obj and cand_obj:
            extract_pair_features_fast(s1_obj, cand_obj, mask, c_cnt, X, idx)
    return X


def test_parallel():
    print("Connecting SQLite and loading sample...")
    conn = build_or_connect_sqlite_entities(DB_PATH)
    rows = conn.execute("SELECT eid, country, name, addr FROM entities LIMIT 20000").fetchall()
    
    entities = [CompactEntity(eid, c, n, a) for eid, c, n, a in rows]
    s1 = entities[0]
    cands = entities[1:10001]
    
    s1_dict = {s1.eid: s1}
    cand_dict = {c.eid: c for c in cands}
    pairs = [(s1.eid, c.eid, 1, len(cands)) for c in cands] * 10 # 100,000 pairs
    
    # 1. Single core benchmark
    t0 = time.time()
    X_single = extract_features_worker(s1_dict, cand_dict, pairs)
    t_single = time.time() - t0
    print(f"Single Core: 100k pairs in {t_single:.3f}s ({len(pairs)/t_single:.1f} pairs/s)")
    
    # 2. Multi-core benchmark
    for num_workers in [4, 8, 12]:
        t0 = time.time()
        chunk_size = (len(pairs) + num_workers - 1) // num_workers
        chunks = [pairs[i : i + chunk_size] for i in range(0, len(pairs), chunk_size)]
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(extract_features_worker, s1_dict, cand_dict, ch) for ch in chunks]
            results = [f.result() for f in futures]
        X_par = np.vstack(results)
        t_par = time.time() - t0
        print(f"Parallel ({num_workers} workers): 100k pairs in {t_par:.3f}s ({len(pairs)/t_par:.1f} pairs/s) -> {t_single/t_par:.2f}x speedup!")
        np.testing.assert_allclose(X_single, X_par, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_parallel()
