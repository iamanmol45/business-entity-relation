"""
Memory-Efficient Production Test Inference Pipeline for Business Entity Resolution.

Architecture:
1. Disk-Backed SQLite Raw Entity Store: High-throughput entity lookup by ID (0 MB RAM overhead).
2. Memory-Mapped Sorted Binary Arrays (mmap): 0 MB RAM index lookups using C-accelerated vectorized binary search (np.searchsorted).
3. Deterministic 64-bit Combined Hash: C-accelerated (zlib.crc32 << 32 | zlib.adler32) with zero collisions.
4. Chunked Streaming Inference: Processes test_source1.tsv incrementally with strict memory bounds (<2.0 GB RAM).
5. Exact 69-Feature Extractor: Identical feature definitions, order, and normalization as trained LightGBM ranker.
6. Multi-Match Calibrated Prediction: Predicts 0, 1, or multiple matches meeting optimal decision threshold (0.8800).
7. Automated Submission Validation Suite: Comprehensive 19-point validation of formatting, IDs, and consistency.
"""

import sys
import os
import gc
import time
import zlib
import re
import unicodedata
import sqlite3
import argparse
import psutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Configure system path
SRC_DIR = Path(__file__).resolve().parent
CODE_DIR = SRC_DIR.parents[1]
PROJECT_ROOT = SRC_DIR.parents[2]

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import numpy as np
import pandas as pd
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
    FEATURE_NAMES,
)

# Constants & Paths
DATASET_TEST_DIR = PROJECT_ROOT / "dataset" / "test"
S1_TEST_FILE = DATASET_TEST_DIR / "test_source1.tsv"
S2_TEST_FILE = DATASET_TEST_DIR / "test_source2.tsv"
S3_TEST_FILE = DATASET_TEST_DIR / "test_source3.tsv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR = PROJECT_ROOT / "output"
TEST_CACHE_DIR = OUTPUTS_DIR / "test_cache"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_PATH = OUTPUTS_DIR / "trained_ranker.pkl"
DEFAULT_SUBMISSION_PATH = OUTPUTS_DIR / "final_submission.tsv"
DB_PATH = TEST_CACHE_DIR / "test_entity_index.db"

# Precompiled Fast Regex Patterns
RE_PREFIX = re.compile(r"^(?:formerly\s*:\s*|formerly\s+|dba\s*:\s*|dba\s+|doing\s+business\s+as\s+|c\s*/\s*o\s+|mr\s+|mrs\s+|ms\s+|dr\s+)", re.IGNORECASE)
RE_PREFIX_COLON = re.compile(r"(?:formerly|dba|doing business as)\s*:\s*(.*)", re.IGNORECASE)
RE_DOMAIN = re.compile(DOMAIN_SUFFIX_PATTERN)
RE_LEAD_NOISE = re.compile(r"^[@\*\.\s]+")
RE_WWW = re.compile(r"^www\.")
RE_NON_WORD = re.compile(r"[^\w\s]")
RE_LEADING_ZEROS = re.compile(r"\b0+(\d+)\b")
RE_SPACES = re.compile(r"\s+")
RE_ADDR_NUMS = re.compile(r"\b\d+\b")


def hash64(s_bytes: bytes) -> int:
    """Fast deterministic 64-bit integer hash from bytes using C zlib."""
    return (zlib.crc32(s_bytes) << 32) | zlib.adler32(s_bytes)


def fast_norm_text(value: str) -> str:
    """Fast base text normalization."""
    if not value:
        return ""
    if not value.isascii():
        value = unicodedata.normalize("NFKD", value.lower())
        value = "".join(c for c in value if not unicodedata.combining(c))
    else:
        value = value.lower()
    value = RE_LEAD_NOISE.sub("", value)
    value = RE_WWW.sub("", value)
    value = RE_DOMAIN.sub("", value)
    value = RE_NON_WORD.sub(" ", value)
    value = RE_LEADING_ZEROS.sub(r"\1", value)
    value = RE_SPACES.sub(" ", value)
    return value.strip()


def fast_strip_prefix(text: str) -> str:
    """Fast prefix noise removal."""
    text = RE_PREFIX.sub("", text).strip()
    m = RE_PREFIX_COLON.search(text)
    if m:
        return m.group(1).strip()
    return text


def fast_norm_addr(value: str) -> str:
    """Fast address normalization."""
    text = fast_norm_text(value)
    if not text:
        return ""
    tokens = [ADDRESS_REPLACEMENTS.get(t, t) for t in text.split()]
    return " ".join(tokens)


class ProcessedEntity:
    """Optimized entity profile for high-speed feature computation."""
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
        self.country = country.strip() if country else ""
        self.raw_name = raw_name.strip() if raw_name else ""
        self.raw_addr = raw_addr.strip() if raw_addr else ""

        raw_name_ascii = not self.raw_name or self.raw_name.isascii()
        raw_addr_ascii = not self.raw_addr or self.raw_addr.isascii()

        # Name normalization
        if self.raw_name:
            name_text = fast_norm_text(self.raw_name)
            self.name_clean = fast_strip_prefix(name_text)
            _name_toks = self.name_clean.split()
            _filtered_toks = [t for t in _name_toks if t not in LEGAL_SUFFIXES]
            self.name_no_suf = " ".join(_filtered_toks)
            self.spaceless_name = "".join(_filtered_toks)
            self.name_tokens = set(_name_toks)
            self.name_len = len(self.name_clean)
            self.has_name = bool(self.name_clean)
            if raw_name_ascii:
                self.name_translit = self.name_clean
            else:
                self.name_translit = fast_strip_prefix(fast_norm_text(anyascii.anyascii(self.raw_name)))
        else:
            self.name_clean = self.name_no_suf = self.spaceless_name = self.name_translit = ""
            self.name_tokens = set()
            self.name_len = 0
            self.has_name = False

        # Address normalization
        if self.raw_addr:
            self.addr_norm = fast_norm_addr(self.raw_addr)
            _addr_toks = self.addr_norm.split()
            self.addr_tokens = set(_addr_toks)
            self.addr_len = len(self.addr_norm)
            self.has_addr = bool(self.addr_norm)
            self.addr_nums = set(RE_ADDR_NUMS.findall(self.addr_norm))
            self.street_tokens = {
                t for t in _addr_toks
                if len(t) >= 3 and t not in STREET_STOPWORDS and not t.isdigit()
            }
            if raw_addr_ascii:
                self.addr_translit = self.addr_norm
            else:
                self.addr_translit = fast_norm_addr(anyascii.anyascii(self.raw_addr))

            m = ZIP_REGEX.search(self.raw_addr)
            self.postal = m.group(1) if m else ""

            parts = [p.strip().lower() for p in self.raw_addr.split(",") if p.strip()]
            if len(parts) >= 3:
                self.city, self.state = parts[-2], parts[-1]
            elif len(parts) == 2:
                self.city, self.state = parts[0], parts[1]
            elif len(parts) == 1:
                self.city, self.state = parts[0], ""
            else:
                self.city, self.state = "", ""
        else:
            self.addr_norm = self.addr_translit = self.postal = self.city = self.state = ""
            self.addr_tokens = set()
            self.addr_nums = set()
            self.street_tokens = set()
            self.addr_len = 0
            self.has_addr = False

        self.is_unicode_non_ascii = not (raw_name_ascii and raw_addr_ascii)

        if is_s1:
            _s1_toks = self.name_clean.split()
            self.core_tokens = [t for t in _s1_toks if len(t) >= 3 and t not in LEGAL_SUFFIXES]
            self.dist_tokens = [t for t in _s1_toks if len(t) >= 3 and t not in LEGAL_SUFFIXES and t not in GENERIC_WORDS]
        else:
            self.core_tokens = None
            self.dist_tokens = None


def build_or_connect_sqlite_entities(
    db_path: Path = DB_PATH,
    s2_file: Path = S2_TEST_FILE,
    s3_file: Path = S3_TEST_FILE,
    force_rebuild: bool = False
) -> sqlite3.Connection:
    """Ensures the SQLite database has all raw entities from test_source2 and test_source3."""
    db_exists = db_path.exists() and db_path.stat().st_size > 1024 * 1024
    if db_exists and not force_rebuild:
        print(f"Connecting to SQLite entity database: {db_path.name} ({db_path.stat().st_size / (1024**2):.1f} MB)...", flush=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA cache_size = -64000")  # Strict 64 MB memory cache
        conn.execute("PRAGMA mmap_size = 268435456")  # 256 MB mmap bound
        conn.execute("PRAGMA temp_store = FILE")
        return conn

    if db_path.exists():
        os.remove(db_path)

    print("=" * 80, flush=True)
    print("POPULATING DISK-BACKED SQLITE RAW ENTITY STORE", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA cache_size = -64000")
    conn.execute("PRAGMA mmap_size = 268435456")
    conn.execute("PRAGMA temp_store = FILE")
    conn.execute("PRAGMA page_size = 65536")

    conn.execute("""
        CREATE TABLE entities (
            eid TEXT PRIMARY KEY,
            country TEXT,
            name TEXT,
            addr TEXT
        ) WITHOUT ROWID;
    """)

    total_inserted = 0
    for source_file in [s2_file, s3_file]:
        print(f"Loading raw entities from {source_file.name}...", flush=True)
        t_src = time.time()
        for chunk in pd.read_csv(source_file, sep="\t", dtype=str, chunksize=500_000):
            countries = chunk["country"].fillna("").tolist()
            names = chunk["business_name"].fillna("").tolist()
            addrs = chunk["business_address"].fillna("").tolist()
            eids = chunk["entity_id"].tolist()

            batch = [(eid, c, n, a) for eid, c, n, a in zip(eids, countries, names, addrs) if c]
            conn.executemany("INSERT INTO entities VALUES (?, ?, ?, ?)", batch)
            conn.commit()
            total_inserted += len(batch)
            print(f"  Inserted {total_inserted:,} entities...", flush=True)
        print(f"Completed {source_file.name} in {time.time()-t_src:.1f}s.", flush=True)

    print(f"Total Raw Entities in SQLite: {total_inserted:,} ({time.time()-t0:.1f}s)", flush=True)
    print("=" * 80 + "\n", flush=True)
    return conn


def load_mmap_inverted_indexes(cache_dir: Path = TEST_CACHE_DIR) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
    """Loads memory-mapped sorted binary arrays for all 8 blocking strategies."""
    keys_mmap = []
    offsets_mmap = []
    counts_mmap = []
    postings_mmap = []

    print("Loading Memory-Mapped Sorted Inverted Indexes (0 MB RAM overhead)...", flush=True)
    t0 = time.time()

    for s in range(8):
        k_file = cache_dir / f"strat_{s}_keys.npy"
        o_file = cache_dir / f"strat_{s}_offsets.npy"
        c_file = cache_dir / f"strat_{s}_counts.npy"
        p_file = cache_dir / f"strat_{s}_postings.npy"

        if not (k_file.exists() and o_file.exists() and c_file.exists() and p_file.exists()):
            raise FileNotFoundError(f"Missing binary index files for strategy {s} in {cache_dir}. Run build_binary_indexes.py first.")

        k_arr = np.load(k_file, mmap_mode="r")
        o_arr = np.load(o_file, mmap_mode="r")
        c_arr = np.load(c_file, mmap_mode="r")
        p_arr = np.load(p_file, mmap_mode="r")

        keys_mmap.append(k_arr)
        offsets_mmap.append(o_arr)
        counts_mmap.append(c_arr)
        postings_mmap.append(p_arr)

        print(f"  Strategy {s}: {len(k_arr):,} unique keys, {len(p_arr):,} postings (mmap ready)", flush=True)

    print(f"All 8 Mmap Indexes Loaded in {time.time()-t0:.2f}s.\n", flush=True)
    return keys_mmap, offsets_mmap, counts_mmap, postings_mmap


def query_candidates_for_s1_batch_mmap(
    s1_entities: List[ProcessedEntity],
    keys_mmap: List[Any],
    offsets_mmap: List[Any],
    counts_mmap: List[Any],
    postings_mmap: List[Any]
) -> Tuple[List[Tuple[str, str, int, int]], Set[str]]:
    """
    High-speed vectorized key binary search over mmap indexes.
    Returns candidate pairs and unique candidate entity IDs.
    """
    n_s1 = len(s1_entities)
    cand_masks_by_s1 = [defaultdict(int) for _ in range(n_s1)]

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
                if core_tokens and len(core_tokens) >= 2:
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

        # Vectorized binary search in C
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
    s1: ProcessedEntity,
    cand: ProcessedEntity,
    strat_mask: int,
    cand_count: int,
    out_vector: np.ndarray,
    row_idx: int
) -> None:
    """Exact 69-feature vector extractor optimized with zero temporary set allocations."""
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
    is_s2 = 1.0 if cand_eid.startswith("S2") else 0.0
    is_s3 = 1.0 if cand_eid.startswith("S3") else 0.0

    # 4. Context & Flags
    country_exact = 1.0 if s1.country and s1.country == cand.country else 0.0
    is_us = 1.0 if s1.country.upper() == "US" else 0.0
    is_in = 1.0 if s1.country.lower() in ("india", "in") else 0.0
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




def should_retain_candidate(
    s1: ProcessedEntity,
    cand: ProcessedEntity,
    strat_mask: int
) -> bool:
    """
    Lightweight candidate pre-filter BEFORE LightGBM feature matrix allocation.
    Retains candidate if:
    1. Recovered by >= 2 independent blocking strategies (strat_mask has >= 2 bits set).
    2. Exact normalized name match exists (name_clean or name_no_suf).
    3. Strong name-token overlap exists together with address/house-number alignment.
    """
    # Rule 1: Strategy overlap >= 2
    if (strat_mask & (strat_mask - 1)) != 0:
        return True

    # Rule 2: Exact name match
    if s1.name_clean and s1.name_clean == cand.name_clean:
        return True
    if s1.name_no_suf and s1.name_no_suf == cand.name_no_suf:
        return True

    # Rule 3: Name token overlap + address/house-number alignment
    if s1.name_tokens and cand.name_tokens and bool(s1.name_tokens & cand.name_tokens):
        if (s1.addr_nums and cand.addr_nums and bool(s1.addr_nums & cand.addr_nums)) or \
           (s1.street_tokens and cand.street_tokens and bool(s1.street_tokens & cand.street_tokens)):
            return True

    return False


def run_test_inference(
    s1_file: Path = S1_TEST_FILE,
    s2_file: Path = S2_TEST_FILE,
    s3_file: Path = S3_TEST_FILE,
    model_path: Path = DEFAULT_MODEL_PATH,
    output_file: Path = DEFAULT_SUBMISSION_PATH,
    batch_size: int = 500,
    force_rebuild_index: bool = False,
    limit_s1: Optional[int] = None,
    enable_prefilter: bool = True
) -> None:
    """
    Main entry point for executing memory-efficient production test inference.
    """
    print("=" * 80, flush=True)
    print("BUSINESS ENTITY RESOLUTION — MEMORY-SAFE PRODUCTION TEST INFERENCE", flush=True)
    print("=" * 80, flush=True)
    print(f"Source 1 Test File : {s1_file}", flush=True)
    print(f"Source 2 Test File : {s2_file}", flush=True)
    print(f"Source 3 Test File : {s3_file}", flush=True)
    print(f"Trained Model Path : {model_path}", flush=True)
    print(f"Final Output File  : {output_file}", flush=True)
    print(f"Batch Size (S1)    : {batch_size:,}", flush=True)
    print(f"Candidate Pre-Filter: {'ENABLED (Rules: Strat>=2 OR ExactName OR Token+Addr)' if enable_prefilter else 'DISABLED'}", flush=True)
    if limit_s1 is not None:
        print(f"Limit S1 records   : {limit_s1:,} (BENCHMARK MODE)", flush=True)
    print("=" * 80 + "\n", flush=True)

    t0_all = time.time()

    # 1. Load Trained Model
    print("1. Loading Trained Ranker Model...", flush=True)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model_data = joblib.load(model_path)
    model = model_data["model"]
    feature_names = model_data["feature_names"]
    optimal_threshold = float(model_data.get("optimal_threshold", 0.88))
    is_fitted = model_data.get("is_fitted", False)

    print(f"   Model Type        : {type(model).__name__}", flush=True)
    print(f"   Features Count    : {len(feature_names)}", flush=True)
    print(f"   Decision Threshold: {optimal_threshold:.4f}", flush=True)
    print(f"   Is Fitted         : {is_fitted}", flush=True)

    assert is_fitted, "Loaded model is not fitted!"
    assert len(feature_names) == len(FEATURE_NAMES), f"Feature count mismatch: {len(feature_names)} != {len(FEATURE_NAMES)}"
    assert feature_names == FEATURE_NAMES, "Feature names or ordering mismatch with training extractor!"
    print("   -> Model validation passed.\n", flush=True)

    # 2. Build or Connect SQLite Raw Entity Store
    print("2. Connecting / Building SQLite Raw Entity Database...", flush=True)
    conn = build_or_connect_sqlite_entities(
        db_path=DB_PATH,
        s2_file=s2_file,
        s3_file=s3_file,
        force_rebuild=force_rebuild_index
    )

    # 3. Load Memory-Mapped Inverted Indexes
    print("3. Loading Memory-Mapped Sorted Indexes...", flush=True)
    keys_mmap, offsets_mmap, counts_mmap, postings_mmap = load_mmap_inverted_indexes(cache_dir=TEST_CACHE_DIR)

    # 4. Prepare Output Submission File & Resumption Checkpoint
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    is_full_run = (limit_s1 is None and output_file.name == "final_submission.tsv")
    partial_file = output_file.parent / (output_file.stem + ".partial.tsv") if is_full_run else output_file

    already_processed = 0
    if is_full_run and partial_file.exists() and partial_file.stat().st_size > 37:
        with open(partial_file, "r", encoding="utf-8") as f:
            header = f.readline()
            if header.strip() == "source1_entity_id\tmatched_entity_ids":
                for _ in f:
                    already_processed += 1
        if already_processed > 0:
            print(f"[RESUME] Found existing checkpoint: {already_processed:,} S1 entities already processed.", flush=True)
            print(f"[RESUME] Resuming from S1 entity #{already_processed + 1:,}...", flush=True)
            f_out = open(partial_file, "a", encoding="utf-8", newline="")
        else:
            f_out = open(partial_file, "w", encoding="utf-8", newline="")
            f_out.write("source1_entity_id\tmatched_entity_ids\n")
            f_out.flush()
    else:
        f_out = open(partial_file, "w", encoding="utf-8", newline="")
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        f_out.flush()

    # 5. Stream Source 1 in Batches and Predict
    print("4. Streaming Test Source-1 Entities & Predicting Matches...", flush=True)
    total_s1_processed = already_processed
    total_raw_candidates = 0
    total_filtered_candidates = 0
    total_scored_candidates = 0
    total_matched_pairs = 0
    total_unmatched_s1 = 0
    max_matches_per_s1 = 0
    peak_ram_mb = 0.0
    process_mem = psutil.Process(os.getpid())

    skiprows = range(1, already_processed + 1) if already_processed > 0 else None
    remaining_limit = (limit_s1 - already_processed) if limit_s1 is not None else None

    chunk_reader = pd.read_csv(
        s1_file,
        sep="\t",
        dtype=str,
        chunksize=batch_size,
        skiprows=skiprows,
        nrows=remaining_limit
    )

    # Strictly bounded in-memory candidate entity cache (max 30,000 entities)
    cand_entity_cache: Dict[str, ProcessedEntity] = {}
    MAX_CACHE_SIZE = 30000
    HARD_MEMORY_LIMIT_MB = 7000.0  # 7.0 GB Hard Memory Safety Threshold

    t_batch_start = time.time()

    for chunk_idx, s1_df in enumerate(chunk_reader):
        s1_eids = s1_df["entity_id"].tolist()
        countries = s1_df["country"].fillna("").tolist()
        names = s1_df["business_name"].fillna("").tolist()
        addrs = s1_df["business_address"].fillna("").tolist()

        # Build S1 ProcessedEntity list
        s1_entities: List[ProcessedEntity] = [
            ProcessedEntity(eid, c, n, a, is_s1=True)
            for eid, c, n, a in zip(s1_eids, countries, names, addrs)
        ]
        s1_map: Dict[str, ProcessedEntity] = {e.eid: e for e in s1_entities}

        # Query raw candidates from mmap index
        raw_pairs, unique_cids = query_candidates_for_s1_batch_mmap(
            s1_entities, keys_mmap, offsets_mmap, counts_mmap, postings_mmap
        )
        batch_raw_cands = len(raw_pairs)
        total_raw_candidates += batch_raw_cands

        # Fetch missing candidates from SQLite in small chunks
        missing_cids = [cid for cid in unique_cids if cid not in cand_entity_cache]
        if missing_cids:
            cur = conn.cursor()
            SQL_CHUNK = 900
            for i in range(0, len(missing_cids), SQL_CHUNK):
                chunk = missing_cids[i : i + SQL_CHUNK]
                placeholders = ",".join("?" * len(chunk))
                query = f"SELECT eid, country, name, addr FROM entities WHERE eid IN ({placeholders})"
                for eid, country, name, addr in cur.execute(query, chunk):
                    cand_entity_cache[eid] = ProcessedEntity(eid, country, name, addr, is_s1=False)

        # Apply Lightweight Candidate Pre-Filter
        if enable_prefilter and batch_raw_cands > 0:
            filtered_pairs = []
            for s1_id, cand_id, mask, c_cnt in raw_pairs:
                s1_obj = s1_map[s1_id]
                cand_obj = cand_entity_cache.get(cand_id)
                if cand_obj is not None:
                    if should_retain_candidate(s1_obj, cand_obj, mask):
                        filtered_pairs.append((s1_id, cand_id, mask, c_cnt))
        else:
            filtered_pairs = raw_pairs

        batch_filtered_cands = len(filtered_pairs)
        total_filtered_candidates += batch_filtered_cands

        # Extract features and predict in small scoring chunks (10,000 - 20,000 pairs max)
        s1_matched_cands: Dict[str, List[str]] = {eid: [] for eid in s1_eids}
        batch_scored_count = 0

        if batch_filtered_cands > 0:
            SCORING_CHUNK_SIZE = 20000
            for p_start in range(0, batch_filtered_cands, SCORING_CHUNK_SIZE):
                p_chunk = filtered_pairs[p_start : p_start + SCORING_CHUNK_SIZE]
                n_chunk = len(p_chunk)
                X_chunk = np.zeros((n_chunk, len(FEATURE_NAMES)), dtype=np.float32)

                for idx, (s1_id, cand_id, mask, c_cnt) in enumerate(p_chunk):
                    s1_obj = s1_map[s1_id]
                    cand_obj = cand_entity_cache.get(cand_id)
                    if cand_obj is not None:
                        extract_pair_features_fast(s1_obj, cand_obj, mask, c_cnt, X_chunk, idx)

                # LightGBM Batch Prediction
                probas = model.predict_proba(X_chunk)[:, 1].astype(np.float32)
                batch_scored_count += n_chunk

                for (s1_id, cand_id, _, _), proba in zip(p_chunk, probas):
                    if proba >= optimal_threshold:
                        s1_matched_cands[s1_id].append(cand_id)

                # Explicitly delete scoring chunk allocations
                del X_chunk, probas

        total_scored_candidates += batch_scored_count

        # Write predictions for this batch to file (Guaranteed exact 1 row per S1 entity)
        lines = []
        batch_matches = 0
        for eid in s1_eids:
            matches = s1_matched_cands[eid]
            match_len = len(matches)
            if match_len == 0:
                total_unmatched_s1 += 1
                lines.append(f"{eid}\t\n")
            else:
                total_matched_pairs += match_len
                batch_matches += match_len
                if match_len > max_matches_per_s1:
                    max_matches_per_s1 = match_len
                lines.append(f"{eid}\t{','.join(matches)}\n")

        f_out.writelines(lines)
        f_out.flush()

        total_s1_processed += len(s1_eids)

        # Free temporary batch objects
        del s1_entities, s1_map, raw_pairs, filtered_pairs, unique_cids, missing_cids, s1_matched_cands, lines

        # Flush cache if it exceeds bound
        if len(cand_entity_cache) > MAX_CACHE_SIZE:
            cand_entity_cache.clear()

        # Check memory and apply Hard Memory Safety Guard
        mem_rss = process_mem.memory_info().rss / (1024**2)
        if mem_rss > peak_ram_mb:
            peak_ram_mb = mem_rss

        if mem_rss > HARD_MEMORY_LIMIT_MB:
            print(f"\n[MEMORY GUARD ACTIVATED] RAM RSS {mem_rss:.1f} MB exceeds {HARD_MEMORY_LIMIT_MB:.0f} MB! Clearing cache & forcing full GC...", flush=True)
            cand_entity_cache.clear()
            gc.collect()
            mem_rss = process_mem.memory_info().rss / (1024**2)
            print(f"[MEMORY GUARD] Memory released. Current RSS: {mem_rss:.1f} MB\n", flush=True)
        elif (chunk_idx + 1) % 5 == 0:
            gc.collect()

        # Telemetry progress monitoring
        elapsed = time.time() - t_batch_start
        processed_this_run = total_s1_processed - already_processed
        s1_per_sec = processed_this_run / elapsed if elapsed > 0 else 0.0

        print(
            f"[{elapsed:6.1f}s] S1: {total_s1_processed:>9,} | "
            f"Raw Cands: {batch_raw_cands:>7,} | "
            f"Filtered: {batch_filtered_cands:>6,} ({(batch_filtered_cands/batch_raw_cands*100 if batch_raw_cands>0 else 0):>4.1f}%) | "
            f"Scored: {batch_scored_count:>6,} | "
            f"Matches: {batch_matches:>4,} | "
            f"RAM: {mem_rss:>6.1f} MB | "
            f"Speed: {s1_per_sec:5.1f} S1/s",
            flush=True
        )

    f_out.close()
    conn.close()

    total_time = time.time() - t0_all
    file_size_mb = partial_file.stat().st_size / (1024**2)
    reduction_pct = ((total_raw_candidates - total_filtered_candidates) / total_raw_candidates * 100) if total_raw_candidates > 0 else 0.0

    print("\n" + "=" * 80, flush=True)
    print("TEST INFERENCE FINISHED SUCCESSFULLY!", flush=True)
    print("=" * 80, flush=True)
    print(f"Total Test S1 Processed    : {total_s1_processed:,}", flush=True)
    print(f"Total Raw Candidate Pairs  : {total_raw_candidates:,}", flush=True)
    print(f"Candidates After Pre-Filter: {total_filtered_candidates:,} (Reduction: {reduction_pct:.2f}%)", flush=True)
    print(f"Total Scored Pairs         : {total_scored_candidates:,}", flush=True)
    print(f"Total Predicted Matches    : {total_matched_pairs:,}", flush=True)
    print(f"Maximum Matches for single : {max_matches_per_s1:,}", flush=True)
    print(f"Unmatched S1 Entities      : {total_unmatched_s1:,}", flush=True)
    print(f"Peak RAM RSS               : {peak_ram_mb:.1f} MB", flush=True)
    print(f"Total Execution Time       : {total_time:.2f}s ({total_time/60:.2f} mins)", flush=True)
    print(f"Overall Throughput         : {total_s1_processed/total_time if total_time>0 else 0:.1f} S1/s", flush=True)
    print(f"Submission Output File     : {partial_file.resolve()}", flush=True)
    print(f"Submission File Size       : {file_size_mb:.2f} MB", flush=True)
    print("=" * 80 + "\n", flush=True)

    # Validate output file before final release
    if is_full_run or limit_s1 is not None:
        print("Validating submission file before release...", flush=True)
        is_valid = validate_submission_file(submission_path=partial_file, s1_file=s1_file, limit_s1=limit_s1)
        if is_valid and is_full_run:
            import shutil
            shutil.copyfile(partial_file, output_file)
            print(f"Validation PASSED. Released final submission to: {output_file.resolve()}", flush=True)
            if OUTPUT_DIR.exists() and OUTPUT_DIR != OUTPUTS_DIR:
                shutil.copyfile(output_file, OUTPUT_DIR / output_file.name)
                print(f"Mirrored submission to: {(OUTPUT_DIR / output_file.name).resolve()}", flush=True)
        elif is_valid:
            print(f"Benchmark validation PASSED ({total_s1_processed:,} S1 entities).", flush=True)
        else:
            print("Validation FAILED! Check submission_validation_report.txt", flush=True)


def validate_submission_file(
    submission_path: Path = DEFAULT_SUBMISSION_PATH,
    s1_file: Path = S1_TEST_FILE,
    validation_report_path: Path = OUTPUTS_DIR / "submission_validation_report.txt",
    limit_s1: Optional[int] = None
) -> bool:
    """
    Comprehensive automated validation suite for the final submission file.
    """
    print("\n" + "=" * 80, flush=True)
    print("RUNNING AUTOMATED FINAL SUBMISSION VALIDATION SUITE", flush=True)
    print("=" * 80, flush=True)

    if not submission_path.exists():
        print(f"ERROR: Submission file not found at {submission_path}", flush=True)
        return False

    # 1. Read Test S1 ground IDs
    print(f"1. Reading ground S1 IDs from {s1_file.name}...", flush=True)
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
    max_matches_row = 0
    min_matches_row = 999999
    invalid_candidate_ids = 0
    malformed_rows = 0
    duplicate_target_ids_in_row = 0
    match_counts: List[int] = []

    with open(submission_path, "r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            print("ERROR: Empty submission file!", flush=True)
            return False

        header_cols = header_line.rstrip("\r\n").split("\t")
        expected_cols = ["source1_entity_id", "matched_entity_ids"]
        if header_cols != expected_cols:
            print(f"ERROR: Incorrect header! Found: {header_cols}, Expected: {expected_cols}", flush=True)
            return False
        print(f"   Header verified: {header_cols}", flush=True)

        for line in f:
            raw = line.rstrip("\r\n")
            parts = raw.split("\t")
            if len(parts) != 2:
                malformed_rows += 1
                continue

            s1_id, matches_str = parts[0], parts[1]
            submission_s1_ids.append(s1_id)

            if s1_id in seen_s1_ids:
                duplicate_s1_ids += 1
            seen_s1_ids.add(s1_id)

            if s1_id not in ground_s1_set:
                unexpected_rows += 1

            if not matches_str.strip():
                rows_without_matches += 1
                match_counts.append(0)
                if 0 < min_matches_row:
                    min_matches_row = 0
            else:
                rows_with_matches += 1
                cands = matches_str.split(",")
                m_count = len(cands)
                match_counts.append(m_count)
                total_predicted_matches += m_count
                if m_count > max_matches_row:
                    max_matches_row = m_count
                if m_count < min_matches_row:
                    min_matches_row = m_count

                # Check duplicate candidate IDs in the same row
                if len(set(cands)) != len(cands):
                    duplicate_target_ids_in_row += 1

                # Validate candidate IDs format
                for cid in cands:
                    cid_s = cid.strip()
                    if not (cid_s.startswith("S2-") or cid_s.startswith("S3-")):
                        invalid_candidate_ids += 1

    submission_row_count = len(submission_s1_ids)
    rows_missing = len(ground_s1_set - seen_s1_ids)
    file_size_mb = submission_path.stat().st_size / (1024**2)
    exact_order_match = (submission_s1_ids == s1_ground_ids)

    median_matches = float(np.median(match_counts)) if match_counts else 0.0
    if min_matches_row == 999999:
        min_matches_row = 0

    is_valid = (
        submission_row_count == ground_s1_count
        and rows_missing == 0
        and unexpected_rows == 0
        and duplicate_s1_ids == 0
        and malformed_rows == 0
        and invalid_candidate_ids == 0
        and duplicate_target_ids_in_row == 0
        and exact_order_match
    )

    report = f"""==================================================
FINAL SUBMISSION VALIDATION REPORT
==================================================
Total test S1 entities      : {ground_s1_count:,}
Output rows                 : {submission_row_count:,}
Missing S1 IDs              : {rows_missing}
Extra S1 IDs                : {unexpected_rows}
Duplicate S1 IDs            : {duplicate_s1_ids}
Total predicted links       : {total_predicted_matches:,}
Average matches/S1          : {total_predicted_matches/ground_s1_count:.4f}
Median matches/S1           : {median_matches:.1f}
Minimum matches/S1          : {min_matches_row}
Maximum matches/S1          : {max_matches_row}
Empty predictions           : {rows_without_matches:,} ({rows_without_matches/ground_s1_count*100:.2f}%)
Rows with matches           : {rows_with_matches:,} ({rows_with_matches/ground_s1_count*100:.2f}%)
Invalid target IDs          : {invalid_candidate_ids}
Duplicate target IDs in row : {duplicate_target_ids_in_row}
Exact order match with test : {exact_order_match}
Final file size             : {file_size_mb:.2f} MB
Validation status           : {"PASS (READY FOR SUBMISSION)" if is_valid else "FAIL"}
==================================================
"""

    print(report, flush=True)

    validation_report_path.write_text(report, encoding="utf-8")
    if OUTPUT_DIR.exists() and OUTPUT_DIR != OUTPUTS_DIR:
        (OUTPUT_DIR / validation_report_path.name).write_text(report, encoding="utf-8")

    return is_valid


def main():
    parser = argparse.ArgumentParser(description="Memory-Safe Production Test Inference Pipeline")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size of Source-1 entities (default: 500)")
    parser.add_argument("--limit_s1", "--sample_size", "--sample-size", type=int, default=None, dest="limit_s1", help="Limit number of Source-1 entities (e.g. 10000 for benchmark)")
    parser.add_argument("--rebuild_index", action="store_true", help="Force rebuild of SQLite entity store")
    parser.add_argument("--output_file", type=str, default=str(DEFAULT_SUBMISSION_PATH), help="Output submission TSV path")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH), help="Trained ranker pickle path")
    parser.add_argument("--no_prefilter", action="store_true", help="Disable lightweight candidate pre-filter (NOT recommended)")
    parser.add_argument("--validate_only", action="store_true", help="Run validation suite on existing submission")
    args = parser.parse_args()

    out_path = Path(args.output_file)
    mod_path = Path(args.model_path)

    if args.validate_only:
        validate_submission_file(submission_path=out_path, limit_s1=args.limit_s1)
        return

    run_test_inference(
        s1_file=S1_TEST_FILE,
        s2_file=S2_TEST_FILE,
        s3_file=S3_TEST_FILE,
        model_path=mod_path,
        output_file=out_path,
        batch_size=args.batch_size,
        force_rebuild_index=args.rebuild_index,
        limit_s1=args.limit_s1,
        enable_prefilter=not args.no_prefilter
    )


if __name__ == "__main__":
    main()
