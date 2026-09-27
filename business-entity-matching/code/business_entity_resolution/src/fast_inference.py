"""
Fast Inference Engine - Amazon ML Challenge Business Entity Resolution
=======================================================================
Three combined optimizations over sharded_inference.py:

1. EXACT-MATCH FAST-PATH
   Any candidate with exact name_clean + country match is immediately
   emitted as a match without touching LightGBM at all.

2. HARD CANDIDATE CAP + SMART PRE-FILTER
   After exact-match extraction, surviving candidates are capped at
   MAX_CANDS_PER_S1 (default 300). Selection priority:
     a) Multi-strategy candidates first (mask popcount > 1)
     b) Within each tier: name token overlap descending
   This transforms worst-case O(5750 x 69 features) -> O(300 x 69).

3. TWO-STAGE LIGHTGBM PIPELINE
   Stage 1: Score 12 cheap features (exact flags + token sets, zero fuzzy
            string calls) via a weighted dot-product heuristic. Prune to
            top-STAGE1_TOP_K candidates per S1 entity.
   Stage 2: Compute full 69 features on Stage-1 survivors only.
            Use the calibrated OPTIMAL threshold (default 0.88).

Expected throughput improvement: 8-15x over sharded_inference.py
Target: >= 120 S1/s -> full 1.73M dataset in < 4 hours.
Memory target: < 1.5 GB RSS (single process, safe on 16 GB DDR5).
"""

import sys
import os
import gc
import time
import mmap
import zlib
import re
import unicodedata
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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

import numpy as np
import pandas as pd
import joblib
import psutil
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

# Paths
DATASET_TEST_DIR = PROJECT_ROOT / "dataset" / "test"
S1_TEST_FILE = DATASET_TEST_DIR / "test_source1.tsv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TEST_CACHE_DIR = OUTPUTS_DIR / "test_cache"
DEFAULT_MODEL_PATH = OUTPUTS_DIR / "trained_ranker.pkl"
DEFAULT_SUBMISSION_PATH = OUTPUTS_DIR / "final_submission.tsv"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Tunable constants
MAX_CANDS_PER_S1 = 300
STAGE1_TOP_K = 50
STAGE1_THRESHOLD = 0.10
DEFAULT_BATCH_SIZE = 500

# Stage-1 feature weights (dot-product heuristic, no model call)
S1_WEIGHTS = np.array([
    4.0,   # 0  exact_clean
    3.0,   # 1  exact_no_suf
    3.0,   # 2  exact_spaceless
    2.0,   # 3  name_jaccard
    2.0,   # 4  name_overlap_ratio
    0.5,   # 5  token_inter
    1.5,   # 6  addr_exact
    0.8,   # 7  hn_overlap
    1.0,   # 8  hn_exact
    0.5,   # 9  country_exact
    0.3,   # 10 popcount
    0.0,   # 11 log_cand_count
], dtype=np.float32)
S1_MAX_SCORE = float(S1_WEIGHTS[:10].sum())

# Pre-compiled regex
RE_PREFIX = re.compile(
    r"^(?:formerly\s*:\s*|formerly\s+|dba\s*:\s*|dba\s+|doing\s+business\s+as\s+"
    r"|c\s*/\s*o\s+|mr\s+|mrs\s+|ms\s+|dr\s+)",
    re.IGNORECASE,
)
RE_PREFIX_COLON = re.compile(r"(?:formerly|dba|doing business as)\s*:\s*(.*)", re.IGNORECASE)
RE_DOMAIN = re.compile(DOMAIN_SUFFIX_PATTERN)
RE_LEAD_NOISE = re.compile(r"^[@\*\.\s]+")
RE_WWW = re.compile(r"^www\.")
RE_NON_WORD = re.compile(r"[^\w\s]")
RE_LEADING_ZEROS = re.compile(r"\b0+(\d+)\b")
RE_SPACES = re.compile(r"\s+")
RE_ADDR_NUMS = re.compile(r"\b\d+\b")


def hash64(s_bytes: bytes) -> int:
    return (zlib.crc32(s_bytes) << 32) | zlib.adler32(s_bytes)


def fast_norm_text(value: str) -> str:
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
    text = RE_PREFIX.sub("", text).strip()
    m = RE_PREFIX_COLON.search(text)
    if m:
        return m.group(1).strip()
    return text


def fast_norm_addr(value: str) -> str:
    text = fast_norm_text(value)
    if not text:
        return ""
    tokens = [ADDRESS_REPLACEMENTS.get(t, t) for t in text.split()]
    return " ".join(tokens)


class Source1Entity:
    __slots__ = (
        "eid", "country", "raw_name", "raw_addr",
        "name_clean", "name_no_suf", "spaceless_name", "name_translit",
        "name_tokens", "name_len",
        "addr_norm", "addr_translit", "addr_nums", "street_tokens",
        "addr_tokens", "addr_len", "postal", "city", "state",
        "has_name", "has_addr", "is_unicode_non_ascii",
        "core_tokens", "dist_tokens",
    )

    def __init__(self, eid: str, country: str, raw_name: str, raw_addr: str):
        self.eid = eid
        self.country = country.strip() if country else ""
        self.raw_name = raw_name.strip() if raw_name else ""
        self.raw_addr = raw_addr.strip() if raw_addr else ""
        raw_name_ascii = not self.raw_name or self.raw_name.isascii()
        raw_addr_ascii = not self.raw_addr or self.raw_addr.isascii()
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
            self.name_translit = (
                self.name_clean if raw_name_ascii
                else fast_strip_prefix(fast_norm_text(anyascii.anyascii(self.raw_name)))
            )
            self.core_tokens = [t for t in _name_toks if len(t) >= 3 and t not in LEGAL_SUFFIXES]
            self.dist_tokens = [
                t for t in _name_toks
                if len(t) >= 3 and t not in LEGAL_SUFFIXES and t not in GENERIC_WORDS
            ]
        else:
            self.name_clean = self.name_no_suf = self.spaceless_name = self.name_translit = ""
            self.name_tokens = set()
            self.name_len = 0
            self.has_name = False
            self.core_tokens = []
            self.dist_tokens = []
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
            self.addr_translit = (
                self.addr_norm if raw_addr_ascii
                else fast_norm_addr(anyascii.anyascii(self.raw_addr))
            )
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


class CandidateEntity:
    __slots__ = (
        "eid", "country", "raw_name", "name_clean", "name_no_suf", "spaceless_name",
        "name_translit", "name_tokens", "name_len", "addr_norm", "addr_translit",
        "addr_tokens", "addr_nums", "street_tokens", "addr_len", "postal", "city",
        "state", "has_name", "has_addr", "is_unicode_non_ascii",
    )

    def __init__(self, eid: str, rec: List[str]):
        self.eid = eid
        self.country = rec[0]
        self.raw_name = rec[1]
        self.name_clean = rec[2]
        self.name_no_suf = rec[3]
        self.spaceless_name = rec[4]
        self.name_translit = rec[5]
        self.addr_norm = rec[6]
        self.addr_translit = rec[7]
        self.postal = rec[8]
        self.city = rec[9]
        self.state = rec[10]
        self.is_unicode_non_ascii = rec[11] == "1"
        if self.name_clean:
            self.name_tokens = set(self.name_clean.split())
            self.name_len = len(self.name_clean)
            self.has_name = True
        else:
            self.name_tokens = set()
            self.name_len = 0
            self.has_name = False
        if self.addr_norm:
            _addr_toks = self.addr_norm.split()
            self.addr_tokens = set(_addr_toks)
            self.addr_nums = set(RE_ADDR_NUMS.findall(self.addr_norm))
            self.street_tokens = {
                t for t in _addr_toks
                if len(t) >= 3 and t not in STREET_STOPWORDS and not t.isdigit()
            }
            self.addr_len = len(self.addr_norm)
            self.has_addr = True
        else:
            self.addr_tokens = set()
            self.addr_nums = set()
            self.street_tokens = set()
            self.addr_len = 0
            self.has_addr = False


class MmapCandidateStore:
    def __init__(self, cache_dir: Path = TEST_CACHE_DIR):
        files = {
            "s2_offsets":    cache_dir / "s2_offsets.npy",
            "s2_records":    cache_dir / "s2_records.bin",
            "s3_offsets":    cache_dir / "s3_offsets.npy",
            "s3_records":    cache_dir / "s3_records.bin",
            "s2_sorted_ids": cache_dir / "s2_sorted_ids.npy",
            "s2_orig_rows":  cache_dir / "s2_orig_rows.npy",
            "s3_sorted_ids": cache_dir / "s3_sorted_ids.npy",
            "s3_orig_rows":  cache_dir / "s3_orig_rows.npy",
        }
        for name, f in files.items():
            if not f.exists():
                raise FileNotFoundError(f"Missing: {f}\nRun precompute_mmap_candidates first.")
        self.s2_offsets    = np.load(files["s2_offsets"],    mmap_mode="r")
        self.s3_offsets    = np.load(files["s3_offsets"],    mmap_mode="r")
        self.s2_sorted_ids = np.load(files["s2_sorted_ids"], mmap_mode="r")
        self.s2_orig_rows  = np.load(files["s2_orig_rows"],  mmap_mode="r")
        self.s3_sorted_ids = np.load(files["s3_sorted_ids"], mmap_mode="r")
        self.s3_orig_rows  = np.load(files["s3_orig_rows"],  mmap_mode="r")
        self.f_s2    = open(files["s2_records"], "rb")
        self.f_s3    = open(files["s3_records"], "rb")
        self.s2_mmap = mmap.mmap(self.f_s2.fileno(), 0, access=mmap.ACCESS_READ)
        self.s3_mmap = mmap.mmap(self.f_s3.fileno(), 0, access=mmap.ACCESS_READ)

    def unpack_candidates_batch(self, missing_vals: List[int], cache_dict: Dict[int, "CandidateEntity"]) -> None:
        if not missing_vals:
            return
        s2_vals: List[int] = []
        s3_vals: List[int] = []
        for v in missing_vals:
            (s3_vals if v & 0x80000000 else s2_vals).append(v)
        if s2_vals:
            arr  = np.array(s2_vals, dtype=np.uint32)
            pos  = np.searchsorted(self.s2_sorted_ids, arr)
            orig = self.s2_orig_rows[pos]
            sos  = self.s2_offsets[orig]
            eos  = self.s2_offsets[orig + 1]
            m    = self.s2_mmap
            for val, so, eo in zip(s2_vals, sos, eos):
                cache_dict[val] = CandidateEntity(f"S2-{val}", m[so:eo].decode("utf-8").split("\x00"))
        if s3_vals:
            ids  = np.array([v & 0x7FFFFFFF for v in s3_vals], dtype=np.uint32)
            pos  = np.searchsorted(self.s3_sorted_ids, ids)
            orig = self.s3_orig_rows[pos]
            sos  = self.s3_offsets[orig]
            eos  = self.s3_offsets[orig + 1]
            m    = self.s3_mmap
            for val, so, eo in zip(s3_vals, sos, eos):
                cache_dict[val] = CandidateEntity(f"S3-{val & 0x7FFFFFFF}", m[so:eo].decode("utf-8").split("\x00"))

    def close(self) -> None:
        try:
            self.s2_mmap.close(); self.s3_mmap.close()
            self.f_s2.close();    self.f_s3.close()
        except Exception:
            pass


def load_mmap_inverted_indexes(cache_dir: Path = TEST_CACHE_DIR):
    keys_mmap, offsets_mmap, counts_mmap, postings_mmap = [], [], [], []
    for s in range(8):
        keys_mmap.append(np.load(cache_dir / f"strat_{s}_keys.npy",      mmap_mode="r"))
        offsets_mmap.append(np.load(cache_dir / f"strat_{s}_offsets.npy",  mmap_mode="r"))
        counts_mmap.append(np.load(cache_dir / f"strat_{s}_counts.npy",    mmap_mode="r"))
        postings_mmap.append(np.load(cache_dir / f"strat_{s}_postings.npy", mmap_mode="r"))
    return keys_mmap, offsets_mmap, counts_mmap, postings_mmap


def query_candidates_for_batch(
    s1_entities: List[Source1Entity],
    keys_mmap, offsets_mmap, counts_mmap, postings_mmap,
) -> Tuple[List[Tuple[str, int, int, int]], Set[int]]:
    n_s1 = len(s1_entities)
    cand_masks_by_s1 = [defaultdict(int) for _ in range(n_s1)]
    for s_idx in range(8):
        k_arr   = keys_mmap[s_idx]
        o_arr   = offsets_mmap[s_idx]
        c_arr   = counts_mmap[s_idx]
        p_arr   = postings_mmap[s_idx]
        k_len   = len(k_arr)
        bit_val = 1 << s_idx
        query_hashes: List[int] = []
        query_s1_idx: List[int] = []
        for i, s1 in enumerate(s1_entities):
            c = s1.country
            nc = s1.name_clean
            nns = s1.name_no_suf
            sp = s1.spaceless_name
            an = s1.addr_norm
            anum = s1.addr_nums
            st = s1.street_tokens
            ct = s1.core_tokens
            dt = s1.dist_tokens
            if s_idx == 0:
                if nc:
                    query_hashes.append(hash64(f"{c}\t{nc}".encode()))
                    query_s1_idx.append(i)
            elif s_idx == 1:
                if nns and nns != nc:
                    query_hashes.append(hash64(f"{c}\t{nns}".encode()))
                    query_s1_idx.append(i)
            elif s_idx == 2:
                if sp and len(sp) >= 5:
                    query_hashes.append(hash64(f"{c}\t{sp}".encode()))
                    query_s1_idx.append(i)
            elif s_idx == 3:
                if an and len(an) >= 5:
                    query_hashes.append(hash64(f"{c}\t{an}".encode()))
                    query_s1_idx.append(i)
            elif s_idx == 4:
                if dt and anum:
                    for num in anum:
                        for tok in dt:
                            query_hashes.append(hash64(f"{c}\t{num}\t{tok}".encode()))
                            query_s1_idx.append(i)
            elif s_idx == 5:
                if st and anum:
                    for num in anum:
                        for stok in st:
                            query_hashes.append(hash64(f"{c}\t{num}\t{stok}".encode()))
                            query_s1_idx.append(i)
            elif s_idx == 6:
                if ct and len(ct) >= 2:
                    query_hashes.append(hash64(f"{c}\t{ct[0]}\t{ct[1]}".encode()))
                    query_s1_idx.append(i)
            elif s_idx == 7:
                if dt and len(dt[0]) >= 5:
                    query_hashes.append(hash64(f"{c}\t{dt[0]}".encode()))
                    query_s1_idx.append(i)
        if not query_hashes:
            continue
        q_hashes = np.array(query_hashes, dtype=np.uint64)
        q_s1     = np.array(query_s1_idx, dtype=np.int32)
        idx_arr  = np.searchsorted(k_arr, q_hashes)
        valid    = (idx_arr < k_len) & (k_arr[idx_arr] == q_hashes)
        if not np.any(valid):
            continue
        matched_idx = idx_arr[valid]
        matched_s1  = q_s1[valid]
        offs = o_arr[matched_idx]
        cnts = c_arr[matched_idx]
        for s1_i, off, cnt in zip(matched_s1, offs, cnts):
            m_dict = cand_masks_by_s1[s1_i]
            # Cap postings per key hit to limit Python-loop work for common names.
            # Entities with unique names are unaffected (cnt is small).
            # Common names ("walmart US") may have thousands of postings — we
            # take the first MAX_POSTINGS_PER_KEY which are the most canonical.
            scan_end = off + min(int(cnt), 75)
            for p in p_arr[off: scan_end]:
                m_dict[int(p)] |= bit_val
    pairs: List[Tuple[str, int, int, int]] = []
    unique_vals: Set[int] = set()
    for i, s1 in enumerate(s1_entities):
        cand_masks = cand_masks_by_s1[i]
        cand_count = len(cand_masks)
        for val, mask in cand_masks.items():
            pairs.append((s1.eid, val, mask, cand_count))
            unique_vals.add(val)
    return pairs, unique_vals


# Fast 8-bit popcount lookup table (built once at import time)
_PC8 = bytes(bin(i).count('1') for i in range(256))
_PC8_NP = np.frombuffer(_PC8, dtype=np.uint8)  # numpy version for vectorised indexing

def _popcount(x: int) -> int:
    """Fast popcount using 8-bit lookup table."""
    n = 0
    while x:
        n += _PC8[x & 0xFF]
        x >>= 8
    return n


def smart_cap_candidates(
    pairs: List[Tuple[str, int, int, int]],
    s1_map: Dict[str, Source1Entity],
    cand_cache: Dict[int, CandidateEntity],
    max_cands: int = MAX_CANDS_PER_S1,
) -> Tuple[List[Tuple[str, int, int, int]], Dict[str, List[str]]]:
    """
    For each S1 entity:
      1. Exact-match fast-path: name_clean + country + (addr_num OR street_tok overlap)
         -> emit immediately, skip LightGBM entirely.
      2. Remaining candidates: cap at max_cands using NumPy argsort on
         priority = popcount * 32 + token_overlap  (no Python sort loop).
    """
    by_s1: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
    for s1_id, val, mask, c_cnt in pairs:
        by_s1[s1_id].append((val, mask, c_cnt))

    scored_pairs: List[Tuple[str, int, int, int]] = []
    exact_by_s1: Dict[str, List[str]] = {}

    for s1_id, cand_list in by_s1.items():
        s1         = s1_map[s1_id]
        s1_nc      = s1.name_clean
        s1_country = s1.country
        s1_toks    = s1.name_tokens
        s1_anum    = s1.addr_nums
        s1_stok    = s1.street_tokens
        s1_has_addr = s1.has_addr

        exact_matches: List[str] = []
        rem_vals:  List[int] = []
        rem_masks: List[int] = []

        for val, mask, _c_cnt in cand_list:
            cand = cand_cache.get(val)
            if cand is None:
                rem_vals.append(val)
                rem_masks.append(mask)
                continue

            # Exact-match fast-path: name_clean + country match PLUS
            # at least one grounding signal (address number OR street token overlap).
            # This prevents over-matching on generic names like "walmart" in "US".
            if (s1_nc and cand.name_clean and s1_nc == cand.name_clean
                    and s1_country == cand.country):
                # Require at least one address anchor if both entities have addresses
                if not s1_has_addr or not cand.has_addr:
                    # One side has no address -> accept on name match alone
                    exact_matches.append(cand.eid)
                elif (s1_anum and cand.addr_nums and bool(s1_anum & cand.addr_nums)):
                    exact_matches.append(cand.eid)
                elif (s1_stok and cand.street_tokens and bool(s1_stok & cand.street_tokens)):
                    exact_matches.append(cand.eid)
                else:
                    # Same name, same country, but no address overlap -> send to scoring
                    rem_vals.append(val)
                    rem_masks.append(mask)
            else:
                rem_vals.append(val)
                rem_masks.append(mask)

        if exact_matches:
            exact_by_s1[s1_id] = exact_matches

        n_rem = len(rem_vals)
        if n_rem == 0:
            continue

        if n_rem > max_cands:
            # Fast popcount-only priority — NO cand_cache lookups here.
            # Eliminates 920K dict.get() + set intersection calls per batch.
            # Popcount is a perfectly good proxy: multi-strategy candidates
            # (high popcount) are far more likely to be true matches.
            masks_arr = np.array(rem_masks, dtype=np.uint32)
            # Vectorised popcount via numpy byte lookup table (no Python loop)
            pri = (_PC8_NP[masks_arr & 0xFF].astype(np.int32)
                   + _PC8_NP[(masks_arr >> 8) & 0xFF].astype(np.int32))
            top_idx = np.argpartition(-pri, max_cands)[:max_cands]
            rem_vals  = [rem_vals[i]  for i in top_idx]
            rem_masks = [rem_masks[i] for i in top_idx]
            n_rem = max_cands

        for val, mask in zip(rem_vals, rem_masks):
            scored_pairs.append((s1_id, val, mask, n_rem))

    return scored_pairs, exact_by_s1


def build_stage1_features(
    pairs: List[Tuple[str, int, int, int]],
    s1_map: Dict[str, Source1Entity],
    cand_cache: Dict[int, CandidateEntity],
) -> np.ndarray:
    n = len(pairs)
    X = np.zeros((n, 12), dtype=np.float32)
    for i, (s1_id, val, mask, c_cnt) in enumerate(pairs):
        s1   = s1_map[s1_id]
        cand = cand_cache.get(val)
        if cand is None:
            continue
        s1_nc = s1.name_clean
        c_nc  = cand.name_clean
        ex_cl = 1.0 if s1_nc and s1_nc == c_nc else 0.0
        ex_ns = 1.0 if s1.name_no_suf and s1.name_no_suf == cand.name_no_suf else 0.0
        ex_sp = 1.0 if (s1.spaceless_name and len(s1.spaceless_name) >= 4
                        and s1.spaceless_name == cand.spaceless_name) else 0.0
        s1t   = s1.name_tokens
        ct    = cand.name_tokens
        s1t_n = len(s1t)
        ct_n  = len(ct)
        inter = len(s1t & ct) if s1t and ct else 0
        union = s1t_n + ct_n - inter
        jac   = inter / union if union > 0 else 0.0
        min_t = min(s1t_n, ct_n)
        ovlp  = inter / min_t if min_t > 0 else 0.0
        s1_ad = s1.addr_norm
        c_ad  = cand.addr_norm
        ad_ex = 1.0 if s1_ad and len(s1_ad) >= 4 and s1_ad == c_ad else 0.0
        hn_int = s1.addr_nums & cand.addr_nums
        hn_ov  = 1.0 if hn_int else 0.0
        hn_ex  = 1.0 if s1.addr_nums and s1.addr_nums == cand.addr_nums else 0.0
        co_ex  = 1.0 if s1.country and s1.country == cand.country else 0.0
        X[i] = (
            ex_cl, ex_ns, ex_sp,
            jac, ovlp, float(inter),
            ad_ex, hn_ov, hn_ex,
            co_ex, float(_popcount(mask)), float(np.log1p(c_cnt)),
        )
    return X


def extract_pair_features_fast(
    s1: Source1Entity,
    cand: CandidateEntity,
    strat_mask: int,
    cand_count: int,
    out_vector: np.ndarray,
    row_idx: int,
) -> None:
    s1_nc    = s1.name_clean
    cand_nc  = cand.name_clean
    s1_nns   = s1.name_no_suf
    cand_nns = cand.name_no_suf
    s1_sp    = s1.spaceless_name
    cand_sp  = cand.spaceless_name
    s1_tr    = s1.name_translit
    cand_tr  = cand.name_translit
    exact_clean     = 1.0 if s1_nc and s1_nc == cand_nc else 0.0
    exact_raw       = 1.0 if s1.raw_name and s1.raw_name == cand.raw_name else 0.0
    exact_no_suf    = 1.0 if s1_nns and s1_nns == cand_nns else 0.0
    exact_spaceless = 1.0 if s1_sp and len(s1_sp) >= 4 and s1_sp == cand_sp else 0.0
    exact_translit  = 1.0 if s1_tr and s1_tr == cand_tr else 0.0
    if exact_clean == 1.0:
        name_fuzz = name_partial = name_sort = name_set = name_jw = name_lev = 1.0
    elif s1_nc and cand_nc:
        name_fuzz    = fuzz.ratio(s1_nc, cand_nc) / 100.0
        name_partial = fuzz.partial_ratio(s1_nc, cand_nc) / 100.0
        name_sort    = fuzz.token_sort_ratio(s1_nc, cand_nc) / 100.0
        name_set     = fuzz.token_set_ratio(s1_nc, cand_nc) / 100.0
        name_jw      = distance.JaroWinkler.similarity(s1_nc, cand_nc)
        name_lev     = distance.Levenshtein.normalized_similarity(s1_nc, cand_nc)
    else:
        name_fuzz = name_partial = name_sort = name_set = name_jw = name_lev = 0.0
    if exact_no_suf == 1.0:
        no_suf_fuzz = no_suf_sort = 1.0
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
        translit_fuzz = translit_sort = 1.0
    elif s1_tr and cand_tr:
        translit_fuzz = fuzz.ratio(s1_tr, cand_tr) / 100.0
        translit_sort = fuzz.token_sort_ratio(s1_tr, cand_tr) / 100.0
    else:
        translit_fuzz = translit_sort = 0.0
    s1_nt_len   = len(s1.name_tokens)
    cand_nt_len = len(cand.name_tokens)
    t_inter     = len(s1.name_tokens & cand.name_tokens)
    t_union     = s1_nt_len + cand_nt_len - t_inter
    name_jaccard       = t_inter / t_union if t_union > 0 else 0.0
    min_t              = min(s1_nt_len, cand_nt_len)
    name_overlap_ratio = t_inter / min_t if min_t > 0 else 0.0
    name_containment   = 1.0 if (s1_nc and cand_nc and (s1_nc in cand_nc or cand_nc in s1_nc)) else 0.0
    spaceless_cont     = 1.0 if (s1_sp and cand_sp and len(s1_sp) >= 4 and len(cand_sp) >= 4
                                 and (s1_sp in cand_sp or cand_sp in s1_sp)) else 0.0
    p3 = 1.0 if (s1_nc and cand_nc and s1_nc[:3] == cand_nc[:3]) else 0.0
    p5 = 1.0 if (s1_nc and cand_nc and len(s1_nc) >= 5 and len(cand_nc) >= 5
                 and s1_nc[:5] == cand_nc[:5]) else 0.0
    max_nl        = max(s1.name_len, cand.name_len, 1)
    name_len_diff = abs(s1.name_len - cand.name_len) / max_nl
    s1_ad    = s1.addr_norm
    cand_ad  = cand.addr_norm
    s1_ad_tr = s1.addr_translit
    cad_tr   = cand.addr_translit
    addr_exact = 1.0 if s1_ad and len(s1_ad) >= 4 and s1_ad == cand_ad else 0.0
    if addr_exact == 1.0:
        addr_fuzz = addr_partial = addr_sort = addr_set = addr_jw = addr_lev = addr_tr_fuzz = 1.0
    elif s1_ad and cand_ad:
        addr_fuzz    = fuzz.ratio(s1_ad, cand_ad) / 100.0
        addr_partial = fuzz.partial_ratio(s1_ad, cand_ad) / 100.0
        addr_sort    = fuzz.token_sort_ratio(s1_ad, cand_ad) / 100.0
        addr_set     = fuzz.token_set_ratio(s1_ad, cand_ad) / 100.0
        addr_jw      = distance.JaroWinkler.similarity(s1_ad, cand_ad)
        addr_lev     = distance.Levenshtein.normalized_similarity(s1_ad, cand_ad)
        addr_tr_fuzz = fuzz.ratio(s1_ad_tr, cad_tr) / 100.0 if s1_ad_tr and cad_tr else 0.0
    else:
        addr_fuzz = addr_partial = addr_sort = addr_set = addr_jw = addr_lev = addr_tr_fuzz = 0.0
    s1_at_len   = len(s1.addr_tokens)
    cand_at_len = len(cand.addr_tokens)
    ad_inter    = len(s1.addr_tokens & cand.addr_tokens)
    ad_union    = s1_at_len + cand_at_len - ad_inter
    addr_jaccard       = ad_inter / ad_union if ad_union > 0 else 0.0
    min_ad             = min(s1_at_len, cand_at_len)
    addr_overlap_ratio = ad_inter / min_ad if min_ad > 0 else 0.0
    s1_an_len   = len(s1.addr_nums)
    cand_an_len = len(cand.addr_nums)
    num_inter   = len(s1.addr_nums & cand.addr_nums)
    num_union   = s1_an_len + cand_an_len - num_inter
    hn_overlap  = 1.0 if num_inter > 0 else 0.0
    hn_jaccard  = num_inter / num_union if num_union > 0 else 0.0
    hn_exact    = 1.0 if s1.addr_nums and s1.addr_nums == cand.addr_nums else 0.0
    has_hn      = 1.0 if (s1_an_len > 0 and cand_an_len > 0) else 0.0
    s1_st_len   = len(s1.street_tokens)
    cand_st_len = len(cand.street_tokens)
    st_inter    = len(s1.street_tokens & cand.street_tokens)
    st_union    = s1_st_len + cand_st_len - st_inter
    st_jaccard  = st_inter / st_union if st_union > 0 else 0.0
    city_sim     = fuzz.ratio(s1.city, cand.city) / 100.0 if s1.city and cand.city else 0.0
    state_sim    = fuzz.ratio(s1.state, cand.state) / 100.0 if s1.state and cand.state else 0.0
    postal_match = 1.0 if s1.postal and cand.postal and s1.postal == cand.postal else 0.0
    max_al        = max(s1.addr_len, cand.addr_len, 1)
    addr_len_diff = abs(s1.addr_len - cand.addr_len) / max_al
    s_mask        = int(strat_mask)
    strat_en      = 1.0 if s_mask & 1   else 0.0
    strat_nns     = 1.0 if s_mask & 2   else 0.0
    strat_sp      = 1.0 if s_mask & 4   else 0.0
    strat_ea      = 1.0 if s_mask & 8   else 0.0
    strat_num_tok = 1.0 if s_mask & 16  else 0.0
    strat_num_st  = 1.0 if s_mask & 32  else 0.0
    strat_two     = 1.0 if s_mask & 64  else 0.0
    strat_first   = 1.0 if s_mask & 128 else 0.0
    strat_count   = float(strat_en + strat_nns + strat_sp + strat_ea +
                          strat_num_tok + strat_num_st + strat_two + strat_first)
    log_cands    = float(np.log1p(cand_count))
    cand_eid     = cand.eid
    is_s2        = 1.0 if cand_eid.startswith("S2") else 0.0
    is_s3        = 1.0 if cand_eid.startswith("S3") else 0.0
    country_exact = 1.0 if s1.country and s1.country == cand.country else 0.0
    is_us         = 1.0 if s1.country.upper() == "US" else 0.0
    is_in         = 1.0 if s1.country.lower() in ("india", "in") else 0.0
    missing_name  = 1.0 if (not s1.has_name or not cand.has_name) else 0.0
    missing_addr  = 1.0 if (not s1.has_addr or not cand.has_addr) else 0.0
    has_unicode   = 1.0 if (s1.is_unicode_non_ascii or cand.is_unicode_non_ascii) else 0.0
    mean_sim = 0.5 * (name_sort + addr_sort)
    mult_sim = name_sort * addr_sort
    max_sim  = max(name_fuzz, addr_fuzz)
    min_sim  = min(name_fuzz, addr_fuzz)
    out_vector[row_idx, :] = (
        exact_clean, exact_raw, exact_no_suf, exact_spaceless, exact_translit,
        name_fuzz, name_partial, name_sort, name_set, name_jw, name_lev,
        no_suf_fuzz, no_suf_sort, spaceless_fuzz, translit_fuzz, translit_sort,
        name_jaccard, name_overlap_ratio, float(t_inter), name_containment,
        spaceless_cont, p3, p5, name_len_diff,
        float(min(s1.name_len, cand.name_len)),
        float(max(s1.name_len, cand.name_len)),
        addr_exact, addr_fuzz, addr_partial, addr_sort, addr_set,
        addr_jw, addr_lev, addr_tr_fuzz, addr_jaccard, addr_overlap_ratio,
        hn_overlap, hn_jaccard, hn_exact, has_hn, st_jaccard,
        float(st_inter), city_sim, state_sim, postal_match, addr_len_diff,
        strat_en, strat_nns, strat_sp, strat_ea, strat_num_tok,
        strat_num_st, strat_two, strat_first, strat_count,
        float(cand_count), log_cands, is_s2, is_s3,
        country_exact, is_us, is_in, missing_name, missing_addr, has_unicode,
        mean_sim, mult_sim, max_sim, min_sim,
    )


def run_fast_inference(
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit_s1: Optional[int] = None,
    s1_file: Path = S1_TEST_FILE,
    cache_dir: Path = TEST_CACHE_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    output_file: Path = DEFAULT_SUBMISSION_PATH,
    max_cands_per_s1: int = MAX_CANDS_PER_S1,
    stage1_top_k: int = STAGE1_TOP_K,
    stage1_threshold: float = STAGE1_THRESHOLD,
) -> Dict[str, Any]:
    t_start = time.time()
    pid     = os.getpid()
    proc    = psutil.Process(pid)
    print("=" * 80, flush=True)
    print("FAST INFERENCE ENGINE  (Two-Stage + Exact-Match Fast-Path)", flush=True)
    print("=" * 80, flush=True)
    print(f"PID                        : {pid}", flush=True)
    print(f"Batch Size (S1)            : {batch_size:,}", flush=True)
    print(f"Max Candidates / S1        : {max_cands_per_s1}", flush=True)
    print(f"Stage-1 Top-K              : {stage1_top_k}", flush=True)
    print(f"Stage-1 Score Threshold    : {stage1_threshold}", flush=True)
    print(f"Source-1 File              : {s1_file}", flush=True)
    print(f"Output File                : {output_file}", flush=True)
    with open(s1_file, "r", encoding="utf-8") as fh:
        total_s1_ground = sum(1 for _ in fh) - 1
    total_s1_target = min(limit_s1, total_s1_ground) if limit_s1 else total_s1_ground
    print(f"Target S1 Entities         : {total_s1_target:,}", flush=True)
    print("=" * 80 + "\n", flush=True)
    print("Loading model...", flush=True)
    model_data        = joblib.load(model_path)
    model             = model_data["model"]
    optimal_threshold = float(model_data.get("optimal_threshold", 0.88))
    print(f"  Model loaded. Optimal threshold = {optimal_threshold}", flush=True)
    print("Loading mmap indexes and candidate store...", flush=True)
    keys_mmap, offsets_mmap, counts_mmap, postings_mmap = load_mmap_inverted_indexes(cache_dir)
    cand_store = MmapCandidateStore(cache_dir)
    print("  Loaded.\n", flush=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    f_out = open(output_file, "w", encoding="utf-8", newline="")
    f_out.write("source1_entity_id\tmatched_entity_ids\n")
    cand_cache: Dict[int, CandidateEntity] = {}
    HARD_RAM_LIMIT_MB = 3000.0   # Evict cache if RSS exceeds this (was 4500)
    total_s1 = 0
    total_raw = 0
    total_after_cap = 0
    total_after_s1_f = 0
    total_after_s2_f = 0
    total_exact = 0
    total_matches = 0
    peak_ram_mb = 0.0
    chunk_reader = pd.read_csv(
        s1_file, sep="\t", dtype=str,
        nrows=total_s1_target,
        chunksize=batch_size,
    )
    for b_idx, s1_df in enumerate(chunk_reader):
        s1_eids   = s1_df["entity_id"].tolist()
        countries = s1_df["country"].fillna("").tolist()
        names     = s1_df["business_name"].fillna("").tolist()
        addrs     = s1_df["business_address"].fillna("").tolist()
        s1_entities = [
            Source1Entity(eid, c, n, a)
            for eid, c, n, a in zip(s1_eids, countries, names, addrs)
        ]
        s1_map = {e.eid: e for e in s1_entities}
        raw_pairs, unique_vals = query_candidates_for_batch(
            s1_entities, keys_mmap, offsets_mmap, counts_mmap, postings_mmap
        )
        n_raw = len(raw_pairs)
        total_raw += n_raw
        missing_vals = [v for v in unique_vals if v not in cand_cache]
        cand_store.unpack_candidates_batch(missing_vals, cand_cache)
        scored_pairs, exact_by_s1 = smart_cap_candidates(
            raw_pairs, s1_map, cand_cache, max_cands=max_cands_per_s1
        )
        n_cap  = len(scored_pairs)
        ex_cnt = sum(len(v) for v in exact_by_s1.values())
        total_after_cap += n_cap
        total_exact     += ex_cnt
        s1_survivors: List[Tuple[str, int, int, int]] = []
        if scored_pairs:
            X1          = build_stage1_features(scored_pairs, s1_map, cand_cache)
            raw_scores  = X1 @ S1_WEIGHTS
            norm_scores = np.clip(raw_scores / max(S1_MAX_SCORE, 1e-6), 0.0, 1.0)
            pairs_by_s1: Dict[str, List[Tuple[float, int, int, int]]] = defaultdict(list)
            for idx, (s1_id, val, mask, c_cnt) in enumerate(scored_pairs):
                pairs_by_s1[s1_id].append((float(norm_scores[idx]), val, mask, c_cnt))
            for s1_id, scored in pairs_by_s1.items():
                above = [t for t in scored if t[0] >= stage1_threshold]
                if not above:
                    continue
                above.sort(key=lambda x: x[0], reverse=True)
                for _, val, mask, c_cnt in above[:stage1_top_k]:
                    s1_survivors.append((s1_id, val, mask, c_cnt))
        n_s1f = len(s1_survivors)
        total_after_s1_f += n_s1f
        s1_matched: Dict[str, List[str]] = {eid: [] for eid in s1_eids}
        for s1_id, eids in exact_by_s1.items():
            s1_matched[s1_id].extend(eids)
            total_matches += len(eids)
        if s1_survivors:
            PAIR_CHUNK = 10000
            for p_start in range(0, n_s1f, PAIR_CHUNK):
                chunk = s1_survivors[p_start: p_start + PAIR_CHUNK]
                n_c   = len(chunk)
                X2    = np.zeros((n_c, len(FEATURE_NAMES)), dtype=np.float32)
                for idx, (s1_id, val, mask, c_cnt) in enumerate(chunk):
                    extract_pair_features_fast(
                        s1_map[s1_id], cand_cache[val], mask, c_cnt, X2, idx
                    )
                probas = model.predict_proba(X2)[:, 1].astype(np.float32)
                for (s1_id, val, _, _), proba in zip(chunk, probas):
                    if proba >= optimal_threshold:
                        s1_matched[s1_id].append(cand_cache[val].eid)
                        total_matches += 1
                total_after_s2_f += n_c
                del X2, probas
        lines = [f"{eid}\t{','.join(s1_matched[eid])}\n" for eid in s1_eids]
        f_out.writelines(lines)
        f_out.flush()
        total_s1 += len(s1_eids)
        if len(cand_cache) > 8000:   # Evict to keep RAM low (was 40000)
            cand_cache.clear()
        del s1_entities, s1_map, raw_pairs, scored_pairs, s1_survivors, lines, s1_matched
        mem_rss = proc.memory_info().rss / (1024 ** 2)
        if mem_rss > peak_ram_mb:
            peak_ram_mb = mem_rss
        if mem_rss > HARD_RAM_LIMIT_MB:
            cand_cache.clear()
            gc.collect()
        elapsed    = time.time() - t_start
        s1_per_sec = total_s1 / elapsed if elapsed > 0 else 0.0
        pct        = total_s1 / total_s1_target * 100
        cap_r      = (1 - n_cap / n_raw) * 100 if n_raw > 0 else 0.0
        s1_r       = (1 - n_s1f / max(n_cap, 1)) * 100
        print(
            f"[Batch {b_idx+1:>4}] "
            f"S1:{total_s1:>7,}/{total_s1_target:>7,}({pct:>5.1f}%) "
            f"Raw:{n_raw:>6,}->Cap:{n_cap:>5,}({cap_r:>4.1f}%dn) "
            f"->S1:{n_s1f:>4,}({s1_r:>4.1f}%dn) "
            f"Exact:{ex_cnt:>3,} Mtch:{total_matches:>6,} "
            f"RAM:{mem_rss:>5.0f}MB Spd:{s1_per_sec:>6.1f}S1/s",
            flush=True,
        )
    f_out.close()
    cand_store.close()
    total_time = time.time() - t_start
    cap_red = (1 - total_after_cap / total_raw) * 100 if total_raw else 0.0
    s1_red  = (1 - total_after_s1_f / max(total_after_cap, 1)) * 100
    print("\n" + "=" * 80, flush=True)
    print("FAST INFERENCE SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"Total S1 Entities          : {total_s1:,}")
    print(f"Total Raw Pairs            : {total_raw:,}  ({total_raw/total_s1:.1f}/S1)")
    print(f"After Hard Cap             : {total_after_cap:,}  ({total_after_cap/total_s1:.1f}/S1)  [{cap_red:.1f}% reduction]")
    print(f"After Stage-1 Filter       : {total_after_s1_f:,}  ({total_after_s1_f/total_s1:.1f}/S1)  [{s1_red:.1f}% reduction]")
    print(f"After Stage-2 (LightGBM)   : {total_after_s2_f:,}  ({total_after_s2_f/total_s1:.1f}/S1)")
    print(f"Exact-Match Fast-Path      : {total_exact:,} matches (skipped LightGBM)")
    print(f"Total Matches Found        : {total_matches:,}  ({total_matches/total_s1:.4f}/S1)")
    print(f"Peak RAM                   : {peak_ram_mb:.1f} MB")
    print(f"Total Wall Time            : {total_time:.1f}s  ({total_time/60:.2f} min)")
    print(f"Throughput                 : {total_s1/total_time:,.1f} S1/s")
    if limit_s1 and limit_s1 < total_s1_ground:
        proj = total_s1_ground / (total_s1 / total_time)
        print(f"Projected Full (1.73M S1)  : {proj:.0f}s  ({proj/3600:.2f} hours)")
    print("=" * 80, flush=True)
    return {
        "total_s1": total_s1, "total_raw": total_raw,
        "total_after_cap": total_after_cap, "total_after_s1": total_after_s1_f,
        "total_matches": total_matches, "total_time": total_time,
        "s1_per_sec": total_s1 / total_time if total_time > 0 else 0.0,
        "peak_ram_mb": peak_ram_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="Fast Two-Stage Inference Engine")
    parser.add_argument("--batch_size",       type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit_s1",         type=int,   default=None)
    parser.add_argument("--max_cands",        type=int,   default=MAX_CANDS_PER_S1)
    parser.add_argument("--stage1_top_k",     type=int,   default=STAGE1_TOP_K)
    parser.add_argument("--stage1_threshold", type=float, default=STAGE1_THRESHOLD)
    parser.add_argument("--s1_file",    type=str, default=str(S1_TEST_FILE))
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output_file",type=str, default=str(DEFAULT_SUBMISSION_PATH))
    parser.add_argument("--cache_dir",  type=str, default=str(TEST_CACHE_DIR))
    parser.add_argument("--skip_validate", action="store_true")
    args = parser.parse_args()
    result = run_fast_inference(
        batch_size=args.batch_size, limit_s1=args.limit_s1,
        s1_file=Path(args.s1_file), cache_dir=Path(args.cache_dir),
        model_path=Path(args.model_path), output_file=Path(args.output_file),
        max_cands_per_s1=args.max_cands, stage1_top_k=args.stage1_top_k,
        stage1_threshold=args.stage1_threshold,
    )
    if not args.skip_validate:
        from business_entity_resolution.src.merge_and_validate import validate_submission_file
        validate_submission_file(
            submission_path=Path(args.output_file),
            s1_file=Path(args.s1_file),
            limit_s1=args.limit_s1,
        )


if __name__ == "__main__":
    main()
