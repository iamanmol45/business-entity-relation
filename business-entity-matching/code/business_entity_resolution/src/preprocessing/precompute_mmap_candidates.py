"""
Precompute and store normalized candidate entities (S2 & S3) into compact memory-mapped binary stores.

Zero SQLite runtime overhead during inference:
- Reads test_source2.tsv and test_source3.tsv
- Pre-normalizes all names, addresses, transliterations, postal codes, and city/state
- Stores packed null-delimited records in binary files (.bin) with 64-bit offsets (.npy)
- Uses <1.5 GB RAM during precomputation and 0 MB RAM during inference via mmap
"""

import sys
import os
import gc
import time
import re
import unicodedata
from pathlib import Path
from typing import List, Tuple

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
import anyascii

from business_entity_resolution.src.features.feature_extractor import (
    DOMAIN_SUFFIX_PATTERN,
    ADDRESS_REPLACEMENTS,
    LEGAL_SUFFIXES,
    ZIP_REGEX
)

from business_entity_resolution.src.test_inference import (
    S2_TEST_FILE,
    S3_TEST_FILE,
    TEST_CACHE_DIR
)

# Precompiled Fast Regex Patterns
RE_PREFIX = re.compile(r"^(?:formerly\s*:\s*|formerly\s+|dba\s*:\s*|dba\s+|doing\s+business\s+as\s+|c\s*/\s*o\s+|mr\s+|mrs\s+|ms\s+|dr\s+)", re.IGNORECASE)
RE_PREFIX_COLON = re.compile(r"(?:formerly|dba|doing business as)\s*:\s*(.*)", re.IGNORECASE)
RE_DOMAIN = re.compile(DOMAIN_SUFFIX_PATTERN)
RE_LEAD_NOISE = re.compile(r"^[@\*\.\s]+")
RE_WWW = re.compile(r"^www\.")
RE_NON_WORD = re.compile(r"[^\w\s]")
RE_LEADING_ZEROS = re.compile(r"\b0+(\d+)\b")
RE_SPACES = re.compile(r"\s+")


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


def process_and_pack_entity(country: str, raw_name: str, raw_addr: str) -> bytes:
    """Computes all normalized fields and returns a packed UTF-8 byte record."""
    country = country.strip() if country else ""
    raw_name = raw_name.strip() if raw_name else ""
    raw_addr = raw_addr.strip() if raw_addr else ""

    raw_name_ascii = not raw_name or raw_name.isascii()
    raw_addr_ascii = not raw_addr or raw_addr.isascii()

    # 1. Name normalization
    if raw_name:
        name_text = fast_norm_text(raw_name)
        name_clean = fast_strip_prefix(name_text)
        _name_toks = name_clean.split()
        _filtered_toks = [t for t in _name_toks if t not in LEGAL_SUFFIXES]
        name_no_suf = " ".join(_filtered_toks)
        spaceless_name = "".join(_filtered_toks)
        if raw_name_ascii:
            name_translit = name_clean
        else:
            name_translit = fast_strip_prefix(fast_norm_text(anyascii.anyascii(raw_name)))
    else:
        name_clean = name_no_suf = spaceless_name = name_translit = ""

    # 2. Address normalization
    if raw_addr:
        addr_norm = fast_norm_addr(raw_addr)
        if raw_addr_ascii:
            addr_translit = addr_norm
        else:
            addr_translit = fast_norm_addr(anyascii.anyascii(raw_addr))

        m = ZIP_REGEX.search(raw_addr)
        postal = m.group(1) if m else ""

        parts = [p.strip().lower() for p in raw_addr.split(",") if p.strip()]
        if len(parts) >= 3:
            city, state = parts[-2], parts[-1]
        elif len(parts) == 2:
            city, state = parts[0], parts[1]
        elif len(parts) == 1:
            city, state = parts[0], ""
        else:
            city, state = "", ""
    else:
        addr_norm = addr_translit = postal = city = state = ""

    is_unicode_flag = "0" if (raw_name_ascii and raw_addr_ascii) else "1"

    # Packed format:
    # 0: country
    # 1: raw_name
    # 2: name_clean
    # 3: name_no_suf
    # 4: spaceless_name
    # 5: name_translit
    # 6: addr_norm
    # 7: addr_translit
    # 8: postal
    # 9: city
    # 10: state
    # 11: is_unicode
    record_str = f"{country}\x00{raw_name}\x00{name_clean}\x00{name_no_suf}\x00{spaceless_name}\x00{name_translit}\x00{addr_norm}\x00{addr_translit}\x00{postal}\x00{city}\x00{state}\x00{is_unicode_flag}"
    return record_str.encode("utf-8")


def precompute_source_file(tsv_file: Path, prefix: str, out_dir: Path = TEST_CACHE_DIR, chunk_size: int = 250_000):
    """Streams a Source TSV file, pre-normalizes entities in chunks, and writes binary mmap files."""
    offsets_file = out_dir / f"{prefix}_offsets.npy"
    bin_file = out_dir / f"{prefix}_records.bin"

    if offsets_file.exists() and bin_file.exists():
        size_mb = (offsets_file.stat().st_size + bin_file.stat().st_size) / (1024**2)
        print(f"[{prefix.upper()}] Precomputed mmap files already exist: {bin_file.name} ({size_mb:.1f} MB). Skipping.", flush=True)
        return

    print("=" * 80, flush=True)
    print(f"PRECOMPUTING CANDIDATE DATA FOR {prefix.upper()} ({tsv_file.name})", flush=True)
    print("=" * 80, flush=True)

    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    offsets: List[int] = [0]
    total_records = 0
    current_offset = 0

    with open(bin_file, "wb") as f_bin:
        for chunk in pd.read_csv(tsv_file, sep="\t", dtype=str, chunksize=chunk_size):
            t_c0 = time.time()
            countries = chunk["country"].fillna("").tolist()
            names = chunk["business_name"].fillna("").tolist()
            addrs = chunk["business_address"].fillna("").tolist()
            eids = chunk["entity_id"].tolist()

            batch_bytes = bytearray()
            for c, n, a in zip(countries, names, addrs):
                rec = process_and_pack_entity(c, n, a)
                batch_bytes.extend(rec)
                current_offset += len(rec)
                offsets.append(current_offset)

            f_bin.write(batch_bytes)
            total_records += len(eids)
            t_c1 = time.time()
            print(f"  Processed {total_records:,} entities... ({len(eids)/(t_c1-t_c0):,.0f} ent/s, offset={current_offset/(1024**2):.1f} MB)", flush=True)

    # Save 64-bit offsets array
    arr_offsets = np.array(offsets, dtype=np.uint64)
    np.save(offsets_file, arr_offsets)

    total_time = time.time() - t0
    total_mb = (bin_file.stat().st_size + offsets_file.stat().st_size) / (1024**2)
    print(f"[{prefix.upper()}] Precomputed {total_records:,} entities in {total_time:.1f}s ({total_records/total_time:,.0f} ent/s)")
    print(f"[{prefix.upper()}] Total Binary Size: {total_mb:.1f} MB (Records: {bin_file.stat().st_size/(1024**2):.1f} MB, Offsets: {offsets_file.stat().st_size/(1024**2):.1f} MB)")
    print("=" * 80 + "\n", flush=True)


def precompute_all():
    print("=" * 80, flush=True)
    print("PRECOMPUTING CANDIDATE STORES (S2 & S3) FOR 0-SQLITE PARALLEL INFERENCE", flush=True)
    print("=" * 80, flush=True)
    t0 = time.time()

    precompute_source_file(S2_TEST_FILE, prefix="s2", out_dir=TEST_CACHE_DIR)
    precompute_source_file(S3_TEST_FILE, prefix="s3", out_dir=TEST_CACHE_DIR)

    total_sec = time.time() - t0
    print(f"ALL CANDIDATE PRECOMPUTATION COMPLETE IN {total_sec:.1f}s ({total_sec/60:.2f} mins)!\n", flush=True)


if __name__ == "__main__":
    precompute_all()
