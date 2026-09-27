"""
Pairwise Feature Extractor for Business Entity Resolution.

Extracts rich, deterministic lexical, phonetic, syntactic, transliteration,
structural, address-component, strategy-origin, and metadata features
between Source 1 entities and Candidate entities.
"""

import re
import unicodedata
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import rapidfuzz
from rapidfuzz import fuzz, distance
import anyascii

# Legal Suffixes
LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "co", "company", "plc", "private", "pvt", "services", "service",
    "center", "centre", "associates", "consulting", "enterprises",
    "group", "holdings", "partners", "solutions", "international", "intl",
    "management", "consultants", "technologies", "tech"
}

GENERIC_WORDS = {
    "the", "and", "for", "with", "all", "one", "pro", "new", "top", "best",
    "first", "national", "global", "american", "general", "united",
    "inc", "llc", "ltd", "corp", "co", "services", "service", "center",
    "group", "solutions", "partners", "associates", "consulting", "holdings",
    "private", "pvt", "limited", "technologies", "enterprises", "management"
}

PREFIX_PATTERNS = [
    r"^formerly\s*:\s*",
    r"^formerly\s+",
    r"^dba\s*:\s*",
    r"^dba\s+",
    r"^doing\s+business\s+as\s+",
    r"^c\s*/\s*o\s+",
    r"^mr\s+",
    r"^mrs\s+",
    r"^ms\s+",
    r"^dr\s+",
]

DOMAIN_SUFFIX_PATTERN = r"\.(?:com|org|net|biz|co|in|us|info|io|me|ai|gov|edu|co\.in|org\.in|net\.in)\b"

ADDRESS_REPLACEMENTS = {
    "street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd",
    "drive": "dr", "lane": "ln", "highway": "hwy", "parkway": "pkwy",
    "place": "pl", "apartment": "apt", "suite": "ste", "court": "ct",
    "circle": "cir", "building": "bldg"
}

STREET_STOPWORDS = {
    "st", "rd", "ave", "blvd", "dr", "ln", "hwy", "pkwy", "pl", "apt", "ste",
    "unit", "floor", "suite", "road", "street", "avenue", "drive", "lane",
    "near", "opp", "opposite", "behind", "beside", "post", "box", "po", "pmb",
    "court", "ct", "cir", "circle", "bldg", "building", "null", "highway",
    "parkway", "hn", "plot", "house", "no", "shop", "vill", "block", "sector",
    "state", "route", "rt", "city"
}

ZIP_REGEX = re.compile(r"\b(\d{5}(?:-\d{4})?|\d{6})\b")


def normalize_text_base(value: Any) -> str:
    """Base unicode decomposition and cleaning."""
    if pd.isna(value) or not value:
        return ""
    value = unicodedata.normalize("NFKD", str(value).lower())
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = re.sub(r"^[@\*\.\s]+", "", value)
    value = re.sub(r"^www\.", "", value)
    value = re.sub(DOMAIN_SUFFIX_PATTERN, "", value)
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\b0+(\d+)\b", r"\1", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def strip_prefix_noise(text: str) -> str:
    for pattern in PREFIX_PATTERNS:
        text = re.sub(pattern, "", text).strip()
    m = re.search(r"(?:formerly|dba|doing business as)\s*:\s*(.*)", text)
    if m:
        return m.group(1).strip()
    return text


def normalize_name_clean(value: Any) -> str:
    text = normalize_text_base(value)
    return strip_prefix_noise(text)


def normalize_name_no_suffix(value: Any) -> str:
    text = normalize_name_clean(value)
    tokens = [t for t in text.split() if t not in LEGAL_SUFFIXES]
    return " ".join(tokens)


def normalize_spaceless_name(value: Any) -> str:
    text = normalize_name_clean(value)
    tokens = [t for t in text.split() if t not in LEGAL_SUFFIXES]
    spaceless = "".join(tokens)
    return spaceless


def normalize_transliteration(value: Any) -> str:
    if pd.isna(value) or not value:
        return ""
    ascii_val = anyascii.anyascii(str(value))
    return normalize_name_clean(ascii_val)


def normalize_address(value: Any) -> str:
    text = normalize_text_base(value)
    tokens = [ADDRESS_REPLACEMENTS.get(t, t) for t in text.split()]
    return " ".join(tokens)


def extract_address_numbers(addr_norm: str) -> set:
    return set(re.findall(r"\b\d+\b", addr_norm))


def extract_name_core_tokens(name_norm: str) -> List[str]:
    return [t for t in name_norm.split() if len(t) >= 3 and t not in LEGAL_SUFFIXES]


def extract_distinctive_name_tokens(name_clean: str) -> List[str]:
    return [
        t for t in name_clean.split()
        if len(t) >= 3 and t not in LEGAL_SUFFIXES and t not in GENERIC_WORDS
    ]


def extract_street_tokens(addr_norm: str) -> set:
    return {
        t for t in addr_norm.split()
        if len(t) >= 3 and t not in STREET_STOPWORDS and not t.isdigit()
    }


def extract_postal_code(raw_addr: str) -> str:
    if pd.isna(raw_addr) or not raw_addr:
        return ""
    m = ZIP_REGEX.search(str(raw_addr))
    return m.group(1) if m else ""


def extract_address_segments(raw_addr: str) -> Tuple[str, str]:
    """Extract rough city and state from comma-separated address."""
    if pd.isna(raw_addr) or not raw_addr:
        return "", ""
    parts = [p.strip().lower() for p in str(raw_addr).split(",") if p.strip()]
    if len(parts) >= 3:
        city = parts[-2]
        state = parts[-1]
        return city, state
    elif len(parts) == 2:
        return parts[0], parts[1]
    elif len(parts) == 1:
        return parts[0], ""
    return "", ""


class ProcessedEntity:
    """Precomputed entity profile for high-speed feature computation."""
    __slots__ = (
        'eid', 'country', 'raw_name', 'raw_addr',
        'name_clean', 'name_no_suf', 'spaceless_name', 'name_translit',
        'name_tokens', 'name_len',
        'addr_norm', 'addr_translit', 'addr_nums', 'street_tokens',
        'addr_tokens', 'addr_len', 'postal', 'city', 'state',
        'has_name', 'has_addr', 'is_unicode_non_ascii'
    )

    def __init__(self, eid: str, country: str, raw_name: str, raw_addr: str):
        self.eid = eid
        self.country = str(country).strip() if pd.notna(country) else ""
        self.raw_name = str(raw_name).strip() if pd.notna(raw_name) else ""
        self.raw_addr = str(raw_addr).strip() if pd.notna(raw_addr) else ""

        # Name fields
        self.name_clean = normalize_name_clean(self.raw_name)
        _name_tokens = [t for t in self.name_clean.split() if t not in LEGAL_SUFFIXES]
        self.name_no_suf = " ".join(_name_tokens)
        self.spaceless_name = "".join(_name_tokens)
        if not self.raw_name or self.raw_name.isascii():
            self.name_translit = self.name_clean
        else:
            self.name_translit = normalize_transliteration(self.raw_name)
        self.name_tokens = set(self.name_clean.split())
        self.name_len = len(self.name_clean)

        # Address fields
        self.addr_norm = normalize_address(self.raw_addr)
        if not self.raw_addr or self.raw_addr.isascii():
            self.addr_translit = self.addr_norm
        else:
            self.addr_translit = normalize_address(anyascii.anyascii(self.raw_addr))
        self.addr_nums = extract_address_numbers(self.addr_norm)
        self.street_tokens = extract_street_tokens(self.addr_norm)
        self.addr_tokens = set(self.addr_norm.split())
        self.addr_len = len(self.addr_norm)
        self.postal = extract_postal_code(self.raw_addr)
        self.city, self.state = extract_address_segments(self.raw_addr)

        # Flags
        self.has_name = bool(self.name_clean)
        self.has_addr = bool(self.addr_norm)
        self.is_unicode_non_ascii = not (self.raw_name + self.raw_addr).isascii()


FEATURE_NAMES = [
    # 1. Name Similarity Features
    "name_exact_clean",
    "name_exact_raw",
    "name_exact_no_suf",
    "name_exact_spaceless",
    "name_exact_translit",
    "name_fuzz_ratio",
    "name_fuzz_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_jaro_winkler",
    "name_levenshtein_sim",
    "name_no_suf_fuzz_ratio",
    "name_no_suf_token_sort",
    "name_spaceless_fuzz_ratio",
    "name_translit_fuzz_ratio",
    "name_translit_token_sort",
    "name_token_jaccard",
    "name_token_overlap_ratio",
    "name_token_overlap_count",
    "name_containment",
    "name_spaceless_containment",
    "name_prefix_3_match",
    "name_prefix_5_match",
    "name_len_diff_ratio",
    "name_len_min",
    "name_len_max",

    # 2. Address Similarity Features
    "addr_exact_match",
    "addr_fuzz_ratio",
    "addr_fuzz_partial_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_jaro_winkler",
    "addr_levenshtein_sim",
    "addr_translit_fuzz_ratio",
    "addr_token_jaccard",
    "addr_token_overlap_ratio",
    "addr_house_num_overlap",
    "addr_house_num_jaccard",
    "addr_house_num_exact",
    "addr_has_house_num",
    "addr_street_token_jaccard",
    "addr_street_token_overlap_count",
    "addr_city_fuzz_ratio",
    "addr_state_fuzz_ratio",
    "addr_postal_match",
    "addr_len_diff_ratio",

    # 3. Blocking Strategy Origin & Meta
    "strat_exact_name",
    "strat_name_no_suf",
    "strat_spaceless",
    "strat_exact_addr",
    "strat_num_any_tok",
    "strat_num_street",
    "strat_two_tok",
    "strat_distinctive_first",
    "strat_count_match",
    "s1_cand_count",
    "s1_log_cand_count",
    "cand_is_s2",
    "cand_is_s3",

    # 4. Context & Missingness Flags
    "country_exact_match",
    "country_is_us",
    "country_is_india",
    "missing_name_either",
    "missing_addr_either",
    "has_unicode_variation",

    # 5. Interaction & Composite Features
    "name_addr_mean_sim",
    "name_addr_mult_sim",
    "name_addr_max_sim",
    "name_addr_min_sim",
]


def extract_pair_feature_vector(
    s1: ProcessedEntity,
    cand: ProcessedEntity,
    strat_flags: Dict[str, bool],
    cand_count: int,
    out_vector: np.ndarray,
    row_idx: int
) -> None:
    """Extract full 63-feature array for a single candidate pair directly into out_vector."""
    
    # 1. Name Features
    s1_nc = s1.name_clean
    cand_nc = cand.name_clean
    s1_n_ns = s1.name_no_suf
    cand_n_ns = cand.name_no_suf
    s1_sp = s1.spaceless_name
    cand_sp = cand.spaceless_name
    s1_tr = s1.name_translit
    cand_tr = cand.name_translit

    exact_clean = 1.0 if s1_nc and s1_nc == cand_nc else 0.0
    exact_raw = 1.0 if s1.raw_name and s1.raw_name.lower() == cand.raw_name.lower() else 0.0
    exact_no_suf = 1.0 if s1_n_ns and s1_n_ns == cand_n_ns else 0.0
    exact_spaceless = 1.0 if s1_sp and len(s1_sp) >= 3 and s1_sp == cand_sp else 0.0
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
    elif s1_n_ns and cand_n_ns:
        no_suf_fuzz = fuzz.ratio(s1_n_ns, cand_n_ns) / 100.0
        no_suf_sort = fuzz.token_sort_ratio(s1_n_ns, cand_n_ns) / 100.0
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

    # Token overlap
    t_inter = len(s1.name_tokens.intersection(cand.name_tokens))
    t_union = len(s1.name_tokens.union(cand.name_tokens))
    name_jaccard = (t_inter / t_union) if t_union > 0 else 0.0
    min_t = min(len(s1.name_tokens), len(cand.name_tokens))
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

    # Address token overlap
    ad_inter = len(s1.addr_tokens.intersection(cand.addr_tokens))
    ad_union = len(s1.addr_tokens.union(cand.addr_tokens))
    addr_jaccard = (ad_inter / ad_union) if ad_union > 0 else 0.0
    min_ad = min(len(s1.addr_tokens), len(cand.addr_tokens))
    addr_overlap_ratio = (ad_inter / min_ad) if min_ad > 0 else 0.0

    # House numbers
    num_inter = len(s1.addr_nums.intersection(cand.addr_nums))
    num_union = len(s1.addr_nums.union(cand.addr_nums))
    hn_overlap = 1.0 if num_inter > 0 else 0.0
    hn_jaccard = (num_inter / num_union) if num_union > 0 else 0.0
    hn_exact = 1.0 if s1.addr_nums and s1.addr_nums == cand.addr_nums else 0.0
    has_hn = 1.0 if bool(s1.addr_nums and cand.addr_nums) else 0.0

    # Street tokens
    st_inter = len(s1.street_tokens.intersection(cand.street_tokens))
    st_union = len(s1.street_tokens.union(cand.street_tokens))
    st_jaccard = (st_inter / st_union) if st_union > 0 else 0.0

    # City & State & Postal
    city_sim = fuzz.ratio(s1.city, cand.city) / 100.0 if s1.city and cand.city else 0.0
    state_sim = fuzz.ratio(s1.state, cand.state) / 100.0 if s1.state and cand.state else 0.0
    postal_match = 1.0 if s1.postal and cand.postal and s1.postal == cand.postal else 0.0

    max_al = max(s1.addr_len, cand.addr_len, 1)
    addr_len_diff = abs(s1.addr_len - cand.addr_len) / max_al

    # 3. Strategy Features
    if isinstance(strat_flags, (int, np.integer)):
        s_mask = int(strat_flags)
        strat_en = 1.0 if (s_mask & 1) else 0.0
        strat_nns = 1.0 if (s_mask & 2) else 0.0
        strat_sp = 1.0 if (s_mask & 4) else 0.0
        strat_ea = 1.0 if (s_mask & 8) else 0.0
        strat_num_tok = 1.0 if (s_mask & 16) else 0.0
        strat_num_st = 1.0 if (s_mask & 32) else 0.0
        strat_two = 1.0 if (s_mask & 64) else 0.0
        strat_first = 1.0 if (s_mask & 128) else 0.0
    else:
        strat_en = 1.0 if strat_flags.get("exact_name", False) else 0.0
        strat_nns = 1.0 if strat_flags.get("name_no_suffix", False) else 0.0
        strat_sp = 1.0 if strat_flags.get("spaceless_name", False) else 0.0
        strat_ea = 1.0 if strat_flags.get("exact_address", False) else 0.0
        strat_num_tok = 1.0 if strat_flags.get("addr_num_any_name_tok", False) else 0.0
        strat_num_st = 1.0 if strat_flags.get("addr_num_street_tok", False) else 0.0
        strat_two = 1.0 if strat_flags.get("name_two_tok", False) else 0.0
        strat_first = 1.0 if strat_flags.get("distinctive_first_tok", False) else 0.0
    strat_count = float(strat_en + strat_nns + strat_sp + strat_ea + strat_num_tok + strat_num_st + strat_two + strat_first)

    log_cands = float(np.log1p(cand_count))
    is_s2 = 1.0 if cand.eid.startswith("S2") else 0.0
    is_s3 = 1.0 if cand.eid.startswith("S3") else 0.0

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

    # Direct row assignment in preallocated numpy matrix
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
