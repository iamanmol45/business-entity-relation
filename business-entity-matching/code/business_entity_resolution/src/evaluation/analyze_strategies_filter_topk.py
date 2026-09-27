"""
Comprehensive Analysis Script:
1. Strategy-by-strategy recall & candidate volume breakdown (unique & incremental contribution)
2. Cheap first-pass candidate filter calibration
3. Top-K candidate ranking vs recall tradeoff
"""

import sys
import os
import re
import time
import unicodedata
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance

SRC_DIR = Path(__file__).resolve().parents[1]
PKG_DIR = SRC_DIR.parent
CODE_DIR = PKG_DIR.parent
PROJECT_ROOT = CODE_DIR.parent

TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
S1_TRAIN = TRAIN_DIR / "train_source1.tsv"
S2_TRAIN = TRAIN_DIR / "train_source2.tsv"
S3_TRAIN = TRAIN_DIR / "train_source3.tsv"
GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"

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
    r"^formerly\s*:\s*", r"^formerly\s+", r"^dba\s*:\s*", r"^dba\s+",
    r"^doing\s+business\s+as\s+", r"^c\s*/\s*o\s+", r"^mr\s+", r"^mrs\s+",
    r"^ms\s+", r"^dr\s+"
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

RE_ADDR_NUMS = re.compile(r"\b\d+\b")

def fast_norm_text(value: str) -> str:
    if not value or pd.isna(value):
        return ""
    val = str(value)
    if not val.isascii():
        val = unicodedata.normalize("NFKD", val.lower())
        val = "".join(c for c in val if not unicodedata.combining(c))
    else:
        val = val.lower()
    val = re.sub(r"^[@\*\.\s]+", "", val)
    val = re.sub(r"^www\.", "", val)
    val = re.sub(DOMAIN_SUFFIX_PATTERN, "", val)
    val = re.sub(r"[^\w\s]", " ", val)
    val = re.sub(r"\b0+(\d+)\b", r"\1", val)
    val = re.sub(r"\s+", " ", val)
    return val.strip()

def fast_strip_prefix(text: str) -> str:
    for pat in PREFIX_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE).strip()
    m = re.search(r"(?:formerly|dba|doing business as)\s*:\s*(.*)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text

def fast_norm_addr(value: str) -> str:
    text = fast_norm_text(value)
    if not text:
        return ""
    tokens = [ADDRESS_REPLACEMENTS.get(t, t) for t in text.split()]
    return " ".join(tokens)

class TrainEntity:
    __slots__ = (
        'eid', 'country', 'raw_name', 'raw_addr',
        'name_clean', 'name_no_suf', 'spaceless_name',
        'name_tokens', 'name_len',
        'addr_norm', 'addr_nums', 'street_tokens',
        'addr_tokens', 'addr_len',
        'core_tokens', 'dist_tokens'
    )
    def __init__(self, eid: str, country: str, raw_name: str, raw_addr: str):
        self.eid = eid
        self.country = country.strip() if country and pd.notna(country) else ""
        self.raw_name = raw_name.strip() if raw_name and pd.notna(raw_name) else ""
        self.raw_addr = raw_addr.strip() if raw_addr and pd.notna(raw_addr) else ""

        if self.raw_name:
            t = fast_norm_text(self.raw_name)
            self.name_clean = fast_strip_prefix(t)
            toks = self.name_clean.split()
            filtered = [x for x in toks if x not in LEGAL_SUFFIXES]
            self.name_no_suf = " ".join(filtered)
            self.spaceless_name = "".join(filtered)
            self.name_tokens = set(toks)
            self.name_len = len(self.name_clean)
            self.core_tokens = [x for x in toks if len(x) >= 3 and x not in LEGAL_SUFFIXES]
            self.dist_tokens = [x for x in toks if len(x) >= 3 and x not in LEGAL_SUFFIXES and x not in GENERIC_WORDS]
        else:
            self.name_clean = self.name_no_suf = self.spaceless_name = ""
            self.name_tokens = set()
            self.name_len = 0
            self.core_tokens = []
            self.dist_tokens = []

        if self.raw_addr:
            self.addr_norm = fast_norm_addr(self.raw_addr)
            a_toks = self.addr_norm.split()
            self.addr_tokens = set(a_toks)
            self.addr_nums = set(RE_ADDR_NUMS.findall(self.addr_norm))
            self.street_tokens = {
                t for t in a_toks
                if len(t) >= 3 and t not in STREET_STOPWORDS and not t.isdigit()
            }
            self.addr_len = len(self.addr_norm)
        else:
            self.addr_norm = ""
            self.addr_tokens = set()
            self.addr_nums = set()
            self.street_tokens = set()
            self.addr_len = 0

def run_analysis(sample_size: int = 10000):
    print("=" * 80)
    print(f"ANALYSIS OF STRATEGIES, CANDIDATE REDUCTION & TOP-K (N={sample_size:,})")
    print("=" * 80)
    t0 = time.time()

    # 1. Load S1 sample
    print(f"Loading {sample_size:,} train Source-1 entities...")
    df_s1 = pd.read_csv(S1_TRAIN, sep="\t", dtype=str, nrows=sample_size)
    s1_entities = [
        TrainEntity(eid, c, n, a)
        for eid, c, n, a in zip(df_s1["entity_id"], df_s1["country"], df_s1["business_name"], df_s1["business_address"])
    ]
    s1_ids_set = {e.eid for e in s1_entities}

    # 2. Load Ground Truth for this sample
    print("Loading Ground Truth for sample...")
    ground_truth = defaultdict(set)
    total_true_matches = 0
    with open(GT_FILE, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 2 and parts[0] in s1_ids_set:
                s1_id = parts[0]
                if parts[1].strip():
                    targets = set(parts[1].split(","))
                    ground_truth[s1_id] = targets
                    total_true_matches += len(targets)
    print(f"Total True Matches in {sample_size:,} sample: {total_true_matches:,}")

    # 3. Build In-Memory Test Indexes for S2 & S3
    print("Building blocking indexes over train S2 and S3...")
    t_idx0 = time.time()
    STRAT_NAMES = [
        "0_exact_name",
        "1_name_no_suf",
        "2_spaceless_name",
        "3_exact_addr",
        "4_num_dist_tok",
        "5_num_street_tok",
        "6_two_name_tok",
        "7_dist_first_tok"
    ]
    
    # We will build inverted indexes: strat -> dict of key -> list of eids
    indexes = [defaultdict(list) for _ in range(8)]
    cand_entity_store: Dict[str, TrainEntity] = {}

    for src_file in [S2_TRAIN, S3_TRAIN]:
        print(f"  Indexing {src_file.name}...")
        for chunk in pd.read_csv(src_file, sep="\t", dtype=str, chunksize=500_000):
            eids = chunk["entity_id"].tolist()
            countries = chunk["country"].tolist()
            names = chunk["business_name"].tolist()
            addrs = chunk["business_address"].tolist()

            for eid, c, n, a in zip(eids, countries, names, addrs):
                ent = TrainEntity(eid, c, n, a)
                cand_entity_store[eid] = ent
                country = ent.country
                nc = ent.name_clean
                nns = ent.name_no_suf
                sp = ent.spaceless_name
                ad = ent.addr_norm
                an = ent.addr_nums
                st = ent.street_tokens
                ct = ent.core_tokens
                dt = ent.dist_tokens

                if nc:
                    indexes[0][(country, nc)].append(eid)
                if nns and nns != nc:
                    indexes[1][(country, nns)].append(eid)
                if sp and len(sp) >= 5:
                    indexes[2][(country, sp)].append(eid)
                if ad and len(ad) >= 5:
                    indexes[3][(country, ad)].append(eid)
                if dt and an:
                    for num in an:
                        for tok in dt:
                            indexes[4][(country, num, tok)].append(eid)
                if st and an:
                    for num in an:
                        for s_tok in st:
                            indexes[5][(country, num, s_tok)].append(eid)
                if ct and len(ct) >= 2:
                    indexes[6][(country, ct[0], ct[1])].append(eid)
                if dt and len(dt[0]) >= 5:
                    indexes[7][(country, dt[0])].append(eid)
    print(f"Indexing completed in {time.time()-t_idx0:.1f}s.")

    # 4. Evaluate Strategy Contributions
    print("\nEvaluating individual and combined strategy recall & candidate counts...")
    strat_cands = [defaultdict(set) for _ in range(8)] # strat -> s1_id -> set of cids
    strat_recovered = [0] * 8
    strat_total_cands = [0] * 8

    for s1 in s1_entities:
        sid = s1.eid
        gt_set = ground_truth[sid]
        country = s1.country
        nc = s1.name_clean
        nns = s1.name_no_suf
        sp = s1.spaceless_name
        ad = s1.addr_norm
        an = s1.addr_nums
        st = s1.street_tokens
        ct = s1.core_tokens
        dt = s1.dist_tokens

        # S0
        if nc:
            c = set(indexes[0].get((country, nc), []))
            strat_cands[0][sid] = c
            strat_total_cands[0] += len(c)
            strat_recovered[0] += len(gt_set.intersection(c))

        # S1
        if nns and nns != nc:
            c = set(indexes[1].get((country, nns), []))
            strat_cands[1][sid] = c
            strat_total_cands[1] += len(c)
            strat_recovered[1] += len(gt_set.intersection(c))

        # S2
        if sp and len(sp) >= 5:
            c = set(indexes[2].get((country, sp), []))
            strat_cands[2][sid] = c
            strat_total_cands[2] += len(c)
            strat_recovered[2] += len(gt_set.intersection(c))

        # S3
        if ad and len(ad) >= 5:
            c = set(indexes[3].get((country, ad), []))
            strat_cands[3][sid] = c
            strat_total_cands[3] += len(c)
            strat_recovered[3] += len(gt_set.intersection(c))

        # S4
        if dt and an:
            c = set()
            for num in an:
                for tok in dt:
                    c.update(indexes[4].get((country, num, tok), []))
            strat_cands[4][sid] = c
            strat_total_cands[4] += len(c)
            strat_recovered[4] += len(gt_set.intersection(c))

        # S5
        if st and an:
            c = set()
            for num in an:
                for s_tok in st:
                    c.update(indexes[5].get((country, num, s_tok), []))
            strat_cands[5][sid] = c
            strat_total_cands[5] += len(c)
            strat_recovered[5] += len(gt_set.intersection(c))

        # S6
        if ct and len(ct) >= 2:
            c = set(indexes[6].get((country, ct[0], ct[1]), []))
            strat_cands[6][sid] = c
            strat_total_cands[6] += len(c)
            strat_recovered[6] += len(gt_set.intersection(c))

        # S7
        if dt and len(dt[0]) >= 5:
            c = set(indexes[7].get((country, dt[0]), []))
            strat_cands[7][sid] = c
            strat_total_cands[7] += len(c)
            strat_recovered[7] += len(gt_set.intersection(c))

    # Print Strategy Table
    print("\n" + "=" * 95)
    print(f"{'Strategy':<20} | {'Recall (%)':<10} | {'Recovered':<10} | {'Total Cands':<14} | {'Avg Cands/S1':<12}")
    print("-" * 95)
    for s_idx in range(8):
        rec = strat_recovered[s_idx] / total_true_matches * 100
        avg_c = strat_total_cands[s_idx] / sample_size
        print(f"{STRAT_NAMES[s_idx]:<20} | {rec:>8.2f} % | {strat_recovered[s_idx]:>10,} | {strat_total_cands[s_idx]:>14,} | {avg_c:>12.1f}")

    # Incremental / Cumulative Recall Calculation
    cum_cands_by_s1 = defaultdict(set)
    cum_recovered = 0
    print("\n--- INCREMENTAL RECALL CONTRIBUTION ---")
    print(f"{'Added Strategy':<20} | {'Cumulative Recall (%)':<22} | {'Incremental True Matches':<26} | {'Total Union Cands':<18}")
    print("-" * 95)
    for s_idx in range(8):
        new_matches = 0
        for s1 in s1_entities:
            sid = s1.eid
            gt_set = ground_truth[sid]
            old_set = cum_cands_by_s1[sid]
            curr_cands = strat_cands[s_idx][sid]
            
            # newly added
            newly_added_cands = curr_cands - old_set
            new_true = (gt_set.intersection(newly_added_cands))
            new_matches += len(new_true)
            old_set.update(curr_cands)
            
        cum_recovered += new_matches
        cum_rec = cum_recovered / total_true_matches * 100
        tot_c = sum(len(c) for c in cum_cands_by_s1.values())
        print(f"{STRAT_NAMES[s_idx]:<20} | {cum_rec:>19.2f} % | {new_matches:>24,} | {tot_c:>18,}")

    # 5. Evaluate Cheap First-Pass Similarity & Top-K Truncation
    print("\n" + "=" * 95)
    print("CHEAP FIRST-PASS SIMILARITY & TOP-K RANKING TRADEOFF ANALYSIS")
    print("=" * 95)
    
    # For every S1, rank candidates by cheap similarity:
    # Cheap score = 0.5 * token_overlap_ratio + 0.3 * (hn_overlap) + 0.2 * rapidfuzz_quick_ratio
    top_k_values = [20, 50, 100, 150, 200, 300, 500, 1000]
    top_k_recovered = {k: 0 for k in top_k_values}
    top_k_total_cands = {k: 0 for k in top_k_values}

    t_eval_topk = time.time()
    for s1 in s1_entities:
        sid = s1.eid
        gt_set = ground_truth[sid]
        all_cands = cum_cands_by_s1[sid]
        
        if not all_cands:
            continue
            
        # Compute cheap score for each candidate
        scored = []
        s1_nt = s1.name_tokens
        s1_an = s1.addr_nums
        s1_nc = s1.name_clean
        
        for cid in all_cands:
            c_obj = cand_entity_store[cid]
            c_nt = c_obj.name_tokens
            c_an = c_obj.addr_nums
            c_nc = c_obj.name_clean
            
            # 1. Exact name match gets max priority
            if s1_nc and s1_nc == c_nc:
                score = 10.0
            else:
                # 2. Token overlap
                inter = len(s1_nt.intersection(c_nt))
                min_tok = min(len(s1_nt), len(c_nt))
                tok_ov = (inter / min_tok) if min_tok > 0 else 0.0
                
                # 3. House number overlap
                hn_ov = 1.0 if (s1_an and c_an and (s1_an.intersection(c_an))) else 0.0
                
                # 4. Fast prefix / cheap similarity
                pref = 1.0 if (s1_nc and c_nc and s1_nc[:3] == c_nc[:3]) else 0.0
                
                score = 2.0 * tok_ov + 1.0 * hn_ov + 0.5 * pref
                
            scored.append((score, cid))
            
        # Sort descending by cheap score
        scored.sort(key=lambda x: x[0], reverse=True)
        ranked_cids = [x[1] for x in scored]
        
        for k in top_k_values:
            k_cands = set(ranked_cids[:k])
            top_k_total_cands[k] += len(k_cands)
            top_k_recovered[k] += len(gt_set.intersection(k_cands))

    print(f"{'Top-K':<8} | {'Recall (%)':<10} | {'Matches Recovered':<18} | {'Missed vs Total':<18} | {'Total Cands':<14} | {'Avg Cands/S1':<12} | {'Cand Reduction':<15}")
    print("-" * 110)
    full_union_cands = sum(len(c) for c in cum_cands_by_s1.values())
    for k in top_k_values:
        rec = top_k_recovered[k] / total_true_matches * 100
        missed = total_true_matches - top_k_recovered[k]
        tot_c = top_k_total_cands[k]
        avg_c = tot_c / sample_size
        cand_red = (1.0 - (tot_c / full_union_cands)) * 100
        print(f"K = {k:<4} | {rec:>8.2f} % | {top_k_recovered[k]:>17,} | {missed:>17,} | {tot_c:>14,} | {avg_c:>12.1f} | {cand_red:>13.1f} %")
    print("=" * 110)

if __name__ == "__main__":
    run_analysis()
