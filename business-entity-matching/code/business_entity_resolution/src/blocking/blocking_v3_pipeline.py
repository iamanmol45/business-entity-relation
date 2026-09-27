import sys
import os
import re
import csv
import time
import json
import string
import unicodedata
from collections import defaultdict, Counter
from pathlib import Path
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[4]
TRAIN_DIR = ROOT / "dataset" / "train"
OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

S1_FILE = TRAIN_DIR / "train_source1.tsv"
S2_FILE = TRAIN_DIR / "train_source2.tsv"
S3_FILE = TRAIN_DIR / "train_source3.tsv"
GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"

SAMPLE_SIZE = 10_000
MAX_POSTINGS_STD = 400
MAX_POSTINGS_RELAXED = 150
CANDIDATE_BUDGET = 280

LEGAL_SUFFIXES = {
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "co", "company", "plc", "private", "pvt", "services", "service",
    "center", "centre", "associates", "consulting", "enterprises",
    "group", "holdings", "partners", "solutions", "international", "intl",
    "management", "consultants", "technologies", "tech", "lp", "llp", "pa", "pc"
}

GENERIC_WORDS = {
    "the", "and", "for", "with", "all", "one", "pro", "new", "top", "best",
    "first", "national", "global", "american", "general", "united",
    "inc", "llc", "ltd", "corp", "co", "services", "service", "center",
    "group", "solutions", "partners", "associates", "consulting", "holdings",
    "private", "pvt", "limited", "technologies", "enterprises", "management",
    "club", "shop", "store", "india", "usa", "america", "city", "state", "center", "centre",
    "world", "star", "royal", "super", "prime", "direct", "express", "apex", "care", "health"
}

DOMAIN_SUFFIX_PATTERN = re.compile(r"\.(?:com|org|net|biz|co|in|us|info|io|me|ai|gov|edu|co\.in|org\.in|net\.in)\b", re.I)
TRANS_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))

ADDR_MAP = {
    "street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd",
    "drive": "dr", "lane": "ln", "highway": "hwy", "parkway": "pkwy",
    "place": "pl", "apartment": "apt", "suite": "ste", "court": "ct",
    "circle": "cir", "building": "bldg", "floor": "fl", "no": "no", "hn": "hn"
}

STREET_STOPWORDS = {
    "st", "rd", "ave", "blvd", "dr", "ln", "hwy", "pkwy", "pl", "apt", "ste",
    "unit", "floor", "suite", "road", "street", "avenue", "drive", "lane",
    "near", "opp", "opposite", "behind", "beside", "post", "box", "po", "pmb",
    "court", "ct", "cir", "circle", "bldg", "building", "null", "highway",
    "parkway", "hn", "plot", "house", "no", "shop", "vill", "block", "sector",
    "state", "route", "rt", "city", "east", "west", "north", "south", "e", "w", "n", "s"
}

PREFIX_STOPWORDS = {"formerly", "dba", "mr", "mrs", "ms", "dr", "doing", "business", "as", "co"}


def normalize_and_extract_features(raw_name, raw_addr, raw_country):
    country = str(raw_country).strip().upper() if raw_country else ""
    
    # 1. Normalize Name
    n_str = str(raw_name or "")
    has_unicode = not n_str.isascii()
    n_low = n_str.lower()
    if has_unicode:
        n_low = "".join(c for c in unicodedata.normalize("NFKD", n_low) if not unicodedata.combining(c))
    if "." in n_low:
        n_low = DOMAIN_SUFFIX_PATTERN.sub("", n_low)
    n_clean = n_low.translate(TRANS_TABLE)
    n_toks = n_clean.split()
    
    # Strip prefix noise
    while n_toks and n_toks[0] in PREFIX_STOPWORDS:
        n_toks = n_toks[1:]
        
    clean_name = " ".join(n_toks)
    tokens_no_suf = [t for t in n_toks if t not in LEGAL_SUFFIXES]
    name_no_suf = " ".join(tokens_no_suf)
    spaceless_name = "".join(tokens_no_suf)
    
    dist_tokens = [t for t in tokens_no_suf if len(t) >= 3 and t not in GENERIC_WORDS and not t.isdigit()][:3]
    core_tokens = [t for t in tokens_no_suf if len(t) >= 3 and not t.isdigit()][:3]
    
    # 2. Normalize Address
    a_str = str(raw_addr or "")
    if not a_str.isascii():
        a_str = "".join(c for c in unicodedata.normalize("NFKD", a_str) if not unicodedata.combining(c))
    a_clean = a_str.lower().translate(TRANS_TABLE)
    a_toks = a_clean.split()
    
    clean_addr_toks = [ADDR_MAP.get(t, t) for t in a_toks]
    clean_addr = " ".join(clean_addr_toks)
    
    nums = [t.lstrip("0") for t in a_toks if t.isdigit() and len(t) <= 6][:3]
    addr_nums = set(nums)
    
    street_toks = [t for t in clean_addr_toks if len(t) >= 4 and t not in STREET_STOPWORDS and not t.isdigit()][:3]
    
    # Location/city tokens from end of address
    loc_tokens = []
    if len(clean_addr_toks) >= 2:
        for t in reversed(clean_addr_toks[-3:]):
            if len(t) >= 3 and t not in STREET_STOPWORDS and not t.isdigit() and t not in loc_tokens:
                loc_tokens.append(t)
    loc_tokens = loc_tokens[:2]
    
    # Check if URL / domain present
    is_url = bool(re.search(DOMAIN_SUFFIX_PATTERN, str(raw_name or "")) or "@" in str(raw_name or "") or "www." in str(raw_name or "").lower())
    is_missing_addr = not a_toks or len(a_toks) < 2
    
    return {
        "country": country,
        "raw_name": str(raw_name or ""),
        "raw_addr": str(raw_addr or ""),
        "clean_name": clean_name,
        "name_no_suf": name_no_suf,
        "spaceless_name": spaceless_name,
        "dist_tokens": dist_tokens,
        "core_tokens": core_tokens,
        "clean_addr": clean_addr,
        "addr_nums": addr_nums,
        "street_toks": street_toks,
        "loc_tokens": loc_tokens,
        "has_unicode": has_unicode,
        "is_url": is_url,
        "is_missing_addr": is_missing_addr,
    }


def build_v3_indexes(file_path):
    print(f"\nBuilding V3 Indexes for {file_path.name} (safe streaming)...", flush=True)
    t0 = time.time()
    
    indexes = {
        # Stage 1: High Precision Blocks
        "exact_name": defaultdict(list),
        "name_no_suffix": defaultdict(list),
        "exact_address": defaultdict(list),
        "name_and_addr": defaultdict(list),
        "hn_dist_tok": defaultdict(list),
        "hn_street_tok": defaultdict(list),
        "dist_tok_loc": defaultdict(list),
        "domain_loc": defaultdict(list),
        
        # Stage 2: Controlled Relaxation Blocks
        "spaceless_name": defaultdict(list),
        "name_two_tok": defaultdict(list),
        "country_num_name_tok": defaultdict(list),
        "distinctive_first_tok": defaultdict(list),
    }
    
    row_count = 0
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4:
                continue
            eid, raw_name, raw_addr, raw_country = parts[0], parts[1], parts[2], parts[3]
            if not raw_country:
                continue
            row_count += 1
            
            feat = normalize_and_extract_features(raw_name, raw_addr, raw_country)
            country = feat["country"]
            clean_name = feat["clean_name"]
            name_no_suf = feat["name_no_suf"]
            spaceless_name = feat["spaceless_name"]
            clean_addr = feat["clean_addr"]
            dist_tokens = feat["dist_tokens"]
            core_tokens = feat["core_tokens"]
            addr_nums = feat["addr_nums"]
            street_toks = feat["street_toks"]
            loc_tokens = feat["loc_tokens"]
            
            # STAGE 1: High Precision
            # 1. Exact Name
            if clean_name:
                k = (country, clean_name)
                lst = indexes["exact_name"][k]
                if len(lst) < MAX_POSTINGS_STD:
                    lst.append(eid)
                    
            # 2. Name No Suffix
            if name_no_suf and name_no_suf != clean_name:
                k = (country, name_no_suf)
                lst = indexes["name_no_suffix"][k]
                if len(lst) < MAX_POSTINGS_STD:
                    lst.append(eid)
                    
            # 3. Exact Address
            if clean_addr and len(clean_addr) >= 6:
                k = (country, clean_addr)
                lst = indexes["exact_address"][k]
                if len(lst) < MAX_POSTINGS_STD:
                    lst.append(eid)
                    
            # 4. Exact Name + Address
            if name_no_suf and clean_addr:
                k = (country, name_no_suf, clean_addr)
                lst = indexes["name_and_addr"][k]
                if len(lst) < MAX_POSTINGS_STD:
                    lst.append(eid)
                    
            # 5. House Number + Distinctive Token
            if dist_tokens and addr_nums:
                for num in addr_nums:
                    for d_tok in dist_tokens:
                        k = (country, num, d_tok)
                        lst = indexes["hn_dist_tok"][k]
                        if len(lst) < MAX_POSTINGS_STD:
                            lst.append(eid)
                            
            # 6. House Number + Street Token
            if street_toks and addr_nums:
                for num in addr_nums:
                    for s_tok in street_toks:
                        k = (country, num, s_tok)
                        lst = indexes["hn_street_tok"][k]
                        if len(lst) < MAX_POSTINGS_STD:
                            lst.append(eid)
                            
            # 7. Distinctive Token + Location / City Token
            if dist_tokens and loc_tokens:
                for d_tok in dist_tokens:
                    for l_tok in loc_tokens:
                        k = (country, d_tok, l_tok)
                        lst = indexes["dist_tok_loc"][k]
                        if len(lst) < MAX_POSTINGS_STD:
                            lst.append(eid)
                            
            # 8. Domain / Spaceless Name + Location / Street Token
            if spaceless_name and len(spaceless_name) >= 5 and street_toks:
                s_tok = street_toks[0]
                k = (country, spaceless_name, s_tok)
                lst = indexes["domain_loc"][k]
                if len(lst) < MAX_POSTINGS_STD:
                    lst.append(eid)
                    
            # STAGE 2: Controlled Relaxation
            # 9. Spaceless Name
            if spaceless_name and len(spaceless_name) >= 5:
                k = (country, spaceless_name)
                lst = indexes["spaceless_name"][k]
                if len(lst) < MAX_POSTINGS_STD:
                    lst.append(eid)
                    
            # 10. First Two Core Name Tokens
            if len(core_tokens) >= 2:
                k = (country, core_tokens[0], core_tokens[1])
                lst = indexes["name_two_tok"][k]
                if len(lst) < MAX_POSTINGS_STD:
                    lst.append(eid)
                    
            # 11. Country + House Number + Core Name Token
            if core_tokens and addr_nums:
                first_tok = core_tokens[0]
                for num in addr_nums:
                    k = (country, num, first_tok)
                    lst = indexes["country_num_name_tok"][k]
                    if len(lst) < MAX_POSTINGS_STD:
                        lst.append(eid)
                        
            # 12. Distinctive First Token (Rare keys only, capped postings)
            if dist_tokens and len(dist_tokens[0]) >= 6:
                d_first = dist_tokens[0]
                k = (country, d_first)
                lst = indexes["distinctive_first_tok"][k]
                if len(lst) < MAX_POSTINGS_RELAXED:
                    lst.append(eid)
                    
            if row_count % 1_000_000 == 0:
                print(f"  Indexed {row_count:,} rows in {time.time()-t0:.1f}s...", flush=True)
                
    print(f"Done indexing {file_path.name}: {row_count:,} rows in {time.time()-t0:.1f}s.", flush=True)
    return indexes


def load_ground_truth():
    gt = {}
    with open(GT_FILE, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 2:
                continue
            s1_id, m_ids = parts[0], parts[1]
            if not m_ids or not m_ids.strip():
                gt[s1_id] = set()
            else:
                gt[s1_id] = {x.strip() for x in m_ids.split(",") if x.strip()}
    return gt


def rank_and_budget_candidates(cand_strats, budget=CANDIDATE_BUDGET):
    """
    STAGE 4: Adaptive Candidate Budget & Deterministic Evidence Ranking.
    Ranks candidates by strategy strength, consensus, and deterministic tie-breaking.
    """
    if len(cand_strats) <= budget:
        return set(cand_strats.keys())
        
    scored = []
    for cid, strats in cand_strats.items():
        score = 0.0
        if "name_and_addr" in strats:
            score += 120.0
        if "exact_name" in strats:
            score += 100.0
        if "name_no_suffix" in strats:
            score += 95.0
        if "exact_address" in strats:
            score += 90.0
        if "hn_dist_tok" in strats:
            score += 75.0
        if "domain_loc" in strats:
            score += 70.0
        if "hn_street_tok" in strats:
            score += 60.0
        if "dist_tok_loc" in strats:
            score += 55.0
        if "spaceless_name" in strats:
            score += 45.0
        if "country_num_name_tok" in strats:
            score += 40.0
        if "name_two_tok" in strats:
            score += 35.0
        if "distinctive_first_tok" in strats:
            score += 25.0
            
        # Consensus bonus across multiple independent strategies
        if len(strats) > 1:
            score += 25.0 * (len(strats) - 1)
            
        scored.append((score, cid))
        
    # Sort descending by score, deterministic tie-break by candidate entity_id
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return {cid for _, cid in scored[:budget]}


def main():
    print("=" * 105)
    print("AMAZON ML CHALLENGE 2026 - PRODUCTION MULTI-STAGE BLOCKING BENCHMARK (V3)")
    print("=" * 105)
    t_start = time.time()
    
    # 1. Build S2 and S3 Indexes
    s2_idx = build_v3_indexes(S2_FILE)
    s3_idx = build_v3_indexes(S3_FILE)
    
    # 2. Load Ground Truth
    print("\nLoading ground truth...", flush=True)
    ground_truth = load_ground_truth()
    
    # 3. Load S1 Sample
    print(f"\nLoading Source 1 benchmark sample ({SAMPLE_SIZE:,} entities)...", flush=True)
    s1_entities = []
    with open(S1_FILE, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                s1_entities.append(parts[:4])
            if len(s1_entities) >= SAMPLE_SIZE:
                break
                
    print(f"Loaded {len(s1_entities):,} Source 1 entities.", flush=True)
    
    # 4. Execute Benchmark Evaluation
    print("\nExecuting Stage 1-4 Candidate Generation & Ranking across benchmark...", flush=True)
    t_eval = time.time()
    
    strategies_list = [
        "exact_name", "name_no_suffix", "exact_address", "name_and_addr",
        "hn_dist_tok", "hn_street_tok", "dist_tok_loc", "domain_loc",
        "spaceless_name", "name_two_tok", "country_num_name_tok", "distinctive_first_tok"
    ]
    
    # Metrics accumulators
    total_true_matches = 0
    recovered_matches = 0
    missed_matches_list = []
    
    total_candidate_pairs = 0
    candidate_counts = []
    
    strategy_recovered = defaultdict(int)
    country_metrics = defaultdict(lambda: {"true": 0, "rec": 0})
    source_metrics = {"S2": {"true": 0, "rec": 0}, "S3": {"true": 0, "rec": 0}}
    cardinality_metrics = defaultdict(lambda: {"true": 0, "rec": 0})
    
    slice_metrics = {
        "unicode": {"true": 0, "rec": 0},
        "missing_addr": {"true": 0, "rec": 0},
        "url_domain": {"true": 0, "rec": 0},
        "house_number": {"true": 0, "rec": 0},
    }
    
    for idx, (s1_eid, raw_name, raw_addr, raw_country) in enumerate(s1_entities):
        true_matches = ground_truth.get(s1_eid, set())
        n_true = len(true_matches)
        total_true_matches += n_true
        
        # Track cardinality slice
        card_bucket = "1" if n_true == 1 else ("2-3" if 2 <= n_true <= 3 else ("4-6" if 4 <= n_true <= 6 else "7+"))
        cardinality_metrics[card_bucket]["true"] += n_true
        
        # Extract features
        feat = normalize_and_extract_features(raw_name, raw_addr, raw_country)
        country = feat["country"]
        clean_name = feat["clean_name"]
        name_no_suf = feat["name_no_suf"]
        spaceless_name = feat["spaceless_name"]
        clean_addr = feat["clean_addr"]
        dist_tokens = feat["dist_tokens"]
        core_tokens = feat["core_tokens"]
        addr_nums = feat["addr_nums"]
        street_toks = feat["street_toks"]
        loc_tokens = feat["loc_tokens"]
        
        # Slices
        if country:
            country_metrics[country]["true"] += n_true
        for tm in true_matches:
            if tm.startswith("S2"):
                source_metrics["S2"]["true"] += 1
            elif tm.startswith("S3"):
                source_metrics["S3"]["true"] += 1
                
        if feat["has_unicode"]:
            slice_metrics["unicode"]["true"] += n_true
        if feat["is_missing_addr"]:
            slice_metrics["missing_addr"]["true"] += n_true
        if feat["is_url"]:
            slice_metrics["url_domain"]["true"] += n_true
        if addr_nums:
            slice_metrics["house_number"]["true"] += n_true
            
        cand_strats = defaultdict(set)
        
        if country:
            # STAGE 1: High Precision Blocks
            if clean_name:
                k = (country, clean_name)
                for cid in s2_idx["exact_name"].get(k, []): cand_strats[cid].add("exact_name")
                for cid in s3_idx["exact_name"].get(k, []): cand_strats[cid].add("exact_name")
                
            if name_no_suf and name_no_suf != clean_name:
                k = (country, name_no_suf)
                for cid in s2_idx["name_no_suffix"].get(k, []): cand_strats[cid].add("name_no_suffix")
                for cid in s3_idx["name_no_suffix"].get(k, []): cand_strats[cid].add("name_no_suffix")
                
            if clean_addr and len(clean_addr) >= 6:
                k = (country, clean_addr)
                for cid in s2_idx["exact_address"].get(k, []): cand_strats[cid].add("exact_address")
                for cid in s3_idx["exact_address"].get(k, []): cand_strats[cid].add("exact_address")
                
            if name_no_suf and clean_addr:
                k = (country, name_no_suf, clean_addr)
                for cid in s2_idx["name_and_addr"].get(k, []): cand_strats[cid].add("name_and_addr")
                for cid in s3_idx["name_and_addr"].get(k, []): cand_strats[cid].add("name_and_addr")
                
            if dist_tokens and addr_nums:
                for num in addr_nums:
                    for d_tok in dist_tokens:
                        k = (country, num, d_tok)
                        for cid in s2_idx["hn_dist_tok"].get(k, []): cand_strats[cid].add("hn_dist_tok")
                        for cid in s3_idx["hn_dist_tok"].get(k, []): cand_strats[cid].add("hn_dist_tok")
                        
            if street_toks and addr_nums:
                for num in addr_nums:
                    for s_tok in street_toks:
                        k = (country, num, s_tok)
                        for cid in s2_idx["hn_street_tok"].get(k, []): cand_strats[cid].add("hn_street_tok")
                        for cid in s3_idx["hn_street_tok"].get(k, []): cand_strats[cid].add("hn_street_tok")
                        
            if dist_tokens and loc_tokens:
                for d_tok in dist_tokens:
                    for l_tok in loc_tokens:
                        k = (country, d_tok, l_tok)
                        for cid in s2_idx["dist_tok_loc"].get(k, []): cand_strats[cid].add("dist_tok_loc")
                        for cid in s3_idx["dist_tok_loc"].get(k, []): cand_strats[cid].add("dist_tok_loc")
                        
            if spaceless_name and len(spaceless_name) >= 5 and street_toks:
                s_tok = street_toks[0]
                k = (country, spaceless_name, s_tok)
                for cid in s2_idx["domain_loc"].get(k, []): cand_strats[cid].add("domain_loc")
                for cid in s3_idx["domain_loc"].get(k, []): cand_strats[cid].add("domain_loc")
                
            # STAGE 2: Controlled Relaxation Blocks
            if spaceless_name and len(spaceless_name) >= 5:
                k = (country, spaceless_name)
                for cid in s2_idx["spaceless_name"].get(k, []): cand_strats[cid].add("spaceless_name")
                for cid in s3_idx["spaceless_name"].get(k, []): cand_strats[cid].add("spaceless_name")
                
            if len(core_tokens) >= 2:
                k = (country, core_tokens[0], core_tokens[1])
                for cid in s2_idx["name_two_tok"].get(k, []): cand_strats[cid].add("name_two_tok")
                for cid in s3_idx["name_two_tok"].get(k, []): cand_strats[cid].add("name_two_tok")
                
            if core_tokens and addr_nums:
                first_tok = core_tokens[0]
                for num in addr_nums:
                    k = (country, num, first_tok)
                    for cid in s2_idx["country_num_name_tok"].get(k, []): cand_strats[cid].add("country_num_name_tok")
                    for cid in s3_idx["country_num_name_tok"].get(k, []): cand_strats[cid].add("country_num_name_tok")
                    
            if dist_tokens and len(dist_tokens[0]) >= 6:
                d_first = dist_tokens[0]
                k = (country, d_first)
                for cid in s2_idx["distinctive_first_tok"].get(k, []): cand_strats[cid].add("distinctive_first_tok")
                for cid in s3_idx["distinctive_first_tok"].get(k, []): cand_strats[cid].add("distinctive_first_tok")
                
        # Track raw strategy recovery
        for strat in strategies_list:
            sc = {cid for cid, s_set in cand_strats.items() if strat in s_set}
            strategy_recovered[strat] += len(true_matches.intersection(sc))
            
        # STAGE 4: Adaptive Candidate Budget & Ranking
        final_cands = rank_and_budget_candidates(cand_strats, budget=CANDIDATE_BUDGET)
        
        n_cands = len(final_cands)
        total_candidate_pairs += n_cands
        candidate_counts.append(n_cands)
        
        # Recovered vs Missed
        matched_set = true_matches.intersection(final_cands)
        rec_count = len(matched_set)
        recovered_matches += rec_count
        
        if country:
            country_metrics[country]["rec"] += rec_count
        cardinality_metrics[card_bucket]["rec"] += rec_count
        
        for tm in matched_set:
            if tm.startswith("S2"):
                source_metrics["S2"]["rec"] += 1
            elif tm.startswith("S3"):
                source_metrics["S3"]["rec"] += 1
                
        if feat["has_unicode"]:
            slice_metrics["unicode"]["rec"] += rec_count
        if feat["is_missing_addr"]:
            slice_metrics["missing_addr"]["rec"] += rec_count
        if feat["is_url"]:
            slice_metrics["url_domain"]["rec"] += rec_count
        if addr_nums:
            slice_metrics["house_number"]["rec"] += rec_count
            
        # Missed matches logging
        missed = true_matches - final_cands
        for m_id in missed:
            missed_matches_list.append({
                "source1_entity_id": s1_eid,
                "matched_entity_id": m_id,
                "country": country,
                "business_name": raw_name,
                "business_address": raw_addr,
                "generated_cands_before_budget": len(cand_strats),
                "in_unbudgeted_pool": m_id in cand_strats,
            })
            
        if (idx + 1) % 2500 == 0:
            cur_rec = recovered_matches / total_true_matches * 100 if total_true_matches > 0 else 0
            cur_avg = total_candidate_pairs / (idx + 1)
            print(f"  Processed {idx + 1:,} / {len(s1_entities):,} queries (Recall: {cur_rec:.2f}%, Avg Cands: {cur_avg:.1f})...", flush=True)
            
    eval_duration = time.time() - t_eval
    total_duration = time.time() - t_start
    print(f"\nEvaluation completed in {eval_duration:.1f}s (Total pipeline: {total_duration:.1f}s).", flush=True)
    
    # 5. Calculate Benchmark Summary Statistics
    blocking_recall = (recovered_matches / total_true_matches * 100) if total_true_matches > 0 else 0
    missed_matches_count = total_true_matches - recovered_matches
    avg_cands = total_candidate_pairs / len(s1_entities)
    
    cand_series = pd.Series(candidate_counts)
    median_cands = float(cand_series.median())
    p95_cands = float(cand_series.quantile(0.95))
    p99_cands = float(cand_series.quantile(0.99))
    max_cands = int(cand_series.max())
    
    # V2 Baseline constants for direct comparison
    v2_recall = 94.740
    v2_total_cands = 22_558_930
    v2_avg_cands = 2255.89
    v2_max_cands = 38_161
    
    reduction_ratio = (v2_total_cands - total_candidate_pairs) / v2_total_cands * 100
    candidate_fold_reduction = v2_avg_cands / avg_cands if avg_cands > 0 else 0
    
    # 6. Save Missed Matches TSV
    missed_df = pd.DataFrame(missed_matches_list)
    missed_tsv_path = OUTPUTS_DIR / "blocking_v3_missed_matches.tsv"
    missed_df.to_csv(missed_tsv_path, sep="\t", index=False)
    print(f"Saved missed matches to: {missed_tsv_path} ({len(missed_df):,} rows)")
    
    # 7. Construct Benchmark JSON
    benchmark_data = {
        "benchmark_metadata": {
            "version": "V3",
            "date": "2026-09-25",
            "source1_sample_size": SAMPLE_SIZE,
            "max_postings_standard": MAX_POSTINGS_STD,
            "max_postings_relaxed": MAX_POSTINGS_RELAXED,
            "candidate_budget_per_s1": CANDIDATE_BUDGET,
            "runtime_seconds": round(total_duration, 2),
        },
        "summary_metrics": {
            "total_ground_truth_matches": total_true_matches,
            "recovered_matches": recovered_matches,
            "missed_matches": missed_matches_count,
            "blocking_recall_pct": round(blocking_recall, 4),
            "total_candidate_pairs": total_candidate_pairs,
            "average_candidates_per_s1": round(avg_cands, 2),
            "median_candidates_per_s1": round(median_cands, 1),
            "p95_candidates_per_s1": round(p95_cands, 1),
            "p99_candidates_per_s1": round(p99_cands, 1),
            "maximum_candidates_per_s1": max_cands,
            "candidate_reduction_ratio_pct": round(reduction_ratio, 2),
            "candidate_fold_reduction": round(candidate_fold_reduction, 2),
        },
        "comparison_v2_vs_v3": {
            "v2_recall_pct": v2_recall,
            "v3_recall_pct": round(blocking_recall, 4),
            "recall_difference_pct": round(blocking_recall - v2_recall, 4),
            "v2_avg_candidates": v2_avg_cands,
            "v3_avg_candidates": round(avg_cands, 2),
            "v2_max_candidates": v2_max_cands,
            "v3_max_candidates": max_cands,
            "candidate_reduction_pct": round(reduction_ratio, 2),
        },
        "recall_by_strategy": {
            strat: {
                "recovered": strategy_recovered[strat],
                "recall_pct": round(strategy_recovered[strat] / total_true_matches * 100, 2)
            }
            for strat in strategies_list
        },
        "recall_by_country": {
            cntry: {
                "true_matches": data["true"],
                "recovered": data["rec"],
                "recall_pct": round(data["rec"] / data["true"] * 100, 2) if data["true"] > 0 else 0
            }
            for cntry, data in sorted(country_metrics.items(), key=lambda x: x[1]["true"], reverse=True)
        },
        "recall_by_source": {
            src: {
                "true_matches": data["true"],
                "recovered": data["rec"],
                "recall_pct": round(data["rec"] / data["true"] * 100, 2) if data["true"] > 0 else 0
            }
            for src, data in source_metrics.items()
        },
        "recall_by_match_cardinality": {
            card: {
                "true_matches": data["true"],
                "recovered": data["rec"],
                "recall_pct": round(data["rec"] / data["true"] * 100, 2) if data["true"] > 0 else 0
            }
            for card, data in cardinality_metrics.items()
        },
        "recall_by_record_slice": {
            slice_name: {
                "true_matches": data["true"],
                "recovered": data["rec"],
                "recall_pct": round(data["rec"] / data["true"] * 100, 2) if data["true"] > 0 else 0
            }
            for slice_name, data in slice_metrics.items()
        }
    }
    
    benchmark_json_path = OUTPUTS_DIR / "blocking_v3_benchmark.json"
    with open(benchmark_json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"Saved benchmark JSON to: {benchmark_json_path}")
    
    # 8. Generate Benchmark Text Report
    report_lines = []
    report_lines.append("=" * 105)
    report_lines.append("AMAZON ML CHALLENGE 2026 - PRODUCTION MULTI-STAGE BLOCKING BENCHMARK (V3)")
    report_lines.append("=" * 105)
    report_lines.append(f"Source-1 Sample Size        : {SAMPLE_SIZE:,}")
    report_lines.append(f"Total Ground-Truth Matches  : {total_true_matches:,}")
    report_lines.append(f"Recovered Matches           : {recovered_matches:,}")
    report_lines.append(f"Missed Matches              : {missed_matches_count:,}")
    report_lines.append(f"Blocking Recall             : {blocking_recall:.4f}%")
    report_lines.append("-" * 105)
    report_lines.append(f"Total Candidate Pairs       : {total_candidate_pairs:,}")
    report_lines.append(f"Average Candidates / S1     : {avg_cands:.2f}")
    report_lines.append(f"Median Candidates / S1      : {median_cands:.1f}")
    report_lines.append(f"P95 Candidates / S1         : {p95_cands:.1f}")
    report_lines.append(f"P99 Candidates / S1         : {p99_cands:.1f}")
    report_lines.append(f"Maximum Candidates / S1     : {max_cands:,}")
    report_lines.append(f"Candidate Reduction Ratio   : {reduction_ratio:.2f}% ({candidate_fold_reduction:.1f}x reduction)")
    report_lines.append("=" * 105)
    
    report_lines.append("\n" + "=" * 105)
    report_lines.append("DIRECT COMPARISON: V2 BASELINE vs V3 MULTI-STAGE BLOCKING")
    report_lines.append("=" * 105)
    report_lines.append(f"{'Metric':35s}{'V2 Baseline':>20s}{'V3 Multi-Stage':>20s}{'Change':>25s}")
    report_lines.append("-" * 105)
    report_lines.append(f"{'Blocking Recall (%)':35s}{v2_recall:>19.2f}%{blocking_recall:>19.2f}%{blocking_recall - v2_recall:>+24.2f}%")
    report_lines.append(f"{'Total Candidate Pairs':35s}{v2_total_cands:>20,}{total_candidate_pairs:>20,}{f'-{reduction_ratio:.1f}%':>25s}")
    report_lines.append(f"{'Average Candidates / S1':35s}{v2_avg_cands:>20.1f}{avg_cands:>20.1f}{f'-{v2_avg_cands - avg_cands:.1f} ({(avg_cands - v2_avg_cands)/v2_avg_cands*100:+.1f}%)':>25s}")
    report_lines.append(f"{'Maximum Candidates / S1':35s}{v2_max_cands:>20,}{max_cands:>20,}{f'-{v2_max_cands - max_cands:,}':>25s}")
    report_lines.append("=" * 105)
    
    report_lines.append("\n" + "-" * 105)
    report_lines.append("RECALL BY BLOCKING STRATEGY")
    report_lines.append("-" * 105)
    report_lines.append(f"{'Strategy Name':35s}{'Recovered Matches':>20s}{'Strategy Recall (%)':>22s}")
    report_lines.append("-" * 105)
    for strat in strategies_list:
        rec = strategy_recovered[strat]
        pct = rec / total_true_matches * 100 if total_true_matches > 0 else 0
        report_lines.append(f"{strat:35s}{rec:>20,}{pct:>21.2f}%")
        
    report_lines.append("\n" + "-" * 105)
    report_lines.append("RECALL BY COUNTRY")
    report_lines.append("-" * 105)
    report_lines.append(f"{'Country':15s}{'True Matches':>18s}{'Recovered':>18s}{'Recall (%)':>18s}")
    report_lines.append("-" * 105)
    for cntry, data in sorted(country_metrics.items(), key=lambda x: x[1]["true"], reverse=True):
        pct = data["rec"] / data["true"] * 100 if data["true"] > 0 else 0
        report_lines.append(f"{cntry:15s}{data['true']:>18,}{data['rec']:>18,}{pct:>17.2f}%")
        
    report_lines.append("\n" + "-" * 105)
    report_lines.append("RECALL BY MATCH SOURCE")
    report_lines.append("-" * 105)
    report_lines.append(f"{'Source':15s}{'True Matches':>18s}{'Recovered':>18s}{'Recall (%)':>18s}")
    report_lines.append("-" * 105)
    for src in ["S2", "S3"]:
        data = source_metrics[src]
        pct = data["rec"] / data["true"] * 100 if data["true"] > 0 else 0
        report_lines.append(f"{src:15s}{data['true']:>18,}{data['rec']:>18,}{pct:>17.2f}%")
        
    report_lines.append("\n" + "-" * 105)
    report_lines.append("RECALL BY MATCH CARDINALITY")
    report_lines.append("-" * 105)
    report_lines.append(f"{'Cardinality Bucket':25s}{'True Matches':>18s}{'Recovered':>18s}{'Recall (%)':>18s}")
    report_lines.append("-" * 105)
    for card in ["1", "2-3", "4-6", "7+"]:
        data = cardinality_metrics[card]
        pct = data["rec"] / data["true"] * 100 if data["true"] > 0 else 0
        report_lines.append(f"{card:25s}{data['true']:>18,}{data['rec']:>18,}{pct:>17.2f}%")
        
    report_lines.append("\n" + "-" * 105)
    report_lines.append("RECALL BY HARD RECORD SLICES")
    report_lines.append("-" * 105)
    report_lines.append(f"{'Record Slice':35s}{'True Matches':>18s}{'Recovered':>18s}{'Recall (%)':>18s}")
    report_lines.append("-" * 105)
    for s_name, label in [
        ("unicode", "Unicode / Transliterated"),
        ("missing_addr", "Missing / Sparse Address"),
        ("url_domain", "URL / Domain / Handle"),
        ("house_number", "Has House Number"),
    ]:
        data = slice_metrics[s_name]
        pct = data["rec"] / data["true"] * 100 if data["true"] > 0 else 0
        report_lines.append(f"{label:35s}{data['true']:>18,}{data['rec']:>18,}{pct:>17.2f}%")
    report_lines.append("=" * 105)
    
    benchmark_txt_path = OUTPUTS_DIR / "blocking_v3_benchmark.txt"
    report_text = "\n".join(report_lines)
    with open(benchmark_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved benchmark report to: {benchmark_txt_path}")
    
    # Print report to stdout
    print("\n" + report_text)


if __name__ == "__main__":
    main()
