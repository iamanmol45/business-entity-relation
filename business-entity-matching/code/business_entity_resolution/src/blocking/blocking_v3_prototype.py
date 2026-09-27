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
MAX_POSTINGS = 400
CANDIDATE_BUDGET = 250

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
    "club", "shop", "store", "india", "usa", "america", "city", "state", "center", "centre"
}

DOMAIN_SUFFIX_PATTERN = re.compile(r"\.(?:com|org|net|biz|co|in|us|info|io|me|ai|gov|edu|co\.in|org\.in|net\.in)\b", re.I)
PREFIX_PATTERN = re.compile(r"^(?:formerly\s*:\s*|formerly\s+|dba\s*:\s*|dba\s+|doing\s+business\s+as\s+|c\s*/\s*o\s+|mr\s+|mrs\s+|ms\s+|dr\s+)", re.I)
TRANS_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))

ADDRESS_REPLACEMENTS = {
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

def normalize_text_fast(text):
    if not text:
        return ""
    val = str(text).lower()
    if not val.isascii():
        val = unicodedata.normalize("NFKD", val)
        val = "".join(c for c in val if not unicodedata.combining(c))
    if "." in val:
        val = DOMAIN_SUFFIX_PATTERN.sub("", val)
    val = val.translate(TRANS_TABLE)
    # Strip leading zeros from numeric tokens
    tokens = val.split()
    clean_tokens = [t.lstrip("0") if t.isdigit() and len(t) > 1 else t for t in tokens]
    return " ".join(clean_tokens)

def extract_features(raw_name, raw_addr, raw_country):
    country = str(raw_country).strip().upper() if raw_country else ""
    
    # Name normalization
    norm_name = normalize_text_fast(raw_name)
    norm_name = PREFIX_PATTERN.sub("", norm_name).strip()
    name_tokens = norm_name.split()
    tokens_no_suf = [t for t in name_tokens if t not in LEGAL_SUFFIXES]
    name_no_suf = " ".join(tokens_no_suf)
    spaceless_name = "".join(tokens_no_suf)
    dist_tokens = [t for t in tokens_no_suf if len(t) >= 3 and t not in GENERIC_WORDS and not t.isdigit()]
    core_tokens = [t for t in tokens_no_suf if len(t) >= 3 and not t.isdigit()]
    
    # Address normalization
    addr_norm = normalize_text_fast(raw_addr)
    raw_addr_tokens = addr_norm.split()
    addr_tokens = [ADDRESS_REPLACEMENTS.get(t, t) for t in raw_addr_tokens]
    clean_addr = " ".join(addr_tokens)
    
    # Numbers (cap to first 3)
    nums = [t for t in raw_addr_tokens if t.isdigit() and 1 <= len(t) <= 6][:3]
    addr_nums = set(nums)
    
    # Street tokens (cap to first 3)
    street_toks = [t for t in addr_tokens if len(t) >= 4 and t not in STREET_STOPWORDS and not t.isdigit()][:3]
    
    # City / Location tokens (last parts of address)
    loc_tokens = []
    if raw_addr and "," in str(raw_addr):
        parts = str(raw_addr).split(",")
        if len(parts) >= 2:
            for p in parts[-2:]:
                for t in normalize_text_fast(p).split():
                    if len(t) >= 3 and t not in STREET_STOPWORDS and not t.isdigit() and t not in loc_tokens:
                        loc_tokens.append(t)
    loc_tokens = loc_tokens[:3]
    
    return {
        "country": country,
        "name_clean": norm_name,
        "name_no_suf": name_no_suf,
        "spaceless_name": spaceless_name,
        "dist_tokens": dist_tokens[:3],
        "core_tokens": core_tokens[:3],
        "clean_addr": clean_addr,
        "addr_nums": addr_nums,
        "street_toks": street_toks,
        "loc_tokens": loc_tokens,
    }

def build_v3_indexes(file_path):
    print(f"\nBuilding V3 indexes: {file_path.name} (fast streaming)...", flush=True)
    t0 = time.time()
    
    indexes = {
        # Stage 1: High Precision
        "exact_name": defaultdict(list),
        "name_no_suffix": defaultdict(list),
        "exact_address": defaultdict(list),
        "name_and_addr": defaultdict(list),
        "hn_dist_tok": defaultdict(list),
        "hn_street_tok": defaultdict(list),
        "dist_tok_loc": defaultdict(list),
        "domain_loc": defaultdict(list),
        
        # Stage 2: Controlled Relaxation
        "spaceless_name": defaultdict(list),
        "name_two_tok": defaultdict(list),
        "country_num_name_tok": defaultdict(list),
    }
    
    row_count = 0
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) < 4:
                continue
            eid, raw_name, raw_addr, raw_country = row[0], row[1], row[2], row[3]
            row_count += 1
            
            feat = extract_features(raw_name, raw_addr, raw_country)
            country = feat["country"]
            if not country:
                continue
                
            name_clean = feat["name_clean"]
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
            if name_clean:
                k = (country, name_clean)
                lst = indexes["exact_name"][k]
                if len(lst) < MAX_POSTINGS:
                    lst.append(eid)
                    
            # 2. Name No Suffix
            if name_no_suf and name_no_suf != name_clean:
                k = (country, name_no_suf)
                lst = indexes["name_no_suffix"][k]
                if len(lst) < MAX_POSTINGS:
                    lst.append(eid)
                    
            # 3. Exact Address
            if clean_addr and len(clean_addr) >= 6:
                k = (country, clean_addr)
                lst = indexes["exact_address"][k]
                if len(lst) < MAX_POSTINGS:
                    lst.append(eid)
                    
            # 4. Exact Name + Address
            if name_no_suf and clean_addr:
                k = (country, name_no_suf, clean_addr)
                lst = indexes["name_and_addr"][k]
                if len(lst) < MAX_POSTINGS:
                    lst.append(eid)
                    
            # 5. House Number + Distinctive Token
            if dist_tokens and addr_nums:
                for num in addr_nums:
                    for d_tok in dist_tokens:
                        k = (country, num, d_tok)
                        lst = indexes["hn_dist_tok"][k]
                        if len(lst) < MAX_POSTINGS:
                            lst.append(eid)
                            
            # 6. House Number + Street Token
            if street_toks and addr_nums:
                for num in addr_nums:
                    for s_tok in street_toks:
                        k = (country, num, s_tok)
                        lst = indexes["hn_street_tok"][k]
                        if len(lst) < MAX_POSTINGS:
                            lst.append(eid)
                            
            # 7. Distinctive Token + Location / City Token
            if dist_tokens and loc_tokens:
                for d_tok in dist_tokens:
                    for l_tok in loc_tokens:
                        k = (country, d_tok, l_tok)
                        lst = indexes["dist_tok_loc"][k]
                        if len(lst) < MAX_POSTINGS:
                            lst.append(eid)
                            
            # 8. Domain / Spaceless Name + Location / Street Token
            if spaceless_name and len(spaceless_name) >= 5 and street_toks:
                s_tok = street_toks[0]
                k = (country, spaceless_name, s_tok)
                lst = indexes["domain_loc"][k]
                if len(lst) < MAX_POSTINGS:
                    lst.append(eid)
                    
            # STAGE 2: Controlled Relaxation
            # 9. Spaceless Name
            if spaceless_name and len(spaceless_name) >= 5:
                k = (country, spaceless_name)
                lst = indexes["spaceless_name"][k]
                if len(lst) < MAX_POSTINGS:
                    lst.append(eid)
                    
            # 10. First Two Core Name Tokens
            if len(core_tokens) >= 2:
                k = (country, core_tokens[0], core_tokens[1])
                lst = indexes["name_two_tok"][k]
                if len(lst) < MAX_POSTINGS:
                    lst.append(eid)
                    
            # 11. Country + House Number + Core Name Token
            if core_tokens and addr_nums:
                first_tok = core_tokens[0]
                for num in addr_nums:
                    k = (country, num, first_tok)
                    lst = indexes["country_num_name_tok"][k]
                    if len(lst) < MAX_POSTINGS:
                        lst.append(eid)
                        
            if row_count % 1_000_000 == 0:
                print(f"  Indexed {row_count:,} rows in {time.time()-t0:.1f}s...", flush=True)
                
    print(f"Finished indexing {file_path.name} ({row_count:,} rows) in {time.time()-t0:.1f}s.", flush=True)
    return indexes

def load_ground_truth():
    gt = {}
    with open(GT_FILE, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            s1_id, m_ids = row[0], row[1]
            if not m_ids or not m_ids.strip():
                gt[s1_id] = set()
            else:
                gt[s1_id] = {x.strip() for x in m_ids.split(",") if x.strip()}
    return gt

def score_candidate(cand_id, matched_strats):
    score = 0.0
    if "exact_name" in matched_strats:
        score += 100.0
    if "name_no_suffix" in matched_strats:
        score += 95.0
    if "name_and_addr" in matched_strats:
        score += 110.0
    if "exact_address" in matched_strats:
        score += 90.0
    if "hn_dist_tok" in matched_strats:
        score += 70.0
    if "domain_loc" in matched_strats:
        score += 65.0
    if "hn_street_tok" in matched_strats:
        score += 55.0
    if "dist_tok_loc" in matched_strats:
        score += 50.0
    if "spaceless_name" in matched_strats:
        score += 45.0
    if "country_num_name_tok" in matched_strats:
        score += 40.0
    if "name_two_tok" in matched_strats:
        score += 35.0
        
    if len(matched_strats) > 1:
        score += 25.0 * (len(matched_strats) - 1)
        
    return score

def main():
    print("=" * 105)
    print("AMAZON ML CHALLENGE 2026 - BLOCKING V3 PROTOTYPE BENCHMARK")
    print("=" * 105)
    
    # 1. Build S2 and S3 indexes
    s2_idx = build_v3_indexes(S2_FILE)
    s3_idx = build_v3_indexes(S3_FILE)
    
    # 2. Load Ground Truth
    print("\nLoading ground truth...", flush=True)
    gt = load_ground_truth()
    
    # 3. Load S1 Sample
    print(f"\nLoading Source 1 sample ({SAMPLE_SIZE:,} rows)...", flush=True)
    s1_rows = []
    with open(S1_FILE, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) >= 4:
                s1_rows.append(row[:4])
            if len(s1_rows) >= SAMPLE_SIZE:
                break
                
    print(f"Loaded {len(s1_rows):,} Source 1 rows.", flush=True)
    
    # 4. Evaluate Blocking V3
    print("\nEvaluating Blocking V3 across sample...", flush=True)
    t0 = time.time()
    
    total_true_matches = 0
    recovered_matches = 0
    total_candidate_pairs = 0
    candidate_counts = []
    
    strategy_recovered = defaultdict(int)
    
    for idx, row in enumerate(s1_rows):
        eid, raw_name, raw_addr, raw_country = row[0], row[1], row[2], row[3]
        true_m = gt.get(eid, set())
        total_true_matches += len(true_m)
        
        feat = extract_features(raw_name, raw_addr, raw_country)
        country = feat["country"]
        name_clean = feat["name_clean"]
        name_no_suf = feat["name_no_suf"]
        spaceless_name = feat["spaceless_name"]
        clean_addr = feat["clean_addr"]
        dist_tokens = feat["dist_tokens"]
        core_tokens = feat["core_tokens"]
        addr_nums = feat["addr_nums"]
        street_toks = feat["street_toks"]
        loc_tokens = feat["loc_tokens"]
        
        cand_strats = defaultdict(set)
        
        if country:
            # 1. Exact name
            if name_clean:
                k = (country, name_clean)
                for cid in s2_idx["exact_name"].get(k, []):
                    cand_strats[cid].add("exact_name")
                for cid in s3_idx["exact_name"].get(k, []):
                    cand_strats[cid].add("exact_name")
                    
            # 2. Name no suffix
            if name_no_suf and name_no_suf != name_clean:
                k = (country, name_no_suf)
                for cid in s2_idx["name_no_suffix"].get(k, []):
                    cand_strats[cid].add("name_no_suffix")
                for cid in s3_idx["name_no_suffix"].get(k, []):
                    cand_strats[cid].add("name_no_suffix")
                    
            # 3. Exact Address
            if clean_addr and len(clean_addr) >= 6:
                k = (country, clean_addr)
                for cid in s2_idx["exact_address"].get(k, []):
                    cand_strats[cid].add("exact_address")
                for cid in s3_idx["exact_address"].get(k, []):
                    cand_strats[cid].add("exact_address")
                    
            # 4. Exact Name + Address
            if name_no_suf and clean_addr:
                k = (country, name_no_suf, clean_addr)
                for cid in s2_idx["name_and_addr"].get(k, []):
                    cand_strats[cid].add("name_and_addr")
                for cid in s3_idx["name_and_addr"].get(k, []):
                    cand_strats[cid].add("name_and_addr")
                    
            # 5. House Number + Distinctive Token
            if dist_tokens and addr_nums:
                for num in addr_nums:
                    for d_tok in dist_tokens:
                        k = (country, num, d_tok)
                        for cid in s2_idx["hn_dist_tok"].get(k, []):
                            cand_strats[cid].add("hn_dist_tok")
                        for cid in s3_idx["hn_dist_tok"].get(k, []):
                            cand_strats[cid].add("hn_dist_tok")
                            
            # 6. House Number + Street Token
            if street_toks and addr_nums:
                for num in addr_nums:
                    for s_tok in street_toks:
                        k = (country, num, s_tok)
                        for cid in s2_idx["hn_street_tok"].get(k, []):
                            cand_strats[cid].add("hn_street_tok")
                        for cid in s3_idx["hn_street_tok"].get(k, []):
                            cand_strats[cid].add("hn_street_tok")
                            
            # 7. Distinctive Token + Location Token
            if dist_tokens and loc_tokens:
                for d_tok in dist_tokens:
                    for l_tok in loc_tokens:
                        k = (country, d_tok, l_tok)
                        for cid in s2_idx["dist_tok_loc"].get(k, []):
                            cand_strats[cid].add("dist_tok_loc")
                        for cid in s3_idx["dist_tok_loc"].get(k, []):
                            cand_strats[cid].add("dist_tok_loc")
                            
            # 8. Domain + Location
            if spaceless_name and len(spaceless_name) >= 5 and street_toks:
                s_tok = street_toks[0]
                k = (country, spaceless_name, s_tok)
                for cid in s2_idx["domain_loc"].get(k, []):
                    cand_strats[cid].add("domain_loc")
                for cid in s3_idx["domain_loc"].get(k, []):
                    cand_strats[cid].add("domain_loc")
                    
            # 9. Spaceless Name
            if spaceless_name and len(spaceless_name) >= 5:
                k = (country, spaceless_name)
                for cid in s2_idx["spaceless_name"].get(k, []):
                    cand_strats[cid].add("spaceless_name")
                for cid in s3_idx["spaceless_name"].get(k, []):
                    cand_strats[cid].add("spaceless_name")
                    
            # 10. First Two Core Name Tokens
            if len(core_tokens) >= 2:
                k = (country, core_tokens[0], core_tokens[1])
                for cid in s2_idx["name_two_tok"].get(k, []):
                    cand_strats[cid].add("name_two_tok")
                for cid in s3_idx["name_two_tok"].get(k, []):
                    cand_strats[cid].add("name_two_tok")
                    
            # 11. Country + House Number + Core Name Token
            if core_tokens and addr_nums:
                first_tok = core_tokens[0]
                for num in addr_nums:
                    k = (country, num, first_tok)
                    for cid in s2_idx["country_num_name_tok"].get(k, []):
                        cand_strats[cid].add("country_num_name_tok")
                    for cid in s3_idx["country_num_name_tok"].get(k, []):
                        cand_strats[cid].add("country_num_name_tok")
                        
        # Track individual strategy recall
        for s_name in [
            "exact_name", "name_no_suffix", "exact_address", "name_and_addr",
            "hn_dist_tok", "hn_street_tok", "dist_tok_loc", "domain_loc",
            "spaceless_name", "name_two_tok", "country_num_name_tok"
        ]:
            s_cands = {c for c, strats in cand_strats.items() if s_name in strats}
            strategy_recovered[s_name] += len(true_m.intersection(s_cands))
            
        # Stage 4: Candidate Scoring and Budget Cap
        if len(cand_strats) <= CANDIDATE_BUDGET:
            final_cands = set(cand_strats.keys())
        else:
            scored = []
            for cid, strats in cand_strats.items():
                sc = score_candidate(cid, strats)
                scored.append((sc, cid))
            # Sort descending by score, tie-break by cid
            scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
            final_cands = {cid for _, cid in scored[:CANDIDATE_BUDGET]}
            
        c_count = len(final_cands)
        total_candidate_pairs += c_count
        candidate_counts.append(c_count)
        
        recovered_matches += len(true_m.intersection(final_cands))
        
        if (idx + 1) % 2500 == 0:
            print(f"  Processed {idx + 1:,} / {len(s1_rows):,} queries ({recovered_matches/total_true_matches*100:.2f}% recall, {total_candidate_pairs/(idx+1):.1f} avg cands)...", flush=True)
            
    print(f"\nEvaluation completed in {time.time()-t0:.1f}s.")
    
    # Summary Metrics
    recall = (recovered_matches / total_true_matches * 100) if total_true_matches > 0 else 0
    avg_cands = total_candidate_pairs / len(s1_rows)
    cand_series = pd.Series(candidate_counts)
    
    print("\n" + "=" * 105)
    print("V3 PROTOTYPE RESULTS:")
    print("=" * 105)
    print(f"Total True Matches : {total_true_matches:,}")
    print(f"Recovered Matches  : {recovered_matches:,}")
    print(f"Blocking Recall    : {recall:.4f}%")
    print(f"Total Cand Pairs   : {total_candidate_pairs:,}")
    print(f"Avg Cands / S1     : {avg_cands:.2f}")
    print(f"Median Cands / S1  : {cand_series.median():.1f}")
    print(f"P95 Cands / S1     : {cand_series.quantile(0.95):.1f}")
    print(f"P99 Cands / S1     : {cand_series.quantile(0.99):.1f}")
    print(f"Max Cands / S1     : {cand_series.max():,}")
    print("=" * 105)
    
    print("\nStrategy Breakdown:")
    for s_name, rec in sorted(strategy_recovered.items(), key=lambda x: x[1], reverse=True):
        print(f"  {s_name:25s}: {rec:6,} matches ({rec/total_true_matches*100:6.2f}%)")

if __name__ == "__main__":
    main()
