import sys
from pathlib import Path
import re
import unicodedata
import pandas as pd
import time
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[4]
TRAIN_DIR = ROOT / "dataset" / "train"

S1_FILE = TRAIN_DIR / "train_source1.tsv"
S2_FILE = TRAIN_DIR / "train_source2.tsv"
S3_FILE = TRAIN_DIR / "train_source3.tsv"
GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"

SAMPLE_SIZE = 10_000
MAX_POSTINGS = 500  # Strict cap on postings per key to prevent memory thrashing

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


def normalize_text(value):
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


def strip_prefix_noise(text):
    for pattern in PREFIX_PATTERNS:
        text = re.sub(pattern, "", text).strip()
    m = re.search(r"(?:formerly|dba|doing business as)\s*:\s*(.*)", text)
    if m:
        return m.group(1).strip()
    return text


def normalize_name_clean(value):
    text = normalize_text(value)
    return strip_prefix_noise(text)


def normalize_name_no_suffix(value):
    text = normalize_name_clean(value)
    tokens = [t for t in text.split() if t not in LEGAL_SUFFIXES]
    return " ".join(tokens)


def normalize_spaceless_name(value):
    text = normalize_name_clean(value)
    tokens = [t for t in text.split() if t not in LEGAL_SUFFIXES]
    spaceless = "".join(tokens)
    return spaceless if len(spaceless) >= 4 else ""


def normalize_address(value):
    text = normalize_text(value)
    tokens = [ADDRESS_REPLACEMENTS.get(t, t) for t in text.split()]
    return " ".join(tokens)


def extract_address_numbers(addr_norm):
    return set(re.findall(r"\b\d+\b", addr_norm))


def extract_name_core_tokens(name_norm):
    return [t for t in name_norm.split() if len(t) >= 3 and t not in LEGAL_SUFFIXES]


def extract_distinctive_name_tokens(name_clean):
    return [
        t for t in name_clean.split()
        if len(t) >= 3 and t not in LEGAL_SUFFIXES and t not in GENERIC_WORDS
    ]


def extract_street_tokens(addr_norm):
    return [
        t for t in addr_norm.split()
        if len(t) >= 4 and t not in STREET_STOPWORDS and not t.isdigit()
    ]


def build_indexes(file_path):
    print(f"\nBuilding indexes: {file_path.name} (streaming chunks)...", flush=True)
    t0 = time.time()

    indexes = {
        "exact_name": defaultdict(list),
        "name_no_suffix": defaultdict(list),
        "spaceless_name": defaultdict(list),
        "exact_address": defaultdict(list),
        "addr_num_any_name_tok": defaultdict(list),
        "addr_num_street_tok": defaultdict(list),
        "name_two_tok": defaultdict(list),
        "distinctive_first_tok": defaultdict(list),
    }

    chunk_count = 0
    row_count = 0

    for chunk in pd.read_csv(file_path, sep="\t", dtype=str, chunksize=500_000):
        chunk_count += 1
        countries = chunk["country"].fillna("").tolist()
        names = chunk["business_name"].fillna("").tolist()
        addrs = chunk["business_address"].fillna("").tolist()
        eids = chunk["entity_id"].tolist()

        for country, raw_name, raw_addr, eid in zip(countries, names, addrs, eids):
            row_count += 1
            if not country:
                continue

            name_clean = normalize_name_clean(raw_name)
            name_no_suf = normalize_name_no_suffix(raw_name)
            spaceless_name = normalize_spaceless_name(raw_name)
            addr_norm = normalize_address(raw_addr)

            # 1. Exact Clean Name
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

            # 3. Spaceless Name
            if spaceless_name and len(spaceless_name) >= 5:
                k = (country, spaceless_name)
                lst = indexes["spaceless_name"][k]
                if len(lst) < MAX_POSTINGS:
                    lst.append(eid)

            # 4. Exact Address
            if addr_norm and len(addr_norm) >= 5:
                k = (country, addr_norm)
                lst = indexes["exact_address"][k]
                if len(lst) < MAX_POSTINGS:
                    lst.append(eid)

            addr_nums = extract_address_numbers(addr_norm)
            core_tokens = extract_name_core_tokens(name_clean)
            dist_tokens = extract_distinctive_name_tokens(name_clean)
            street_tokens = extract_street_tokens(addr_norm)

            # 5. House Number + ANY Distinctive Name Token
            if dist_tokens and addr_nums:
                for num in addr_nums:
                    for tok in dist_tokens:
                        k = (country, num, tok)
                        lst = indexes["addr_num_any_name_tok"][k]
                        if len(lst) < MAX_POSTINGS:
                            lst.append(eid)

            # 6. House Number + Street / City Token
            if street_tokens and addr_nums:
                for num in addr_nums:
                    for s_tok in street_tokens:
                        k = (country, num, s_tok)
                        lst = indexes["addr_num_street_tok"][k]
                        if len(lst) < MAX_POSTINGS:
                            lst.append(eid)

            # 7. First Two Name Tokens
            if len(core_tokens) >= 2:
                k = (country, core_tokens[0], core_tokens[1])
                lst = indexes["name_two_tok"][k]
                if len(lst) < MAX_POSTINGS:
                    lst.append(eid)

            # 8. Distinctive First Token
            if dist_tokens and len(dist_tokens[0]) >= 5:
                first_tok = dist_tokens[0]
                k = (country, first_tok)
                lst = indexes["distinctive_first_tok"][k]
                if len(lst) < MAX_POSTINGS:
                    lst.append(eid)

        print(f"  Processed chunk {chunk_count} ({row_count:,} rows)...", flush=True)

    print(f"Done indexing {file_path.name} in {time.time()-t0:.1f}s.", flush=True)
    return indexes


def load_ground_truth():
    gt = {}
    for chunk in pd.read_csv(GT_FILE, sep="\t", dtype=str, chunksize=500_000):
        for s1_id, m_ids in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            if pd.isna(m_ids) or not str(m_ids).strip():
                gt[s1_id] = set()
            else:
                gt[s1_id] = {x.strip() for x in str(m_ids).split(",") if x.strip()}
    return gt


def main():
    print("=" * 105)
    print("AMAZON ML CHALLENGE 2026 - BLOCKING BENCHMARK (V2 vs V1 BASELINE)")
    print("=" * 105)

    # 1. Load S1 Sample
    source1 = pd.read_csv(S1_FILE, sep="\t", dtype=str, nrows=SAMPLE_SIZE)
    print(f"Source 1 sample: {len(source1):,}")

    # 2. Build S2 and S3 indexes
    s2_indexes = build_indexes(S2_FILE)
    s3_indexes = build_indexes(S3_FILE)

    # 3. Load ground truth
    print("\nLoading ground truth...", flush=True)
    ground_truth = load_ground_truth()

    # 4. Evaluate streaming
    print("\nEvaluating strategies across sample...", flush=True)

    strategies = [
        "exact_name",
        "name_no_suffix",
        "spaceless_name",
        "exact_address",
        "addr_num_any_name_tok",
        "addr_num_street_tok",
        "name_two_tok",
        "distinctive_first_tok",
        "old_combined",
        "new_combined",
    ]

    metrics = {
        strat: {"recovered": 0, "total_candidates": 0, "max_candidates": 0, "with_cands": 0}
        for strat in strategies
    }

    total_true_matches = 0
    t0 = time.time()

    for idx, (_, row) in enumerate(source1.iterrows()):
        entity_id = row["entity_id"]
        true_matches = ground_truth.get(entity_id, set())
        total_true_matches += len(true_matches)

        country = row["country"] if pd.notna(row["country"]) else ""
        name_clean = normalize_name_clean(row["business_name"])
        name_no_suf = normalize_name_no_suffix(row["business_name"])
        spaceless_name = normalize_spaceless_name(row["business_name"])
        addr_norm = normalize_address(row["business_address"])

        addr_nums = extract_address_numbers(addr_norm)
        core_tokens = extract_name_core_tokens(name_clean)
        dist_tokens = extract_distinctive_name_tokens(name_clean)
        street_tokens = extract_street_tokens(addr_norm)

        # 1. Exact Name
        c_exact_name = set(s2_indexes["exact_name"].get((country, name_clean), []))
        c_exact_name.update(s3_indexes["exact_name"].get((country, name_clean), []))

        # 2. Name No Suffix
        c_no_suf = set(s2_indexes["name_no_suffix"].get((country, name_no_suf), []))
        c_no_suf.update(s3_indexes["name_no_suffix"].get((country, name_no_suf), []))

        # 3. Spaceless Name
        c_spaceless = set()
        if spaceless_name and len(spaceless_name) >= 5:
            c_spaceless.update(s2_indexes["spaceless_name"].get((country, spaceless_name), []))
            c_spaceless.update(s3_indexes["spaceless_name"].get((country, spaceless_name), []))

        # 4. Exact Address
        c_addr = set()
        if addr_norm and len(addr_norm) >= 5:
            c_addr.update(s2_indexes["exact_address"].get((country, addr_norm), []))
            c_addr.update(s3_indexes["exact_address"].get((country, addr_norm), []))

        # 5. Addr Num + ANY Name Token
        c_num_any_tok = set()
        if dist_tokens and addr_nums:
            for num in addr_nums:
                for tok in dist_tokens:
                    c_num_any_tok.update(s2_indexes["addr_num_any_name_tok"].get((country, num, tok), []))
                    c_num_any_tok.update(s3_indexes["addr_num_any_name_tok"].get((country, num, tok), []))

        # 6. Addr Num + Street Token
        c_num_street = set()
        if street_tokens and addr_nums:
            for num in addr_nums:
                for s_tok in street_tokens:
                    c_num_street.update(s2_indexes["addr_num_street_tok"].get((country, num, s_tok), []))
                    c_num_street.update(s3_indexes["addr_num_street_tok"].get((country, num, s_tok), []))

        # 7. First Two Name Tokens
        c_two_tok = set()
        if len(core_tokens) >= 2:
            c_two_tok.update(s2_indexes["name_two_tok"].get((country, core_tokens[0], core_tokens[1]), []))
            c_two_tok.update(s3_indexes["name_two_tok"].get((country, core_tokens[0], core_tokens[1]), []))

        # 8. Distinctive First Token
        c_first_tok = set()
        if dist_tokens and len(dist_tokens[0]) >= 5:
            first_tok = dist_tokens[0]
            c_first_tok.update(s2_indexes["distinctive_first_tok"].get((country, first_tok), []))
            c_first_tok.update(s3_indexes["distinctive_first_tok"].get((country, first_tok), []))

        # Old Combined
        c_old_comb = set()
        c_old_comb.update(c_exact_name)
        c_old_comb.update(c_no_suf)
        c_old_comb.update(c_addr)
        if core_tokens and addr_nums:
            first_tok = core_tokens[0]
            for num in addr_nums:
                c_old_comb.update(s2_indexes["addr_num_any_name_tok"].get((country, num, first_tok), []))
                c_old_comb.update(s3_indexes["addr_num_any_name_tok"].get((country, num, first_tok), []))
        c_old_comb.update(c_two_tok)

        # New Combined (All targeted strategies)
        c_new_comb = set()
        c_new_comb.update(c_exact_name)
        c_new_comb.update(c_no_suf)
        c_new_comb.update(c_spaceless)
        c_new_comb.update(c_addr)
        c_new_comb.update(c_num_any_tok)
        c_new_comb.update(c_num_street)
        c_new_comb.update(c_two_tok)
        c_new_comb.update(c_first_tok)

        cands_map = {
            "exact_name": c_exact_name,
            "name_no_suffix": c_no_suf,
            "spaceless_name": c_spaceless,
            "exact_address": c_addr,
            "addr_num_any_name_tok": c_num_any_tok,
            "addr_num_street_tok": c_num_street,
            "name_two_tok": c_two_tok,
            "distinctive_first_tok": c_first_tok,
            "old_combined": c_old_comb,
            "new_combined": c_new_comb,
        }

        for strat in strategies:
            cands = cands_map[strat]
            num_cands = len(cands)
            metrics[strat]["total_candidates"] += num_cands
            metrics[strat]["max_candidates"] = max(metrics[strat]["max_candidates"], num_cands)
            if num_cands > 0:
                metrics[strat]["with_cands"] += 1
            metrics[strat]["recovered"] += len(true_matches.intersection(cands))

        if (idx + 1) % 2500 == 0:
            print(f"  Processed {idx + 1:,} / {len(source1):,} queries...", flush=True)

    print(f"\nEvaluation completed in {time.time()-t0:.1f}s.")

    # 5. Print Results
    results = []
    for strat in strategies:
        recall = (
            metrics[strat]["recovered"] / total_true_matches * 100
            if total_true_matches > 0 else 0
        )
        avg_cand = metrics[strat]["total_candidates"] / len(source1)
        results.append({
            "strategy": strat,
            "recall": recall,
            "avg_candidates": avg_cand,
            "max_candidates": metrics[strat]["max_candidates"],
            "with_candidates": metrics[strat]["with_cands"],
            "recovered": metrics[strat]["recovered"],
        })

    results.sort(key=lambda x: x["recall"], reverse=True)

    old_rec = next(r["recall"] for r in results if r["strategy"] == "old_combined")
    new_rec = next(r["recall"] for r in results if r["strategy"] == "new_combined")
    improvement = new_rec - old_rec

    print("\n" + "=" * 105)
    print("BENCHMARK RESULTS SUMMARY (V2)")
    print("=" * 105)
    print(f"Total True Matches in Sample : {total_true_matches:,}")
    print(f"Old Combined Recall (V1)     : {old_rec:.4f}%")
    print(f"New Combined Recall (V2)     : {new_rec:.4f}%")
    print(f"Absolute Recall Improvement  : +{improvement:.4f}% ({(new_rec - old_rec)/old_rec*100:+.2f}% relative)")
    print("-" * 105)
    print(f"{'Strategy':25s}{'Recall (%)':>12s}{'Recovered':>12s}{'Avg Cand/S1':>15s}{'Max Cand/S1':>15s}{'Entities w/ Cands':>20s}")
    print("-" * 105)

    for r in results:
        print(
            f"{r['strategy']:25s}"
            f"{r['recall']:>11.2f}%"
            f"{r['recovered']:>12,}"
            f"{r['avg_candidates']:>15.1f}"
            f"{r['max_candidates']:>15,}"
            f"{r['with_candidates']:>20,}"
        )
    print("=" * 105)


if __name__ == "__main__":
    main()