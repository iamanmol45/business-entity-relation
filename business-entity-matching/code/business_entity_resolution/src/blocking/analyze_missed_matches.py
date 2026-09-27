import sys
from pathlib import Path
import re
import unicodedata
import pandas as pd
import time
from collections import defaultdict, Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[4]
TRAIN_DIR = ROOT / "dataset" / "train"
OUTPUT_DIR_PLURAL = ROOT / "outputs"
OUTPUT_DIR_SINGULAR = ROOT / "output"

OUTPUT_DIR_PLURAL.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_SINGULAR.mkdir(parents=True, exist_ok=True)

S1_FILE = TRAIN_DIR / "train_source1.tsv"
S2_FILE = TRAIN_DIR / "train_source2.tsv"
S3_FILE = TRAIN_DIR / "train_source3.tsv"
GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"

MISSED_MATCHES_V2_TSV = OUTPUT_DIR_PLURAL / "blocking_missed_matches_v2.tsv"
ANALYSIS_V2_TXT = OUTPUT_DIR_PLURAL / "blocking_missed_match_analysis_v2.txt"

SAMPLE_SIZE = 10_000
MAX_POSTINGS = 500

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


def build_indexes_with_overflow(file_path):
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

    overflow_counts = {k: defaultdict(int) for k in indexes}

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
                overflow_counts["exact_name"][k] += 1
                if len(lst) < MAX_POSTINGS:
                    lst.append(eid)

            # 2. Name No Suffix
            if name_no_suf and name_no_suf != name_clean:
                k = (country, name_no_suf)
                lst = indexes["name_no_suffix"][k]
                overflow_counts["name_no_suffix"][k] += 1
                if len(lst) < MAX_POSTINGS:
                    lst.append(eid)

            # 3. Spaceless Name
            if spaceless_name and len(spaceless_name) >= 5:
                k = (country, spaceless_name)
                lst = indexes["spaceless_name"][k]
                overflow_counts["spaceless_name"][k] += 1
                if len(lst) < MAX_POSTINGS:
                    lst.append(eid)

            # 4. Exact Address
            if addr_norm and len(addr_norm) >= 5:
                k = (country, addr_norm)
                lst = indexes["exact_address"][k]
                overflow_counts["exact_address"][k] += 1
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
                        overflow_counts["addr_num_any_name_tok"][k] += 1
                        if len(lst) < MAX_POSTINGS:
                            lst.append(eid)

            # 6. House Number + Street / City Token
            if street_tokens and addr_nums:
                for num in addr_nums:
                    for s_tok in street_tokens:
                        k = (country, num, s_tok)
                        lst = indexes["addr_num_street_tok"][k]
                        overflow_counts["addr_num_street_tok"][k] += 1
                        if len(lst) < MAX_POSTINGS:
                            lst.append(eid)

            # 7. First Two Name Tokens
            if len(core_tokens) >= 2:
                k = (country, core_tokens[0], core_tokens[1])
                lst = indexes["name_two_tok"][k]
                overflow_counts["name_two_tok"][k] += 1
                if len(lst) < MAX_POSTINGS:
                    lst.append(eid)

            # 8. Distinctive First Token
            if dist_tokens and len(dist_tokens[0]) >= 5:
                first_tok = dist_tokens[0]
                k = (country, first_tok)
                lst = indexes["distinctive_first_tok"][k]
                overflow_counts["distinctive_first_tok"][k] += 1
                if len(lst) < MAX_POSTINGS:
                    lst.append(eid)

        print(f"  Processed chunk {chunk_count} ({row_count:,} rows)...", flush=True)

    print(f"Done indexing {file_path.name} in {time.time()-t0:.1f}s.", flush=True)
    return indexes, overflow_counts


def load_ground_truth():
    gt = {}
    for chunk in pd.read_csv(GT_FILE, sep="\t", dtype=str, chunksize=500_000):
        for s1_id, m_ids in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            if pd.isna(m_ids) or not str(m_ids).strip():
                gt[s1_id] = set()
            else:
                gt[s1_id] = {x.strip() for x in str(m_ids).split(",") if x.strip()}
    return gt


def load_candidate_records_by_ids(file_path, target_ids):
    records = {}
    remaining_ids = set(target_ids)
    print(f"Streaming {file_path.name} to extract {len(target_ids):,} records...", flush=True)

    for chunk in pd.read_csv(file_path, sep="\t", dtype=str, chunksize=500_000):
        if not remaining_ids:
            break
        matched_chunk = chunk[chunk["entity_id"].isin(remaining_ids)]
        for _, row in matched_chunk.iterrows():
            eid = row["entity_id"]
            records[eid] = {
                "entity_id": eid,
                "business_name": str(row["business_name"]) if pd.notna(row["business_name"]) else "",
                "business_address": str(row["business_address"]) if pd.notna(row["business_address"]) else "",
                "country": str(row["country"]) if pd.notna(row["country"]) else "",
            }
            remaining_ids.remove(eid)

    print(f"Extracted {len(records):,} records from {file_path.name}.", flush=True)
    return records


def classify_missed_pair(
    s1_row,
    match_row,
    s2_overflow,
    s3_overflow,
):
    flags = []

    s1_name_raw = str(s1_row.get("business_name", ""))
    s1_addr_raw = str(s1_row.get("business_address", ""))
    s1_country = str(s1_row.get("country", ""))

    m_name_raw = str(match_row.get("business_name", ""))
    m_addr_raw = str(match_row.get("business_address", ""))
    m_country = str(match_row.get("country", ""))
    m_id = str(match_row.get("entity_id", ""))

    if not s1_name_raw.strip() or not m_name_raw.strip() or m_name_raw.lower() == "nan":
        flags.append("MISSING_NAME")
    if not s1_addr_raw.strip() or not m_addr_raw.strip() or m_addr_raw.lower() == "nan":
        flags.append("MISSING_ADDRESS")

    s1_name_clean = normalize_name_clean(s1_name_raw)
    m_name_clean = normalize_name_clean(m_name_raw)

    s1_name_nosuf = normalize_name_no_suffix(s1_name_raw)
    m_name_nosuf = normalize_name_no_suffix(m_name_raw)

    s1_spaceless = normalize_spaceless_name(s1_name_raw)
    m_spaceless = normalize_spaceless_name(m_name_raw)

    s1_addr_norm = normalize_address(s1_addr_raw)
    m_addr_norm = normalize_address(m_addr_raw)

    s1_nums = extract_address_numbers(s1_addr_norm)
    m_nums = extract_address_numbers(m_addr_norm)

    s1_tokens = extract_name_core_tokens(s1_name_clean)
    m_tokens = extract_name_core_tokens(m_name_clean)

    s1_dist_tokens = extract_distinctive_name_tokens(s1_name_clean)
    m_dist_tokens = extract_distinctive_name_tokens(m_name_clean)

    if s1_name_raw.strip().lower() == m_name_raw.strip().lower() and s1_name_raw.strip():
        flags.append("EXACT_NAME_MISSED")
    elif s1_name_clean and s1_name_clean == m_name_clean:
        flags.append("NAME_NORMALIZATION_MISSED")

    if s1_name_nosuf and s1_name_nosuf == m_name_nosuf:
        flags.append("LEGAL_SUFFIX_VARIATION")

    if s1_spaceless and s1_spaceless == m_spaceless:
        flags.append("SPACELESS_NAME_MATCH")

    if s1_tokens and m_tokens and sorted(s1_tokens) == sorted(m_tokens) and s1_tokens != m_tokens:
        flags.append("NAME_WORD_ORDER_VARIATION")

    shared_tokens = set(s1_tokens) & set(m_tokens)
    if shared_tokens:
        flags.append("NAME_TOKEN_VARIATION")

    url_pattern = r"(?:\.com|\.org|\.net|\.biz|\.co|\.in|\.us|\.io|\.biz|www\.)"
    if re.search(url_pattern, s1_name_raw.lower()) or re.search(url_pattern, m_name_raw.lower()):
        flags.append("URL_OR_DOMAIN_VARIATION")

    non_ascii_s1 = any(ord(c) > 127 for c in s1_name_raw + s1_addr_raw)
    non_ascii_m = any(ord(c) > 127 for c in m_name_raw + m_addr_raw)
    if non_ascii_s1 or non_ascii_m:
        flags.append("TRANSLITERATION_OR_UNICODE_VARIATION")

    if s1_addr_norm and s1_addr_norm == m_addr_norm:
        flags.append("ADDRESS_NORMALIZATION_MISSED")
        if not shared_tokens and s1_name_clean != m_name_clean:
            flags.append("ADDRESS_ONLY")

    shared_nums = s1_nums & m_nums
    if shared_nums:
        flags.append("HOUSE_NUMBER_MATCH")
        if shared_tokens:
            flags.append("HOUSE_NUMBER_PLUS_NAME_TOKEN")

    if s1_country and s1_country == m_country:
        if shared_tokens:
            flags.append("COUNTRY_PLUS_NAME")

    overflow_dict = s2_overflow if m_id.startswith("S2-") else s3_overflow
    posting_limit_details = []

    # Check posting limit across all strategies
    if s1_name_clean and s1_name_clean == m_name_clean:
        k = (s1_country, s1_name_clean)
        cnt = overflow_dict["exact_name"].get(k, 0)
        if cnt >= MAX_POSTINGS:
            posting_limit_details.append(("exact_name", k, cnt))

    if s1_name_nosuf and s1_name_nosuf == m_name_nosuf:
        k = (s1_country, s1_name_nosuf)
        cnt = overflow_dict["name_no_suffix"].get(k, 0)
        if cnt >= MAX_POSTINGS:
            posting_limit_details.append(("name_no_suffix", k, cnt))

    if s1_spaceless and s1_spaceless == m_spaceless and len(s1_spaceless) >= 5:
        k = (s1_country, s1_spaceless)
        cnt = overflow_dict["spaceless_name"].get(k, 0)
        if cnt >= MAX_POSTINGS:
            posting_limit_details.append(("spaceless_name", k, cnt))

    if s1_dist_tokens and m_dist_tokens and shared_nums:
        for num in shared_nums:
            for tok in (set(s1_dist_tokens) & set(m_dist_tokens)):
                k = (s1_country, num, tok)
                cnt = overflow_dict["addr_num_any_name_tok"].get(k, 0)
                if cnt >= MAX_POSTINGS:
                    posting_limit_details.append(("addr_num_any_name_tok", k, cnt))

    if posting_limit_details:
        flags.append("POSTING_LIMIT")

    if not flags:
        flags.append("NO_OBVIOUS_BLOCK")

    return flags, posting_limit_details


def main():
    print("=" * 80)
    print("MISSED MATCH SYSTEMATIC ANALYSIS PIPELINE (V2)")
    print("=" * 80)

    # 1. Load S1 Sample
    print(f"\n1. Loading Source-1 sample ({SAMPLE_SIZE:,} rows)...", flush=True)
    source1 = pd.read_csv(S1_FILE, sep="\t", dtype=str, nrows=SAMPLE_SIZE)
    s1_dict = {
        row["entity_id"]: {
            "entity_id": row["entity_id"],
            "business_name": str(row["business_name"]) if pd.notna(row["business_name"]) else "",
            "business_address": str(row["business_address"]) if pd.notna(row["business_address"]) else "",
            "country": str(row["country"]) if pd.notna(row["country"]) else "",
        }
        for _, row in source1.iterrows()
    }
    print(f"Loaded {len(s1_dict):,} S1 entities.")

    # 2. Build S2 and S3 indexes with overflow tracking
    s2_indexes, s2_overflow = build_indexes_with_overflow(S2_FILE)
    s3_indexes, s3_overflow = build_indexes_with_overflow(S3_FILE)

    # 3. Load ground truth
    print("\n2. Loading ground truth...", flush=True)
    ground_truth = load_ground_truth()

    # 4. Generate candidates & identify missed matches under NEW combined strategy
    print("\n3. Identifying missed true matches under V2 combined strategy...", flush=True)
    total_true_matches = 0
    recovered_matches = 0
    missed_pairs_ids = []

    # Map candidate sets per S1 ID for quick target verification
    s1_candidates_map = {}

    for s1_id, s1_row in s1_dict.items():
        true_matches = ground_truth.get(s1_id, set())
        total_true_matches += len(true_matches)

        country = s1_row["country"]
        name_clean = normalize_name_clean(s1_row["business_name"])
        name_no_suf = normalize_name_no_suffix(s1_row["business_name"])
        spaceless_name = normalize_spaceless_name(s1_row["business_name"])
        addr_norm = normalize_address(s1_row["business_address"])

        addr_nums = extract_address_numbers(addr_norm)
        core_tokens = extract_name_core_tokens(name_clean)
        dist_tokens = extract_distinctive_name_tokens(name_clean)
        street_tokens = extract_street_tokens(addr_norm)

        candidates = set()

        # 1. Exact Name
        candidates.update(s2_indexes["exact_name"].get((country, name_clean), []))
        candidates.update(s3_indexes["exact_name"].get((country, name_clean), []))

        # 2. Name No Suffix
        candidates.update(s2_indexes["name_no_suffix"].get((country, name_no_suf), []))
        candidates.update(s3_indexes["name_no_suffix"].get((country, name_no_suf), []))

        # 3. Spaceless Name
        if spaceless_name and len(spaceless_name) >= 5:
            candidates.update(s2_indexes["spaceless_name"].get((country, spaceless_name), []))
            candidates.update(s3_indexes["spaceless_name"].get((country, spaceless_name), []))

        # 4. Exact Address
        if addr_norm and len(addr_norm) >= 5:
            candidates.update(s2_indexes["exact_address"].get((country, addr_norm), []))
            candidates.update(s3_indexes["exact_address"].get((country, addr_norm), []))

        # 5. Addr Num + ANY Name Token
        if dist_tokens and addr_nums:
            for num in addr_nums:
                for tok in dist_tokens:
                    candidates.update(s2_indexes["addr_num_any_name_tok"].get((country, num, tok), []))
                    candidates.update(s3_indexes["addr_num_any_name_tok"].get((country, num, tok), []))

        # 6. Addr Num + Street Token
        if street_tokens and addr_nums:
            for num in addr_nums:
                for s_tok in street_tokens:
                    candidates.update(s2_indexes["addr_num_street_tok"].get((country, num, s_tok), []))
                    candidates.update(s3_indexes["addr_num_street_tok"].get((country, num, s_tok), []))

        # 7. First Two Name Tokens
        if len(core_tokens) >= 2:
            candidates.update(s2_indexes["name_two_tok"].get((country, core_tokens[0], core_tokens[1]), []))
            candidates.update(s3_indexes["name_two_tok"].get((country, core_tokens[0], core_tokens[1]), []))

        # 8. Distinctive First Token
        if dist_tokens and len(dist_tokens[0]) >= 5:
            first_tok = dist_tokens[0]
            candidates.update(s2_indexes["distinctive_first_tok"].get((country, first_tok), []))
            candidates.update(s3_indexes["distinctive_first_tok"].get((country, first_tok), []))

        s1_candidates_map[s1_id] = candidates

        recovered = true_matches.intersection(candidates)
        recovered_matches += len(recovered)

        missed = true_matches - candidates
        for m_id in missed:
            missed_pairs_ids.append((s1_id, m_id))

    missed_count = len(missed_pairs_ids)
    recall = (recovered_matches / total_true_matches * 100) if total_true_matches else 0

    print(f"Total True Matches   : {total_true_matches:,}")
    print(f"Recovered Matches    : {recovered_matches:,} ({recall:.2f}%)")
    print(f"Missed Matches       : {missed_count:,} ({100 - recall:.2f}%)")

    # 5. Extract only the needed matched records from S2 and S3
    print("\n4. Loading raw details of missed candidates...", flush=True)
    target_s2_ids = {m_id for _, m_id in missed_pairs_ids if m_id.startswith("S2-")}
    target_s3_ids = {m_id for _, m_id in missed_pairs_ids if m_id.startswith("S3-")}

    s2_records = load_candidate_records_by_ids(S2_FILE, target_s2_ids)
    s3_records = load_candidate_records_by_ids(S3_FILE, target_s3_ids)
    all_matched_records = {**s2_records, **s3_records}

    # 6. Build detailed missed-matches dataset
    print("\n5. Classifying missed pairs and computing diagnostics...", flush=True)
    detailed_rows = []
    category_counts = Counter()
    posting_limit_losses = []

    for s1_id, m_id in missed_pairs_ids:
        s1_row = s1_dict.get(s1_id, {})
        m_row = all_matched_records.get(m_id, {
            "entity_id": m_id,
            "business_name": "",
            "business_address": "",
            "country": "",
        })

        flags, p_details = classify_missed_pair(s1_row, m_row, s2_overflow, s3_overflow)
        for f in flags:
            category_counts[f] += 1

        if p_details:
            for strat, key, cnt in p_details:
                posting_limit_losses.append({
                    "strategy": strat,
                    "blocking_key": str(key),
                    "posting_list_size": cnt,
                    "source1_entity_id": s1_id,
                    "matched_entity_id": m_id,
                    "source1_name": s1_row.get("business_name", ""),
                    "matched_name": m_row.get("business_name", ""),
                })

        s1_name_clean = normalize_name_clean(s1_row.get("business_name", ""))
        m_name_clean = normalize_name_clean(m_row.get("business_name", ""))
        s1_addr_norm = normalize_address(s1_row.get("business_address", ""))
        m_addr_norm = normalize_address(m_row.get("business_address", ""))

        s1_nums = sorted(extract_address_numbers(s1_addr_norm))
        m_nums = sorted(extract_address_numbers(m_addr_norm))

        name_exact = int(str(s1_row.get("business_name", "")).strip().lower() == str(m_row.get("business_name", "")).strip().lower())
        addr_exact = int(str(s1_row.get("business_address", "")).strip().lower() == str(m_row.get("business_address", "")).strip().lower())
        country_match = int(s1_row.get("country", "") == m_row.get("country", "") and bool(s1_row.get("country", "")))

        detailed_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_id": m_id,
            "source1_name": s1_row.get("business_name", ""),
            "matched_name": m_row.get("business_name", ""),
            "source1_address": s1_row.get("business_address", ""),
            "matched_address": m_row.get("business_address", ""),
            "source1_country": s1_row.get("country", ""),
            "matched_country": m_row.get("country", ""),
            "name_exact": name_exact,
            "normalized_name_source1": s1_name_clean,
            "normalized_name_match": m_name_clean,
            "address_exact": addr_exact,
            "normalized_address_source1": s1_addr_norm,
            "normalized_address_match": m_addr_norm,
            "house_number_source1": ",".join(s1_nums),
            "house_number_match": ",".join(m_nums),
            "country_match": country_match,
            "failure_categories": "|".join(flags),
        })

    # Save to TSV
    df_missed = pd.DataFrame(detailed_rows)
    df_missed.to_csv(MISSED_MATCHES_V2_TSV, sep="\t", index=False)
    df_missed.to_csv(OUTPUT_DIR_SINGULAR / "blocking_missed_matches_v2.tsv", sep="\t", index=False)
    print(f"Saved {len(df_missed):,} missed matches to {MISSED_MATCHES_V2_TSV}")

    # 7. Generate diagnostic text report
    print("\n6. Generating diagnostic report V2...", flush=True)
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MISSED MATCH ANALYSIS REPORT (V2)")
    report_lines.append("=" * 80)
    report_lines.append(f"Source-1 entities evaluated : {len(s1_dict):,}")
    report_lines.append(f"Total True Matches          : {total_true_matches:,}")
    report_lines.append(f"Recovered Matches           : {recovered_matches:,} ({recall:.2f}%)")
    report_lines.append(f"Missed Matches              : {missed_count:,} ({100 - recall:.2f}%)")
    report_lines.append(f"Blocking Recall             : {recall:.4f}%")
    report_lines.append("-" * 80)
    report_lines.append("\nMISSED-MATCH CATEGORIES BREAKDOWN:")
    report_lines.append(f"{'Category':38s}{'Count':>12s}{'Pct of Missed':>18s}")
    report_lines.append("-" * 80)

    for cat, cnt in category_counts.most_common():
        pct = (cnt / missed_count * 100) if missed_count else 0
        report_lines.append(f"{cat:38s}{cnt:>12,}{pct:>17.2f}%")

    report_lines.append("-" * 80)
    report_lines.append(f"\nPOSTING-LIMIT (MAX_POSTINGS={MAX_POSTINGS}) LOSS INVESTIGATION:")
    report_lines.append(f"Total true matches lost due to posting cap: {len(posting_limit_losses):,} ({len(posting_limit_losses)/missed_count*100:.2f}% of missed)")

    # 8. Target Examples Verification Section
    report_lines.append("\n" + "=" * 80)
    report_lines.append("TARGET VALIDATION CASES CHECK:")
    report_lines.append("=" * 80)

    target_checks = [
        ("S1-925783039", "S3-698172821", "Orelee's Barbershop vs Orelee'S Services"),
        ("S1-773889195", "S3-202893869", "Prime Money vs primemoney.com"),
        ("S1-773889195", "S2-970528089", "Prime Money vs @primemoney"),
        ("S1-500841209", "S3-61034637", "Bautista Properties LLC vs bautistaproperties.com"),
    ]

    for s1_id, m_id, desc in target_checks:
        cands = s1_candidates_map.get(s1_id, set())
        recovered = m_id in cands
        status_str = "RECOVERED (SUCCESS)" if recovered else "MISSED (FAIL)"
        report_lines.append(f"  [{status_str}] {desc} ({s1_id} -> {m_id})")

    report_lines.append("\n" + "=" * 80)
    report_lines.append("REPRESENTATIVE MISSED MATCH EXAMPLES (Top 35 Cases)")
    report_lines.append("=" * 80)

    seen_cats = set()
    sample_examples = []
    for row in detailed_rows:
        cats = set(row["failure_categories"].split("|"))
        new_cats = cats - seen_cats
        if new_cats or len(sample_examples) < 35:
            seen_cats.update(cats)
            sample_examples.append(row)
        if len(sample_examples) >= 40:
            break

    for idx, ex in enumerate(sample_examples[:35], 1):
        report_lines.append(f"\n[Example {idx}] Categories: {ex['failure_categories']}")
        report_lines.append(f"  Source 1 ({ex['source1_entity_id']}):")
        report_lines.append(f"    Name   : {ex['source1_name']}")
        report_lines.append(f"    Address: {ex['source1_address']}")
        report_lines.append(f"    Country: {ex['source1_country']}")
        report_lines.append(f"  Match ({ex['matched_entity_id']}):")
        report_lines.append(f"    Name   : {ex['matched_name']}")
        report_lines.append(f"    Address: {ex['matched_address']}")
        report_lines.append(f"    Country: {ex['matched_country']}")
        report_lines.append(f"  House Numbers: S1=[{ex['house_number_source1']}] Match=[{ex['house_number_match']}]")
        report_lines.append(f"  Country Match: {bool(ex['country_match'])}")

    report_content = "\n".join(report_lines)

    with open(ANALYSIS_V2_TXT, "w", encoding="utf-8") as f:
        f.write(report_content)
    with open(OUTPUT_DIR_SINGULAR / "blocking_missed_match_analysis_v2.txt", "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Saved diagnostic report to {ANALYSIS_V2_TXT}")
    print("\n" + report_content)


if __name__ == "__main__":
    main()