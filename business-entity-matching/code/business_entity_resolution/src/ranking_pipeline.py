"""
End-to-End Business Entity Resolution Ranking & Matching Pipeline.

Executes:
1. Candidate Pair Generation (using V2 Multi-Strategy Blocking)
2. Train / Validation Split without Entity Leakage
3. High-Performance Pairwise Feature Extraction (Rapidfuzz + anyascii)
4. Tree-based Model Training (LightGBM Ranker)
5. Validation Threshold Calibration for Multi-Candidate Matching
6. Multi-Match Entity Prediction Generation
7. Detailed Evaluation (Micro/Macro F1, Per-Category Breakdowns)
8. Model, Metrics, Predictions, and Feature Importance Persistence
"""

import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Configure path so ranking_pipeline can be run from any working directory
SRC_DIR = Path(__file__).resolve().parent
CODE_DIR = SRC_DIR.parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pickle
import numpy as np
import pandas as pd

# Package Imports
from business_entity_resolution.src.features.feature_extractor import (
    ProcessedEntity,
    extract_pair_feature_vector,
    normalize_name_clean,
    normalize_name_no_suffix,
    normalize_spaceless_name,
    normalize_address,
    extract_address_numbers,
    extract_name_core_tokens,
    extract_distinctive_name_tokens,
    extract_street_tokens,
    FEATURE_NAMES,
    LEGAL_SUFFIXES,
)
from business_entity_resolution.src.models.entity_ranker import EntityRanker
from business_entity_resolution.src.evaluation.ranker_evaluation import (
    evaluate_ranking_performance,
    format_evaluation_report,
)

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = PROJECT_ROOT / "dataset" / "train"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = OUTPUTS_DIR / "cache"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

S1_FILE = DATASET_DIR / "train_source1.tsv"
S2_FILE = DATASET_DIR / "train_source2.tsv"
S3_FILE = DATASET_DIR / "train_source3.tsv"
GT_FILE = DATASET_DIR / "train_ground_truth.tsv"

MAX_POSTINGS = 500


def build_streaming_indexes_and_entities(
    file_path: Path,
    cache_dir: Optional[Path] = CACHE_DIR
) -> Tuple[Dict[str, Dict[Any, List[str]]], Dict[str, Tuple[str, str, str]]]:
    """Build V2 blocking inverted indexes and lightweight raw entity store from streaming source file with disk caching."""
    cache_file = None
    if cache_dir is not None:
        cache_file = cache_dir / f"{file_path.stem}_v2_cache.pkl"
        if cache_file.exists():
            print(f"Loading cached indexes for {file_path.name} from {cache_file.name}...", flush=True)
            t0 = time.time()
            with open(cache_file, "rb") as f:
                indexes, raw_entities = pickle.load(f)
            print(f"Loaded {len(raw_entities):,} records in {time.time()-t0:.1f}s.", flush=True)
            return indexes, raw_entities

    print(f"Indexing {file_path.name} (streaming chunks)...", flush=True)
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

    raw_entities: Dict[str, Tuple[str, str, str]] = {}
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

            raw_entities[eid] = (country, raw_name, raw_addr)

            name_clean = normalize_name_clean(raw_name)
            tokens_no_suf = [t for t in name_clean.split() if t not in LEGAL_SUFFIXES] if name_clean else []
            name_no_suf = " ".join(tokens_no_suf)
            spaceless_name = "".join(tokens_no_suf)
            addr_norm = normalize_address(raw_addr)

            # Key 1: Exact Clean Name
            if name_clean:
                k = (country, name_clean)
                lst = indexes["exact_name"][k]
                if len(lst) < MAX_POSTINGS:
                    lst.append(eid)

            # Key 2: Name No Suffix
            if name_no_suf and name_no_suf != name_clean:
                k = (country, name_no_suf)
                lst = indexes["name_no_suffix"][k]
                if len(lst) < MAX_POSTINGS:
                    lst.append(eid)

            # Key 3: Spaceless Name
            if spaceless_name and len(spaceless_name) >= 5:
                k = (country, spaceless_name)
                lst = indexes["spaceless_name"][k]
                if len(lst) < MAX_POSTINGS:
                    lst.append(eid)

            # Key 4: Exact Address
            if addr_norm and len(addr_norm) >= 5:
                k = (country, addr_norm)
                lst = indexes["exact_address"][k]
                if len(lst) < MAX_POSTINGS:
                    lst.append(eid)

            # Extract tokens
            addr_nums = extract_address_numbers(addr_norm)
            core_tokens = extract_name_core_tokens(name_clean)
            dist_tokens = extract_distinctive_name_tokens(name_clean)
            street_tokens = extract_street_tokens(addr_norm)

            # Key 5: House Number + ANY Distinctive Name Token
            if dist_tokens and addr_nums:
                for num in addr_nums:
                    for tok in dist_tokens:
                        k = (country, num, tok)
                        lst = indexes["addr_num_any_name_tok"][k]
                        if len(lst) < MAX_POSTINGS:
                            lst.append(eid)

            # Key 6: House Number + Street Token
            if street_tokens and addr_nums:
                for num in addr_nums:
                    for s_tok in street_tokens:
                        k = (country, num, s_tok)
                        lst = indexes["addr_num_street_tok"][k]
                        if len(lst) < MAX_POSTINGS:
                            lst.append(eid)

            # Key 7: First Two Name Tokens
            if len(core_tokens) >= 2:
                k = (country, core_tokens[0], core_tokens[1])
                lst = indexes["name_two_tok"][k]
                if len(lst) < MAX_POSTINGS:
                    lst.append(eid)

            # Key 8: Distinctive Single First Name Token
            if dist_tokens and len(dist_tokens[0]) >= 5:
                first_tok = dist_tokens[0]
                k = (country, first_tok)
                lst = indexes["distinctive_first_tok"][k]
                if len(lst) < MAX_POSTINGS:
                    lst.append(eid)

        print(f"  Processed chunk {chunk_count} ({row_count:,} rows)...", flush=True)

    print(f"Indexed {row_count:,} records in {time.time()-t0:.1f}s.", flush=True)
    if cache_file is not None:
        print(f"Saving index cache to {cache_file.name}...", flush=True)
        with open(cache_file, "wb") as f:
            pickle.dump((indexes, raw_entities), f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Cache saved successfully.", flush=True)
    return indexes, raw_entities


def load_ground_truth(s1_filter_set: Optional[Set[str]] = None) -> Dict[str, Set[str]]:
    """Load ground truth mapping source1_entity_id -> set of matched_entity_ids using ultra-fast streaming."""
    gt = {}
    target_count = len(s1_filter_set) if s1_filter_set is not None else None
    with open(GT_FILE, "r", encoding="utf-8") as f:
        header_line = f.readline()
        if not header_line:
            return gt
        headers = [h.strip() for h in header_line.rstrip("\r\n").split("\t")]
        s1_idx = headers.index("source1_entity_id")
        m_idx = headers.index("matched_entity_ids")
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) <= s1_idx:
                continue
            s1_id = parts[s1_idx]
            if s1_filter_set is not None and s1_id not in s1_filter_set:
                continue
            m_str = parts[m_idx] if len(parts) > m_idx else ""
            if not m_str.strip():
                gt[s1_id] = set()
            else:
                gt[s1_id] = {x.strip() for x in m_str.split(",") if x.strip()}
            if target_count is not None and len(gt) >= target_count:
                break
    return gt


def generate_candidates_for_s1(
    s1_entities: List[ProcessedEntity],
    s2_indexes: Dict[str, Dict[Any, List[str]]],
    s3_indexes: Dict[str, Dict[Any, List[str]]],
    gt_map: Optional[Dict[str, Set[str]]] = None
) -> List[Tuple[str, str, Dict[str, bool], int]]:
    """
    Generate candidate pairs with strategy origin tracking for each S1 entity incrementally.
    Prints progress every 500 records and per-strategy elapsed timings.
    Returns list of (s1_id, cand_id, strat_flags, s1_cand_count).
    """
    print("\nCandidate generation started", flush=True)
    t0_gen = time.time()
    all_pairs: List[Tuple[str, str, Dict[str, bool], int]] = []

    # Direct references for high-speed dict lookup
    s2_en = s2_indexes["exact_name"]
    s3_en = s3_indexes["exact_name"]
    s2_nns = s2_indexes["name_no_suffix"]
    s3_nns = s3_indexes["name_no_suffix"]
    s2_sn = s2_indexes["spaceless_name"]
    s3_sn = s3_indexes["spaceless_name"]
    s2_ea = s2_indexes["exact_address"]
    s3_ea = s3_indexes["exact_address"]
    s2_num_tok = s2_indexes["addr_num_any_name_tok"]
    s3_num_tok = s3_indexes["addr_num_any_name_tok"]
    s2_num_st = s2_indexes["addr_num_street_tok"]
    s3_num_st = s3_indexes["addr_num_street_tok"]
    s2_two = s2_indexes["name_two_tok"]
    s3_two = s3_indexes["name_two_tok"]
    s2_first = s2_indexes["distinctive_first_tok"]
    s3_first = s3_indexes["distinctive_first_tok"]

    strategy_names = [
        "exact_name",
        "name_no_suffix",
        "spaceless_name",
        "exact_address",
        "addr_num_any_name_tok",
        "addr_num_street_tok",
        "name_two_tok",
        "distinctive_first_tok",
    ]
    strategy_timings = {name: 0.0 for name in strategy_names}
    max_cands_per_s1 = 0
    total_s1 = len(s1_entities)
    total_true = 0
    recovered_true = 0

    import gc
    gc.disable()
    try:
        for idx, s1 in enumerate(s1_entities):
            country = s1.country
            name_clean = s1.name_clean
            name_no_suf = s1.name_no_suf
            spaceless_name = s1.spaceless_name
            addr_norm = s1.addr_norm

            addr_nums = s1.addr_nums
            core_tokens = extract_name_core_tokens(name_clean)
            dist_tokens = extract_distinctive_name_tokens(name_clean)
            street_tokens = s1.street_tokens

            # Strategy 1: Exact Clean Name
            t_s = time.perf_counter()
            s_en = set()
            if name_clean:
                k = (country, name_clean)
                if k in s2_en: s_en.update(s2_en[k])
                if k in s3_en: s_en.update(s3_en[k])
            strategy_timings["exact_name"] += time.perf_counter() - t_s

            # Strategy 2: Name No Suffix
            t_s = time.perf_counter()
            s_nns = set()
            if name_no_suf and name_no_suf != name_clean:
                k = (country, name_no_suf)
                if k in s2_nns: s_nns.update(s2_nns[k])
                if k in s3_nns: s_nns.update(s3_nns[k])
            strategy_timings["name_no_suffix"] += time.perf_counter() - t_s

            # Strategy 3: Spaceless Name
            t_s = time.perf_counter()
            s_sn = set()
            if spaceless_name and len(spaceless_name) >= 5:
                k = (country, spaceless_name)
                if k in s2_sn: s_sn.update(s2_sn[k])
                if k in s3_sn: s_sn.update(s3_sn[k])
            strategy_timings["spaceless_name"] += time.perf_counter() - t_s

            # Strategy 4: Exact Address
            t_s = time.perf_counter()
            s_ea = set()
            if addr_norm and len(addr_norm) >= 5:
                k = (country, addr_norm)
                if k in s2_ea: s_ea.update(s2_ea[k])
                if k in s3_ea: s_ea.update(s3_ea[k])
            strategy_timings["exact_address"] += time.perf_counter() - t_s

            # Strategy 5: Addr Num + ANY Distinctive Name Token
            t_s = time.perf_counter()
            s_num_tok = set()
            if dist_tokens and addr_nums:
                for num in addr_nums:
                    for tok in dist_tokens:
                        k = (country, num, tok)
                        if k in s2_num_tok: s_num_tok.update(s2_num_tok[k])
                        if k in s3_num_tok: s_num_tok.update(s3_num_tok[k])
            strategy_timings["addr_num_any_name_tok"] += time.perf_counter() - t_s

            # Strategy 6: Addr Num + Street Token
            t_s = time.perf_counter()
            s_num_st = set()
            if street_tokens and addr_nums:
                for num in addr_nums:
                    for s_tok in street_tokens:
                        k = (country, num, s_tok)
                        if k in s2_num_st: s_num_st.update(s2_num_st[k])
                        if k in s3_num_st: s_num_st.update(s3_num_st[k])
            strategy_timings["addr_num_street_tok"] += time.perf_counter() - t_s

            # Strategy 7: First Two Name Tokens
            t_s = time.perf_counter()
            s_two = set()
            if len(core_tokens) >= 2:
                k = (country, core_tokens[0], core_tokens[1])
                if k in s2_two: s_two.update(s2_two[k])
                if k in s3_two: s_two.update(s3_two[k])
            strategy_timings["name_two_tok"] += time.perf_counter() - t_s

            # Strategy 8: Distinctive Single First Name Token
            t_s = time.perf_counter()
            s_first = set()
            if dist_tokens and len(dist_tokens[0]) >= 5:
                first_tok = dist_tokens[0]
                k = (country, first_tok)
                if k in s2_first: s_first.update(s2_first[k])
                if k in s3_first: s_first.update(s3_first[k])
            strategy_timings["distinctive_first_tok"] += time.perf_counter() - t_s

            # Efficient Candidate Union & Origin Tracking
            all_s1_cands = s_en | s_nns | s_sn | s_ea | s_num_tok | s_num_st | s_two | s_first
            cand_count = len(all_s1_cands)
            if cand_count > max_cands_per_s1:
                max_cands_per_s1 = cand_count

            s1_eid = s1.eid
            if gt_map is not None:
                true_set = gt_map.get(s1_eid, set())
                total_true += len(true_set)
                recovered_true += len(true_set.intersection(all_s1_cands))

            for cid in all_s1_cands:
                mask = 0
                if cid in s_en: mask |= 1
                if cid in s_nns: mask |= 2
                if cid in s_sn: mask |= 4
                if cid in s_ea: mask |= 8
                if cid in s_num_tok: mask |= 16
                if cid in s_num_st: mask |= 32
                if cid in s_two: mask |= 64
                if cid in s_first: mask |= 128
                all_pairs.append((s1_eid, cid, mask, cand_count))

            # Progress reporting every 500 S1 records
            if (idx + 1) % 500 == 0 or (idx + 1) == total_s1:
                elapsed = time.time() - t0_gen
                avg_cands = len(all_pairs) / (idx + 1)
                timings_str = ", ".join(f"{k}: {v:.2f}s" for k, v in strategy_timings.items())
                print(f"\nS1 {idx + 1}/{total_s1}", flush=True)
                print(f"  candidates: {len(all_pairs):,}", flush=True)
                print(f"  avg candidates/S1: {avg_cands:.1f}", flush=True)
                print(f"  elapsed: {elapsed:.2f}s", flush=True)
                print(f"  strategy timings: {timings_str}", flush=True)
    finally:
        gc.enable()

    print("\nCandidate generation completed", flush=True)
    print(f"Total candidate pairs: {len(all_pairs):,}", flush=True)
    print(f"Average candidates/S1: {len(all_pairs) / total_s1:.1f}", flush=True)
    print(f"Maximum candidates/S1: {max_cands_per_s1:,}", flush=True)

    # Evaluate against ground truth if provided
    if gt_map is not None:
        recall = (recovered_true / total_true * 100) if total_true > 0 else 0.0
        missed_true = total_true - recovered_true
        print(f"\nTotal true matches: {total_true:,}", flush=True)
        print(f"Recovered true matches: {recovered_true:,}", flush=True)
        print(f"Blocking recall: {recall:.2f}%", flush=True)
        print(f"Missed true matches: {missed_true:,}", flush=True)

    return all_pairs


def extract_features_matrix(
    pairs: List[Tuple[str, str, Dict[str, bool], int]],
    s1_map: Dict[str, ProcessedEntity],
    cand_map: Dict[str, ProcessedEntity]
) -> np.ndarray:
    """Extract features in preallocated compact float32 matrix."""
    n_samples = len(pairs)
    n_features = len(FEATURE_NAMES)
    X = np.zeros((n_samples, n_features), dtype=np.float32)

    for i, (s1_id, cand_id, flags, cand_count) in enumerate(pairs):
        s1 = s1_map.get(s1_id)
        cand = cand_map.get(cand_id)
        if s1 is not None and cand is not None:
            extract_pair_feature_vector(s1, cand, flags, cand_count, X, i)

    return X


def main():
    parser = argparse.ArgumentParser(description="Deterministic Candidate Ranking & Entity Matching Pipeline")
    parser.add_argument("--sample_size", type=int, default=10000, help="Number of Source-1 entities to evaluate")
    parser.add_argument("--val_split", type=float, default=0.20, help="Fraction of S1 entities for validation")
    parser.add_argument("--n_estimators", type=int, default=300, help="Number of gradient boosting trees")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate")
    args = parser.parse_args()

    print("=" * 80)
    print("LOCAL DETERMINISTIC CANDIDATE RANKING & MATCHING PIPELINE")
    print("=" * 80)

    # 1. Load Source 1 Sample
    print(f"\n1. Loading Source 1 ({args.sample_size:,} rows)...", flush=True)
    s1_df = pd.read_csv(S1_FILE, sep="\t", dtype=str, nrows=args.sample_size)
    s1_entities = [
        ProcessedEntity(r["entity_id"], r["country"], r["business_name"], r["business_address"])
        for _, r in s1_df.iterrows()
    ]
    s1_map = {e.eid: e for e in s1_entities}
    all_s1_ids = [e.eid for e in s1_entities]
    print(f"Loaded {len(s1_entities):,} Source 1 entities.", flush=True)

    # 2. Load Ground Truth
    print("\n2. Loading Ground Truth...", flush=True)
    gt_map = load_ground_truth(s1_filter_set=set(all_s1_ids))
    total_true_sample = sum(len(gt_map.get(eid, set())) for eid in all_s1_ids)
    print(f"Total True Matches in Source-1 sample: {total_true_sample:,}", flush=True)

    # 3. Build S2 and S3 Indexes and Entity Maps
    print("\n3. Building Inverted Indexes for Candidate Sources...", flush=True)
    s2_indexes, s2_raw_entities = build_streaming_indexes_and_entities(S2_FILE)
    s3_indexes, s3_raw_entities = build_streaming_indexes_and_entities(S3_FILE)

    # 4. Generate Candidate Pairs
    print("\n4. Generating Candidate Pairs via V2 Blocking...", flush=True)
    all_candidate_pairs = generate_candidates_for_s1(s1_entities, s2_indexes, s3_indexes, gt_map=gt_map)

    # Precompute ProcessedEntity profiles for unique retrieved candidates
    unique_cand_ids = {p[1] for p in all_candidate_pairs}
    print(f"Instantiating entity profiles for {len(unique_cand_ids):,} unique candidates...", flush=True)
    t0 = time.time()
    cand_entities_map = {}
    for cid in unique_cand_ids:
        raw_ent = s2_raw_entities.get(cid) or s3_raw_entities.get(cid)
        if raw_ent is not None:
            c_country, c_name, c_addr = raw_ent
            cand_entities_map[cid] = ProcessedEntity(cid, c_country, c_name, c_addr)
    print(f"Candidate entity profiles created in {time.time()-t0:.2f}s.", flush=True)

    # 5. Entity-Level Train / Validation Split (Zero-Leakage)
    print(f"\n5. Performing Leak-Free Train / Validation Split ({(1-args.val_split)*100:.0f}% / {args.val_split*100:.0f}%)...", flush=True)
    np.random.seed(42)
    shuffled_s1_ids = np.random.permutation(all_s1_ids)
    n_val = int(len(all_s1_ids) * args.val_split)

    val_s1_set = set(shuffled_s1_ids[:n_val])
    train_s1_set = set(shuffled_s1_ids[n_val:])
    val_s1_ids = [eid for eid in all_s1_ids if eid in val_s1_set]
    train_s1_ids = [eid for eid in all_s1_ids if eid in train_s1_set]

    train_pairs_meta = [p for p in all_candidate_pairs if p[0] in train_s1_set]
    val_pairs_meta = [p for p in all_candidate_pairs if p[0] in val_s1_set]

    print(f"Train Entities: {len(train_s1_ids):,} ({len(train_pairs_meta):,} pairs)", flush=True)
    print(f"Val Entities  : {len(val_s1_ids):,} ({len(val_pairs_meta):,} pairs)", flush=True)

    # 6. High-Performance Feature Extraction
    print("\n6. Extracting Pairwise Features...", flush=True)
    t0 = time.time()
    X_train = extract_features_matrix(train_pairs_meta, s1_map, cand_entities_map)
    y_train = np.array([1 if p[1] in gt_map.get(p[0], set()) else 0 for p in train_pairs_meta], dtype=np.int32)

    X_val = extract_features_matrix(val_pairs_meta, s1_map, cand_entities_map)
    y_val = np.array([1 if p[1] in gt_map.get(p[0], set()) else 0 for p in val_pairs_meta], dtype=np.int32)
    print(f"Feature extraction completed in {time.time()-t0:.2f}s. (Train positives: {y_train.sum():,}, Val positives: {y_val.sum():,})", flush=True)

    # 7. Train Tree-Based Entity Ranker
    print("\n7. Training LightGBM Entity Ranker...", flush=True)
    t0 = time.time()
    ranker = EntityRanker(
        feature_names=FEATURE_NAMES,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate
    )
    ranker.fit(X_train, y_train, X_val, y_val)
    print(f"Model trained in {time.time()-t0:.2f}s.", flush=True)

    # 8. Calibrate Threshold for Multi-Candidate Matching
    print("\n8. Calibrating Decision Threshold on Validation Set...", flush=True)
    val_probas = ranker.predict_proba(X_val)
    val_pairs_tuples = [(p[0], p[1]) for p in val_pairs_meta]

    optimal_threshold, calib_metrics = ranker.calibrate_threshold(
        val_pairs=val_pairs_tuples,
        val_probas=val_probas,
        val_s1_ids=val_s1_ids,
        ground_truth=gt_map
    )
    print(f"Optimal Threshold: {optimal_threshold:.4f} (Validation Set F1: {calib_metrics['f1']*100:.2f}%)", flush=True)

    # 9. Comprehensive Evaluation
    print("\n9. Evaluating Final Model Performance...", flush=True)
    eval_results = evaluate_ranking_performance(
        val_s1_ids=val_s1_ids,
        val_pairs=val_pairs_tuples,
        val_probas=val_probas,
        ground_truth=gt_map,
        s1_entities_map=s1_map,
        cand_entities_map=cand_entities_map,
        threshold=optimal_threshold
    )

    report_text = format_evaluation_report(eval_results)
    print("\n" + report_text)

    # 10. Generate Validation Predictions
    print("\n10. Generating and Saving Validation Predictions...", flush=True)
    val_predictions = ranker.predict_entity_matches(
        pairs=val_pairs_tuples,
        probas=val_probas,
        all_s1_ids=val_s1_ids,
        threshold=optimal_threshold
    )

    pred_rows = [
        {"source1_entity_id": s1_id, "matched_entity_ids": ",".join(matches)}
        for s1_id, matches in val_predictions.items()
    ]
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(OUTPUTS_DIR / "validation_predictions.tsv", sep="\t", index=False)
    pred_df.to_csv(OUTPUT_DIR / "validation_predictions.tsv", sep="\t", index=False)

    # 11. Save Feature Importance
    print("11. Saving Feature Importance...", flush=True)
    feat_df = ranker.get_feature_importance_df()
    feat_df.to_csv(OUTPUTS_DIR / "ranking_feature_importance.csv", index=False)
    feat_df.to_csv(OUTPUT_DIR / "ranking_feature_importance.csv", index=False)
    print(f"Top 10 Features by Gain Importance:\n{feat_df.head(10).to_string(index=False)}")

    # 12. Save Trained Model
    print("\n12. Saving Trained Model...", flush=True)
    ranker.save(OUTPUTS_DIR / "trained_ranker.pkl")
    ranker.save(OUTPUT_DIR / "trained_ranker.pkl")

    # 13. Save Metrics Report
    print("13. Saving Ranking Metrics Report...", flush=True)
    (OUTPUTS_DIR / "ranking_metrics.txt").write_text(report_text, encoding="utf-8")
    (OUTPUT_DIR / "ranking_metrics.txt").write_text(report_text, encoding="utf-8")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Output files written to: {OUTPUTS_DIR.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
