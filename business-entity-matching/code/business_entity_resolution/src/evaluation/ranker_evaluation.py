"""
Comprehensive Evaluation Suite for Business Entity Ranking and Resolution.

Calculates pair-level metrics, entity-level set metrics (micro/macro precision,
recall, F1), exact match accuracy, and per-category performance breakdowns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Any
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix
)
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = SRC_DIR.parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from business_entity_resolution.src.features.feature_extractor import ProcessedEntity
except ImportError:
    from features.feature_extractor import ProcessedEntity



def evaluate_ranking_performance(
    val_s1_ids: List[str],
    val_pairs: List[Tuple[str, str]],
    val_probas: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    s1_entities_map: Dict[str, ProcessedEntity],
    cand_entities_map: Dict[str, ProcessedEntity],
    threshold: float = 0.50
) -> Dict[str, Any]:
    """Compute exhaustive pairwise, entity-level, and per-category evaluation metrics."""

    # 1. Pairwise Evaluation
    y_true_pairs = np.array([
        1 if cand_id in ground_truth.get(s1_id, set()) else 0
        for s1_id, cand_id in val_pairs
    ], dtype=np.int32)

    y_pred_pairs = (val_probas >= threshold).astype(np.int32)

    pair_prec = float(precision_score(y_true_pairs, y_pred_pairs, zero_division=0))
    pair_rec = float(recall_score(y_true_pairs, y_pred_pairs, zero_division=0))
    pair_f1 = float(f1_score(y_true_pairs, y_pred_pairs, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true_pairs, val_probas))
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true_pairs, val_probas))
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y_true_pairs, y_pred_pairs, labels=[0, 1])
    pair_tn, pair_fp, pair_fn, pair_tp = cm.ravel()

    # 2. Entity-level Set Evaluation
    s1_to_preds: Dict[str, Set[str]] = {s1: set() for s1 in val_s1_ids}
    for (s1_id, cand_id), p in zip(val_pairs, val_probas):
        if p >= threshold and s1_id in s1_to_preds:
            s1_to_preds[s1_id].add(cand_id)

    total_gt_matches = sum(len(ground_truth.get(s1, set())) for s1 in val_s1_ids)
    total_pred_matches = sum(len(preds) for preds in s1_to_preds.values())
    total_tp = 0
    total_fp = 0
    total_fn = 0

    macro_precs = []
    macro_recs = []
    macro_f1s = []
    exact_matches = 0

    # Per-category accumulators
    cat_metrics = {
        "country_us": {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "preds": 0, "s1_count": 0},
        "country_india": {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "preds": 0, "s1_count": 0},
        "source_s2": {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "preds": 0},
        "source_s3": {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "preds": 0},
        "unicode_variation": {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "preds": 0, "s1_count": 0},
        "standard_ascii": {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "preds": 0, "s1_count": 0},
        "cardinality_zero": {"exact": 0, "total": 0},
        "cardinality_single": {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "preds": 0, "total": 0},
        "cardinality_multi": {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "preds": 0, "total": 0},
    }

    for s1_id in val_s1_ids:
        true_set = ground_truth.get(s1_id, set())
        pred_set = s1_to_preds.get(s1_id, set())

        tp = len(pred_set.intersection(true_set))
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        if pred_set == true_set:
            exact_matches += 1

        # Macro precision/recall/F1 for this entity
        p_i = tp / len(pred_set) if len(pred_set) > 0 else (1.0 if len(true_set) == 0 else 0.0)
        r_i = tp / len(true_set) if len(true_set) > 0 else 1.0
        f_i = (2 * p_i * r_i) / (p_i + r_i) if (p_i + r_i) > 0 else 0.0

        macro_precs.append(p_i)
        macro_recs.append(r_i)
        macro_f1s.append(f_i)

        # Context features
        s1_ent = s1_entities_map.get(s1_id)
        country = s1_ent.country.upper() if s1_ent else ""
        is_unicode = s1_ent.is_unicode_non_ascii if s1_ent else False

        # Country category
        if country == "US":
            c_key = "country_us"
        elif country in ("INDIA", "IN"):
            c_key = "country_india"
        else:
            c_key = None

        if c_key:
            cat_metrics[c_key]["tp"] += tp
            cat_metrics[c_key]["fp"] += fp
            cat_metrics[c_key]["fn"] += fn
            cat_metrics[c_key]["gt"] += len(true_set)
            cat_metrics[c_key]["preds"] += len(pred_set)
            cat_metrics[c_key]["s1_count"] += 1

        # Unicode category
        u_key = "unicode_variation" if is_unicode else "standard_ascii"
        cat_metrics[u_key]["tp"] += tp
        cat_metrics[u_key]["fp"] += fp
        cat_metrics[u_key]["fn"] += fn
        cat_metrics[u_key]["gt"] += len(true_set)
        cat_metrics[u_key]["preds"] += len(pred_set)
        cat_metrics[u_key]["s1_count"] += 1

        # Cardinality category
        gt_len = len(true_set)
        if gt_len == 0:
            cat_metrics["cardinality_zero"]["total"] += 1
            if len(pred_set) == 0:
                cat_metrics["cardinality_zero"]["exact"] += 1
        elif gt_len == 1:
            cat_metrics["cardinality_single"]["total"] += 1
            cat_metrics["cardinality_single"]["tp"] += tp
            cat_metrics["cardinality_single"]["fp"] += fp
            cat_metrics["cardinality_single"]["fn"] += fn
            cat_metrics["cardinality_single"]["gt"] += gt_len
            cat_metrics["cardinality_single"]["preds"] += len(pred_set)
        else:
            cat_metrics["cardinality_multi"]["total"] += 1
            cat_metrics["cardinality_multi"]["tp"] += tp
            cat_metrics["cardinality_multi"]["fp"] += fp
            cat_metrics["cardinality_multi"]["fn"] += fn
            cat_metrics["cardinality_multi"]["gt"] += gt_len
            cat_metrics["cardinality_multi"]["preds"] += len(pred_set)

        # Source breakdown (S2 vs S3)
        for cand_id in true_set:
            if cand_id.startswith("S2"):
                cat_metrics["source_s2"]["gt"] += 1
                if cand_id in pred_set:
                    cat_metrics["source_s2"]["tp"] += 1
                else:
                    cat_metrics["source_s2"]["fn"] += 1
            elif cand_id.startswith("S3"):
                cat_metrics["source_s3"]["gt"] += 1
                if cand_id in pred_set:
                    cat_metrics["source_s3"]["tp"] += 1
                else:
                    cat_metrics["source_s3"]["fn"] += 1

        for cand_id in pred_set - true_set:
            if cand_id.startswith("S2"):
                cat_metrics["source_s2"]["fp"] += 1
                cat_metrics["source_s2"]["preds"] += 1
            elif cand_id.startswith("S3"):
                cat_metrics["source_s3"]["fp"] += 1
                cat_metrics["source_s3"]["preds"] += 1

    # Aggregate Micro & Macro
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec) / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    macro_prec_val = float(np.mean(macro_precs))
    macro_rec_val = float(np.mean(macro_recs))
    macro_f1_val = float(np.mean(macro_f1s))
    exact_match_ratio = exact_matches / len(val_s1_ids) if len(val_s1_ids) > 0 else 0.0

    # Format per-category summary
    def calc_p_r_f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        return p, r, f

    breakdown_results = {}
    for k in ["country_us", "country_india", "source_s2", "source_s3", "unicode_variation", "standard_ascii", "cardinality_single", "cardinality_multi"]:
        d = cat_metrics[k]
        p, r, f = calc_p_r_f1(d["tp"], d["fp"], d["fn"])
        breakdown_results[k] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "tp": d["tp"],
            "fp": d["fp"],
            "fn": d["fn"],
            "gt": d["gt"],
            "count": d.get("total", d.get("s1_count", 0))
        }

    return {
        "threshold": threshold,
        "pairwise": {
            "precision": pair_prec,
            "recall": pair_rec,
            "f1": pair_f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "tp": int(pair_tp),
            "fp": int(pair_fp),
            "fn": int(pair_fn),
            "tn": int(pair_tn),
        },
        "entity_level": {
            "micro_precision": micro_prec,
            "micro_recall": micro_rec,
            "micro_f1": micro_f1,
            "macro_precision": macro_prec_val,
            "macro_recall": macro_rec_val,
            "macro_f1": macro_f1_val,
            "exact_match_ratio": exact_match_ratio,
            "total_s1_evaluated": len(val_s1_ids),
            "total_true_matches": total_gt_matches,
            "total_predicted_matches": total_pred_matches,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
        },
        "per_category": breakdown_results
    }


def format_evaluation_report(results: Dict[str, Any]) -> str:
    """Format full results into structured text output."""
    pw = results["pairwise"]
    el = results["entity_level"]
    cat = results["per_category"]
    t = results["threshold"]

    lines = []
    lines.append("=" * 80)
    lines.append("BUSINESS ENTITY RESOLUTION - CANDIDATE RANKING & MATCHING BENCHMARK")
    lines.append("=" * 80)
    lines.append(f"Optimal Calibrated Threshold : {t:.4f}")
    lines.append(f"Total S1 Entities Evaluated  : {el['total_s1_evaluated']:,}")
    lines.append(f"Total True Matches in GT     : {el['total_true_matches']:,}")
    lines.append(f"Total Predicted Matches      : {el['total_predicted_matches']:,}")
    lines.append("-" * 80)

    lines.append("\n1. ENTITY-LEVEL SET MATCHING METRICS (PRIMARY BENCHMARK)")
    lines.append("-" * 80)
    lines.append(f"{'Metric':<30s}{'Micro Value':>15s}{'Macro Value':>15s}")
    lines.append("-" * 80)
    lines.append(f"{'Precision':<30s}{el['micro_precision']*100:>14.2f}%{el['macro_precision']*100:>14.2f}%")
    lines.append(f"{'Recall':<30s}{el['micro_recall']*100:>14.2f}%{el['macro_recall']*100:>14.2f}%")
    lines.append(f"{'F1-Score':<30s}{el['micro_f1']*100:>14.2f}%{el['macro_f1']*100:>14.2f}%")
    lines.append(f"{'Exact Match Set Accuracy':<30s}{el['exact_match_ratio']*100:>14.2f}%{'--':>15s}")
    lines.append(f"{'True Positive Matches':<30s}{el['true_positives']:>15,}")
    lines.append(f"{'False Positive Matches':<30s}{el['false_positives']:>15,}")
    lines.append(f"{'False Negative Matches':<30s}{el['false_negatives']:>15,}")

    lines.append("\n2. PAIRWISE CLASSIFICATION METRICS (CANDIDATE POOL)")
    lines.append("-" * 80)
    lines.append(f"{'Pair Precision':<30s}: {pw['precision']*100:>8.2f}%")
    lines.append(f"{'Pair Recall':<30s}: {pw['recall']*100:>8.2f}%")
    lines.append(f"{'Pair F1 Score':<30s}: {pw['f1']*100:>8.2f}%")
    lines.append(f"{'ROC-AUC Score':<30s}: {pw['roc_auc']:>8.4f}")
    lines.append(f"{'PR-AUC (Average Precision)':<30s}: {pw['pr_auc']:>8.4f}")
    lines.append(f"{'Pair Confusion Matrix':<30s}: TP={pw['tp']:,}, FP={pw['fp']:,}, FN={pw['fn']:,}, TN={pw['tn']:,}")

    lines.append("\n3. PER-CATEGORY PERFORMANCE BREAKDOWN")
    lines.append("-" * 80)
    lines.append(f"{'Category':<25s}{'Count / GT':>15s}{'Precision':>13s}{'Recall':>13s}{'F1-Score':>13s}")
    lines.append("-" * 80)

    category_labels = [
        ("country_us", "Country: US"),
        ("country_india", "Country: India"),
        ("source_s2", "Source: S2 Matches"),
        ("source_s3", "Source: S3 Matches"),
        ("unicode_variation", "Script: Translit/Unicode"),
        ("standard_ascii", "Script: Standard ASCII"),
        ("cardinality_single", "Cardinality: 1 Match"),
        ("cardinality_multi", "Cardinality: 2+ Matches"),
    ]

    for key, label in category_labels:
        if key in cat:
            d = cat[key]
            lines.append(
                f"{label:<25s}"
                f"{d['gt']:>15,}"
                f"{d['precision']*100:>12.2f}%"
                f"{d['recall']*100:>12.2f}%"
                f"{d['f1']*100:>12.2f}%"
            )

    lines.append("=" * 80)
    return "\n".join(lines)
