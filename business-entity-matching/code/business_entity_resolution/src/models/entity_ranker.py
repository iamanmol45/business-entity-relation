"""
Entity Ranking and Matching Model using LightGBM.

Trains a gradient boosted decision tree classifier to predict match probabilities,
calibrates decision thresholds to maximize entity-level F1 score,
and supports multi-candidate match predictions.
"""

import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
import lightgbm as lgb


class EntityRanker:
    """LightGBM-based pairwise entity ranker and multi-match calibrator."""

    def __init__(
        self,
        feature_names: List[str],
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        max_depth: int = 8,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        class_weight: Optional[str] = None,
        optimal_threshold: float = 0.50
    ):
        self.feature_names = feature_names
        self.optimal_threshold = optimal_threshold
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            subsample_freq=1,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
            class_weight=class_weight,
            importance_type="gain",
            verbosity=-1
        )
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 30
    ) -> "EntityRanker":
        """Fit the LightGBM classifier with optional early stopping."""
        eval_set = [(X_train, y_train)]
        eval_names = ["train"]

        callbacks = []
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            eval_names.append("val")
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_names=eval_names,
            callbacks=callbacks
        )
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return match probability for candidate pairs."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict_proba.")
        return self.model.predict_proba(X)[:, 1].astype(np.float32)

    def calibrate_threshold(
        self,
        val_pairs: List[Tuple[str, str]],
        val_probas: np.ndarray,
        val_s1_ids: List[str],
        ground_truth: Dict[str, Set[str]],
        thresholds: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calibrate decision threshold to maximize entity-level F1 score on validation set.
        Supports 0, 1, or multiple matches per Source 1 entity.
        """
        if thresholds is None:
            thresholds = np.arange(0.10, 0.92, 0.02)

        # Map predictions per s1_id: s1_id -> list of (cand_id, proba)
        s1_to_cand_probas: Dict[str, List[Tuple[str, float]]] = {s1: [] for s1 in val_s1_ids}
        for (s1_id, cand_id), proba in zip(val_pairs, val_probas):
            if s1_id in s1_to_cand_probas:
                s1_to_cand_probas[s1_id].append((cand_id, float(proba)))

        best_threshold = 0.50
        best_f1 = -1.0
        best_metrics = {}

        for t in thresholds:
            total_tp = 0
            total_fp = 0
            total_fn = 0

            for s1_id in val_s1_ids:
                true_set = ground_truth.get(s1_id, set())
                cand_list = s1_to_cand_probas.get(s1_id, [])
                pred_set = {cand_id for cand_id, p in cand_list if p >= t}

                tp = len(pred_set.intersection(true_set))
                fp = len(pred_set - true_set)
                fn = len(true_set - pred_set)

                total_tp += tp
                total_fp += fp
                total_fn += fn

            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(t)
                best_metrics = {
                    "threshold": float(t),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "true_positives": total_tp,
                    "false_positives": total_fp,
                    "false_negatives": total_fn,
                }

        self.optimal_threshold = best_threshold
        return best_threshold, best_metrics

    def predict_entity_matches(
        self,
        pairs: List[Tuple[str, str]],
        probas: np.ndarray,
        all_s1_ids: List[str],
        threshold: Optional[float] = None
    ) -> Dict[str, List[str]]:
        """
        Generate final matched entity predictions for all Source 1 entities.
        Returns dict: s1_id -> list of matched candidate IDs.
        """
        t = threshold if threshold is not None else self.optimal_threshold
        predictions: Dict[str, List[str]] = {s1: [] for s1 in all_s1_ids}

        for (s1_id, cand_id), p in zip(pairs, probas):
            if p >= t and s1_id in predictions:
                predictions[s1_id].append(cand_id)

        return predictions

    def get_feature_importance_df(self) -> pd.DataFrame:
        """Extract and format feature importances."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance.")
        
        gain_imp = self.model.feature_importances_
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance_gain": gain_imp
        }).sort_values(by="importance_gain", ascending=False).reset_index(drop=True)
        
        total_gain = df["importance_gain"].sum()
        df["importance_pct"] = (df["importance_gain"] / total_gain * 100) if total_gain > 0 else 0.0
        return df

    def save(self, file_path: Path) -> None:
        """Save ranker model and metadata locally."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "optimal_threshold": self.optimal_threshold,
            "is_fitted": self.is_fitted
        }
        joblib.dump(data, file_path)

    @classmethod
    def load(cls, file_path: Path) -> "EntityRanker":
        """Load ranker model and metadata from local path."""
        data = joblib.load(file_path)
        instance = cls(feature_names=data["feature_names"])
        instance.model = data["model"]
        instance.optimal_threshold = data["optimal_threshold"]
        instance.is_fitted = data["is_fitted"]
        return instance
