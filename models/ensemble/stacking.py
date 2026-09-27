"""
stacking.py
===========
Implements advanced meta-features extraction and Stratified K-Fold Out-of-Fold (OOF)
training for Stacking Ensemble Classifiers.

Author  : Antigravity
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import joblib
import numpy as np

# Make config importable
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG
from utils import get_logger

logger = get_logger(__name__)

def prepare_stacking_features(probs: Dict[str, np.ndarray], texts: List[str] = None) -> np.ndarray:
    """
    Format base prediction probabilities and calculate advanced statistical & structural meta-features.
    If texts list is provided, joins text length, sentence count, average sentence length,
    lexical diversity, and sentence burstiness metrics.

    Returns:
        Feature matrix (N, 18).
    """
    models = ["roberta", "deberta", "distilbert", "xgboost"]

    # 1. Base probabilities (positive class AI) with robust shape normalization
    probs_clean = {}
    for m in models:
        if m not in probs:
            raise KeyError(f"Stacking features require probability arrays for model '{m}'.")
        arr = np.asarray(probs[m])
        if arr.ndim == 1:
            if arr.shape[0] == 2:
                arr = arr.reshape(1, 2)
            else:
                arr = arr.reshape(-1, 1)
        probs_clean[m] = arr

    n_samples = probs_clean["roberta"].shape[0]
    base_probs = np.zeros((n_samples, 4))
    for idx, m in enumerate(models):
        arr = probs_clean[m]
        if arr.ndim == 2 and arr.shape[1] == 2:
            base_probs[:, idx] = arr[:, 1]
        elif arr.ndim == 2 and arr.shape[1] == 1:
            base_probs[:, idx] = arr[:, 0]
        else:
            base_probs[:, idx] = arr.ravel()

    # 2. Compute statistical meta-features
    prob_mean = np.mean(base_probs, axis=1, keepdims=True)
    prob_std = np.std(base_probs, axis=1, keepdims=True)
    prob_max = np.max(base_probs, axis=1, keepdims=True)
    prob_min = np.min(base_probs, axis=1, keepdims=True)
    prob_range = prob_max - prob_min
    prob_variance = np.var(base_probs, axis=1, keepdims=True)

    # Shannon entropy of the average prediction probability
    epsilon = 1e-15
    p_avg = np.clip(prob_mean, epsilon, 1.0 - epsilon)
    entropy = -p_avg * np.log2(p_avg) - (1.0 - p_avg) * np.log2(1.0 - p_avg)

    # Agreement count: count of models where prob > 0.5
    agreement_count = np.sum(base_probs > 0.5, axis=1, keepdims=True)

    # Distance between top two probabilities
    sorted_probs = np.sort(base_probs, axis=1)
    confidence_spread = (sorted_probs[:, -1] - sorted_probs[:, -2]).reshape(-1, 1)

    features = [
        base_probs,
        prob_mean,
        prob_std,
        prob_max,
        prob_min,
        prob_range,
        prob_variance,
        entropy,
        agreement_count,
        confidence_spread
    ]

    # 3. Document-level structural meta-features
    if texts is not None:
        if isinstance(texts, str):
            texts = [texts]
        try:
            try:
                from feature_engineering.utils import extract_features_batch
                hc_features = extract_features_batch(texts)
            except (ImportError, Exception):
                import re
                hc_features = []
                for t in texts:
                    t_str = str(t)
                    words = re.findall(r'\b\w+\b', t_str)
                    sents = [s for s in re.split(r'[.!?]+', t_str) if s.strip()]
                    w_count = len(words)
                    s_count = len(sents) if len(sents) > 0 else (1 if w_count > 0 else 0)
                    avg_slen = w_count / s_count if s_count > 0 else 0
                    u_words = set(w.lower() for w in words)
                    ttr = len(u_words) / w_count if w_count > 0 else 0
                    lens = [len(re.findall(r'\b\w+\b', s)) for s in sents] if sents else [0]
                    burstiness = float(np.std(lens)) if len(lens) > 1 else 0.0
                    hc_features.append({
                        "char_count": len(t_str),
                        "sentence_count": s_count,
                        "avg_sentence_length": avg_slen,
                        "ttr": ttr,
                        "sentence_length_burstiness": burstiness
                    })

            hc_array = np.zeros((n_samples, 5))
            for i in range(min(n_samples, len(hc_features))):
                f = hc_features[i]
                char_count = f.get("char_count", 0)
                sent_count = f.get("sentence_count", f.get("sent_count", 0))
                avg_sent_len = f.get("avg_sentence_length", f.get("avg_sent_len", 0))
                ttr = f.get("ttr", 0)
                burstiness = f.get("sentence_length_burstiness", f.get("burstiness", 0))

                hc_array[i, 0] = float(np.clip(char_count / 1000.0, 0.0, 3.5))       # 13: norm_char_count
                hc_array[i, 1] = float(np.clip(sent_count / 10.0, 0.0, 3.5))        # 14: norm_sent_count
                hc_array[i, 2] = float(np.clip(avg_sent_len / 20.0, 0.0, 3.5))      # 15: norm_avg_sent_len
                hc_array[i, 3] = float(ttr)                                          # 16: ttr
                hc_array[i, 4] = float(burstiness)                                   # 17: burstiness
            features.append(hc_array)
        except Exception as e:
            logger.warning(f"Failed to extract document meta-features during stacking: {e}. Filling with zeros.")
            features.append(np.zeros((n_samples, 5)))
    else:
        # Fill with zeros for shape consistency
        features.append(np.zeros((n_samples, 5)))

    return np.hstack(features)


class StackingEnsembleMetaClassifier:
    """Wrapper managing Stratified K-Fold cross-validation meta-model training and predictions."""
    def __init__(self, classifier_type: str = "logistic_regression", n_splits: int = 5):
        self.classifier_type = classifier_type
        self.n_splits = n_splits
        self.models = []
        self.best_threshold = 0.5
        self.coef_ = None
        self.intercept_ = None

    def fit(self, val_probs: Dict[str, np.ndarray], y_val: np.ndarray, texts: List[str] = None) -> "StackingEnsembleMetaClassifier":
        from sklearn.model_selection import StratifiedKFold
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        import xgboost as xgb

        seed = getattr(CFG, "seed", 42)
        X_meta = prepare_stacking_features(val_probs, texts)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=seed)

        self.models = []
        oof_probs = np.zeros(len(y_val))

        for train_idx, val_idx in skf.split(X_meta, y_val):
            X_tr, X_va = X_meta[train_idx], X_meta[val_idx]
            y_tr = y_val[train_idx]

            # Select meta-classifier
            if self.classifier_type == "logistic_regression":
                model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed)
            elif self.classifier_type == "random_forest":
                model = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=5, random_state=seed)
            elif self.classifier_type == "xgboost":
                model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, reg_alpha=1.0, reg_lambda=2.0, random_state=seed, eval_metric="logloss")
            else:
                model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=seed)

            model.fit(X_tr, y_tr)
            self.models.append(model)

            # Predict validation fold
            oof_probs[val_idx] = model.predict_proba(X_va)[:, 1]

        # Store coefficients if Logistic Regression for interpretability reporting
        if self.classifier_type == "logistic_regression":
            self.coef_ = np.mean([m.coef_ for m in self.models], axis=0)
            self.intercept_ = np.mean([m.intercept_ for m in self.models], axis=0)

        # Optimize threshold on OOF validation predictions to balance FPR vs F1
        self.best_threshold = self._optimize_threshold(y_val, oof_probs)
        return self

    def _optimize_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        best_threshold = 0.5
        best_score = -9999.0

        thresholds = np.linspace(0.1, 0.9, 81)
        for t in thresholds:
            preds = (y_prob >= t).astype(int)
            from sklearn.metrics import f1_score, confusion_matrix
            f1 = f1_score(y_true, preds, zero_division=0)
            cm = confusion_matrix(y_true, preds)
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0

            # Objective: keep False Positive Rate strictly low while maximizing F1
            if fpr <= 0.01:
                score = f1 + 2.0  # Substantial boost for 0-1% FPR
            elif fpr <= 0.02:
                score = f1 + 1.0  # Moderate boost for 1-2% FPR
            else:
                score = f1 - (fpr * 5.0)  # Heavy penalty for high FPR

            if score > best_score:
                best_score = score
                best_threshold = t

        return float(best_threshold)

    def predict_proba(self, probs: Dict[str, np.ndarray], texts: List[str] = None) -> np.ndarray:
        """Averages prediction probabilities of all K-Fold models cleanly without hard overrides."""
        X_meta = prepare_stacking_features(probs, texts)
        fold_probs = []
        for model in self.models:
            fold_probs.append(model.predict_proba(X_meta))
        return np.mean(fold_probs, axis=0)

    def predict(self, probs: Dict[str, np.ndarray], texts: List[str] = None) -> np.ndarray:
        final_probs = self.predict_proba(probs, texts)
        return (final_probs[:, 1] >= self.best_threshold).astype(int)

    def get_diagnostics(self, probs: Dict[str, np.ndarray], texts: List[str] = None) -> Dict:
        """
        Generate detailed diagnostic information for the Stacking Meta-Classifier.
        Does NOT override the learned model prediction.
        """
        X_meta = prepare_stacking_features(probs, texts)
        fold_probs = [float(model.predict_proba(X_meta)[0, 1]) for model in self.models]
        final_ai_prob = float(np.mean(fold_probs)) if fold_probs else 0.5
        final_human_prob = 1.0 - final_ai_prob

        raw_features = X_meta[0].tolist() if len(X_meta) > 0 else []
        clipped_length_features = [float(X_meta[0, 13]), float(X_meta[0, 14]), float(X_meta[0, 15])] if len(X_meta) > 0 else [0.0, 0.0, 0.0]

        agreement_count = int(X_meta[0, 11]) if len(X_meta) > 0 else 0
        prob_mean = float(X_meta[0, 4]) if len(X_meta) > 0 else 0.5
        unanimous_base_ai = (agreement_count == 4 and prob_mean >= 0.50)

        decision = "AI Generated" if final_ai_prob >= self.best_threshold else "Human Written"
        consensus_conflict = bool(unanimous_base_ai and decision == "Human Written")

        conflict_warning = None
        if consensus_conflict:
            conflict_warning = "Base-model consensus conflict: 4/4 base models predict AI while meta-classifier predicts Human."
            logger.warning(conflict_warning)

        return {
            "raw_features": raw_features,
            "clipped_length_features": clipped_length_features,
            "fold_probabilities": fold_probs,
            "final_ai_probability": final_ai_prob,
            "final_human_probability": final_human_prob,
            "best_threshold": float(self.best_threshold),
            "final_decision": decision,
            "unanimous_base_ai": unanimous_base_ai,
            "consensus_conflict": consensus_conflict,
            "conflict_warning": conflict_warning
        }


def train_meta_classifier(
    val_probs: Dict[str, np.ndarray],
    y_val: np.ndarray,
    output_path: Path,
    texts: List[str] = None,
    classifier_type: str = "logistic_regression"
) -> StackingEnsembleMetaClassifier:
    """Train a Stratified K-Fold meta-classifier on the validation set."""
    logger.info("Initializing Stacking Meta-Classifier...")
    meta_model = StackingEnsembleMetaClassifier(classifier_type=classifier_type, n_splits=5)
    meta_model.fit(val_probs, y_val, texts)

    # Save checkpoint
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(meta_model, str(output_path))
    logger.info(f"Meta-classifier saved → {output_path}")

    # Log coefficients if Logistic Regression
    if meta_model.coef_ is not None:
        logger.info("Stacking Meta-Classifier Mean Coefficients:")
        models = ["roberta", "deberta", "distilbert", "xgboost"]
        for m, coef in zip(models, meta_model.coef_[0]):
            logger.info(f"  * {m:<12} weight: {coef:.4f}")
        logger.info(f"  * Intercept    : {meta_model.intercept_[0]:.4f}")

    return meta_model


def stacking_predict(
    meta_model: StackingEnsembleMetaClassifier,
    test_probs: Dict[str, np.ndarray],
    texts: List[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict probabilities and classes using the trained meta-classifier."""
    final_probs = meta_model.predict_proba(test_probs, texts)
    preds = (final_probs[:, 1] >= meta_model.best_threshold).astype(int)
    return final_probs, preds
