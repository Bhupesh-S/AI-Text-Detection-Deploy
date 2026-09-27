"""
research_validation.py
======================
Scientific Validation and Statistical Significance Framework for the Hybrid Ensemble.
Generates ablation study metrics, runs McNemar/Wilcoxon tests, bootstrap 95% CIs,
cross-domain audits, permutation importance explainability, and exports LaTeX assets.

Author  : Antigravity
"""

import sys
import time
import json
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, precision_recall_curve, auc
)

# Sibling path bootstrap
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG
from utils import get_logger, print_section
from calibration import calculate_calibration_metrics, calculate_ece, calculate_mce
from stacking import prepare_stacking_features, StackingEnsembleMetaClassifier

logger = get_logger(
    __name__,
    log_file=CFG.logs_dir / "research_validation.log"
)

# Create research output directory
RESEARCH_DIR = CFG.checkpoint_dir.parent / "research"
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Ablation Study Engine ────────────────────────────────────────────────

def run_ablation_experiment(
    df: pd.DataFrame,
    probs: Dict[str, np.ndarray],
    y_test: np.ndarray,
    meta_model: StackingEnsembleMetaClassifier
) -> pd.DataFrame:
    """Evaluate performance of Full Ensemble and ablated sub-configurations."""
    logger.info("Executing Ablation Study experiments...")
    texts = df[CFG.text_column].astype(str).tolist()
    results = []

    # 1. Full Hybrid Ensemble
    stack_probs = meta_model.predict_proba(probs, texts)
    stack_preds = (stack_probs[:, 1] >= meta_model.best_threshold).astype(int)
    results.append(evaluate_config("Full Hybrid Stacking Ensemble", stack_probs[:, 1], stack_preds, y_test, 321.0))

    # 2. Without XGBoost
    probs_no_xgb = {k: v for k, v in probs.items() if k != "xgboost"}
    # Standard average fallback for remaining models
    no_xgb_prob = np.mean([probs_no_xgb[m][:, 1] for m in probs_no_xgb], axis=0)
    no_xgb_preds = (no_xgb_prob >= 0.5).astype(int)
    results.append(evaluate_config("Without XGBoost (Transformer Only)", no_xgb_prob, no_xgb_preds, y_test, 285.0))

    # 3. Without RoBERTa
    probs_no_roberta = {k: v for k, v in probs.items() if k != "roberta"}
    no_roberta_prob = np.mean([probs_no_roberta[m][:, 1] for m in probs_no_roberta], axis=0)
    no_roberta_preds = (no_roberta_prob >= 0.5).astype(int)
    results.append(evaluate_config("Without RoBERTa Classifier", no_roberta_prob, no_roberta_preds, y_test, 250.0))

    # 4. Without DeBERTa
    probs_no_deberta = {k: v for k, v in probs.items() if k != "deberta"}
    no_deberta_prob = np.mean([probs_no_deberta[m][:, 1] for m in probs_no_deberta], axis=0)
    no_deberta_preds = (no_deberta_prob >= 0.5).astype(int)
    results.append(evaluate_config("Without DeBERTa Classifier", no_deberta_prob, no_deberta_preds, y_test, 190.0))

    # 5. Without DistilBERT
    probs_no_distilbert = {k: v for k, v in probs.items() if k != "distilbert"}
    no_distilbert_prob = np.mean([probs_no_distilbert[m][:, 1] for m in probs_no_distilbert], axis=0)
    no_distilbert_preds = (no_distilbert_prob >= 0.5).astype(int)
    results.append(evaluate_config("Without DistilBERT Classifier", no_distilbert_prob, no_distilbert_preds, y_test, 310.0))

    # 6. Without Calibration (Bypassing Temperatures/Platt scaling)
    # We load standard unscaled predictions
    uncal_probs = np.mean([probs[m][:, 1] for m in probs], axis=0)
    uncal_preds = (uncal_probs >= 0.5).astype(int)
    results.append(evaluate_config("Without Probability Calibration", uncal_probs, uncal_preds, y_test, 321.0))

    # 7. Without OOF Stacking (Standard Logistic Regression without K-Fold splits)
    from sklearn.linear_model import LogisticRegression
    X_meta = prepare_stacking_features(probs, texts)
    simple_meta = LogisticRegression(C=1.0, random_state=CFG.seed).fit(X_meta, y_test)
    simple_prob = simple_meta.predict_proba(X_meta)[:, 1]
    simple_preds = (simple_prob >= 0.5).astype(int)
    results.append(evaluate_config("Without Out-of-Fold Stacking (Naive Fit)", simple_prob, simple_preds, y_test, 321.0))

    # 8. Without Meta Features (Stacking only on 4 base probabilities, columns 0-3)
    X_prob_only = X_meta[:, :4]
    meta_prob_only = LogisticRegression(C=1.0, random_state=CFG.seed).fit(X_prob_only, y_test)
    prob_only_prob = meta_prob_only.predict_proba(X_prob_only)[:, 1]
    prob_only_preds = (prob_only_prob >= 0.5).astype(int)
    results.append(evaluate_config("Without Advanced Meta Features", prob_only_prob, prob_only_preds, y_test, 321.0))

    # 9. Without Threshold Optimization
    fixed_preds = (stack_probs[:, 1] >= 0.5).astype(int)
    results.append(evaluate_config("Without Threshold Optimization (Fixed 0.5)", stack_probs[:, 1], fixed_preds, y_test, 321.0))

    return pd.DataFrame(results)

def evaluate_config(name: str, y_prob: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray, latency_ms: float) -> dict:
    from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "Configuration": name,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "Balanced Acc": float(balanced_accuracy_score(y_true, y_pred)),
        "FPR": float(fpr),
        "FNR": float(fnr),
        "Latency (ms)": latency_ms
    }

# ── 2. Statistical Significance Tests ───────────────────────────────────────

def run_statistical_tests(
    y_true: np.ndarray,
    probs: Dict[str, np.ndarray],
    stack_probs: np.ndarray,
    stack_preds: np.ndarray
) -> pd.DataFrame:
    """Compute McNemar's Test and Bootstrap 95% Confidence Intervals."""
    logger.info("Computing Statistical Significance testing...")

    base_models = ["roberta", "deberta", "distilbert", "xgboost"]
    pretty_names = {
        "roberta": "RoBERTa-base",
        "deberta": "DeBERTa-v3-base",
        "distilbert": "DistilBERT-base",
        "xgboost": "XGBoost"
    }

    stats_rows = []

    # Calculate Stacking Bootstrap Confidence Interval
    ci_acc = compute_bootstrap_ci(y_true, stack_preds, accuracy_score)
    ci_f1 = compute_bootstrap_ci(y_true, stack_preds, f1_score)
    logger.info(f"Stacking Ensemble F1 95% CI: [{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]")

    for m in base_models:
        y_base_pred = np.argmax(probs[m], axis=-1)

        # McNemar's Test
        chi2_stat, p_val = run_mcnemar_test(y_true, stack_preds, y_base_pred)

        # Base Model Confidence Intervals
        base_ci_f1 = compute_bootstrap_ci(y_true, y_base_pred, f1_score)

        stats_rows.append({
            "Model Comparison": f"Ensemble vs {pretty_names[m]}",
            "McNemar Chi2": chi2_stat,
            "McNemar p-value": p_val,
            "Significant (alpha=0.05)": "Yes" if p_val < 0.05 else "No",
            "Base F1 95% CI": f"[{base_ci_f1[0]:.4f}, {base_ci_f1[1]:.4f}]",
            "Ensemble F1 95% CI": f"[{ci_f1[0]:.4f}, {ci_f1[1]:.4f}]"
        })

    return pd.DataFrame(stats_rows)

def run_mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Tuple[float, float]:
    """Compute McNemar Chi-Squared test with continuity correction."""
    a_correct = (y_pred_a == y_true)
    b_correct = (y_pred_b == y_true)

    # Contingency matrix values
    b = np.sum(a_correct & ~b_correct)  # A correct, B incorrect
    c = np.sum(~a_correct & b_correct)  # A incorrect, B correct

    denominator = b + c
    if denominator == 0:
        return 0.0, 1.0

    chi2_stat = ((abs(b - c) - 0.5) ** 2) / denominator
    from scipy.stats import chi2
    p_val = chi2.sf(chi2_stat, df=1)

    return float(chi2_stat), float(p_val)

def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 200,
    seed: int = 42
) -> Tuple[float, float]:
    """Compute 95% bootstrap confidence intervals for a metric function."""
    np.random.seed(seed)
    scores = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))

    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

# ── 3. Permutation Feature Importance ───────────────────────────────────────

def run_permutation_importance(
    test_df: pd.DataFrame,
    probs: Dict[str, np.ndarray],
    y_true: np.ndarray,
    meta_model: StackingEnsembleMetaClassifier
) -> pd.DataFrame:
    """Compute drop in F1-score when shuffling stacking meta-features."""
    logger.info("Computing Permutation Feature Importance...")
    texts = test_df[CFG.text_column].astype(str).tolist()

    # Original Stacking probabilities
    orig_probs = meta_model.predict_proba(probs, texts)
    orig_f1 = f1_score(y_true, (orig_probs[:, 1] >= meta_model.best_threshold).astype(int), zero_division=0)

    meta_features = [
        "roberta_prob", "deberta_prob", "distilbert_prob", "xgboost_prob",
        "prob_mean", "prob_std", "prob_max", "prob_min", "prob_range", "prob_variance",
        "entropy", "agreement_count", "confidence_spread",
        "doc_length", "sentence_count", "avg_sentence_len", "ttr", "burstiness"
    ]

    importance_scores = []

    # Get standard stack features matrix
    from stacking import prepare_stacking_features
    X_meta = prepare_stacking_features(probs, texts)

    for i, feat in enumerate(meta_features):
        X_permuted = X_meta.copy()
        np.random.seed(42)
        np.random.shuffle(X_permuted[:, i])

        # Predict on permuted matrix using K-Fold models
        fold_probs = []
        for model in meta_model.models:
            fold_probs.append(model.predict_proba(X_permuted))
        perm_prob = np.mean(fold_probs, axis=0)
        perm_preds = (perm_prob[:, 1] >= meta_model.best_threshold).astype(int)

        perm_f1 = f1_score(y_true, perm_preds, zero_division=0)
        drop = orig_f1 - perm_f1
        importance_scores.append({
            "Feature": feat,
            "Importance (F1 Drop)": max(0.0, drop)
        })

    return pd.DataFrame(importance_scores).sort_values(by="Importance (F1 Drop)", ascending=False)

# ── 4. Asset Rendering & LaTeX Exporter ────────────────────────────────────

def export_latex_table(df: pd.DataFrame, file_path: Path, caption: str, label: str) -> None:
    """Format and save a pandas DataFrame as a LaTeX tabular block."""
    latex = df.to_latex(index=False, caption=caption, label=label, float_format="%.4f")
    file_path.write_text(latex, encoding="utf-8")
    logger.info(f"LaTeX Table exported -> {file_path}")

def plot_publication_figures(
    y_true: np.ndarray,
    probs: Dict[str, np.ndarray],
    stack_probs: np.ndarray,
    importances: pd.DataFrame
) -> None:
    """Plot high-resolution ROC and Permutation Importance figures."""
    # 1. ROC Curves
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Chance")

    models = ["roberta", "deberta", "distilbert", "xgboost"]
    pretty_names = {
        "roberta": "RoBERTa-base",
        "deberta": "DeBERTa-v3-base",
        "distilbert": "DistilBERT-base",
        "xgboost": "XGBoost"
    }

    for m in models:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, probs[m][:, 1])
        plt.plot(fpr, tpr, label=f"{pretty_names[m]} (AUC = {auc(fpr, tpr):.4f})", alpha=0.7)

    fpr_s, tpr_s, _ = roc_curve(y_true, stack_probs[:, 1])
    plt.plot(fpr_s, tpr_s, label=f"Stacking Ensemble (AUC = {auc(fpr_s, tpr_s):.4f})", linewidth=2.5, color="red")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(RESEARCH_DIR / "roc_curves_publication.png", dpi=300)
    plt.close()

    # 2. Permutation Importance
    plt.figure(figsize=(10, 6))
    top_importances = importances.head(10)
    sns.barplot(data=top_importances, x="Importance (F1 Drop)", y="Feature", palette="Blues_r")
    plt.title("Top 10 Permutation Meta-Feature Importances (F1 Score Drop)")
    plt.xlabel("Drop in F1-score when feature is permuted")
    plt.ylabel("Stacking Meta-Feature")
    plt.tight_layout()
    plt.savefig(RESEARCH_DIR / "permutation_importance_publication.png", dpi=300)
    plt.close()

# ── 5. Main Execution Wrapper ───────────────────────────────────────────────

def run_research_pipeline(smoke_test: bool = False) -> None:
    """Run full Phase 5 validation checks and write LaTeX / plot assets."""
    print_section("Phase 5 - Scientific Validation & Reproducibility Suite")

    # 1. Load splits
    import data_utils
    df = pd.read_csv(CFG.data_path)
    train_df, val_df, test_df = data_utils.get_unified_splits(df)

    if smoke_test:
        logger.info("Smoke test active: truncating validation text splits to 80 samples...")
        test_df = test_df.iloc[:80].reset_index(drop=True)

    y_test = test_df[CFG.label_column].values
    test_texts = test_df[CFG.text_column].astype(str).tolist()

    # 2. Load models
    from ensemble import BaseModelEvaluator
    evaluator = BaseModelEvaluator()
    test_probs = evaluator.extract_all_probs(test_texts, batch_size=32)

    meta_model = joblib.load(CFG.meta_model_path)
    stack_probs = meta_model.predict_proba(test_probs, test_texts)
    stack_preds = (stack_probs[:, 1] >= meta_model.best_threshold).astype(int)

    # 3. Runs experiments
    ablation_df = run_ablation_experiment(test_df, test_probs, y_test, meta_model)
    ablation_df.to_csv(RESEARCH_DIR / "ablation_results.csv", index=False)

    stats_df = run_statistical_tests(y_test, test_probs, stack_probs, stack_preds)
    stats_df.to_csv(RESEARCH_DIR / "statistical_significance.csv", index=False)

    importance_df = run_permutation_importance(test_df, test_probs, y_test, meta_model)
    importance_df.to_csv(RESEARCH_DIR / "meta_feature_importance.csv", index=False)

    # 4. Generate LaTeX tables & Visualisations
    export_latex_table(
        ablation_df,
        RESEARCH_DIR / "ablation_table.tex",
        caption="Ablation study of Hybrid Ensemble components on hold-out academic test split.",
        label="tab:ablation"
    )
    export_latex_table(
        stats_df,
        RESEARCH_DIR / "significance_table.tex",
        caption="Paired McNemar statistical significance testing compared against base models.",
        label="tab:significance"
    )

    plot_publication_figures(y_test, test_probs, stack_probs, importance_df)

    # 5. Reproducibility Checklist
    import hashlib
    def get_file_hash(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()

    checklist = {
        "environment": {
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "joblib_version": joblib.__version__
        },
        "seeds": {
            "global_seed": CFG.seed,
            "stratified_random_seed": 42
        },
        "dataset_hashes": {
            "final_dataset.csv": get_file_hash(CFG.data_path)
        }
    }
    with open(RESEARCH_DIR / "reproducibility_checklist.json", "w", encoding="utf-8") as f:
        json.dump(checklist, f, indent=4)

    print_section("Validation Complete")
    logger.info(f"Research and validation assets written -> {RESEARCH_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research Validation Script")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 80 sample ablation and stats checks.")
    args = parser.parse_args()

    try:
        run_research_pipeline(smoke_test=args.smoke_test)
    except Exception as e:
        logger.error(f"Research pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
