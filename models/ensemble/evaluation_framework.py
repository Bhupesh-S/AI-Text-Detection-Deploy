"""
evaluation_framework.py
========================
Production-quality evaluation and error analysis framework for the
Hybrid Ensemble Detector. Measures model quality, calibration, robustness,
near-duplicates, domain generalization, and false positive/negative failure modes.

Author  : Antigravity
"""

import os
import sys
import re
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
from sklearn.metrics import confusion_matrix

# Sibling path bootstrap
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG
from utils import get_logger, evaluate_predictions, print_section
from calibration import calculate_calibration_metrics, calculate_ece, calculate_mce
from stacking import prepare_stacking_features

logger = get_logger(
    __name__,
    log_file=CFG.logs_dir / "evaluation_framework.log"
)

# ── 1. Text Annotation & Metadata Heuristics ───────────────────────────────

def annotate_text(text: str, label: int) -> dict:
    """Infer domain source, LLM, academic discipline, and document type from text content."""
    text_lower = text.lower()

    # 1. Domain & LLM Source
    if label == 0:
        generator = "Human"
        if "abstract" in text_lower or re.search(r"\[\d+\]", text):
            domain = "Research Papers"
        elif any(c in text_lower for c in ["opinion", "agree", "disagree", "firstly", "conclude"]):
            domain = "Student Essays"
        elif "micusp" in text_lower or "corpus" in text_lower:
            domain = "MICUSP"
        else:
            domain = "BAWE"
    else:
        domain = "AI Generated"
        # GPT words: delve, testament, foster, pivotal, key, etc.
        gpt_count = sum(1 for w in ["delve", "testament", "foster", "pivotal", "vibrant", "transcend"] if w in text_lower)
        if gpt_count >= 2:
            generator = "GPT-4"
        elif re.search(r"\*\*[^*]+\*\*:", text):
            generator = "Gemini"
        elif "claude" in text_lower or (text.count("\n") > 5 and len(text) > 1200):
            generator = "Claude"
        else:
            generator = "Llama"

    # 2. Academic Discipline
    discipline = "General Academic"
    med_kw = ["patient", "clinical", "cardiac", "therapy", "disease", "receptor", "pulmonary", "treatment"]
    bio_kw = ["cell", "molecular", "gene", "protein", "dna", "rna", "organism", "catalyst", "molecule"]
    eng_kw = ["algorithm", "computational", "matrix", "voltage", "voltage", "signal", "software", "network"]
    law_kw = ["statute", "legal", "court", "jurisdiction", "statutory", "policy", "political", "economic"]

    for kw in med_kw:
        if kw in text_lower:
            discipline = "Medicine"
            break
    if discipline == "General Academic":
        for kw in bio_kw:
            if kw in text_lower:
                discipline = "Biology/Chemistry"
                break
    if discipline == "General Academic":
        for kw in eng_kw:
            if kw in text_lower:
                discipline = "Engineering/CS"
                break
    if discipline == "General Academic":
        for kw in law_kw:
            if kw in text_lower:
                discipline = "Law/Social Science"
                break

    # 3. Document Type
    if "abstract" in text_lower or text_lower.startswith("abstract"):
        doc_type = "Abstract"
    elif len(text.split()) > 750:
        doc_type = "Full Paper Section"
    elif "cite" in text_lower or re.search(r"\[\d+\]", text):
        doc_type = "Research Paper Section"
    else:
        doc_type = "Academic Essay"

    return {
        "domain": domain,
        "generator": generator,
        "discipline": discipline,
        "doc_type": doc_type
    }

# ── 2. Dataset Quality Audit Checks ─────────────────────────────────────────

def run_dataset_audit(df: pd.DataFrame) -> dict:
    """Search for domain/length imbalances and identify exact/near duplicates."""
    logger.info("Auditing dataset distributions and near-duplicates...")
    n_samples = len(df)

    # 1. Class & Length imbalance
    class_dist = df["label"].value_counts(normalize=True).to_dict()

    word_counts = df["text"].apply(lambda t: len(str(t).split()))
    short_ratio = float(np.mean(word_counts < 250))
    long_ratio = float(np.mean(word_counts > 750))

    # 2. Exact Duplicates
    exact_duplicates = int(df.duplicated(subset=["text"]).sum())

    # 3. Near duplicates check on a subset of 800 texts (to prevent quadratic slowups)
    logger.info("Running Jaccard Min-Hash checks for near-duplicates...")
    sample_size = min(len(df), 800)
    sample_df = df.sample(n=sample_size, random_state=42).copy()
    sample_texts = sample_df["text"].astype(str).tolist()

    word_sets = [set(t.lower().split()) for t in sample_texts]
    near_dups_count = 0

    for i in range(len(sample_texts)):
        for j in range(i + 1, len(sample_texts)):
            s1, s2 = word_sets[i], word_sets[j]
            if not s1 or not s2:
                continue
            jaccard = len(s1 & s2) / len(s1 | s2)
            if jaccard >= 0.85:
                near_dups_count += 1

    near_dups_estimated = int(near_dups_count * (len(df) / sample_size))

    return {
        "total_samples": n_samples,
        "class_distribution": class_dist,
        "short_text_ratio": round(short_ratio, 4),
        "long_text_ratio": round(long_ratio, 4),
        "exact_duplicates": exact_duplicates,
        "estimated_near_duplicates": near_dups_estimated
    }

# ── 3. Performance Metrics ──────────────────────────────────────────────────

def compute_all_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate 15 classification and calibration performance metrics."""
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, precision_score, recall_score,
        f1_score, matthews_corrcoef, cohen_kappa_score, log_loss, brier_score_loss
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    cal_metrics = calculate_calibration_metrics(y_true, y_prob)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier_score": float(cal_metrics["brier_score"]),
        "ece": float(cal_metrics["ece"]),
        "mce": float(cal_metrics["mce"]),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "specificity": float(specificity),
        "sensitivity": float(sensitivity)
    }

# ── 4. Error Diagnostics ────────────────────────────────────────────────────

def perform_error_diagnostics(
    df: pd.DataFrame,
    probs: Dict[str, np.ndarray],
    stacking_probs: np.ndarray,
    stacking_preds: np.ndarray,
    y_true: np.ndarray
) -> Tuple[List[dict], List[dict]]:
    """Profile FP and FN predictions and assign logical failure classifications."""
    fps = []
    fns = []

    texts = df[CFG.text_column].astype(str).tolist()

    for idx in range(len(y_true)):
        true_lbl = int(y_true[idx])
        pred_lbl = int(stacking_preds[idx])

        # Check mismatch
        if true_lbl == 0 and pred_lbl == 1:
            # False Positive
            anno = annotate_text(texts[idx], true_lbl)
            w_len = len(texts[idx].split())

            # Diagnose reason
            if w_len < 100:
                reason = "Short text (collapses context window)"
            elif anno["discipline"] in ["Biology/Chemistry", "Medicine"]:
                reason = "Highly technical terminology profile"
            elif any(c in texts[idx].lower() for c in ["firstly", "in my opinion", "agree"]):
                reason = "Uniform argument connectors (essay styling)"
            else:
                reason = "Low burstiness / uniform sentence structures"

            fps.append({
                "text_snippet": texts[idx][:150] + " ...",
                "discipline": anno["discipline"],
                "doc_type": anno["doc_type"],
                "word_count": w_len,
                "entropy": float(calculate_entropy(stacking_probs[idx, 1])),
                "stacking_prob": float(stacking_probs[idx, 1]),
                "roberta_prob": float(probs["roberta"][idx, 1]),
                "deberta_prob": float(probs["deberta"][idx, 1]),
                "distilbert_prob": float(probs["distilbert"][idx, 1]),
                "xgboost_prob": float(probs["xgboost"][idx, 1]),
                "failure_mode": reason
            })

        elif true_lbl == 1 and pred_lbl == 0:
            # False Negative
            anno = annotate_text(texts[idx], true_lbl)
            w_len = len(texts[idx].split())

            if w_len < 120:
                reason = "Short AI output outline"
            elif anno["generator"] == "Claude":
                reason = "High complexity Claude stylometrics"
            else:
                reason = "High entropy generator text shifting"

            fns.append({
                "text_snippet": texts[idx][:150] + " ...",
                "generator": anno["generator"],
                "discipline": anno["discipline"],
                "word_count": w_len,
                "entropy": float(calculate_entropy(stacking_probs[idx, 1])),
                "stacking_prob": float(stacking_probs[idx, 1]),
                "roberta_prob": float(probs["roberta"][idx, 1]),
                "deberta_prob": float(probs["deberta"][idx, 1]),
                "distilbert_prob": float(probs["distilbert"][idx, 1]),
                "xgboost_prob": float(probs["xgboost"][idx, 1]),
                "failure_mode": reason
            })

    return fps, fns

def calculate_entropy(p: float) -> float:
    epsilon = 1e-15
    p_clipped = np.clip(p, epsilon, 1.0 - epsilon)
    return float(-p_clipped * np.log2(p_clipped) - (1.0 - p_clipped) * np.log2(1.0 - p_clipped))

# ── 5. Visualisation Dashboards ─────────────────────────────────────────────

def plot_confusion_dashboard(y_true: np.ndarray, y_pred: np.ndarray, save_dir: Path) -> None:
    """Save standardized and normalized confusion matrix diagrams."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    labels = ["Human", "AI"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0], cbar=False)
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[1], cbar=False)
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    fig.suptitle("Confusion Matrix Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_dir / "confusion_matrix_dashboard.png", dpi=300)
    plt.close(fig)

def plot_calibration_dashboard(y_true: np.ndarray, y_prob: np.ndarray, save_dir: Path) -> None:
    """Save reliability calibration diagram and confidence histogram."""
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Reliability diagram
    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    axes[0].plot(prob_pred, prob_true, "s-", color="#4C72B0", label="Stacking Ensemble")
    axes[0].set_ylabel("Fraction of Positives")
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_title("Calibration Curve (Reliability Diagram)")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Confidence histogram
    axes[1].hist(y_prob, bins=10, range=(0, 1), color="#8172B2", edgecolor="white", rwidth=0.8)
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel("Predicted Probability")
    axes[1].set_title("Confidence Distribution Histogram")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Calibration Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_dir / "calibration_dashboard.png", dpi=300)
    plt.close(fig)

# ── 6. Domain Generalization & Reporting ───────────────────────────────────

def run_domain_generalization(
    df: pd.DataFrame,
    stacking_probs: np.ndarray,
    stacking_preds: np.ndarray,
    y_true: np.ndarray
) -> dict:
    """Compute performance metrics grouped by text domains and simulated LLM generators."""
    texts = df[CFG.text_column].astype(str).tolist()
    domains_data = []

    for idx in range(len(y_true)):
        anno = annotate_text(texts[idx], int(y_true[idx]))
        domains_data.append({
            "domain": anno["domain"],
            "generator": anno["generator"],
            "discipline": anno["discipline"],
            "true_lbl": int(y_true[idx]),
            "pred_lbl": int(stacking_preds[idx]),
            "prob_ai": float(stacking_probs[idx, 1])
        })

    dom_df = pd.DataFrame(domains_data)
    results = {}

    # Domain slices
    for dom in dom_df["domain"].unique():
        sub = dom_df[dom_df["domain"] == dom]
        if len(sub) > 2:
            metrics = compute_all_metrics(sub["true_lbl"].values, sub["prob_ai"].values, sub["pred_lbl"].values)
            results[f"domain_{dom}"] = metrics

    # Discipline slices
    for disc in dom_df["discipline"].unique():
        sub = dom_df[dom_df["discipline"] == disc]
        if len(sub) > 2:
            metrics = compute_all_metrics(sub["true_lbl"].values, sub["prob_ai"].values, sub["pred_lbl"].values)
            results[f"discipline_{disc}"] = metrics

    # Generator slices (AI subset)
    ai_sub = dom_df[dom_df["true_lbl"] == 1]
    for gen in ai_sub["generator"].unique():
        sub = ai_sub[ai_sub["generator"] == gen]
        if len(sub) > 2:
            acc = float(np.mean(sub["pred_lbl"] == 1))
            results[f"llm_{gen}"] = {"accuracy_recall": acc, "count": len(sub)}

    return results

def write_benchmark_report(
    overall_metrics: dict,
    domain_generalization: dict,
    audit_results: dict,
    fps: List[dict],
    fns: List[dict],
    save_dir: Path
) -> None:
    """Save benchmark outputs in Markdown, JSON, and CSV tables."""
    # 1. Save JSON
    benchmark_data = {
        "overall_metrics": overall_metrics,
        "domain_generalization": domain_generalization,
        "audit_results": audit_results,
        "false_positives_count": len(fps),
        "false_negatives_count": len(fns)
    }
    with open(save_dir / "benchmark_report.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=4)

    # 2. Save CSV overall
    overall_df = pd.DataFrame([overall_metrics])
    overall_df.to_csv(save_dir / "overall_metrics.csv", index=False)

    # 3. Save Markdown
    fp_modes = pd.DataFrame(fps)["failure_mode"].value_counts().to_dict() if fps else {}
    fn_modes = pd.DataFrame(fns)["failure_mode"].value_counts().to_dict() if fns else {}

    md_lines = [
        "# Comprehensive Evaluation & Benchmark Report",
        f"Generated At: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Executive Summary",
        f"This report documents the diagnostic auditing of the **Hybrid Ensemble Detector** on the hold-out test split.",
        "",
        "## 2. Dataset Auditing Results",
        f"* **Total Samples evaluated**: {audit_results['total_samples']}",
        f"* **Short Text ratio (< 250 words)**: {audit_results['short_text_ratio']:.2%}",
        f"* **Long Text ratio (> 750 words)**: {audit_results['long_text_ratio']:.2%}",
        f"* **Exact Duplicates in dataset**: {audit_results['exact_duplicates']}",
        f"* **Estimated Near-Duplicates (Jaccard >= 0.85)**: {audit_results['estimated_near_duplicates']}",
        "",
        "## 3. Overall Performance Metrics",
        "| Metric | Value | Meaning |",
        "| :--- | :--- | :--- |",
        f"| Accuracy | {overall_metrics['accuracy']:.2%} | Proportion of correct predictions |",
        f"| Balanced Accuracy | {overall_metrics['balanced_accuracy']:.2%} | Mean recall across both classes |",
        f"| Precision | {overall_metrics['precision']:.2%} | AI Predicted class sharpness |",
        f"| Recall (Sensitivity) | {overall_metrics['recall']:.2%} | Captured proportion of AI texts |",
        f"| Specificity | {overall_metrics['specificity']:.2%} | Captured proportion of Human texts |",
        f"| F1-score | {overall_metrics['f1_score']:.2%} | Harmonized precision/recall score |",
        f"| MCC | {overall_metrics['mcc']:.4f} | Matthews Correlation Coefficient |",
        f"| Cohen's Kappa | {overall_metrics['cohen_kappa']:.4f} | Inter-annotator alignment |",
        f"| Brier Score | {overall_metrics['brier_score']:.4f} | Mean squared probability error |",
        f"| ECE (Calibration) | {overall_metrics['ece']:.4f} | Expected Calibration Error |",
        f"| False Positive Rate (FPR) | {overall_metrics['fpr']:.2%} | Human texts flagged as AI (lower is better) |",
        "",
        "## 4. Failure Mode Diagnostics",
        f"### False Positives Count: {len(fps)}",
        "Most common FP failure triggers:",
    ]
    for mode, cnt in fp_modes.items():
        md_lines.append(f"* **{mode}**: {cnt} times")

    md_lines.extend([
        "",
        f"### False Negatives Count: {len(fns)}",
        "Most common FN failure triggers:"
    ])
    for mode, cnt in fn_modes.items():
        md_lines.append(f"* **{mode}**: {cnt} times")

    md_lines.extend([
        "",
        "## 5. Domain Generalization Slices",
        "### Inferred Academic Disciplines",
        "| Discipline | Accuracy | Precision | Recall | ECE |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ])

    for disc in ["Medicine", "Biology/Chemistry", "Engineering/CS", "Law/Social Science", "General Academic"]:
        k = f"discipline_{disc}"
        if k in domain_generalization:
            m = domain_generalization[k]
            md_lines.append(f"| {disc} | {m['accuracy']:.2%} | {m['precision']:.2%} | {m['recall']:.2%} | {m['ece']:.4f} |")

    md_lines.extend([
        "",
        "### Generator Robustness (AI Recall)",
        "| Simulated Generator LLM | Recall Accuracy | Evaluation Counts |",
        "| :--- | :--- | :--- |"
    ])
    for gen in ["GPT-4", "Gemini", "Claude", "Llama"]:
        k = f"llm_{gen}"
        if k in domain_generalization:
            m = domain_generalization[k]
            md_lines.append(f"| {gen} | {m['accuracy_recall']:.2%} | {m['count']} |")

    with open(save_dir / "Benchmark_Report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info(f"Benchmark Report written successfully to: {save_dir}")

# ── 7. Main Pipeline Runner ────────────────────────────────────────────────

def run_evaluation_pipeline(smoke_test: bool = False) -> None:
    """Run full diagnostic audit check and performance evaluation suite."""
    print_section("Phase 4 - Model Evaluation & Error Analysis Framework")

    # 1. Load raw dataset splits
    import data_utils
    if not CFG.data_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {CFG.data_path}")

    logger.info(f"Loading raw dataset from {CFG.data_path}...")
    df = pd.read_csv(CFG.data_path)
    train_df, val_df, test_df = data_utils.get_unified_splits(df)

    # Generate audit report
    audit_res = run_dataset_audit(df)

    # Slice text data if smoke test
    if smoke_test:
        logger.info("Smoke test active: truncating evaluation split to 80 samples...")
        test_df = test_df.iloc[:80].reset_index(drop=True)

    y_test = test_df[CFG.label_column].values
    test_texts = test_df[CFG.text_column].astype(str).tolist()

    # 2. Get Calibrated Predictions
    from ensemble import BaseModelEvaluator
    evaluator = BaseModelEvaluator()

    logger.info("Extracting calibrated probabilities on Hold-Out test split...")
    test_probs = evaluator.extract_all_probs(test_texts, batch_size=32)

    # Load meta-classifier Stacking models
    from stacking import StackingEnsembleMetaClassifier
    if not CFG.meta_model_path.exists():
         raise FileNotFoundError(f"Stacking Meta-classifier not found: {CFG.meta_model_path}. Run train.py first!")

    meta_model = joblib.load(CFG.meta_model_path)

    logger.info("Computing Stacking predictions...")
    stacking_probs = meta_model.predict_proba(test_probs, test_texts)
    stacking_preds = (stacking_probs[:, 1] >= meta_model.best_threshold).astype(int)

    # 3. Calculate Performance Metrics
    overall_met = compute_all_metrics(y_test, stacking_probs[:, 1], stacking_preds)
    logger.info(f"Stacking Accuracy: {overall_met['accuracy']:.4f} | F1: {overall_met['f1_score']:.4f} | ECE: {overall_met['ece']:.4f}")

    # 4. Error Profiling
    fps, fns = perform_error_diagnostics(test_df, test_probs, stacking_probs, stacking_preds, y_test)
    logger.info(f"Diagnostics: logged {len(fps)} False Positives and {len(fns)} False Negatives.")

    # Save failure modes lists to files
    pd.DataFrame(fps).to_csv(CFG.reports_dir / "false_positives_diagnostics.csv", index=False)
    pd.DataFrame(fns).to_csv(CFG.reports_dir / "false_negatives_diagnostics.csv", index=False)

    # 5. Domain generalization Slices
    dom_gen = run_domain_generalization(test_df, stacking_probs, stacking_preds, y_test)

    # 6. Save dashboards & Benchmark Reports
    plot_confusion_dashboard(y_test, stacking_preds, CFG.figures_dir)
    plot_calibration_dashboard(y_test, stacking_probs[:, 1], CFG.figures_dir)

    write_benchmark_report(overall_met, dom_gen, audit_res, fps, fns, CFG.reports_dir)

    print_section("Framework Diagnostics Complete")
    logger.info(f"Benchmark figures stored -> {CFG.figures_dir}")
    logger.info(f"Audit log reports stored -> {CFG.reports_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detector Evaluation Framework")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 80 sample diagnostic check.")
    args = parser.parse_args()

    try:
        run_evaluation_pipeline(smoke_test=args.smoke_test)
    except Exception as e:
        logger.error(f"Diagnostic pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
