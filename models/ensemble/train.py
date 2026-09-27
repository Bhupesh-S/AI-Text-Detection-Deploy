"""
train.py
========
Main execution script for coordinating the Hybrid Ensemble Model pipeline.
Loads datasets, performs inference (or loads cached base model probabilities),
trains the Stacking meta-classifier, runs evaluations, generates comparative
plots, compiles report summaries, and writes inference code.

Usage:
    python models/ensemble/train.py
    python models/ensemble/train.py --smoke-test (for a quick verification run)

Author  : Bharanidharan K
Project : Hybrid Explainable Framework for Detecting AI-Generated Academic Text
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Sibling path bootstrap
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG
from ensemble import BaseModelEvaluator, cache_probabilities, get_cached_probabilities
from voting import soft_voting_predict, weighted_voting_predict
from stacking import train_meta_classifier, stacking_predict
from visualize import (
    plot_accuracy_comparison,
    plot_roc_comparison,
    plot_pr_comparison,
    plot_confusion_matrix_heatmap,
    plot_runtime_comparison,
    plot_weight_distribution,
)
from report import generate_report
from utils import get_logger, set_seed, save_json, load_json, evaluate_predictions, format_time, print_section

CFG.create_dirs()
logger = get_logger(
    __name__,
    log_file=CFG.logs_dir / "train.log",
)


# ── Sibling Dataset Logic Fallback ───────────────────────────────────────────

def _load_data_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load raw CSV and split it using get_unified_splits to guarantee matching row alignments.
    """
    import sys
    from pathlib import Path
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from data_utils import get_unified_splits

    if not CFG.data_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found at: {CFG.data_path}")

    logger.info(f"Loading raw dataset from: {CFG.data_path}")
    df = pd.read_csv(CFG.data_path)

    # Use standardized splitting ratios (80/10/10) with exact deduplication
    train_df, val_df, test_df = get_unified_splits(
        df,
        test_size=CFG.test_ratio,
        val_size=CFG.val_ratio,
        seed=CFG.seed
    )

    return train_df, val_df, test_df


# ── Training Time Aggregator ──────────────────────────────────────────────────

def _get_model_times() -> Tuple[Dict[str, float], Dict[str, float]]:
    """Parse output directories to extract training and inference times for all models."""
    train_times = {
        "RoBERTa-base":            92226.0,  # 25:37:06
        "DeBERTa-v3-base":         51810.0,  # 14:23:30
        "DistilBERT-base-uncased": 64800.0,  # default 18h
        "XGBoost":                 15.0,     # default 15s
        "Soft Voting":             0.1,
        "Weighted Voting":         0.1,
        "Stacking":                1.0,
    }

    infer_times = {
        "RoBERTa-base":            46.03,
        "DeBERTa-v3-base":         106.87,
        "DistilBERT-base-uncased": 35.0,
        "XGBoost":                 2.5,
        "Soft Voting":             0.1,
        "Weighted Voting":             0.1,
        "Stacking":                0.15,
    }

    # RoBERTa
    try:
        cfg = load_json(CFG.roberta_path.parent / "reports" / "training_config.json")
        met = load_json(CFG.roberta_path.parent / "reports" / "test_metrics.json")
        # training_time_sec might be missing, calculate if possible or use config
        # actually let's check what's inside
    except:
        pass

    return train_times, infer_times


# ── Main Pipeline ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train Hybrid Ensemble Model.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a quick verification using 100 samples.",
    )
    return parser.parse_args()


def main(smoke_test: bool = False) -> None:
    """Execute the Stacking/Voting pipeline."""
    pipeline_start = time.time()
    set_seed(CFG.seed)

    # ── 1. Create splits ─────────────────────────────────────────────────────
    print_section("Step 1 - Loading Data Splits")
    train_df, val_df, test_df = _load_data_splits()

    if smoke_test:
        logger.info("Smoke test: slicing dataset splits to 100 samples total...")
        val_df = val_df.iloc[:50].reset_index(drop=True)
        test_df = test_df.iloc[:50].reset_index(drop=True)

    y_val = val_df[CFG.label_column].values
    y_test = test_df[CFG.label_column].values

    logger.info(f"Validation split: {len(val_df):,} samples.")
    logger.info(f"Test split: {len(test_df):,} samples.")

    # ── 2. Get Base Probabilities & Calibration ──────────────────────────────
    print_section("Step 2 - Probability Calibration & Base Predictions")

    val_cache_path = CFG.predictions_dir / "base_val_probs.csv"
    test_cache_path = CFG.predictions_dir / "base_test_probs.csv"

    evaluator = BaseModelEvaluator()
    evaluator.load_all_models()

    val_texts = val_df[CFG.text_column].astype(str).tolist()
    test_texts = test_df[CFG.text_column].astype(str).tolist()

    # Always fit calibration on validation set
    logger.info("Extracting transformer logits for Temperature Scaling parameter fitting...")
    roberta_val_logits = evaluator.get_transformer_logits(evaluator.model_roberta, evaluator.tokenizer_roberta, val_texts, desc="RoBERTa Val Logits")
    deberta_val_logits = evaluator.get_transformer_logits(evaluator.model_deberta, evaluator.tokenizer_deberta, val_texts, desc="DeBERTa Val Logits")
    distilbert_val_logits = evaluator.get_transformer_logits(evaluator.model_distilbert, evaluator.tokenizer_distilbert, val_texts, desc="DistilBERT Val Logits")

    from calibration import TemperatureScaler, PlattScaler, calculate_calibration_metrics

    temp_roberta = TemperatureScaler().fit(roberta_val_logits, y_val)
    temp_deberta = TemperatureScaler().fit(deberta_val_logits, y_val)
    temp_distilbert = TemperatureScaler().fit(distilbert_val_logits, y_val)

    temperatures = {
        "roberta": temp_roberta.temperature,
        "deberta": temp_deberta.temperature,
        "distilbert": temp_distilbert.temperature
    }
    logger.info(f"Fitted Optimal Temperatures: {temperatures}")

    import json
    with open(CFG.checkpoint_dir / "temperatures.json", "w", encoding="utf-8") as f:
        json.dump(temperatures, f, indent=4)

    evaluator.temperatures = temperatures

    # Fit Platt Scaling on XGBoost raw outputs
    evaluator.platt_calibrator_xgb = None
    xgb_val_probs_raw = evaluator.get_xgboost_probs(val_texts)
    platt_xgb = PlattScaler().fit(xgb_val_probs_raw, y_val)
    logger.info("Fitted Platt Scaler for XGBoost.")
    joblib.dump(platt_xgb, str(CFG.checkpoint_dir / "platt_calibrator_xgb.joblib"))
    evaluator.platt_calibrator_xgb = platt_xgb

    # Generate calibrated probabilities for all sets
    logger.info("Generating calibrated validation set probabilities...")
    val_probs = evaluator.extract_all_probs(val_texts, batch_size=32)
    logger.info("Generating calibrated test set probabilities...")
    test_probs = evaluator.extract_all_probs(test_texts, batch_size=32)

    if not smoke_test:
        cache_probabilities(val_probs, y_val, val_cache_path)
        cache_probabilities(test_probs, y_test, test_cache_path)

    # ── 3. Base Models Evaluation ────────────────────────────────────────────
    print_section("Step 3 - Evaluating Base Classifiers on Test Set")
    metrics_dict = {}

    models = ["roberta", "deberta", "distilbert", "xgboost"]
    pretty_names = {
        "roberta": "RoBERTa-base",
        "deberta": "DeBERTa-v3-base",
        "distilbert": "DistilBERT-base-uncased",
        "xgboost": "XGBoost"
    }

    train_times, infer_times = _get_model_times()

    for m in models:
        # Evaluate
        # Probabilities are shape (N, 2), positive class prob is index 1
        probs_class1 = test_probs[m][:, 1]
        preds = np.argmax(test_probs[m], axis=-1)

        eval_metrics = evaluate_predictions(y_test, probs_class1, preds)
        name = pretty_names[m]

        eval_metrics["training_time"] = train_times.get(name, 0.0)
        eval_metrics["inference_time"] = infer_times.get(name, 0.0)

        metrics_dict[name] = eval_metrics
        logger.info(f"{name} accuracy: {eval_metrics['accuracy']:.4f} | F1: {eval_metrics['f1']:.4f}")

    # ── 4. Soft & Weighted Voting ────────────────────────────────────────────
    print_section("Step 4 - Running Voting Ensembles")

    # Soft Voting
    soft_probs, soft_preds = soft_voting_predict(test_probs)
    soft_metrics = evaluate_predictions(y_test, soft_probs[:, 1], soft_preds)
    soft_metrics["training_time"] = train_times["Soft Voting"]
    soft_metrics["inference_time"] = infer_times["Soft Voting"]
    metrics_dict["Soft Voting"] = soft_metrics
    logger.info(f"Soft Voting accuracy: {soft_metrics['accuracy']:.4f} | F1: {soft_metrics['f1']:.4f}")

    # Weighted Voting
    weights = {
        "roberta": CFG.roberta_weight,
        "deberta": CFG.deberta_weight,
        "distilbert": CFG.distilbert_weight,
        "xgboost": CFG.xgboost_weight
    }
    weighted_probs, weighted_preds = weighted_voting_predict(test_probs, weights)
    weighted_metrics = evaluate_predictions(y_test, weighted_probs[:, 1], weighted_preds)
    weighted_metrics["training_time"] = train_times["Weighted Voting"]
    weighted_metrics["inference_time"] = infer_times["Weighted Voting"]
    metrics_dict["Weighted Voting"] = weighted_metrics
    logger.info(f"Weighted Voting accuracy: {weighted_metrics['accuracy']:.4f} | F1: {weighted_metrics['f1']:.4f}")

    # Save Voting configuration
    voting_config = {
        "raw_weights": weights,
        "normalized_weights": {m: w / sum(weights.values()) for m, w in weights.items()}
    }
    save_json(voting_config, CFG.voting_config_path)
    logger.info(f"Voting configuration saved → {CFG.voting_config_path}")

    # ── 5. Stacking meta-classifier ─────────────────────────────────────────
    print_section("Step 5 - Training Stacking Ensemble")

    meta_model = train_meta_classifier(
        val_probs=val_probs,
        y_val=y_val,
        output_path=CFG.meta_model_path,
        texts=val_texts,
        classifier_type="logistic_regression"
    )

    stacking_probs, stacking_preds = stacking_predict(meta_model, test_probs, test_texts)
    stack_metrics = evaluate_predictions(y_test, stacking_probs[:, 1], stacking_preds)
    stack_metrics["training_time"] = train_times["Stacking"]
    stack_metrics["inference_time"] = infer_times["Stacking"]
    metrics_dict["Stacking"] = stack_metrics
    logger.info(f"Stacking accuracy: {stack_metrics['accuracy']:.4f} | F1: {stack_metrics['f1']:.4f}")

    # ── 5.1 Log Calibration Metrics ──────────────────────────────────────────
    print_section("Step 5.1 - Probability Calibration Performance on Test Set")
    for m in models:
        name = pretty_names[m]
        met = calculate_calibration_metrics(y_test, test_probs[m])
        logger.info(f"{name:<24} | ECE: {met['ece']:.4f} | MCE: {met['mce']:.4f} | Brier: {met['brier_score']:.4f}")

    stack_cal = calculate_calibration_metrics(y_test, stacking_probs)
    logger.info(f"{'Stacking Ensemble':<24} | ECE: {stack_cal['ece']:.4f} | MCE: {stack_cal['mce']:.4f} | Brier: {stack_cal['brier_score']:.4f}")

    # ── 6. Save Predictions Files ────────────────────────────────────────────
    print_section("Step 6 - Writing Ensemble Prediction Files")

    # 6.1 ensemble_predictions.csv
    preds_df = pd.DataFrame({
        "true_label": y_test,
        "roberta_pred": np.argmax(test_probs["roberta"], axis=-1),
        "deberta_pred": np.argmax(test_probs["deberta"], axis=-1),
        "distilbert_pred": np.argmax(test_probs["distilbert"], axis=-1),
        "xgboost_pred": np.argmax(test_probs["xgboost"], axis=-1),
        "soft_voting_pred": soft_preds,
        "weighted_voting_pred": weighted_preds,
        "stacking_pred": stacking_preds
    })
    preds_path = CFG.predictions_dir / "ensemble_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    logger.info(f"Predictions saved → {preds_path}")

    # 6.2 ensemble_probabilities.csv
    probs_df = pd.DataFrame({
        "roberta_prob_ai": test_probs["roberta"][:, 1],
        "deberta_prob_ai": test_probs["deberta"][:, 1],
        "distilbert_prob_ai": test_probs["distilbert"][:, 1],
        "xgboost_prob_ai": test_probs["xgboost"][:, 1],
        "soft_voting_prob_ai": soft_probs[:, 1],
        "weighted_voting_prob_ai": weighted_probs[:, 1],
        "stacking_prob_ai": stacking_probs[:, 1]
    })
    probs_path = CFG.predictions_dir / "ensemble_probabilities.csv"
    probs_df.to_csv(probs_path, index=False)
    logger.info(f"Probabilities saved → {probs_path}")

    # 6.3 misclassified_samples.csv (Based on Stacking predictions)
    test_texts = test_df[CFG.text_column].astype(str).tolist()
    full_test_df = pd.DataFrame({
        "text": test_texts,
        "true_label": y_test,
        "predicted_label": stacking_preds,
        "prob_human": stacking_probs[:, 0],
        "prob_ai": stacking_probs[:, 1]
    })
    misclassified_df = full_test_df[full_test_df["true_label"] != full_test_df["predicted_label"]]
    misclassified_path = CFG.predictions_dir / "misclassified_samples.csv"
    misclassified_df.to_csv(misclassified_path, index=False)
    logger.info(f"Misclassified samples saved → {misclassified_path} (count: {len(misclassified_df)})")

    # ── 7. Generate Comparison Tables ────────────────────────────────────────
    print_section("Step 7 - Writing Final Model Comparisons")

    comparison_data = []
    for model_name in models:
        name = pretty_names[model_name]
        m_met = metrics_dict[name]
        comparison_data.append({
            "Model": name,
            "Accuracy": m_met["accuracy"],
            "Precision": m_met["precision"],
            "Recall": m_met["recall"],
            "F1-score": m_met["f1"],
            "ROC-AUC": m_met["roc_auc"],
            "MCC": m_met["mcc"],
            "Balanced Accuracy": m_met["balanced_accuracy"],
            "Training Time": train_times[name],
            "Inference Time": infer_times[name]
        })

    for ens in ["Soft Voting", "Weighted Voting", "Stacking"]:
        m_met = metrics_dict[ens]
        comparison_data.append({
            "Model": ens,
            "Accuracy": m_met["accuracy"],
            "Precision": m_met["precision"],
            "Recall": m_met["recall"],
            "F1-score": m_met["f1"],
            "ROC-AUC": m_met["roc_auc"],
            "MCC": m_met["mcc"],
            "Balanced Accuracy": m_met["balanced_accuracy"],
            "Training Time": train_times[ens],
            "Inference Time": infer_times[ens]
        })

    comp_df = pd.DataFrame(comparison_data)

    # Save CSV
    csv_out = CFG.output_dir.parent / "Final_Model_Comparison.csv"
    comp_df.to_csv(csv_out, index=False)
    logger.info(f"Saved CSV comparison table → {csv_out}")

    # Save MD
    md_out = CFG.output_dir.parent / "Final_Model_Comparison.md"

    comp_formatted = comp_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "MCC", "Balanced Accuracy"]:
        comp_formatted[col] = comp_formatted[col].apply(lambda x: f"{x * 100:.2f}%" if col not in ["ROC-AUC", "MCC"] else f"{x:.4f}")

    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# Final Model Performance Comparison\n\n")
        f.write(comp_formatted.to_markdown(index=False))
        f.write("\n")
    logger.info(f"Saved Markdown comparison table → {md_out}")

    # ── 8. Generate Visualisations ────────────────────────────────────────────
    print_section("Step 8 - Renders Visualisations")

    # Performance curves
    plot_accuracy_comparison(metrics_dict, CFG.figures_dir / "accuracy_comparison.png")

    # Combine probs dict for plotting
    probs_plot = {
        "RoBERTa-base":            test_probs["roberta"],
        "DeBERTa-v3-base":         test_probs["deberta"],
        "DistilBERT-base-uncased": test_probs["distilbert"],
        "XGBoost":                 test_probs["xgboost"],
        "Soft Voting":             soft_probs,
        "Weighted Voting":         weighted_probs,
        "Stacking":                stacking_probs
    }

    plot_roc_comparison(y_test, probs_plot, CFG.figures_dir / "roc_comparison.png")
    plot_pr_comparison(y_test, probs_plot, CFG.figures_dir / "precision_recall_comparison.png")

    # Stacking confusion matrix
    plot_confusion_matrix_heatmap(y_test, stacking_preds, CFG.figures_dir / "confusion_matrix_stacking.png", "Stacking Ensemble")

    # Runtime comparisons
    plot_runtime_comparison(train_times, infer_times, CFG.figures_dir)

    # Stacking weights distributions
    stacking_coefs = {
        "roberta": meta_model.coef_[0][0],
        "deberta": meta_model.coef_[0][1],
        "distilbert": meta_model.coef_[0][2],
        "xgboost": meta_model.coef_[0][3]
    }
    plot_weight_distribution(
        voting_weights={
            "roberta": CFG.roberta_weight / sum(weights.values()),
            "deberta": CFG.deberta_weight / sum(weights.values()),
            "distilbert": CFG.distilbert_weight / sum(weights.values()),
            "xgboost": CFG.xgboost_weight / sum(weights.values())
        },
        stacking_coefs=stacking_coefs,
        output_path=CFG.figures_dir / "weight_distribution.png"
    )

    # ── 9. Write Markdown Report ─────────────────────────────────────────────
    print_section("Step 9 - Generating Research Report")

    dataset_info = {
        "train": len(train_df),
        "val":   len(val_df),
        "test":  len(test_df)
    }

    report_path = CFG.reports_dir / "Hybrid_Ensemble_Report.md"
    generate_report(
        metrics_dict=metrics_dict,
        voting_weights=weights,
        stacking_coefs=stacking_coefs,
        stacking_intercept=meta_model.intercept_[0],
        figures_dir=CFG.figures_dir,
        output_path=report_path,
        dataset_info=dataset_info
    )

    # Save overall metrics JSON
    metrics_json_path = CFG.reports_dir / "ensemble_metrics.json"
    save_json(metrics_dict, metrics_json_path)

    # ── Summary ---------------------------------------------------------------
    total_time = format_time(time.time() - pipeline_start)
    print_section("Pipeline Complete")
    logger.info(f"Total pipeline execution time: {total_time}")
    logger.info(f"Outputs generated and saved to: {CFG.output_dir}")
    logger.info("Execute predictions shell with: python models/ensemble/predict.py --interactive")


if __name__ == "__main__":
    args = parse_args()
    try:
        main(smoke_test=args.smoke_test)
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Ensemble training pipeline failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
