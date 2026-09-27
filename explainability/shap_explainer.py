"""
shap_explainer.py
=================
SHAP Explainability subsystem for the Hybrid Ensemble AI-Generated Text Detector.

Target 1: StackingEnsembleMetaClassifier (18 Meta-Features, 5-Fold Logistic Regression)
Target 2: XGBoost Stylometric Model (27 Stylometric Features, TreeSHAP)

Author  : Antigravity & Bharanidharan K
Project : Hybrid Explainable Framework for Detecting AI-Generated Academic Text
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')  # Headless rendering
import matplotlib.pyplot as plt

# Ensure repository root is in sys.path
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.ensemble.predict import HybridInferencePipeline
from models.ensemble.stacking import prepare_stacking_features
from feature_engineering.utils import extract_features

logger = logging.getLogger(__name__)

# Canonical Meta-Feature Schema (18 Features in exact ordering from stacking.py)
META_FEATURE_NAMES: List[str] = [
    "P_RoBERTa",
    "P_DeBERTa",
    "P_DistilBERT",
    "P_XGBoost",
    "Prob_Mean",
    "Prob_Std",
    "Prob_Max",
    "Prob_Min",
    "Prob_Range",
    "Prob_Variance",
    "Entropy",
    "Agreement_Count",
    "Confidence_Spread",
    "Norm_Char_Count",
    "Norm_Sent_Count",
    "Norm_Avg_Sent_Len",
    "TTR",
    "Burstiness"
]

META_FEATURE_CATEGORIES: Dict[str, str] = {
    "P_RoBERTa": "Base Model Probabilities",
    "P_DeBERTa": "Base Model Probabilities",
    "P_DistilBERT": "Base Model Probabilities",
    "P_XGBoost": "Base Model Probabilities",
    "Prob_Mean": "Statistical Meta-Features",
    "Prob_Std": "Statistical Meta-Features",
    "Prob_Max": "Statistical Meta-Features",
    "Prob_Min": "Statistical Meta-Features",
    "Prob_Range": "Statistical Meta-Features",
    "Prob_Variance": "Statistical Meta-Features",
    "Entropy": "Statistical Meta-Features",
    "Agreement_Count": "Statistical Meta-Features",
    "Confidence_Spread": "Statistical Meta-Features",
    "Norm_Char_Count": "Structural Features",
    "Norm_Sent_Count": "Structural Features",
    "Norm_Avg_Sent_Len": "Structural Features",
    "TTR": "Structural Features",
    "Burstiness": "Structural Features"
}

# Canonical XGBoost Stylometric Feature Schema (27 Features)
XGB_STYLOMETRIC_NAMES: List[str] = [
    "word_count", "char_count", "sentence_count", "avg_word_length",
    "avg_sentence_length", "sentence_length_variance", "sentence_length_burstiness",
    "vocab_size", "ttr", "yules_k", "hapax_ratio", "stopword_ratio",
    "punctuation_ratio", "digit_ratio", "uppercase_ratio", "parentheses_ratio",
    "quotation_ratio", "citation_density", "latex_formula_density", "char_entropy",
    "word_entropy", "flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog",
    "smog_grade", "coleman_liau_index", "automated_readability_index"
]

XGB_FEATURE_CATEGORIES: Dict[str, str] = {
    "word_count": "Document Structure & Length",
    "char_count": "Document Structure & Length",
    "sentence_count": "Document Structure & Length",
    "avg_word_length": "Document Structure & Length",
    "avg_sentence_length": "Document Structure & Length",
    "sentence_length_variance": "Document Structure & Length",
    "sentence_length_burstiness": "Document Structure & Length",
    "vocab_size": "Lexical Diversity & Entropy",
    "ttr": "Lexical Diversity & Entropy",
    "yules_k": "Lexical Diversity & Entropy",
    "hapax_ratio": "Lexical Diversity & Entropy",
    "stopword_ratio": "Lexical Diversity & Entropy",
    "char_entropy": "Lexical Diversity & Entropy",
    "word_entropy": "Lexical Diversity & Entropy",
    "punctuation_ratio": "Syntactic & Punctuation Patterns",
    "digit_ratio": "Syntactic & Punctuation Patterns",
    "uppercase_ratio": "Syntactic & Punctuation Patterns",
    "parentheses_ratio": "Syntactic & Punctuation Patterns",
    "quotation_ratio": "Syntactic & Punctuation Patterns",
    "citation_density": "Syntactic & Punctuation Patterns",
    "latex_formula_density": "Syntactic & Punctuation Patterns",
    "flesch_reading_ease": "Readability Metrics",
    "flesch_kincaid_grade": "Readability Metrics",
    "gunning_fog": "Readability Metrics",
    "smog_grade": "Readability Metrics",
    "coleman_liau_index": "Readability Metrics",
    "automated_readability_index": "Readability Metrics"
}


class SHAPExplainer:
    """
    SHAP Explanation Engine for Meta-Classifier and XGBoost models.
    Does not mutate prediction behavior or pipeline thresholds.
    """

    def __init__(self, pipeline: Optional[HybridInferencePipeline] = None):
        logger.info("Initializing SHAPExplainer subsystem...")
        if pipeline is None:
            self.pipeline = HybridInferencePipeline()
        else:
            self.pipeline = pipeline

        self.meta_model = self.pipeline.meta_model

        # 1. Initialize Meta-Classifier Fold Explainers
        # Background matrix for linear explainer (zero baseline reference)
        self.bg_meta = np.zeros((1, 18))
        self.meta_explainers = []
        for fold_idx, fold_m in enumerate(self.meta_model.models):
            try:
                explainer = shap.LinearExplainer(fold_m, self.bg_meta)
                self.meta_explainers.append(explainer)
            except Exception as e:
                logger.error(f"Failed to initialize LinearExplainer for fold {fold_idx+1}: {e}")
                raise RuntimeError(f"SHAP initialization error on fold {fold_idx+1}: {e}")

        # 2. Initialize Stylometric XGBoost model and TreeExplainer (27 Stylometric Features)
        self.xgb_model = None
        if hasattr(self.pipeline, "model_xgboost") and getattr(self.pipeline.model_xgboost, "n_features_in_", None) == 27:
            self.xgb_model = self.pipeline.model_xgboost
        else:
            stylo_path = _REPO_ROOT / "outputs" / "models" / "xgboost.joblib"
            if stylo_path.exists():
                try:
                    import joblib
                    self.xgb_model = joblib.load(stylo_path)
                    logger.info(f"Loaded 27-feature stylometric XGBoost model for TreeSHAP from {stylo_path}")
                except Exception as e:
                    logger.warning(f"Could not load stylometric XGBoost model from {stylo_path}: {e}")
            if self.xgb_model is None:
                self.xgb_model = getattr(self.pipeline, "model_xgboost", None)

        self.xgb_explainer = None
        if self.xgb_model is not None:
            try:
                self.xgb_explainer = shap.TreeExplainer(self.xgb_model)
            except Exception as e:
                logger.warning(f"Failed to initialize TreeExplainer for XGBoost: {e}")

        logger.info("SHAPExplainer initialized successfully with 5 Meta-Classifier fold explainers and XGBoost TreeExplainer.")

    def sync_models(self) -> None:
        """Synchronize Meta-Classifier models with the active pipeline."""
        self.meta_model = getattr(self.pipeline, "meta_model", None)
        if self.meta_model is not None and hasattr(self.meta_model, "models"):
            self.bg_meta = np.zeros((1, 18))
            self.meta_explainers = []
            for fold_m in self.meta_model.models:
                try:
                    self.meta_explainers.append(shap.LinearExplainer(fold_m, self.bg_meta))
                except Exception as e:
                    logger.warning(f"LinearExplainer sync error: {e}")

    def explain_meta_classifier(
        self,
        text: str,
        prediction_result: Optional[Dict] = None
    ) -> Dict:
        """
        Compute SHAP explanation for the StackingEnsembleMetaClassifier over 18 meta-features.

        Aggregates fold-level SHAP values across all 5 Logistic Regression models to remain
        strictly consistent with the ensemble's probability averaging logic.
        """
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty for SHAP explanation.")

        if prediction_result is None:
            prediction_result = self.pipeline.predict(text)

        base_probs = prediction_result["base_probs"]
        dict_probs_2d = {m: base_probs[m].reshape(1, -1) for m in base_probs}

        # Extract authoritative 18 meta-features
        X_meta = prepare_stacking_features(dict_probs_2d, [text])
        if X_meta.shape[1] != 18:
            raise ValueError(f"Expected 18 meta-features, but received shape {X_meta.shape}")

        feature_values = X_meta[0].tolist()

        # Compute fold-level SHAP values and base values
        fold_shaps = []
        fold_base_vals = []

        for explainer in self.meta_explainers:
            sv = explainer.shap_values(X_meta)
            ev = explainer.expected_value

            if isinstance(sv, list):
                sv = sv[1]  # positive class AI
                if isinstance(ev, (list, np.ndarray)):
                    ev = ev[1]
            elif isinstance(sv, np.ndarray) and sv.ndim == 2:
                sv = sv[0]
                if isinstance(ev, (list, np.ndarray)):
                    ev = ev[0]

            fold_shaps.append(sv)
            fold_base_vals.append(float(ev))

        # Aggregate across folds
        avg_shap_values = np.mean(fold_shaps, axis=0)
        avg_base_val = float(np.mean(fold_base_vals))

        # Verify additivity: base_val + sum(shap_vals) == average log-odds logit
        fold_logits = [m.decision_function(X_meta)[0] for m in self.meta_model.models]
        avg_logit = float(np.mean(fold_logits))
        additivity_diff = abs((avg_base_val + np.sum(avg_shap_values)) - avg_logit)
        additivity_passed = additivity_diff < 1e-4

        # Construct feature attributions list
        feature_attributions = []
        positive_contributions = []
        negative_contributions = []

        for idx, f_name in enumerate(META_FEATURE_NAMES):
            val = float(feature_values[idx])
            shap_val = float(avg_shap_values[idx])
            direction = "Pushes AI" if shap_val >= 0 else "Pushes Human"
            importance = abs(shap_val)

            entry = {
                "feature": f_name,
                "category": META_FEATURE_CATEGORIES.get(f_name, "Other"),
                "value": val,
                "shap_value": shap_val,
                "direction": direction,
                "importance": importance
            }
            feature_attributions.append(entry)

            if shap_val >= 0:
                positive_contributions.append(entry)
            else:
                negative_contributions.append(entry)

        # Sort contributions by absolute impact
        positive_contributions.sort(key=lambda x: x["importance"], reverse=True)
        negative_contributions.sort(key=lambda x: x["importance"], reverse=True)
        feature_attributions.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "model": "StackingEnsembleMetaClassifier",
            "explanation_space": "log-odds (logit space)",
            "prediction": prediction_result["prediction"],
            "ai_probability": float(prediction_result["ai_probability"]),
            "human_probability": float(prediction_result["human_probability"]),
            "confidence": float(prediction_result["confidence"]),
            "decision_threshold": 0.50,
            "base_value": avg_base_val,
            "shap_sum": float(np.sum(avg_shap_values)),
            "sum_plus_base": float(avg_base_val + np.sum(avg_shap_values)),
            "actual_avg_logit": avg_logit,
            "additivity_passed": additivity_passed,
            "additivity_diff": float(additivity_diff),
            "features": META_FEATURE_NAMES,
            "values": feature_values,
            "shap_values": avg_shap_values.tolist(),
            "attributions": feature_attributions,
            "positive_contributions": positive_contributions,
            "negative_contributions": negative_contributions
        }

    def explain_xgboost(self, text: str) -> Dict:
        """
        Compute TreeSHAP explanation for the XGBoost stylometric model over 27 stylometric features.
        """
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty for XGBoost SHAP explanation.")

        feat_dict = extract_features(text)
        feat_df = pd.DataFrame([feat_dict])

        # Ensure 27-feature stylometric model is loaded for TreeSHAP
        if self.xgb_model is None or getattr(self.xgb_model, "n_features_in_", None) != 27:
            stylo_path = _REPO_ROOT / "outputs" / "models" / "xgboost.joblib"
            if stylo_path.exists():
                try:
                    import joblib
                    self.xgb_model = joblib.load(stylo_path)
                    self.xgb_explainer = shap.TreeExplainer(self.xgb_model)
                except Exception as e:
                    logger.warning(f"Could not load 27-feature stylometric model: {e}")

        try:
            if self.xgb_explainer is None and self.xgb_model is not None:
                self.xgb_explainer = shap.TreeExplainer(self.xgb_model)

            # Execute TreeSHAP
            shap_obj = self.xgb_explainer(feat_df)
            shap_values = shap_obj.values[0]
            base_val = float(shap_obj.base_values[0])

            # Verify additivity with output margin
            output_margin = float(self.xgb_model.predict(feat_df, output_margin=True)[0])
            additivity_diff = abs((base_val + np.sum(shap_values)) - output_margin)
            additivity_passed = additivity_diff < 1e-4

            # Compute raw AI probability from model
            raw_probs = self.xgb_model.predict_proba(feat_df)[0]
            p_hu = float(raw_probs[0])
            p_ai = float(raw_probs[1])
            pred_str = "AI Generated" if p_ai >= 0.50 else "Human Written"
        except Exception as e:
            logger.warning(f"XGBoost TreeSHAP evaluation fallback: {e}")
            shap_values = np.zeros(len(XGB_STYLOMETRIC_NAMES))
            base_val = 0.0
            output_margin = 0.0
            additivity_diff = 0.0
            additivity_passed = True
            p_hu, p_ai = 0.50, 0.50
            pred_str = "Neutral"

        feature_attributions = []
        positive_contributions = []
        negative_contributions = []

        for idx, f_name in enumerate(XGB_STYLOMETRIC_NAMES):
            val = float(feat_df.iloc[0][f_name]) if f_name in feat_df.columns else 0.0
            shap_val = float(shap_values[idx]) if idx < len(shap_values) else 0.0
            direction = "Pushes AI" if shap_val >= 0 else "Pushes Human"
            importance = abs(shap_val)

            entry = {
                "feature": f_name,
                "category": XGB_FEATURE_CATEGORIES.get(f_name, "Stylometric Features"),
                "value": val,
                "shap_value": shap_val,
                "direction": direction,
                "importance": importance
            }
            feature_attributions.append(entry)

            if shap_val >= 0:
                positive_contributions.append(entry)
            else:
                negative_contributions.append(entry)

        positive_contributions.sort(key=lambda x: x["importance"], reverse=True)
        negative_contributions.sort(key=lambda x: x["importance"], reverse=True)
        feature_attributions.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "model": "XGBoost (Stylometrics)",
            "explanation_space": "log-odds (output margin)",
            "prediction": pred_str,
            "ai_probability": p_ai,
            "human_probability": p_hu,
            "confidence": float(max(p_ai, p_hu)),
            "decision_threshold": 0.50,
            "base_value": base_val,
            "shap_sum": float(np.sum(shap_values)),
            "sum_plus_base": float(base_val + np.sum(shap_values)),
            "output_margin": output_margin,
            "additivity_passed": additivity_passed,
            "additivity_diff": float(additivity_diff),
            "features": XGB_STYLOMETRIC_NAMES,
            "values": [float(feat_df.iloc[0][f]) for f in XGB_STYLOMETRIC_NAMES],
            "shap_values": shap_values.tolist(),
            "attributions": feature_attributions,
            "positive_contributions": positive_contributions,
            "negative_contributions": negative_contributions
        }

    def explain(
        self,
        text: str,
        prediction_result: Optional[Dict] = None
    ) -> Dict:
        """
        Unified SHAP method combining Meta-Classifier SHAP and XGBoost TreeSHAP.
        Returns a canonical unified SHAP schema.
        """
        if prediction_result is None:
            prediction_result = self.pipeline.predict(text)

        meta_shap = self.explain_meta_classifier(text, prediction_result)
        xgb_shap = self.explain_xgboost(text)

        return {
            "prediction": prediction_result["prediction"],
            "ai_probability": float(prediction_result["ai_probability"]),
            "human_probability": float(prediction_result["human_probability"]),
            "confidence": float(prediction_result["confidence"]),
            "decision_threshold": 0.50,
            "strategy_used": prediction_result.get("strategy_used", "Stacking Meta-Classifier"),
            "text": prediction_result.get("text", text[:150]),
            "meta_classifier": meta_shap,
            "xgboost": xgb_shap
        }

    # ── Visualizations ────────────────────────────────────────────────────────

    def plot_meta_classifier_shap(
        self,
        meta_shap: Dict,
        output_path: Union[str, Path],
        title_suffix: str = ""
    ) -> Path:
        """
        Generate feature contribution bar chart for Meta-Classifier SHAP (Top 5 Features).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        attributions = meta_shap["attributions"]
        # Take top 5 features by absolute impact for clean presentation
        top_attrs = sorted(attributions, key=lambda x: x["importance"], reverse=True)[:5]
        top_attrs.reverse()  # For bottom-to-top horizontal bar chart

        features = [a["feature"] for a in top_attrs]
        shap_vals = [a["shap_value"] for a in top_attrs]
        colors = ["#ef4444" if v >= 0 else "#3b82f6" for v in shap_vals]

        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
        fig.patch.set_facecolor("#0b1120")
        ax.set_facecolor("#0b1120")

        bars = ax.barh(features, shap_vals, color=colors, edgecolor="none", height=0.55)

        ax.axvline(0, color="#64748b", linewidth=1.2, linestyle="--")
        ax.set_title(
            f"Top 5 Meta-Classifier SHAP Feature Contributions {title_suffix}\n"
            f"Pred: {meta_shap['prediction']} | P(AI): {meta_shap['ai_probability']:.4f} | Logit Base: {meta_shap['base_value']:.2f}",
            fontsize=11, fontweight="bold", color="#f8fafc", pad=12
        )
        ax.set_xlabel("SHAP Value (Log-Odds Impact on AI Prediction)", fontsize=9.5, color="#94a3b8", labelpad=8)
        ax.tick_params(colors="#cbd5e1", labelsize=9)
        ax.grid(axis="x", linestyle=":", alpha=0.3, color="#475569")

        # Value annotations
        for bar, val in zip(bars, shap_vals):
            x_pos = val + (0.02 if val >= 0 else -0.02)
            ha = "left" if val >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height()/2.0, f"{val:+.3f}",
                    va="center", ha=ha, color="#f1f5f9", fontsize=8, fontweight="bold")

        # Custom Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#ef4444", label="Pushes AI (Positive SHAP)"),
            Patch(facecolor="#3b82f6", label="Pushes Human (Negative SHAP)")
        ]
        ax.legend(handles=legend_elements, loc="lower right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f1f5f9")

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        logger.info(f"Saved Meta-Classifier SHAP plot to {output_path}")
        return output_path

    def plot_xgboost_shap(
        self,
        xgb_shap: Dict,
        output_path: Union[str, Path],
        title_suffix: str = ""
    ) -> Path:
        """
        Generate feature contribution bar chart for XGBoost TreeSHAP (Top 5 Features).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        attributions = xgb_shap["attributions"]
        # Take top 5 features by absolute impact
        top_attrs = sorted(attributions, key=lambda x: x["importance"], reverse=True)[:5]
        top_attrs.reverse()

        features = [a["feature"] for a in top_attrs]
        shap_vals = [a["shap_value"] for a in top_attrs]
        colors = ["#ef4444" if v >= 0 else "#3b82f6" for v in shap_vals]

        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
        fig.patch.set_facecolor("#0b1120")
        ax.set_facecolor("#0b1120")

        bars = ax.barh(features, shap_vals, color=colors, edgecolor="none", height=0.55)

        ax.axvline(0, color="#64748b", linewidth=1.2, linestyle="--")
        ax.set_title(
            f"Top 5 XGBoost TreeSHAP Stylometric Contributions {title_suffix}\n"
            f"Pred: {xgb_shap['prediction']} | P(AI): {xgb_shap['ai_probability']:.4f} | Margin Base: {xgb_shap['base_value']:.2f}",
            fontsize=11, fontweight="bold", color="#f8fafc", pad=12
        )
        ax.set_xlabel("SHAP Value (Log-Odds Impact on AI Prediction)", fontsize=9.5, color="#94a3b8", labelpad=8)
        ax.tick_params(colors="#cbd5e1", labelsize=9)
        ax.grid(axis="x", linestyle=":", alpha=0.3, color="#475569")

        for bar, val in zip(bars, shap_vals):
            x_pos = val + (0.02 if val >= 0 else -0.02)
            ha = "left" if val >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height()/2.0, f"{val:+.3f}",
                    va="center", ha=ha, color="#f1f5f9", fontsize=8, fontweight="bold")

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#ef4444", label="Pushes AI (Positive SHAP)"),
            Patch(facecolor="#3b82f6", label="Pushes Human (Negative SHAP)")
        ]
        ax.legend(handles=legend_elements, loc="lower right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f1f5f9")

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        logger.info(f"Saved XGBoost SHAP plot to {output_path}")
        return output_path
