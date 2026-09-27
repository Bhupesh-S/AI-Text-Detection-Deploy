"""
lime_explainer.py
=================
LIME Explainability subsystem for the Hybrid Ensemble AI-Generated Text Detector.

Provides end-to-end word-level local text attribution for the complete HybridEnferencePipeline.

Author  : Antigravity & Bharanidharan K
Project : Hybrid Explainable Framework for Detecting AI-Generated Academic Text
"""

import sys
import time
import logging
import html
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
import matplotlib
matplotlib.use('Agg')  # Headless rendering
import matplotlib.pyplot as plt

# Ensure repository root is in sys.path
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.ensemble.predict import HybridInferencePipeline

logger = logging.getLogger(__name__)


class LIMEExplainer:
    """
    End-to-End LIME Word-Level Text Explainer wrapper for HybridInferencePipeline.
    Executes perturbation samples through the actual full ensemble inference path.
    """

    def __init__(
        self,
        pipeline: Optional[HybridInferencePipeline] = None,
        num_samples: int = 100,
        num_features: int = 10,
        random_state: int = 42
    ):
        logger.info("Initializing LIMEExplainer subsystem...")
        if pipeline is None:
            self.pipeline = HybridInferencePipeline()
        else:
            self.pipeline = pipeline

        self.num_samples = num_samples
        self.num_features = num_features
        self.random_state = random_state

        self.class_names = ["Human Written", "AI Generated"]

        self.lime_explainer = LimeTextExplainer(
            class_names=self.class_names,
            random_state=self.random_state,
            bow=True,
            split_expression=r'\W+'
        )

        logger.info(
            f"LIMEExplainer initialized cleanly. Configuration: num_samples={self.num_samples}, "
            f"num_features={self.num_features}, random_state={self.random_state}"
        )

    def predict_fn_ensemble(self, texts: List[str]) -> np.ndarray:
        """
        Batch prediction function wrapper passed to LimeTextExplainer.
        Evaluates input texts through the authoritative HybridInferencePipeline.

        Returns:
            np.ndarray of shape (N, 2):
              Column 0 = P(Human Written)
              Column 1 = P(AI Generated)
        """
        probs = []
        for text in texts:
            # Handle empty or whitespace perturbation samples gracefully
            if not text or not text.strip():
                probs.append([0.50, 0.50])
                continue

            try:
                res = self.pipeline.predict(text)
                p_hu = float(res["human_probability"])
                p_ai = float(res["ai_probability"])
                probs.append([p_hu, p_ai])
            except Exception as e:
                logger.warning(f"LIME perturbation evaluation failed for sample snippet: '{text[:30]}...': {e}")
                probs.append([0.50, 0.50])

        return np.array(probs)

    def explain_text(
        self,
        text: str,
        prediction_result: Optional[Dict] = None,
        num_samples: Optional[int] = None,
        num_features: Optional[int] = None
    ) -> Dict:
        """
        Compute end-to-end word-level LIME explanation for the given text.

        Measures runtime breakdown: base_prediction_time_ms, lime_explanation_time_ms, total_xai_time_ms.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty for LIME explanation.")

        samples_to_use = num_samples if num_samples is not None else self.num_samples
        features_to_use = num_features if num_features is not None else self.num_features

        # 1. Base Prediction Time
        t_base_0 = time.time()
        if prediction_result is None:
            prediction_result = self.pipeline.predict(text)
        base_prediction_time_ms = (time.time() - t_base_0) * 1000.0

        # Handle long document truncation for perturbation window
        # Base models truncate at max_length=512 tokens (~350-400 words)
        words = text.strip().split()
        is_truncated = False
        explained_text = text

        if len(words) > 500:
            is_truncated = True
            explained_words = words[:500]
            explained_text = " ".join(explained_words)
            logger.info(
                f"Input text contains {len(words)} words. Truncating to active 500-word explanation window "
                f"to prevent computational explosion while keeping full model compatibility."
            )

        # 2. LIME Explanation Time
        t_lime_0 = time.time()

        try:
            exp = self.lime_explainer.explain_instance(
                explained_text,
                self.predict_fn_ensemble,
                labels=(1,),  # Explain class 1 (AI Generated)
                num_features=features_to_use,
                num_samples=samples_to_use
            )
        except Exception as e:
            logger.error(f"LIME explain_instance failed: {e}")
            raise RuntimeError(f"LIME calculation error: {e}")

        lime_explanation_time_ms = (time.time() - t_lime_0) * 1000.0
        total_xai_time_ms = base_prediction_time_ms + lime_explanation_time_ms

        # Extract word weights relative to Class 1 (AI Generated)
        # Positive weight => Pushes prediction toward AI Generated
        # Negative weight => Pushes prediction toward Human Written
        weights_list = exp.as_list(label=1)

        word_attributions = []
        positive_contributions = []
        negative_contributions = []
        word_weight_map = {}

        for word, weight in weights_list:
            w_val = float(weight)
            direction = "Pushes AI" if w_val >= 0 else "Pushes Human"
            importance = abs(w_val)
            word_weight_map[word.lower()] = w_val

            entry = {
                "word": word,
                "weight": w_val,
                "direction": direction,
                "importance": importance
            }
            word_attributions.append(entry)

            if w_val >= 0:
                positive_contributions.append(entry)
            else:
                negative_contributions.append(entry)

        positive_contributions.sort(key=lambda x: x["importance"], reverse=True)
        negative_contributions.sort(key=lambda x: x["importance"], reverse=True)
        word_attributions.sort(key=lambda x: x["importance"], reverse=True)

        # Build HTML text highlight string
        html_highlight = self._generate_html_highlight(explained_text, word_weight_map)

        return {
            "model": "HybridInferencePipeline (End-to-End Ensemble)",
            "explanation_method": "LIME (Local Interpretable Model-agnostic Explanations)",
            "prediction": prediction_result["prediction"],
            "ai_probability": float(prediction_result["ai_probability"]),
            "human_probability": float(prediction_result["human_probability"]),
            "confidence": float(prediction_result["confidence"]),
            "decision_threshold": 0.50,
            "strategy_used": prediction_result.get("strategy_used", "Stacking Meta-Classifier"),
            "num_samples_evaluated": samples_to_use,
            "num_features_extracted": features_to_use,
            "text_truncated": is_truncated,
            "explained_word_count": len(explained_text.split()),
            "runtimes": {
                "base_prediction_ms": round(base_prediction_time_ms, 1),
                "lime_explanation_ms": round(lime_explanation_time_ms, 1),
                "total_xai_ms": round(total_xai_time_ms, 1)
            },
            "word_attributions": word_attributions,
            "positive_contributions": positive_contributions,
            "negative_contributions": negative_contributions,
            "word_weight_map": word_weight_map,
            "html_highlight": html_highlight
        }

    def explain(
        self,
        text: str,
        prediction_result: Optional[Dict] = None
    ) -> Dict:
        """Alias method matching project unified XAI convention."""
        return self.explain_text(text, prediction_result)

    def _generate_html_highlight(self, text: str, weight_map: Dict[str, float]) -> str:
        """Generate HTML string with colored inline word spans based on LIME weights."""
        words = text.split()
        spans = []

        max_weight = max([abs(w) for w in weight_map.values()]) if weight_map else 1.0
        if max_weight == 0:
            max_weight = 1.0

        for w in words:
            clean_w = "".join([c for c in w.lower() if c.isalnum()])
            if clean_w in weight_map:
                wt = weight_map[clean_w]
                alpha = min(abs(wt) / max_weight * 0.5 + 0.15, 0.75)
                if wt >= 0:
                    bg = f"rgba(239, 68, 68, {alpha:.2f})"   # Red shade for AI
                    color = "#ffffff"
                    title = f"Pushes AI: {wt:+.4f}"
                else:
                    bg = f"rgba(59, 130, 246, {alpha:.2f})"  # Blue shade for Human
                    color = "#ffffff"
                    title = f"Pushes Human: {wt:+.4f}"

                escaped_w = html.escape(w)
                spans.append(
                    f'<span style="background-color: {bg}; color: {color}; padding: 2px 4px; '
                    f'border-radius: 4px; font-weight: 600;" title="{title}">{escaped_w}</span>'
                )
            else:
                spans.append(html.escape(w))

        return " ".join(spans)

    # ── Visualizations ────────────────────────────────────────────────────────

    def plot_word_contributions(
        self,
        lime_result: Dict,
        output_path: Union[str, Path],
        title_suffix: str = ""
    ) -> Path:
        """
        Generate feature contribution horizontal bar chart for LIME word attributions.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        pos_words = lime_result.get("positive_contributions", [])[:5]
        neg_words = lime_result.get("negative_contributions", [])[:5]
        top_attrs = sorted(pos_words + neg_words, key=lambda x: x["importance"], reverse=True)
        top_attrs.reverse()  # Bottom-to-top

        words = [a["word"] for a in top_attrs]
        weights = [a["weight"] for a in top_attrs]
        colors = ["#ef4444" if w >= 0 else "#3b82f6" for w in weights]

        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=300)
        fig.patch.set_facecolor("#0b1120")
        ax.set_facecolor("#0b1120")

        bars = ax.barh(words, weights, color=colors, edgecolor="none", height=0.55)

        ax.axvline(0, color="#64748b", linewidth=1.2, linestyle="--")
        runtimes = lime_result["runtimes"]
        ax.set_title(
            f"LIME Word Attributions (Top 5 AI & Top 5 Human Words) {title_suffix}\n"
            f"Pred: {lime_result['prediction']} | P(AI): {lime_result['ai_probability']:.4f}",
            fontsize=11, fontweight="bold", color="#f8fafc", pad=12
        )
        ax.set_xlabel("LIME Weight (Attribution to AI-Generated Prediction)", fontsize=9.5, color="#94a3b8", labelpad=8)
        ax.tick_params(colors="#cbd5e1", labelsize=9)
        ax.grid(axis="x", linestyle=":", alpha=0.3, color="#475569")

        for bar, val in zip(bars, weights):
            x_pos = val + (0.002 if val >= 0 else -0.002)
            ha = "left" if val >= 0 else "right"
            ax.text(x_pos, bar.get_y() + bar.get_height()/2.0, f"{val:+.3f}",
                    va="center", ha=ha, color="#f1f5f9", fontsize=8, fontweight="bold")

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#ef4444", label="Pushes AI (Positive LIME Weight)"),
            Patch(facecolor="#3b82f6", label="Pushes Human (Negative LIME Weight)")
        ]
        ax.legend(handles=legend_elements, loc="lower right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f1f5f9")

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        logger.info(f"Saved LIME word contribution plot to {output_path}")
        return output_path

    def plot_text_highlighting(
        self,
        lime_result: Dict,
        output_path: Union[str, Path],
        title_suffix: str = ""
    ) -> Path:
        """
        Generate text highlight card visualization plot demonstrating token-level impact.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
        fig.patch.set_facecolor("#0b1120")
        ax.set_facecolor("#0b1120")
        ax.axis("off")

        # Header Title
        title_text = (
            f"LIME Word Attribution Visual Highlight {title_suffix}\n"
            f"Pred: {lime_result['prediction']} | P(AI): {lime_result['ai_probability']:.4f} | Conf: {lime_result['confidence']:.4f}"
        )
        ax.text(0.02, 0.92, title_text, transform=ax.transAxes, color="#f8fafc",
                fontsize=11, fontweight="bold", verticalalignment="top")

        # Top Positive and Top Negative Word Summary Cards
        pos_words = [f"{a['word']} ({a['weight']:+.3f})" for a in lime_result["positive_contributions"][:5]]
        neg_words = [f"{a['word']} ({a['weight']:+.3f})" for a in lime_result["negative_contributions"][:5]]

        pos_str = "  ".join(pos_words) if pos_words else "None"
        neg_str = "  ".join(neg_words) if neg_words else "None"

        ax.text(0.02, 0.70, "Words Supporting AI Generated (Red):", transform=ax.transAxes, color="#ef4444", fontsize=9.5, fontweight="bold")
        ax.text(0.02, 0.60, pos_str, transform=ax.transAxes, color="#fca5a5", fontsize=9, wrap=True)

        ax.text(0.02, 0.40, "Words Supporting Human Written (Blue):", transform=ax.transAxes, color="#3b82f6", fontsize=9.5, fontweight="bold")
        ax.text(0.02, 0.30, neg_str, transform=ax.transAxes, color="#93c5fd", fontsize=9, wrap=True)

        # Performance Metadata footer
        runtimes = lime_result["runtimes"]
        footer_text = (
            f"LIME Perturbations: {lime_result['num_samples_evaluated']} | "
            f"Base Pred: {runtimes['base_prediction_ms']} ms | "
            f"LIME Time: {runtimes['lime_explanation_ms']/1000:.2f} s | "
            f"Total XAI Time: {runtimes['total_xai_ms']/1000:.2f} s"
        )
        ax.text(0.02, 0.10, footer_text, transform=ax.transAxes, color="#64748b", fontsize=8.5, style="italic")

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        logger.info(f"Saved LIME text highlight plot to {output_path}")
        return output_path
