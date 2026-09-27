"""
explainability.py
=================
Unified Explainable AI (XAI) Engine for the Hybrid Ensemble AI-Generated Text Detector.

Integrates and orchestrates:
1. Canonical Hybrid Ensemble Inference Pipeline (Authoritative Prediction)
2. Phase 8.2: SHAP Feature Attribution (Meta-Classifier & XGBoost TreeSHAP)
3. Phase 8.3: LIME End-to-End Word Attribution
4. Phase 8.4: Transformer Multi-Model Attention Visualization (RoBERTa, DeBERTa, DistilBERT)

Provides a single unified interface that outputs structured JSON explanations,
organized visualization artifacts, and human-readable narrative summaries.

Author  : Antigravity & Bharanidharan K
Project : Hybrid Explainable Framework for Detecting AI-Generated Academic Text
"""

import os
import io
import sys
import time
import json
import logging
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Ensure repository root is in sys.path
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.ensemble.predict import HybridInferencePipeline
from explainability.shap_explainer import SHAPExplainer
from explainability.lime_explainer import LIMEExplainer
from explainability.attention_explainer import AttentionExplainer

logger = logging.getLogger(__name__)

MANDATORY_INTERPRETABILITY_DISCLAIMER = (
    "SHAP, LIME, and attention visualizations provide complementary interpretability signals. "
    "They should not be interpreted as absolute causal proof of why the model produced a particular classification."
)


def load_text_from_input(input_data: Union[str, Path, bytes], filename: Optional[str] = None) -> str:
    """
    Extract raw plain text from input text strings, file paths, or file bytes.
    Supports TXT, DOCX, and PDF document formats.
    """
    if isinstance(input_data, (str, Path)):
        data_str = str(input_data)
        if os.path.isfile(data_str):
            ext = os.path.splitext(data_str)[1].lower()
            if ext == ".txt":
                with open(data_str, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
            elif ext == ".docx":
                with zipfile.ZipFile(data_str) as z:
                    xml_content = z.read("word/document.xml")
                tree = ET.fromstring(xml_content)
                texts = [elem.text for elem in tree.iter() if elem.tag.endswith("}t") and elem.text]
                return " ".join(texts)
            elif ext == ".pdf":
                try:
                    import pypdf
                    reader = pypdf.PdfReader(data_str)
                    texts = [page.extract_text() for page in reader.pages if page.extract_text()]
                    return "\n".join(texts)
                except Exception:
                    import PyPDF2
                    reader = PyPDF2.PdfReader(data_str)
                    texts = [page.extract_text() for page in reader.pages if page.extract_text()]
                    return "\n".join(texts)
            else:
                with open(data_str, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
        else:
            return data_str

    elif isinstance(input_data, bytes):
        ext = os.path.splitext(filename)[1].lower() if filename else ".txt"
        if ext == ".docx":
            with zipfile.ZipFile(io.BytesIO(input_data)) as z:
                xml_content = z.read("word/document.xml")
            tree = ET.fromstring(xml_content)
            texts = [elem.text for elem in tree.iter() if elem.tag.endswith("}t") and elem.text]
            return " ".join(texts)
        elif ext == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(input_data))
                texts = [page.extract_text() for page in reader.pages if page.extract_text()]
                return "\n".join(texts)
            except Exception:
                import PyPDF2
                reader = PyPDF2.PdfReader(io.BytesIO(input_data))
                texts = [page.extract_text() for page in reader.pages if page.extract_text()]
                return "\n".join(texts)
        else:
            return input_data.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported input type for text loading: {type(input_data)}")


class ExplainabilityEngine:
    """
    Unified Explainable AI Engine orchestrating SHAP, LIME, and Transformer Attention.
    Sits observational above HybridInferencePipeline without mutating predictions or thresholds.
    """

    def __init__(
        self,
        pipeline: Optional[HybridInferencePipeline] = None,
        shap_explainer: Optional[SHAPExplainer] = None,
        lime_explainer: Optional[LIMEExplainer] = None,
        attention_explainer: Optional[AttentionExplainer] = None
    ):
        logger.info("Initializing Unified ExplainabilityEngine...")
        if pipeline is None:
            self.pipeline = HybridInferencePipeline()
        else:
            self.pipeline = pipeline

        self.shap_explainer = shap_explainer or SHAPExplainer(pipeline=self.pipeline)
        self.lime_explainer = lime_explainer or LIMEExplainer(pipeline=self.pipeline)
        self.attention_explainer = attention_explainer or AttentionExplainer(pipeline=self.pipeline)

        logger.info("Unified ExplainabilityEngine initialized successfully.")

    def set_pipeline(self, pipeline: HybridInferencePipeline) -> None:
        """Bind and propagate active HybridInferencePipeline instance across all sub-explainers."""
        self.pipeline = pipeline
        if hasattr(self, "shap_explainer") and self.shap_explainer is not None:
            self.shap_explainer.pipeline = pipeline
            if hasattr(self.shap_explainer, "sync_models"):
                self.shap_explainer.sync_models()
        if hasattr(self, "lime_explainer") and self.lime_explainer is not None:
            self.lime_explainer.pipeline = pipeline
        if hasattr(self, "attention_explainer") and self.attention_explainer is not None:
            if hasattr(self.attention_explainer, "set_pipeline"):
                self.attention_explainer.set_pipeline(pipeline)
            else:
                self.attention_explainer.pipeline = pipeline

    def explain(
        self,
        text_or_file: Union[str, Path, bytes],
        filename: Optional[str] = None,
        save_dir: Optional[Union[str, Path]] = None,
        sample_id: str = "sample",
        num_lime_samples: int = 50,
        num_lime_features: int = 10,
        prediction_result: Optional[Dict] = None
    ) -> Dict:
        """
        Execute the unified XAI pipeline for a given text or document file.
        """
        t_total_0 = time.time()

        # ── 1. Text Extraction & Metadata ─────────────────────────────────────
        text = load_text_from_input(text_or_file, filename=filename)
        if not text or not text.strip():
            raise ValueError("Extracted text input is empty. Cannot generate explanations.")

        words = text.strip().split()
        input_metadata = {
            "source": filename if filename else (str(text_or_file)[:50] if isinstance(text_or_file, (str, Path)) else "raw_bytes"),
            "text_length": len(text),
            "word_count": len(words),
            "approx_token_count": int(len(words) * 1.3)
        }

        # Setup directory paths if saving artifacts
        dir_path = Path(save_dir) if save_dir else None
        if dir_path:
            shap_dir = dir_path / "shap"
            lime_dir = dir_path / "lime"
            attn_dir = dir_path / "attention"
            shap_dir.mkdir(parents=True, exist_ok=True)
            lime_dir.mkdir(parents=True, exist_ok=True)
            attn_dir.mkdir(parents=True, exist_ok=True)
        else:
            shap_dir = lime_dir = attn_dir = None

        # ── 2. Authoritative Hybrid Ensemble Prediction ───────────────────────
        if prediction_result is not None:
            pred_res = prediction_result
            prediction_ms = 0.0
        else:
            t_pred_0 = time.time()
            try:
                pred_res = self.pipeline.predict(text)
            except Exception as e:
                logger.warning(f"explain() pipeline predict error: {e}. Re-instantiating pipeline...")
                from models.ensemble.predict import HybridInferencePipeline
                self.pipeline = HybridInferencePipeline()
                self.set_pipeline(self.pipeline)
                pred_res = self.pipeline.predict(text)
            prediction_ms = (time.time() - t_pred_0) * 1000.0

        pred_summary = {
            "class": pred_res["prediction"],
            "ai_probability": float(pred_res["ai_probability"]),
            "human_probability": float(pred_res["human_probability"]),
            "confidence": float(pred_res["confidence"]),
            "decision_threshold": 0.50,
            "consensus": pred_res.get("consensus", {}),
            "latency_ms": prediction_ms
        }

        bp = pred_res.get("base_probs", {})
        mb = pred_res.get("model_breakdown", {})
        def _get_prob(key, alt_key):
            if key in bp:
                val = bp[key]
                flat = np.ravel(val)
                return float(flat[1] if len(flat) > 1 else flat[0])
            if key in mb:
                v = mb[key]
                return float(v.get("ai_probability", 0.5) if isinstance(v, dict) else v)
            if alt_key in mb:
                v = mb[alt_key]
                return float(v.get("ai_probability", 0.5) if isinstance(v, dict) else v)
            return 0.5

        model_breakdown = {
            "roberta": _get_prob("roberta", "RoBERTa-base"),
            "deberta": _get_prob("deberta", "DeBERTa-v3"),
            "distilbert": _get_prob("distilbert", "DistilBERT"),
            "xgboost": _get_prob("xgboost", "XGBoost (Stylometrics)")
        }

        visualizations = {
            "shap": [],
            "lime": [],
            "attention": []
        }

        # ── 3. SHAP Subsystem Execution ───────────────────────────────────────
        t_shap_0 = time.time()
        shap_result = {}
        try:
            shap_raw = self.shap_explainer.explain(text, prediction_result=pred_res)
            shap_result = {
                "status": "PASS",
                "meta_classifier": shap_raw["meta_classifier"],
                "xgboost": shap_raw["xgboost"],
                "runtime_ms": (time.time() - t_shap_0) * 1000.0
            }

            if dir_path:
                meta_plot = self.shap_explainer.plot_meta_classifier_shap(
                    shap_raw["meta_classifier"],
                    shap_dir / f"{sample_id}_meta_classifier_shap.png",
                    title_suffix=f"({sample_id})"
                )
                xgb_plot = self.shap_explainer.plot_xgboost_shap(
                    shap_raw["xgboost"],
                    shap_dir / f"{sample_id}_xgboost_shap.png",
                    title_suffix=f"({sample_id})"
                )
                visualizations["shap"].extend([str(meta_plot), str(xgb_plot)])

        except Exception as e:
            logger.error(f"SHAP subsystem encountered an error: {e}", exc_info=True)
            shap_result = {
                "status": "FAIL",
                "error": str(e),
                "runtime_ms": (time.time() - t_shap_0) * 1000.0
            }

        # ── 4. LIME Subsystem Execution ───────────────────────────────────────
        t_lime_0 = time.time()
        lime_result = {}
        try:
            lime_raw = self.lime_explainer.explain_text(
                text,
                prediction_result=pred_res,
                num_samples=num_lime_samples,
                num_features=num_lime_features
            )

            lime_result = {
                "status": "PASS",
                "explained_class": lime_raw["prediction"],
                "positive_words": lime_raw["positive_contributions"],
                "negative_words": lime_raw["negative_contributions"],
                "word_attributions": lime_raw["word_attributions"],
                "is_truncated": lime_raw.get("text_truncated", False),
                "runtime_ms": (time.time() - t_lime_0) * 1000.0
            }

            if dir_path:
                lime_plot = self.lime_explainer.plot_word_contributions(
                    lime_raw,
                    lime_dir / f"{sample_id}_lime_attribution.png",
                    title_suffix=f"({sample_id})"
                )
                lime_hl_plot = self.lime_explainer.plot_text_highlighting(
                    lime_raw,
                    lime_dir / f"{sample_id}_lime_highlight.png",
                    title_suffix=f"({sample_id})"
                )
                visualizations["lime"].extend([str(lime_plot), str(lime_hl_plot)])

        except Exception as e:
            logger.error(f"LIME subsystem encountered an error: {e}", exc_info=True)
            lime_result = {
                "status": "FAIL",
                "error": str(e),
                "runtime_ms": (time.time() - t_lime_0) * 1000.0
            }

        # ── 5. Transformer Attention Subsystem Execution ──────────────────────
        t_attn_0 = time.time()
        attention_result = {}
        try:
            attn_all = self.attention_explainer.explain_all_models(
                text,
                prediction_result=pred_res
            )
            roberta_attn = attn_all["models"]["roberta"]
            deberta_attn = attn_all["models"]["deberta"]
            distilbert_attn = attn_all["models"]["distilbert"]

            attention_result = {
                "status": "PASS",
                "roberta": {
                    "num_layers": roberta_attn["num_layers"],
                    "num_heads": roberta_attn["num_heads"],
                    "sequence_length": roberta_attn["sequence_length"],
                    "top_attended_tokens": roberta_attn["top_attended_tokens"]
                },
                "deberta": {
                    "num_layers": deberta_attn["num_layers"],
                    "num_heads": deberta_attn["num_heads"],
                    "sequence_length": deberta_attn["sequence_length"],
                    "top_attended_tokens": deberta_attn["top_attended_tokens"]
                },
                "distilbert": {
                    "num_layers": distilbert_attn["num_layers"],
                    "num_heads": distilbert_attn["num_heads"],
                    "sequence_length": distilbert_attn["sequence_length"],
                    "top_attended_tokens": distilbert_attn["top_attended_tokens"]
                },
                "runtime_ms": (time.time() - t_attn_0) * 1000.0
            }

            if dir_path:
                r_imp = self.attention_explainer.plot_token_importance(
                    roberta_attn, attn_dir / f"{sample_id}_roberta_attention.png", title_suffix=f"({sample_id})"
                )
                r_hm = self.attention_explainer.plot_attention_heatmap(
                    roberta_attn, attn_dir / f"{sample_id}_roberta_heatmap.png", title_suffix=f"({sample_id})"
                )
                d_imp = self.attention_explainer.plot_token_importance(
                    deberta_attn, attn_dir / f"{sample_id}_deberta_attention.png", title_suffix=f"({sample_id})"
                )
                d_hm = self.attention_explainer.plot_attention_heatmap(
                    deberta_attn, attn_dir / f"{sample_id}_deberta_heatmap.png", title_suffix=f"({sample_id})"
                )
                b_imp = self.attention_explainer.plot_token_importance(
                    distilbert_attn, attn_dir / f"{sample_id}_distilbert_attention.png", title_suffix=f"({sample_id})"
                )
                b_hm = self.attention_explainer.plot_attention_heatmap(
                    distilbert_attn, attn_dir / f"{sample_id}_distilbert_heatmap.png", title_suffix=f"({sample_id})"
                )
                comp_plot = self.attention_explainer.plot_cross_model_comparison(
                    attn_all, attn_dir / f"{sample_id}_cross_model_attention.png", title_suffix=f"({sample_id})"
                )
                visualizations["attention"].extend([
                    str(r_imp), str(r_hm), str(d_imp), str(d_hm),
                    str(b_imp), str(b_hm), str(comp_plot)
                ])

        except Exception as e:
            logger.error(f"Attention subsystem encountered an error: {e}", exc_info=True)
            attention_result = {
                "status": "FAIL",
                "error": str(e),
                "runtime_ms": (time.time() - t_attn_0) * 1000.0
            }

        total_ms = (time.time() - t_total_0) * 1000.0
        runtime_summary = {
            "prediction_ms": prediction_ms,
            "shap_ms": shap_result.get("runtime_ms", 0.0),
            "lime_ms": lime_result.get("runtime_ms", 0.0),
            "attention_ms": attention_result.get("runtime_ms", 0.0),
            "total_ms": total_ms
        }

        narrative = self._synthesize_narrative(pred_summary, model_breakdown, shap_result, lime_result, attention_result)

        unified_output = {
            "sample_id": sample_id,
            "input": input_metadata,
            "prediction": pred_summary,
            "model_breakdown": model_breakdown,
            "shap": shap_result,
            "lime": lime_result,
            "attention": attention_result,
            "visualizations": visualizations,
            "runtime": runtime_summary,
            "human_readable_summary": narrative,
            "disclaimer": MANDATORY_INTERPRETABILITY_DISCLAIMER
        }

        if dir_path:
            json_file = dir_path / "result.json"
            summary_file = dir_path / "summary.md"

            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(unified_output, f, indent=2, default=str)

            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(self._format_markdown_summary(unified_output))

            logger.info(f"Persisted unified XAI results and visualizations to {dir_path}")

        return unified_output

    def _synthesize_narrative(
        self,
        pred_summary: Dict,
        model_breakdown: Dict,
        shap_result: Dict,
        lime_result: Dict,
        attention_result: Dict
    ) -> str:
        lines = []
        cls_str = pred_summary["class"]
        conf_pct = pred_summary["confidence"] * 100.0
        ai_pct = pred_summary["ai_probability"] * 100.0

        r_p = float(model_breakdown['roberta'])
        d_p = float(model_breakdown['deberta'])
        b_p = float(model_breakdown['distilbert'])
        x_p = float(model_breakdown['xgboost'])

        lines.append(f"### Classification Summary")
        lines.append(f"The Hybrid Ensemble classifies this document as **{cls_str}** with **{conf_pct:.1f}% confidence** (P(AI) = {ai_pct:.1f}%).")
        lines.append(f"Model Probability Breakdown: RoBERTa: {r_p:.3f} | DeBERTa: {d_p:.3f} | DistilBERT: {b_p:.3f} | XGBoost: {x_p:.3f}\n")

        lines.append(f"### SHAP Feature-Level Evidence")
        if shap_result.get("status") == "PASS":
            top_meta = shap_result["meta_classifier"]["positive_contributions"][:2] if cls_str == "AI Generated" else shap_result["meta_classifier"]["negative_contributions"][:2]
            top_meta_str = ", ".join([f"`{f['feature']}` ({f['shap_value']:+.3f})" for f in top_meta]) if top_meta else "None"
            top_xgb = shap_result["xgboost"]["positive_contributions"][:2] if cls_str == "AI Generated" else shap_result["xgboost"]["negative_contributions"][:2]
            top_xgb_str = ", ".join([f"`{f['feature']}` ({f['shap_value']:+.3f})" for f in top_xgb]) if top_xgb else "None"

            lines.append(f"Top Meta-Classifier SHAP drivers for this classification: {top_meta_str}.")
            lines.append(f"Top XGBoost Stylometric SHAP drivers: {top_xgb_str}.\n")
        else:
            lines.append(f"SHAP explanation unavailable: {shap_result.get('error', 'Error')}\n")

        lines.append(f"### LIME Word-Level Attribution Evidence")
        if lime_result.get("status") == "PASS":
            pos_words = [w["word"] for w in lime_result["positive_words"][:5]]
            neg_words = [w["word"] for w in lime_result["negative_words"][:5]]
            lines.append(f"Words pushing toward AI Generated: {', '.join([f'`{w}`' for w in pos_words]) if pos_words else 'None'}.")
            lines.append(f"Words pushing toward Human Written: {', '.join([f'`{w}`' for w in neg_words]) if neg_words else 'None'}.\n")
        else:
            lines.append(f"LIME explanation unavailable: {lime_result.get('error', 'Error')}\n")

        lines.append(f"### Transformer Self-Attention Evidence")
        if attention_result.get("status") == "PASS":
            rob_tokens = [t["clean_token"] for t in attention_result["roberta"]["top_attended_tokens"][:3]]
            deb_tokens = [t["clean_token"] for t in attention_result["deberta"]["top_attended_tokens"][:3]]
            dis_tokens = [t["clean_token"] for t in attention_result["distilbert"]["top_attended_tokens"][:3]]
            lines.append(f"RoBERTa V2.1 focal tokens: {', '.join([f'`{t}`' for t in rob_tokens])}.")
            lines.append(f"DeBERTa-v3 V2.1 focal tokens: {', '.join([f'`{t}`' for t in deb_tokens])}.")
            lines.append(f"DistilBERT V2.1 focal tokens: {', '.join([f'`{t}`' for t in dis_tokens])}.\n")
        else:
            lines.append(f"Attention explanation unavailable: {attention_result.get('error', 'Error')}\n")

        lines.append(f"> **Important**: {MANDATORY_INTERPRETABILITY_DISCLAIMER}")
        return "\n".join(lines)

    def _format_markdown_summary(self, unified_output: Dict) -> str:
        """
        Format the unified XAI output dictionary into a comprehensive, clean, technical Markdown report.
        Handles optional, missing, or failed subsystem fields safely using .get() accessors.
        Does NOT use raw LaTeX formatting.
        """
        if not isinstance(unified_output, dict):
            unified_output = {}

        lines = []
        lines.append("# Unified XAI Explanation")
        lines.append("")

        # ── 1. Input Metadata ──────────────────────────────────────────────────
        inp = unified_output.get("input", {})
        if isinstance(inp, dict) and inp:
            lines.append("## Input Metadata")
            lines.append("")
            lines.append(f"- **Source**: {inp.get('source', 'Unknown')}")
            lines.append(f"- **Character Count**: {inp.get('text_length', 'N/A')}")
            lines.append(f"- **Word Count**: {inp.get('word_count', 'N/A')}")
            lines.append(f"- **Approx. Token Count**: {inp.get('approx_token_count', 'N/A')}")
            lines.append("")

        # ── 2. Prediction ──────────────────────────────────────────────────────
        pred = unified_output.get("prediction", {})
        if not isinstance(pred, dict):
            pred = {}

        pred_class = pred.get("class", "Unknown")
        ai_prob = float(pred.get("ai_probability", 0.0))
        human_prob = float(pred.get("human_probability", 0.0))
        confidence = float(pred.get("confidence", 0.0))
        threshold = float(pred.get("decision_threshold", 0.50))

        lines.append("## Prediction")
        lines.append("")
        lines.append(f"- **Prediction**: {pred_class}")
        lines.append(f"- **P(AI)**: {ai_prob:.4f} ({ai_prob * 100.0:.1f}%)")
        lines.append(f"- **P(Human)**: {human_prob:.4f} ({human_prob * 100.0:.1f}%)")
        lines.append(f"- **Confidence**: {confidence:.4f} ({confidence * 100.0:.1f}%)")
        lines.append(f"- **Threshold**: {threshold:.2f}")
        lines.append("")

        # ── 3. Model Evidence ──────────────────────────────────────────────────
        mb = unified_output.get("model_breakdown", {})
        if not isinstance(mb, dict):
            mb = {}

        lines.append("## Model Evidence")
        lines.append("")
        lines.append("| Model Name | P(AI) | P(Human) | Predicted Class |")
        lines.append("| :--- | :--- | :--- | :--- |")

        model_display_names = [
            ("RoBERTa-v2.1", "roberta"),
            ("DeBERTa-v3-v2.1", "deberta"),
            ("DistilBERT-v2.1", "distilbert"),
            ("XGBoost Stylometrics", "xgboost")
        ]

        for display_name, key in model_display_names:
            p_ai = float(mb.get(key, 0.0))
            p_hu = 1.0 - p_ai
            m_cls = "AI Generated" if p_ai >= threshold else "Human Written"
            lines.append(f"| {display_name} | {p_ai:.4f} | {p_hu:.4f} | {m_cls} |")

        lines.append("")

        consensus = pred.get("consensus", {})
        if not isinstance(consensus, dict):
            consensus = {}

        agree_count = consensus.get("agreement_count")
        if agree_count is None:
            agree_count = sum(
                1 for _, k in model_display_names
                if (float(mb.get(k, 0.0)) >= threshold if pred_class == "AI Generated" else float(mb.get(k, 0.0)) < threshold)
            )

        agree_pct = consensus.get("agreement_pct") or consensus.get("agree_pct") or (agree_count / 4.0 * 100.0)
        consensus_status = consensus.get("consensus_state") or (
            "UNANIMOUS" if agree_count == 4 else ("MAJORITY CONSENSUS" if agree_count == 3 else "SPLIT EVIDENCE")
        )

        lines.append(f"- **Agreement Count**: {agree_count} / 4 models")
        lines.append(f"- **Agreement Percentage**: {agree_pct:.1f}%")
        lines.append(f"- **Consensus Status**: {consensus_status}")
        lines.append("")

        # ── 4. SHAP Evidence ───────────────────────────────────────────────────
        shap_res = unified_output.get("shap", {})
        if not isinstance(shap_res, dict):
            shap_res = {}

        lines.append("## SHAP Evidence")
        lines.append("")
        if shap_res.get("status") == "PASS":
            # Meta-Classifier SHAP
            meta_shap = shap_res.get("meta_classifier", {})
            meta_attrs = meta_shap.get("attributions", [])
            pos_meta = meta_shap.get("positive_contributions", [])
            neg_meta = meta_shap.get("negative_contributions", [])

            top_meta = (pos_meta if pred_class == "AI Generated" else neg_meta) or meta_attrs
            lines.append("### Meta-Classifier SHAP Attributions")
            if top_meta:
                lines.append("| Feature | Value | SHAP Value | Direction |")
                lines.append("| :--- | :--- | :--- | :--- |")
                for item in top_meta[:5]:
                    feat = item.get("feature", "N/A")
                    val = item.get("value", "N/A")
                    val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
                    sv = float(item.get("shap_value", 0.0))
                    direction = item.get("direction", "N/A")
                    lines.append(f"| {feat} | {val_str} | {sv:+.4f} | {direction} |")
            else:
                lines.append("No Meta-Classifier SHAP features available.")
            lines.append("")

            # XGBoost TreeSHAP
            xgb_shap = shap_res.get("xgboost", {})
            xgb_attrs = xgb_shap.get("attributions", [])
            pos_xgb = xgb_shap.get("positive_contributions", [])
            neg_xgb = xgb_shap.get("negative_contributions", [])

            top_xgb = (pos_xgb if pred_class == "AI Generated" else neg_xgb) or xgb_attrs
            lines.append("### XGBoost TreeSHAP Attributions")
            if top_xgb:
                lines.append("| Feature | Value | TreeSHAP Value | Direction |")
                lines.append("| :--- | :--- | :--- | :--- |")
                for item in top_xgb[:5]:
                    feat = item.get("feature", "N/A")
                    val = item.get("value", "N/A")
                    val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
                    tsv = float(item.get("shap_value", 0.0))
                    direction = item.get("direction", "N/A")
                    lines.append(f"| {feat} | {val_str} | {tsv:+.4f} | {direction} |")
            else:
                lines.append("No XGBoost TreeSHAP features available.")
            lines.append("")
        else:
            err_msg = shap_res.get("error", "SHAP subsystem execution disabled or failed.")
            lines.append(f"SHAP evidence unavailable: {err_msg}")
            lines.append("")

        # ── 5. LIME Evidence ───────────────────────────────────────────────────
        lime_res = unified_output.get("lime", {})
        if not isinstance(lime_res, dict):
            lime_res = {}

        lines.append("## LIME Evidence")
        lines.append("")
        if lime_res.get("status") == "PASS":
            pos_words = lime_res.get("positive_words", [])
            neg_words = lime_res.get("negative_words", [])

            if pos_words or neg_words:
                if pos_words:
                    lines.append("### AI-Supporting Words")
                    lines.append("| Word | LIME Weight |")
                    lines.append("| :--- | :--- |")
                    for item in pos_words[:5]:
                        w = item.get("word", "")
                        wt = float(item.get("weight", 0.0))
                        lines.append(f"| `{w}` | {wt:+.4f} |")
                    lines.append("")

                if neg_words:
                    lines.append("### Human-Supporting Words")
                    lines.append("| Word | LIME Weight |")
                    lines.append("| :--- | :--- |")
                    for item in neg_words[:5]:
                        w = item.get("word", "")
                        wt = float(item.get("weight", 0.0))
                        lines.append(f"| `{w}` | {wt:+.4f} |")
                    lines.append("")
            else:
                lines.append("No directional LIME words were available.")
                lines.append("")
        else:
            err_msg = lime_res.get("error", "LIME subsystem execution disabled or failed.")
            lines.append(f"LIME evidence unavailable: {err_msg}")
            lines.append("")

        # ── 6. Transformer Attention ───────────────────────────────────────────
        attn_res = unified_output.get("attention", {})
        if not isinstance(attn_res, dict):
            attn_res = {}

        lines.append("## Transformer Attention")
        lines.append("")
        if attn_res.get("status") == "PASS":
            for model_key, model_label in [("roberta", "RoBERTa-v2.1"), ("deberta", "DeBERTa-v3-v2.1"), ("distilbert", "DistilBERT-v2.1")]:
                m_attn = attn_res.get(model_key, {})
                if isinstance(m_attn, dict) and m_attn:
                    seq_len = m_attn.get("sequence_length", "N/A")
                    tokens = m_attn.get("top_attended_tokens", [])
                    lines.append(f"### {model_label}")
                    lines.append(f"- **Sequence Length**: {seq_len}")
                    if tokens:
                        lines.append("")
                        lines.append("| Token | Clean Token | Attention Score |")
                        lines.append("| :--- | :--- | :--- |")
                        for t_item in tokens[:5]:
                            raw_t = t_item.get("token", "")
                            clean_t = t_item.get("clean_token", "")
                            score = float(t_item.get("score", 0.0))
                            lines.append(f"| `{raw_t}` | `{clean_t}` | {score:.4f} |")
                    lines.append("")
        else:
            err_msg = attn_res.get("error", "Attention subsystem execution disabled or failed.")
            lines.append(f"Transformer attention evidence unavailable: {err_msg}")
            lines.append("")

        # ── 7. Dynamic Decision Explanation ───────────────────────────────────
        lines.append("## Dynamic Decision Explanation")
        lines.append("")
        narrative = unified_output.get("human_readable_summary") or unified_output.get("dynamic_explanation")
        if narrative:
            lines.append(str(narrative))
        else:
            lines.append("Dynamic decision explanation is available when complete XAI feature attributions are computed.")
        lines.append("")

        # ── 8. Methodological Notes ───────────────────────────────────────────
        lines.append("## Methodological Notes")
        lines.append("")
        lines.append("- **SHAP**: Represents feature-level attribution across ensemble model features.")
        lines.append("- **LIME**: Represents local word-level approximation of overall classifier outputs.")
        lines.append("- **Attention**: Represents internal transformer token-routing patterns across transformer layers.")
        lines.append("- These complementary interpretability signals provide multi-perspective observational insight.")
        lines.append("- They are **not** causal proof of why the model produced a particular classification.")
        lines.append("")
        disclaimer = unified_output.get("disclaimer", MANDATORY_INTERPRETABILITY_DISCLAIMER)
        lines.append(f"> **Important**: {disclaimer}")
        lines.append("")

        return "\n".join(lines)

FEATURE_HUMAN_READABLE_MAP = {
    "mean_transformer_probability": "Average AI Probability across Transformer Models",
    "model_agreement_count": "Number of Base Models Agreeing on Classification",
    "roberta_probability": "RoBERTa V2.1 Sequence Classifier Probability",
    "deberta_probability": "DeBERTa-v3 V2.1 Sequence Classifier Probability",
    "distilbert_probability": "DistilBERT V2.1 Sequence Classifier Probability",
    "xgboost_probability": "XGBoost Stylometrics Probability",
    "std_transformer_probability": "Probability Variance across Transformer Models",
    "max_transformer_probability": "Maximum AI Probability among Transformers",
    "min_transformer_probability": "Minimum AI Probability among Transformers",
    "ttr": "Lexical Diversity (Type-Token Ratio)",
    "avg_sentence_len": "Average Sentence Length (Words / Sentence)",
    "avg_word_len": "Average Word Length (Chars / Word)",
    "word_count": "Total Document Word Count",
    "char_count": "Total Character Count",
    "sentence_count": "Total Sentence Count"
}


def generate_dynamic_decision_explanation(xai_res: Dict) -> Dict:
    """
    Generate dynamic, document-specific decision explanation structure from actual XAI results.
    Phase 8.6.5: Reorganized into a clean, 6-section research-grade hierarchy:
    1. Final Decision
    2. Model Evidence & Consensus
    3. Feature-Level Evidence (SHAP & TreeSHAP)
    4. Word-Level Evidence (LIME)
    5. Transformer Attention (Tokens)
    6. Overall Interpretation & Synthesis
    Strictly avoids raw LaTeX formatting, contradictory model-consensus statements, and causal claims.
    """
    if not isinstance(xai_res, dict):
        xai_res = {}

    pred_data = xai_res.get("prediction", {})
    if not isinstance(pred_data, dict):
        pred_data = {}

    cls_val = pred_data.get("class") or xai_res.get("class", "Human Written")
    threshold = float(pred_data.get("decision_threshold") or xai_res.get("decision_threshold", 0.50))
    p_ai = float(pred_data.get("ai_probability") if "ai_probability" in pred_data else xai_res.get("ai_probability", 0.5))
    p_hu = float(pred_data.get("human_probability") if "human_probability" in pred_data else xai_res.get("human_probability", 0.5))
    conf_val = float(pred_data.get("confidence") if "confidence" in pred_data else xai_res.get("confidence", 0.5)) * 100.0

    # Extract base model probabilities
    mb = pred_data.get("model_breakdown", {}) or xai_res.get("model_breakdown", {})
    if not isinstance(mb, dict):
        mb = {}

    r_p = float(mb.get('roberta', 0.5))
    d_p = float(mb.get('deberta', 0.5))
    b_p = float(mb.get('distilbert', 0.5))
    x_p = float(mb.get('xgboost', 0.5))

    # Evaluate individual base model verdicts
    r_vote = "AI Generated" if r_p >= threshold else "Human Written"
    d_vote = "AI Generated" if d_p >= threshold else "Human Written"
    b_vote = "AI Generated" if b_p >= threshold else "Human Written"
    x_vote = "AI Generated" if x_p >= threshold else "Human Written"

    model_votes = [
        ("RoBERTa-v2.1", r_vote, r_p),
        ("DeBERTa-v3-v2.1", d_vote, d_p),
        ("DistilBERT-v2.1", b_vote, b_p),
        ("XGBoost Stylometrics", x_vote, x_p)
    ]

    supporting_models = [name for name, vote, _ in model_votes if vote == cls_val]
    opposing_models = [name for name, vote, _ in model_votes if vote != cls_val]
    agree_count = len(supporting_models)

    # ── 1. Model Consensus Wording & State ──────────────────────────────────────
    if agree_count == 4:
        consensus_state = f"UNANIMOUS ({agree_count}/4 MODELS)"
        consensus_wording = f"All four base models favor **{cls_val}**, providing unanimous model-level support (4/4 models)."
    elif agree_count == 3:
        consensus_state = f"MAJORITY CONSENSUS ({agree_count}/4 MODELS)"
        supp_str = ", ".join(supporting_models)
        opp_str = ", ".join(opposing_models)
        consensus_wording = f"Three of four base models ({supp_str}) favor **{cls_val}**, providing majority model-level support. {opp_str} favors the opposing class."
    elif agree_count == 2:
        consensus_state = "EVENLY DIVIDED (2/4 MODELS)"
        supp_str = ", ".join(supporting_models)
        opp_str = ", ".join(opposing_models)
        consensus_wording = f"Base-model evidence is evenly divided: {supp_str} favor **{cls_val}**, while {opp_str} favor the opposing class (2/4 models support each class). The stacking meta-classifier synthesizes these probabilities and feature signals rather than relying on a simple majority vote."
    elif agree_count == 1:
        consensus_state = f"SPLIT EVIDENCE ({agree_count}/4 MODELS)"
        supp_str = ", ".join(supporting_models)
        opp_str = ", ".join(opposing_models)
        consensus_wording = f"Base-model evidence is divided: {supp_str} favors **{cls_val}**, while {opp_str} favor the opposing class (1/4 model support). The stacking meta-classifier combines individual model logits and stylometric features to reach the final decision."
    else:
        consensus_state = "META-CLASSIFIER OVERRIDE (0/4 MODELS)"
        opp_str = ", ".join(opposing_models)
        consensus_wording = f"Base models individually favored the opposing class ({opp_str}), but the stacking meta-classifier combined high-dimensional logit and stylometric features to classify the document as **{cls_val}**."

    # ── 2. Margin Separation Wording ──────────────────────────────────────────
    margin = abs(p_ai - threshold)
    if margin < 0.15:
        margin_wording = "The final probability is relatively close to the 0.50 decision boundary, indicating weaker separation between the two classes."
    else:
        margin_wording = "The final probability is substantially separated from the 0.50 decision boundary, indicating strong separation between the classes."

    # ── 3. SHAP Feature Attribution Extraction ─────────────────────────────────
    shap_data = xai_res.get("shap", {})
    meta_attrs = shap_data.get("meta_classifier", {}).get("attributions", []) if shap_data.get("status") == "PASS" else []
    xgb_attrs = shap_data.get("xgboost", {}).get("attributions", []) if shap_data.get("status") == "PASS" else []

    meta_supporting = [
        {
            "feature_raw": f.get("feature", "N/A"),
            "feature_human": FEATURE_HUMAN_READABLE_MAP.get(f.get("feature", ""), str(f.get("feature", "")).replace("_", " ").title()),
            "category": f.get("category", "Meta"),
            "value": f.get("value", 0.0),
            "shap_value": float(f.get("shap_value", 0.0)),
            "direction": str(f.get("direction", "")).replace("Pushes ", "")
        }
        for f in meta_attrs
        if (float(f.get("shap_value", 0.0)) > 0 if cls_val == "AI Generated" else float(f.get("shap_value", 0.0)) < 0)
    ]
    meta_supporting.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    meta_opposing = [
        {
            "feature_raw": f.get("feature", "N/A"),
            "feature_human": FEATURE_HUMAN_READABLE_MAP.get(f.get("feature", ""), str(f.get("feature", "")).replace("_", " ").title()),
            "category": f.get("category", "Meta"),
            "value": f.get("value", 0.0),
            "shap_value": float(f.get("shap_value", 0.0)),
            "direction": str(f.get("direction", "")).replace("Pushes ", "")
        }
        for f in meta_attrs
        if (float(f.get("shap_value", 0.0)) <= 0 if cls_val == "AI Generated" else float(f.get("shap_value", 0.0)) >= 0)
    ]
    meta_opposing.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    xgb_supporting = [
        {
            "feature_raw": f.get("feature", "N/A"),
            "feature_human": FEATURE_HUMAN_READABLE_MAP.get(f.get("feature", ""), str(f.get("feature", "")).replace("_", " ").title()),
            "category": f.get("category", "Stylometrics"),
            "value": f.get("value", 0.0),
            "shap_value": float(f.get("shap_value", 0.0)),
            "direction": str(f.get("direction", "")).replace("Pushes ", "")
        }
        for f in xgb_attrs
        if (float(f.get("shap_value", 0.0)) > 0 if cls_val == "AI Generated" else float(f.get("shap_value", 0.0)) < 0)
    ]
    xgb_supporting.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    xgb_opposing = [
        {
            "feature_raw": f.get("feature", "N/A"),
            "feature_human": FEATURE_HUMAN_READABLE_MAP.get(f.get("feature", ""), str(f.get("feature", "")).replace("_", " ").title()),
            "category": f.get("category", "Stylometrics"),
            "value": f.get("value", 0.0),
            "shap_value": float(f.get("shap_value", 0.0)),
            "direction": str(f.get("direction", "")).replace("Pushes ", "")
        }
        for f in xgb_attrs
        if (float(f.get("shap_value", 0.0)) <= 0 if cls_val == "AI Generated" else float(f.get("shap_value", 0.0)) >= 0)
    ]
    xgb_opposing.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    # ── 4. LIME Word Attribution Extraction ────────────────────────────────────
    lime_data = xai_res.get("lime", {})
    pos_lime = []
    neg_lime = []
    if isinstance(lime_data, dict) and lime_data.get("status") == "PASS":
        pos_lime = [w["word"] for w in lime_data.get("positive_words", [])[:5] if isinstance(w, dict) and "word" in w]
        neg_lime = [w["word"] for w in lime_data.get("negative_words", [])[:5] if isinstance(w, dict) and "word" in w]

    supporting_lime = pos_lime if cls_val == "AI Generated" else neg_lime
    opposing_lime = neg_lime if cls_val == "AI Generated" else pos_lime

    # ── 5. Attention Focal Token Extraction ────────────────────────────────────
    attn_data = xai_res.get("attention", {})
    rob_toks = []
    deb_toks = []
    dis_toks = []
    if isinstance(attn_data, dict) and attn_data.get("status") == "PASS":
        rob_toks = [t["clean_token"] for t in attn_data.get("roberta", {}).get("top_attended_tokens", [])[:3] if isinstance(t, dict) and "clean_token" in t]
        deb_toks = [t["clean_token"] for t in attn_data.get("deberta", {}).get("top_attended_tokens", [])[:3] if isinstance(t, dict) and "clean_token" in t]
        dis_toks = [t["clean_token"] for t in attn_data.get("distilbert", {}).get("top_attended_tokens", [])[:3] if isinstance(t, dict) and "clean_token" in t]

    all_attention_tokens = list(dict.fromkeys(rob_toks + deb_toks + dis_toks))[:5]

    # ── 6. Construct Structured Markdown Sections ──────────────────────────────
    sections = []

    # Section A: Final Decision
    sections.append("### Final Decision\n")
    sections.append(
        f"The Hybrid Stacking Ensemble classified this document as **{cls_val}** with **{conf_val:.1f}% confidence**. "
        f"P(AI) = {p_ai*100.0:.1f}% and P(Human) = {p_hu*100.0:.1f}%, placing the final probability on the **{cls_val}** "
        f"side of the {threshold:.2f} decision boundary. {margin_wording}\n"
    )

    # Section B: Model Evidence
    sections.append("### Model Evidence\n")
    sections.append(f"{consensus_wording}\n")
    sections.append(f"- **RoBERTa-v2.1**: P(AI) = {r_p*100.0:.1f}% | P(Human) = {(1.0-r_p)*100.0:.1f}% ({r_vote})")
    sections.append(f"- **DeBERTa-v3-v2.1**: P(AI) = {d_p*100.0:.1f}% | P(Human) = {(1.0-d_p)*100.0:.1f}% ({d_vote})")
    sections.append(f"- **DistilBERT-v2.1**: P(AI) = {b_p*100.0:.1f}% | P(Human) = {(1.0-b_p)*100.0:.1f}% ({b_vote})")
    sections.append(f"- **XGBoost Stylometrics**: P(AI) = {x_p*100.0:.1f}% | P(Human) = {(1.0-x_p)*100.0:.1f}% ({x_vote})\n")

    # Section C: Feature-Level Evidence
    sections.append("### Feature-Level Evidence\n")
    top_features = meta_supporting[:2] + xgb_supporting[:2]
    if top_features:
        sections.append(f"The strongest attribution signals directed toward **{cls_val}** were:")
        for feat in top_features:
            h_name = feat["feature_human"]
            r_name = feat["feature_raw"]
            val_sv = feat["shap_value"]
            sections.append(f"- **{h_name}** (`{r_name}`) — SHAP: {val_sv:+.4f}")
        sections.append(f"\nThese feature-level contributions were directed toward the **{cls_val}** classification.\n")
    else:
        sections.append("No usable SHAP feature attributions were produced for the active explanation window.\n")

    # Section D: Word-Level Evidence
    sections.append("### Word-Level Evidence\n")
    if supporting_lime:
        words_formatted = ", ".join([f"`{w}`" for w in supporting_lime[:3]])
        sections.append(
            f"LIME identified {words_formatted} as the strongest local **{cls_val}**-directed word-level contributions "
            f"within the active explanation window. These represent local perturbation attribution signals from LIME "
            f"and do not independently determine the overall classification.\n"
        )
    else:
        sections.append("No usable LIME word-level attribution was produced for the active explanation window.\n")

    # Section E: Transformer Attention
    sections.append("### Transformer Attention\n")
    if all_attention_tokens:
        tok_str = ", ".join([f"`{t}`" for t in all_attention_tokens])
        sections.append(
            f"High-attention tokenizer subword tokens included {tok_str}. "
            f"These tokens received relatively high attention under transformer attention aggregation and represent "
            f"structural token-routing signals. They should not be interpreted as causal evidence.\n"
        )
    else:
        sections.append("No usable transformer attention visualization was produced for the active explanation window.\n")

    # Section F: Overall Interpretation
    sections.append("### Overall Interpretation\n")
    sections.append(
        f"Taken together, the final ensemble probability and the available feature-, word-, and transformer-level "
        f"attribution signals provide complementary evidence consistent with the **{cls_val}** classification. "
        f"The SHAP, LIME, and attention outputs represent observational interpretability signals rather than "
        f"independent causal proof.\n"
    )

    narrative_text = "\n".join(sections).strip()

    # Conflict Analysis
    has_conflict = (agree_count < 4) or (len(meta_opposing) > 0 and abs(meta_opposing[0]["shap_value"]) > 0.05) or (len(opposing_lime) > 0)
    conflict_message = ""
    if has_conflict:
        if agree_count < 4:
            conflict_message = f"Note: Base models show split agreement ({agree_count}/4 models support {cls_val}). The stacking meta-classifier synthesized these probabilities into the final logit."
        elif len(opposing_lime) > 0:
            conflict_message = f"Note: While the overall prediction favors {cls_val}, some local words ({', '.join([f'`{w}`' for w in opposing_lime[:2]])}) locally push toward opposing classification. LIME provides a local perturbation approximation."

    return {
        "class_val": cls_val,
        "confidence_val": conf_val,
        "p_ai": p_ai,
        "p_human": p_hu,
        "agree_count": agree_count,
        "consensus_state": consensus_state,
        "consensus_wording": consensus_wording,
        "margin_wording": margin_wording,
        "meta_supporting": meta_supporting,
        "meta_opposing": meta_opposing,
        "xgb_supporting": xgb_supporting,
        "xgb_opposing": xgb_opposing,
        "supporting_lime": supporting_lime,
        "opposing_lime": opposing_lime,
        "pos_lime": pos_lime,
        "neg_lime": neg_lime,
        "roberta_tokens": rob_toks,
        "deberta_tokens": deb_toks,
        "distilbert_tokens": dis_toks,
        "dynamic_narrative": narrative_text,
        "has_conflict": has_conflict,
        "conflict_message": conflict_message
    }



