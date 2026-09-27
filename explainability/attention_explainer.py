"""
attention_explainer.py
======================
Transformer Attention Visualization subsystem for the Hybrid Ensemble AI-Generated Text Detector.

Supports:
- RoBERTa V2.1 (12 layers, 12 heads)
- DeBERTa-v3 V2.1 (12 layers, 12 heads)
- DistilBERT V2.1 (6 layers, 12 heads)

Author  : Antigravity & Bharanidharan K
Project : Hybrid Explainable Framework for Detecting AI-Generated Academic Text
"""

import sys
import time
import logging
import html
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import matplotlib
matplotlib.use('Agg')  # Headless rendering
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure repository root is in sys.path
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.ensemble.config import CFG
from models.ensemble.predict import HybridInferencePipeline

logger = logging.getLogger(__name__)

import unicodedata

SPECIAL_TOKENS = {
    "<s>", "</s>", "[CLS]", "[SEP]", "[PAD]", "<pad>", "<unk>", "[UNK]",
    "[MASK]", "<mask]", "<mask>", "<null>", "[NULL]", "<eos>", "<bos>"
}

def is_meaningful_content_token(raw_token: str, clean_token: str) -> bool:
    """
    Returns True if the token is a valid content-bearing lexical token.
    Returns False for punctuation-only, symbol-only, whitespace, special tokens, or artifacts.
    """
    if raw_token in SPECIAL_TOKENS or clean_token in SPECIAL_TOKENS:
        return False
    if raw_token.upper() in SPECIAL_TOKENS or clean_token.upper() in SPECIAL_TOKENS:
        return False
    if not clean_token or not clean_token.strip():
        return False
    if not any(c.isalnum() for c in clean_token):
        return False
    if all(unicodedata.category(c)[0] in ('P', 'S', 'Z', 'C') for c in clean_token):
        return False
    return True


class AttentionExplainer:
    """
    Subsystem for extracting, aggregating, mapping, and visualizing multi-head self-attention
    tensors from fine-tuned PyTorch transformer models in the Hybrid Ensemble.
    """

    def __init__(self, pipeline: Optional[HybridInferencePipeline] = None):
        logger.info("Initializing AttentionExplainer subsystem...")
        if pipeline is None:
            self.pipeline = HybridInferencePipeline()
        else:
            self.pipeline = pipeline

        self.device = self.pipeline.device

        # Direct binding to pipeline's loaded eager transformer models
        self.model_roberta = self.pipeline.model_roberta
        self.model_deberta = self.pipeline.model_deberta
        self.model_distilbert = self.pipeline.model_distilbert

        # Ensure eager attention implementation flag on configs
        for m in [self.model_roberta, self.model_deberta, self.model_distilbert]:
            if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
                m.config._attn_implementation = "eager"
            m.eval()

        self.models = {
            "roberta": (self.model_roberta, self.pipeline.tokenizer_roberta),
            "deberta": (self.model_deberta, self.pipeline.tokenizer_deberta),
            "distilbert": (self.model_distilbert, self.pipeline.tokenizer_distilbert)
        }

        logger.info("AttentionExplainer subsystem initialized successfully with eager attention models for RoBERTa, DeBERTa, and DistilBERT.")

    def set_pipeline(self, pipeline: HybridInferencePipeline) -> None:
        """Update and bind models to the active pipeline instance."""
        self.pipeline = pipeline
        self.device = pipeline.device
        self.model_roberta = pipeline.model_roberta
        self.model_deberta = pipeline.model_deberta
        self.model_distilbert = pipeline.model_distilbert
        for m in [self.model_roberta, self.model_deberta, self.model_distilbert]:
            if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
                m.config._attn_implementation = "eager"
            m.eval()
        self.models = {
            "roberta": (self.model_roberta, self.pipeline.tokenizer_roberta),
            "deberta": (self.model_deberta, self.pipeline.tokenizer_deberta),
            "distilbert": (self.model_distilbert, self.pipeline.tokenizer_distilbert)
        }

    def extract_raw_attentions(
        self,
        text: str,
        model_name: str = "roberta"
    ) -> Tuple[List[str], torch.Tensor, Dict]:
        """
        Extract raw multi-layer multi-head attention tensors for a given model.

        Returns:
            tokens: List of raw string tokens.
            attentions_stack: Tensor of shape (num_layers, num_heads, seq_len, seq_len).
            metadata: Model metadata dict.
        """
        model_key = model_name.lower()
        if model_key not in self.models:
            raise ValueError(f"Unknown model_name '{model_name}'. Must be one of {list(self.models.keys())}")

        model, tokenizer = self.models[model_key]
        model.eval()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"][0]
        raw_tokens = tokenizer.convert_ids_to_tokens(input_ids)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        if not hasattr(outputs, "attentions") or outputs.attentions is None or len(outputs.attentions) == 0:
            raise RuntimeError(f"Model '{model_name}' did not return attention tensors with output_attentions=True.")

        # attentions is a tuple of (1, num_heads, seq_len, seq_len) tensors per layer
        layer_tensors = [attn[0].cpu() for attn in outputs.attentions]
        attentions_stack = torch.stack(layer_tensors, dim=0)

        num_layers, num_heads, seq_len, _ = attentions_stack.shape
        metadata = {
            "model_name": model_name,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "sequence_length": seq_len,
            "raw_token_count": len(raw_tokens)
        }

        return raw_tokens, attentions_stack, metadata

    def aggregate_attention(
        self,
        attentions_stack: torch.Tensor,
        method: str = "last_layer_head_avg"
    ) -> np.ndarray:
        """
        Aggregate multi-layer multi-head attention matrix into a single (seq_len, seq_len) matrix.

        Supported methods:
        - 'last_layer_head_avg': Average across 12 heads in the final layer.
        - 'all_layer_head_avg': Average across all layers and heads.
        - 'attention_rollout': Abnar & Zuidema (2020) attention rollout algorithm.
        """
        num_layers, num_heads, seq_len, _ = attentions_stack.shape

        if method == "last_layer_head_avg":
            # Select last layer, average across heads => (seq_len, seq_len)
            last_layer = attentions_stack[-1]  # (num_heads, seq_len, seq_len)
            agg_matrix = torch.mean(last_layer, dim=0).numpy()
        elif method == "all_layer_head_avg":
            # Average across all layers and heads => (seq_len, seq_len)
            agg_matrix = torch.mean(attentions_stack, dim=(0, 1)).numpy()
        elif method == "attention_rollout":
            # Rollout: V_l = 0.5 * A_l + 0.5 * I
            ey = torch.eye(seq_len)
            rollout = torch.eye(seq_len)
            for l in range(num_layers):
                layer_avg = torch.mean(attentions_stack[l], dim=0)
                a_reg = 0.5 * layer_avg + 0.5 * ey
                a_norm = a_reg / torch.sum(a_reg, dim=-1, keepdim=True)
                rollout = torch.matmul(a_norm, rollout)
            agg_matrix = rollout.numpy()
        else:
            last_layer = attentions_stack[-1]
            agg_matrix = torch.mean(last_layer, dim=0).numpy()

        return agg_matrix

    def map_tokens_and_importance(
        self,
        raw_tokens: List[str],
        agg_matrix: np.ndarray,
        model_name: str
    ) -> Tuple[List[Dict], List[str], np.ndarray]:
        """
        Map raw tokens to clean human-readable tokens and compute token importance scores.
        Excludes special tokens and punctuation from token importance rankings while preserving matrix shape for heatmaps.
        """
        seq_len = len(raw_tokens)
        
        # Token importance score: total incoming attention received by each token across sequence
        importance_scores = np.mean(agg_matrix, axis=0)

        cleaned_token_items = []
        clean_labels = []

        for idx, t in enumerate(raw_tokens):
            clean_t = t.replace("Ġ", "").replace("▁", "").replace(" ", "").replace("##", "").strip()
            
            # Robust token filtering: check if content-bearing lexical token
            is_content = is_meaningful_content_token(t, clean_t)
            is_special = not is_content

            clean_label = clean_t if clean_t else t
            clean_labels.append(clean_label)

            item = {
                "index": idx,
                "raw_token": t,
                "clean_token": clean_label,
                "is_special": is_special,
                "importance_score": float(importance_scores[idx])
            }
            cleaned_token_items.append(item)

        return cleaned_token_items, clean_labels, importance_scores

    def explain_text(
        self,
        text: str,
        model_name: str = "roberta",
        aggregation_method: str = "last_layer_head_avg",
        prediction_result: Optional[Dict] = None
    ) -> Dict:
        """
        Compute transformer attention explanation for a single model.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty for attention explanation.")

        t0 = time.time()
        if prediction_result is None:
            prediction_result = self.pipeline.predict(text)
        base_pred_ms = (time.time() - t0) * 1000.0

        t1 = time.time()
        raw_tokens, attentions_stack, metadata = self.extract_raw_attentions(text, model_name)
        agg_matrix = self.aggregate_attention(attentions_stack, method=aggregation_method)
        cleaned_token_items, clean_labels, importance_scores = self.map_tokens_and_importance(
            raw_tokens, agg_matrix, model_name
        )
        attention_extract_ms = (time.time() - t1) * 1000.0
        total_ms = base_pred_ms + attention_extract_ms

        # Exclude special tokens and punctuation for non-special token attributions list
        valid_attributions = [item for item in cleaned_token_items if not item["is_special"]]
        valid_attributions.sort(key=lambda x: x["importance_score"], reverse=True)

        return {
            "model_name": model_name,
            "prediction": prediction_result["prediction"],
            "ai_probability": float(prediction_result["ai_probability"]),
            "human_probability": float(prediction_result["human_probability"]),
            "confidence": float(prediction_result["confidence"]),
            "decision_threshold": 0.50,
            "num_layers": metadata["num_layers"],
            "num_heads": metadata["num_heads"],
            "sequence_length": metadata["sequence_length"],
            "aggregation_method": aggregation_method,
            "selected_layer": "Last Layer (12)" if aggregation_method == "last_layer_head_avg" else "All Layers",
            "disclaimer": (
                "Attention is an interpretability signal showing model focus under the selected aggregation strategy. "
                "Attention should not automatically be interpreted as causal evidence or probability contributions for the final classification."
            ),
            "runtimes": {
                "base_prediction_ms": round(base_pred_ms, 1),
                "attention_extraction_ms": round(attention_extract_ms, 1),
                "total_xai_ms": round(total_ms, 1)
            },
            "raw_tokens": raw_tokens,
            "clean_labels": clean_labels,
            "attention_matrix": agg_matrix.tolist(),
            "token_items": cleaned_token_items,
            "top_attended_tokens": valid_attributions[:8]
        }

    def explain_all_models(
        self,
        text: str,
        aggregation_method: str = "last_layer_head_avg",
        prediction_result: Optional[Dict] = None
    ) -> Dict:
        """
        Compute transformer attention explanations across RoBERTa, DeBERTa, and DistilBERT.
        """
        if prediction_result is None:
            prediction_result = self.pipeline.predict(text)

        explanations = {}
        for m_name in ["roberta", "deberta", "distilbert"]:
            explanations[m_name] = self.explain_text(
                text,
                model_name=m_name,
                aggregation_method=aggregation_method,
                prediction_result=prediction_result
            )

        return {
            "prediction": prediction_result["prediction"],
            "ai_probability": float(prediction_result["ai_probability"]),
            "human_probability": float(prediction_result["human_probability"]),
            "confidence": float(prediction_result["confidence"]),
            "models": explanations
        }

    # ── Visualizations ────────────────────────────────────────────────────────

    def plot_token_importance(
        self,
        att_res: Dict,
        output_path: Union[str, Path],
        title_suffix: str = ""
    ) -> Path:
        """
        Generate bar chart of top attended tokens for a model (Top 8 meaningful tokens).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        top_tokens = att_res["top_attended_tokens"][:8]
        top_tokens.reverse()  # Bottom-to-top

        labels = [t["clean_token"] for t in top_tokens]
        scores = [t["importance_score"] for t in top_tokens]

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        fig.patch.set_facecolor("#0b1120")
        ax.set_facecolor("#0b1120")

        m_name = att_res["model_name"].upper()
        color = "#818cf8" if m_name == "ROBERTA" else ("#38bdf8" if m_name == "DEBERTA" else "#a78bfa")

        bars = ax.barh(labels, scores, color=color, edgecolor="none", height=0.6)

        ax.set_title(
            f"{m_name} Top Meaningful Token Attention Scores {title_suffix}\n"
            f"Pred: {att_res['prediction']} | P(AI): {att_res['ai_probability']:.4f} | Method: {att_res['aggregation_method']}",
            fontsize=12, fontweight="bold", color="#f8fafc", pad=15
        )
        ax.set_xlabel("Aggregated Attention Score (Received Attention Weight)", fontsize=10, color="#94a3b8", labelpad=10)
        ax.tick_params(colors="#cbd5e1", labelsize=9)
        if scores:
            ax.set_xlim(0, max(scores) * 1.22)

        for bar, val in zip(bars, scores):
            ax.text(val + (max(scores)*0.015 if scores else 0.001), bar.get_y() + bar.get_height()/2.0, f"{val:.4f}",
                    va="center", ha="left", color="#f1f5f9", fontsize=8, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        logger.info(f"Saved token importance plot for {m_name} to {output_path}")
        return output_path

    def plot_attention_heatmap(
        self,
        att_res: Dict,
        output_path: Union[str, Path],
        title_suffix: str = ""
    ) -> Path:
        """
        Generate compact 10x10 2D matrix heatmap of token-to-token attention weights.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        agg_matrix = np.array(att_res["attention_matrix"])
        labels = att_res["clean_labels"]

        # Limit heatmap size to compact 10 tokens for clean readability
        limit = min(10, len(labels))
        sub_matrix = agg_matrix[:limit, :limit]
        sub_labels = labels[:limit]

        fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
        fig.patch.set_facecolor("#0b1120")
        ax.set_facecolor("#0b1120")

        sns.heatmap(
            sub_matrix,
            xticklabels=sub_labels,
            yticklabels=sub_labels,
            cmap="mako",
            ax=ax,
            cbar_kws={"label": "Attention Weight"}
        )

        m_name = att_res["model_name"].upper()
        ax.set_title(
            f"{m_name} Token-to-Token Attention Matrix Heatmap {title_suffix}\n"
            f"Shown: First {limit} Tokens | Method: {att_res['aggregation_method']}",
            fontsize=11, fontweight="bold", color="#f8fafc", pad=12
        )
        ax.tick_params(colors="#cbd5e1", labelsize=8, rotation=45)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        logger.info(f"Saved attention heatmap for {m_name} to {output_path}")
        return output_path

    def plot_cross_model_comparison(
        self,
        all_res: Dict,
        output_path: Union[str, Path],
        title_suffix: str = ""
    ) -> Path:
        """
        Generate 3-panel side-by-side comparison chart of token attention across RoBERTa, DeBERTa, and DistilBERT.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=300)
        fig.patch.set_facecolor("#0b1120")

        model_colors = {"roberta": "#818cf8", "deberta": "#38bdf8", "distilbert": "#a78bfa"}

        for idx, m_name in enumerate(["roberta", "deberta", "distilbert"]):
            ax = axes[idx]
            ax.set_facecolor("#0b1120")

            m_data = all_res["models"][m_name]
            top_tokens = m_data["top_attended_tokens"][:8]
            top_tokens.reverse()

            labels = [t["clean_token"] for t in top_tokens]
            scores = [t["importance_score"] for t in top_tokens]

            bars = ax.barh(labels, scores, color=model_colors[m_name], edgecolor="none", height=0.6)
            ax.set_title(f"{m_name.upper()} Top Tokens", fontsize=11, fontweight="bold", color="#f8fafc")
            ax.set_xlabel("Attention Score", fontsize=9, color="#94a3b8")
            ax.tick_params(colors="#cbd5e1", labelsize=8)
            ax.grid(axis="x", linestyle=":", alpha=0.3, color="#475569")
            if scores:
                ax.set_xlim(0, max(scores) * 1.25)

            for bar, val in zip(bars, scores):
                ax.text(val + (max(scores)*0.015 if scores else 0.001), bar.get_y() + bar.get_height()/2.0, f"{val:.3f}",
                        va="center", ha="left", color="#f1f5f9", fontsize=7.5, fontweight="bold")

        fig.suptitle(
            f"Cross-Model Transformer Attention Comparison {title_suffix}\n"
            f"Ensemble Pred: {all_res['prediction']} | P(AI): {all_res['ai_probability']:.4f}",
            fontsize=13, fontweight="bold", color="#f8fafc", y=1.02
        )

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        logger.info(f"Saved cross-model comparison plot to {output_path}")
        return output_path
