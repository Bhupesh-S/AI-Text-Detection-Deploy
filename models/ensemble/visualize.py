"""
visualize.py
============
Plotting utilities for comparing models and ensemble strategies.

Author  : Bharanidharan K
Project : Hybrid Explainable Framework for Detecting AI-Generated Academic Text
"""

import sys
from pathlib import Path
from typing import Dict, List, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# Make config importable
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from utils import get_logger

logger = get_logger(__name__)

# Colors
PALETTE = {
    "RoBERTa-base":            "#1f77b4",
    "DeBERTa-v3-base":         "#ff7f0e",
    "DistilBERT-base-uncased": "#2ca02c",
    "XGBoost":                 "#d62728",
    "Soft Voting":             "#9467bd",
    "Weighted Voting":         "#8c564b",
    "Stacking":                "#e377c2"
}

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.framealpha": 0.8,
    }
)

PathLike = Union[str, Path]


def _save(fig: plt.Figure, output_path: PathLike, title: str) -> None:
    """Save figure and close it."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved {title} → {path}")


# ── 1. Model Accuracy Comparison ─────────────────────────────────────────────

def plot_accuracy_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    output_path: PathLike,
) -> None:
    """Generate a bar chart comparing accuracy across models."""
    models = list(metrics_dict.keys())
    accuracies = [metrics_dict[m]["accuracy"] for m in models]
    f1s = [metrics_dict[m]["f1"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', color="#4A90D9")
    rects2 = ax.bar(x + width/2, f1s, width, label='F1 Score', color="#E57373")

    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison across Models and Ensembles', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha='right')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0.0, 1.1)
    ax.legend(loc='lower left')

    # Add text labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height == height:  # NaN check
                ax.annotate(f'{height*100:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)

    _save(fig, output_path, "Accuracy Comparison")


# ── 2. ROC Curves Comparison ─────────────────────────────────────────────────

def plot_roc_comparison(
    labels: np.ndarray,
    probs_dict: Dict[str, np.ndarray],
    output_path: PathLike,
) -> None:
    """Plot ROC curves for all models in a single chart."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, probs in probs_dict.items():
        # probs are (N, 2); extract AI generated probability (index 1)
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        color = PALETTE.get(name, "#7f7f7f")
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Comparison (All Models & Ensembles)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")

    _save(fig, output_path, "ROC Comparison")


# ── 3. Precision-Recall Comparison ───────────────────────────────────────────

def plot_pr_comparison(
    labels: np.ndarray,
    probs_dict: Dict[str, np.ndarray],
    output_path: PathLike,
) -> None:
    """Plot PR curves for all models in a single chart."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for name, probs in probs_dict.items():
        precision, recall, _ = precision_recall_curve(labels, probs[:, 1])
        ap = average_precision_score(labels, probs[:, 1])
        color = PALETTE.get(name, "#7f7f7f")
        ax.plot(recall, precision, color=color, lw=2, label=f"{name} (AP = {ap:.4f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left")

    _save(fig, output_path, "PR Comparison")


# ── 4. Confusion Matrix ──────────────────────────────────────────────────────

def plot_confusion_matrix_heatmap(
    labels: np.ndarray,
    preds: np.ndarray,
    output_path: PathLike,
    model_name: str,
) -> None:
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Human Written", "AI Generated"])
    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=13, fontweight="bold")
    plt.xticks(rotation=15, ha="right")

    _save(fig, output_path, f"Confusion Matrix ({model_name})")


# ── 5. Runtime Comparison ────────────────────────────────────────────────────

def plot_runtime_comparison(
    train_times: Dict[str, float],
    infer_times: Dict[str, float],
    output_dir: PathLike,
) -> None:
    """Plot Training Time and Inference Time comparisons in separate files."""
    out_dir = Path(output_dir)

    # 5.1 Training Times (seconds)
    models = list(train_times.keys())
    t_times = [train_times[m] for m in models]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [PALETTE.get(m, "#7f7f7f") for m in models]
    bars = ax.barh(models, t_times, color=colors, height=0.5)
    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Training Time Comparison (Log Scale)", fontsize=13, fontweight="bold")
    ax.set_xscale("log")  # Ensembles take <1s while deep models take hours

    # Add text labels on bars
    for bar, val in zip(bars, t_times):
        if val > 0:
            if val < 60:
                label_text = f"{val:.1f}s"
            elif val < 3600:
                label_text = f"{val/60:.1f}m"
            else:
                label_text = f"{val/3600:.1f}h"
            ax.annotate(label_text,
                        xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0),
                        textcoords="offset points",
                        ha='left', va='center', fontsize=9)

    _save(fig, out_dir / "training_time_comparison.png", "Training Time Comparison")

    # 5.2 Inference Times (seconds)
    models_inf = list(infer_times.keys())
    i_times = [infer_times[m] for m in models_inf]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors_inf = [PALETTE.get(m, "#7f7f7f") for m in models_inf]
    bars_inf = ax.barh(models_inf, i_times, color=colors_inf, height=0.5)
    ax.set_xlabel("Inference Time (seconds)")
    ax.set_title("Inference Latency Comparison (Test Set)", fontsize=13, fontweight="bold")

    for bar, val in zip(bars_inf, i_times):
        if val > 0:
            ax.annotate(f"{val:.2f}s",
                        xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0),
                        textcoords="offset points",
                        ha='left', va='center', fontsize=9)

    _save(fig, out_dir / "inference_time_comparison.png", "Inference Time Comparison")


# ── 6. Ensemble Weight Distribution ──────────────────────────────────────────

def plot_weight_distribution(
    voting_weights: Dict[str, float],
    stacking_coefs: Dict[str, float],
    output_path: PathLike,
) -> None:
    """Plot comparison of custom voting weights vs stacking Meta-model coefficients."""
    models = list(voting_weights.keys())
    v_w = [voting_weights[m] for m in models]

    # Stacking coefficients might be negative/large, normalize them for display
    s_c = [stacking_coefs[m] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    rects1 = ax.bar(x - width/2, v_w, width, label='Voting Weights (Normalized)', color="#4A90D9")
    rects2 = ax.bar(x + width/2, s_c, width, label='Stacking Coefficients (Raw Logits)', color="#AB47BC")

    ax.set_ylabel('Weight Value / Coefficient')
    ax.set_title('Model Weights: Weighted Voting vs. Stacking Meta-Classifier', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    _save(fig, output_path, "Weight Distribution")
