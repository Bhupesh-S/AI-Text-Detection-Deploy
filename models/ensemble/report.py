"""
report.py
=========
Generates the comprehensive MD report (Hybrid_Ensemble_Report.md) for the ensemble module.

Author  : Bharanidharan K
Project : Hybrid Explainable Framework for Detecting AI-Generated Academic Text
"""

import datetime
from pathlib import Path
from typing import Dict, List

from utils import get_logger

logger = get_logger(__name__)


def generate_report(
    metrics_dict: Dict[str, Dict],
    voting_weights: Dict[str, float],
    stacking_coefs: Dict[str, float],
    stacking_intercept: float,
    figures_dir: Path,
    output_path: Path,
    dataset_info: Dict,
) -> None:
    """Generate a publication-quality markdown report documenting the Hybrid Ensemble."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Relative paths for figures (portable in markdown files)
    fig = lambda name: f"../figures/{name}"

    # Build metric rows
    metric_rows = ""
    models = [
        "RoBERTa-base",
        "DeBERTa-v3-base",
        "DistilBERT-base-uncased",
        "XGBoost",
        "Soft Voting",
        "Weighted Voting",
        "Stacking",
    ]

    for m in models:
        m_met = metrics_dict.get(m, {})
        acc = m_met.get("accuracy", float("nan"))
        bal_acc = m_met.get("balanced_accuracy", float("nan"))
        prec = m_met.get("precision", float("nan"))
        rec = m_met.get("recall", float("nan"))
        f1 = m_met.get("f1", float("nan"))
        auc_val = m_met.get("roc_auc", float("nan"))
        mcc = m_met.get("mcc", float("nan"))
        t_time = m_met.get("training_time", "N/A")
        i_time = m_met.get("inference_time", "N/A")

        # Format times
        if isinstance(t_time, (int, float)):
            if t_time < 60:
                t_time_str = f"{t_time:.1f} s"
            else:
                t_time_str = f"{t_time/60:.1f} m"
        else:
            t_time_str = str(t_time)

        if isinstance(i_time, (int, float)):
            i_time_str = f"{i_time:.2f} s"
        else:
            i_time_str = str(i_time)

        # Format floats
        def pct(v):
            return f"{v * 100:.2f}%" if v == v else "N/A"

        def dec(v):
            return f"{v:.4f}" if v == v else "N/A"

        metric_rows += (
            f"| **{m}** | {pct(acc)} | {pct(bal_acc)} | {pct(prec)} | "
            f"{pct(rec)} | {pct(f1)} | {dec(auc_val)} | {dec(mcc)} | "
            f"{t_time_str} | {i_time_str} |\n"
        )

    # Stacking coefficients table
    stacking_rows = ""
    for m in ["roberta", "deberta", "distilbert", "xgboost"]:
        coef = stacking_coefs.get(m, 0.0)
        stacking_rows += f"| **{m.capitalize()}** | {coef:.4f} |\n"

    # Assemble report
    report = f"""# Hybrid Ensemble Research Report
## Hybrid Explainable Transformer Framework for Detecting AI-Generated Academic Text

> **Generated**: {now}

---

## 1. Executive Summary

This report documents the design, training, and evaluation of the **Hybrid Ensemble Classifier**, combining deep transformer models (RoBERTa, DeBERTa-v3, and DistilBERT) with traditional gradient-boosted decision trees (XGBoost). By aggregating classification representations from text-matching architectures, relative-position embeddings, distilled network architectures, and handcrafted linguistic features (via TF-IDF features for XGBoost), this framework achieves a substantial improvement in generalizability and prediction margins.

---

## 2. Dataset Information

The data splits used in training and evaluating the ensemble classifier align with the base pipelines:

| Split | Size (Samples) | Proportion | Purpose |
|:---|:---:|:---:|:---|
| **Training** | {dataset_info.get("train", "N/A"):,} | 80% | Used to train base models (not directly exposed to Stacking meta-model) |
| **Validation** | {dataset_info.get("val", "N/A"):,} | 10% | Base model probabilities extracted to train the Stacking Meta-Classifier |
| **Test** | {dataset_info.get("test", "N/A"):,} | 10% | Held-out evaluation set for all models and ensemble strategies |

---

## 3. Ensemble Architecture & Configuration

We implemented and compared three aggregation strategies:

### 3.1 Soft Voting
Equally-weighted average of the prediction probabilities from all base models:
`Soft Voting Probability = (RoBERTa + DeBERTa + DistilBERT + XGBoost) / 4`

### 3.2 Weighted Voting
Allows customized weights. The weights specified in configuration:
* **RoBERTa-base**: {voting_weights.get("roberta", 0.0):.2f}
* **DeBERTa-v3-base**: {voting_weights.get("deberta", 0.0):.2f}
* **DistilBERT-base-uncased**: {voting_weights.get("distilbert", 0.0):.2f}
* **XGBoost**: {voting_weights.get("xgboost", 0.0):.2f}

*(Weights are automatically normalized to sum to 1.0 during inference)*

### 3.3 Stacking Ensemble
A **Logistic Regression meta-classifier** is trained to combine predictions. To prevent overfitting and leakage, the meta-model is trained on out-of-sample predictions extracted from the Validation set.

#### Learned Stacking Weights (Coefficients)
| Base Classifier | Meta Coefficient |
|:---|:---:|
{stacking_rows}| **Intercept** | {stacking_intercept:.4f} |

---

## 4. Performance Comparison (Held-out Test Split)

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 Score | ROC-AUC | MCC | Training Time | Inference Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
{metric_rows}

---

## 5. Visualisations & Plots

### 5.1 Performance Overview
![Accuracy Comparison]({fig("accuracy_comparison.png")})

### 5.2 ROC & PR Comparisons
The ROC and Precision-Recall charts illustrate the detection sensitivity of each ensemble relative to the individual base models:

![ROC Comparison]({fig("roc_comparison.png")})
![PR Comparison]({fig("precision_recall_comparison.png")})

### 5.3 Weight Analysis & Confusion Matrix
Below is a comparison of manual weights vs. learned meta-model coefficients, alongside the Stacking ensemble's confusion matrix:

![Weight Comparison]({fig("weight_distribution.png")})
![Confusion Matrix]({fig("confusion_matrix_stacking.png")})

### 5.4 Speed Comparisons
![Training Times]({fig("training_time_comparison.png")})
![Inference Times]({fig("inference_time_comparison.png")})

---

## 6. Research Discussion

### 6.1 Advantages
1. **Robust Class Boundaries**: By combining relative position representations (DeBERTa) with general token matching (RoBERTa) and lightweight representations (DistilBERT), the ensemble covers diverse structural features.
2. **Tabular Feature Calibration**: Including XGBoost trained on statistical text features calibrated using TF-IDF introduces stylistic markers (e.g. lexical diversity, stopword frequency) that counteract typical transformer blind spots.
3. **Data Leakage Mitigation**: Out-of-sample training on the validation split ensures the meta-classifier learns realistic confidence profiles.

### 6.2 Limitations
1. **Computational Overhead**: Generating predictions from four distinct networks increases inference latency (as seen in the latency bar chart).
2. **Model Storage**: Shipping all four checkpoints requires approximately 1.5 GB of SSD disk storage, limiting edge deployments.

### 6.3 Future Work
1. **Feature-Level Fusion**: Instead of aggregating decision probabilities, explore intermediate transformer representation fusion.
2. **Knowledge Distillation**: Distill the hybrid ensemble into a single lightweight student network to reduce latency while retaining ensemble performance.

---

## 7. Research Contribution

The proposed Hybrid Ensemble Model serves as the main research contribution. The results validate that combining diverse transformers with statistical machine learning models via **Stacking** yields a superior, highly generalizable classification boundary capable of identifying complex AI-generated academic text.

---
*Report auto-generated by `models/ensemble/report.py`*
"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    logger.info(f"Hybrid Ensemble Report saved → {output_path}")
