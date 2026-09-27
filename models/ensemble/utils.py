"""
utils.py
=========
Shared utilities and performance evaluation metrics for the Ensemble model.

Author  : Bharanidharan K
Project : Hybrid Explainable Framework for Detecting AI-Generated Academic Text
"""

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Reconfigure stdout/stderr stream encoding for UTF-8 on Windows terminal streams.
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass


# ---------------------------------------------------------------------------
# Logger Factory
# ---------------------------------------------------------------------------

def get_logger(name: str, log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Create a named logger that outputs to console and optionally to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix random seeds across environments for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# JSON Serialization Helpers
# ---------------------------------------------------------------------------

def save_json(data: Dict[str, Any], path: Path) -> None:
    """Save a dict to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=_json_serializer)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON to dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for non-serializable objects (numpy types, Paths, etc.)."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Metrics Calculator
# ---------------------------------------------------------------------------

def evaluate_predictions(
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics:
      - Accuracy, Balanced Accuracy, Precision, Recall, F1
      - ROC-AUC
      - Matthews Correlation Coefficient (MCC)
      - Confusion Matrix
      - Classification Report
    """
    # Safety casting
    labels = np.array(labels, dtype=int)
    preds = np.array(preds, dtype=int)
    probs = np.array(probs, dtype=float)

    acc = accuracy_score(labels, preds)
    bal_acc = balanced_accuracy_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = float("nan")

    cm = confusion_matrix(labels, preds)
    report = classification_report(
        labels, preds, target_names=["Human Written", "AI Generated"], output_dict=True
    )
    report_str = classification_report(
        labels, preds, target_names=["Human Written", "AI Generated"]
    )

    return {
        "accuracy": round(float(acc), 6),
        "balanced_accuracy": round(float(bal_acc), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "roc_auc": round(float(roc_auc), 6),
        "mcc": round(float(mcc), 6),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "classification_report_str": report_str,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    """Format elapsed seconds to a readable string (H:MM:SS)."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def print_section(title: str, width: int = 70) -> None:
    """Print section headers."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
