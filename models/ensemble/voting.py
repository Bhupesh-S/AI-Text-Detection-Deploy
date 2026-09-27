"""
voting.py
=========
Implements Soft Voting and Weighted Voting ensemble methods.

Author  : Bharanidharan K
Project : Hybrid Explainable Framework for Detecting AI-Generated Academic Text
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Make config importable
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG
from utils import get_logger

logger = get_logger(__name__)


def soft_voting_predict(probs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Average prediction probabilities from all models (Soft Voting).

    Args:
        probs : Dict containing model names mapped to probabilities arrays of shape (N, 2).
                Expected models: "roberta", "deberta", "distilbert", "xgboost".

    Returns:
        Tuple of:
          - Final combined probabilities (N, 2)
          - Final class predictions (N,)
    """
    logger.info("Computing Soft Voting Ensemble predictions...")

    # Verify models
    models = ["roberta", "deberta", "distilbert", "xgboost"]
    for m in models:
        if m not in probs:
            raise KeyError(f"Soft voting requires probability arrays for model '{m}'.")

    # Sum up probabilities
    sum_probs = np.zeros_like(probs["roberta"])
    for m in models:
        sum_probs += probs[m]

    final_probs = sum_probs / len(models)
    preds = np.argmax(final_probs, axis=-1)

    return final_probs, preds


def weighted_voting_predict(
    probs: Dict[str, np.ndarray],
    weights: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Average prediction probabilities using user-defined weights.
    Weights are automatically normalized to sum to 1.0.

    Args:
        probs   : Dict mapping model names to class probabilities (N, 2).
        weights : Dict mapping model names to float weights.
                  Expected keys: "roberta", "deberta", "distilbert", "xgboost".

    Returns:
        Tuple of:
          - Weighted combined probabilities (N, 2)
          - Weighted class predictions (N,)
    """
    logger.info("Computing Weighted Voting Ensemble predictions...")

    models = ["roberta", "deberta", "distilbert", "xgboost"]
    for m in models:
        if m not in probs:
            raise KeyError(f"Weighted voting requires probability arrays for model '{m}'.")
        if m not in weights:
            raise KeyError(f"Weights dict missing weight for model '{m}'.")

    # Extract weights and normalize
    raw_weights = [weights[m] for m in models]
    sum_w = sum(raw_weights)
    if sum_w <= 0.0:
        raise ValueError("Sum of weights must be greater than zero.")

    normalized_weights = {m: weights[m] / sum_w for m in models}
    logger.info(f"Normalized weights → {normalized_weights}")

    # Calculate weighted sum
    weighted_probs = np.zeros_like(probs["roberta"])
    for m in models:
        weighted_probs += probs[m] * normalized_weights[m]

    preds = np.argmax(weighted_probs, axis=-1)
    return weighted_probs, preds
