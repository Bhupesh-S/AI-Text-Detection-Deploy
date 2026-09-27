"""
calibration.py
==============
Implements probability calibration techniques (Temperature Scaling and Platt Scaling)
and calibration evaluation metrics (Expected Calibration Error, Maximum Calibration Error, Brier Score).

Author  : Antigravity
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = (y_prob >= bin_edges[b]) & (y_prob < bin_edges[b+1])
        if b == n_bins - 1:
            mask = mask | (y_prob == 1.0)
        bin_size = np.sum(mask)
        if bin_size > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_size / n) * abs(bin_acc - bin_conf)
    return float(ece)

def calculate_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Maximum Calibration Error (MCE)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    mce = 0.0
    for b in range(n_bins):
        mask = (y_prob >= bin_edges[b]) & (y_prob < bin_edges[b+1])
        if b == n_bins - 1:
            mask = mask | (y_prob == 1.0)
        bin_size = np.sum(mask)
        if bin_size > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            mce = max(mce, abs(bin_acc - bin_conf))
    return float(mce)

def calculate_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Compute ECE, MCE, and Brier Score calibration metrics."""
    # Ensure 1D positive probabilities
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]
    ece = calculate_ece(y_true, y_prob, n_bins)
    mce = calculate_mce(y_true, y_prob, n_bins)
    brier = brier_score_loss(y_true, y_prob)
    return {
        "ece": round(ece, 6),
        "mce": round(mce, 6),
        "brier_score": round(brier, 6)
    }

class TemperatureScaler:
    """Calibrates deep learning models by scaling pre-softmax logits by a temperature T."""
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        def loss_fn(T):
            scaled = logits / T[0]
            # Softmax
            exp_logits = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            return log_loss(y_true, probs[:, 1])

        res = minimize(loss_fn, [1.0], bounds=[(0.1, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        exp_logits = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return probs

class PlattScaler:
    """Calibrates model output probabilities using logistic calibration (Platt Scaling)."""
    def __init__(self):
        self.lr = None

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "PlattScaler":
        if probs.ndim == 2:
            probs = probs[:, 1]
        epsilon = 1e-15
        probs_clipped = np.clip(probs, epsilon, 1.0 - epsilon)
        logits = np.log(probs_clipped / (1.0 - probs_clipped)).reshape(-1, 1)
        self.lr = LogisticRegression(C=1.0, solver="lbfgs")
        self.lr.fit(logits, y_true)
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        if probs.ndim == 2:
            p_cls1 = probs[:, 1]
        else:
            p_cls1 = probs
        epsilon = 1e-15
        p_clipped = np.clip(p_cls1, epsilon, 1.0 - epsilon)
        logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
        return self.lr.predict_proba(logits)


class IsotonicScaler:
    """Calibrates model output probabilities using non-parametric Isotonic Regression."""
    def __init__(self):
        from sklearn.isotonic import IsotonicRegression
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "IsotonicScaler":
        if probs.ndim == 2:
            probs = probs[:, 1]
        self.iso.fit(probs, y_true)
        return self

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        if probs.ndim == 2:
            p_cls1 = probs[:, 1]
        else:
            p_cls1 = probs
        calibrated_p1 = self.iso.predict(p_cls1)
        return np.column_stack([1.0 - calibrated_p1, calibrated_p1])

