import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import shap
import matplotlib.pyplot as plt

# Path bootstrap
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
ENSEMBLE_INPUT = _REPO_ROOT / "outputs" / "ensemble" / "base_model_probs.csv"
META_MODEL_PATH = _REPO_ROOT / "outputs" / "ensemble" / "meta_classifier.joblib"
ENSEMBLE_OUTPUT_DIR = _REPO_ROOT / "outputs" / "ensemble"

def main():
    if not ENSEMBLE_INPUT.exists() or not META_MODEL_PATH.exists():
        logger.error("Missing input probabilities or meta-model. Please run predictions and train scripts first.")
        return

    df = pd.read_csv(ENSEMBLE_INPUT)
    prob_cols = [col for col in df.columns if col.endswith("_prob")]
    X = df[prob_cols]

    meta_model = joblib.load(META_MODEL_PATH)

    # Create SHAP explainer
    # For Logistic Regression, we can use LinearExplainer
    explainer = shap.LinearExplainer(meta_model, X)
    shap_values = explainer.shap_values(X)

    # Plot Summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("Influence of Base Models on Hybrid Ensemble Decision")
    plt.savefig(ENSEMBLE_OUTPUT_DIR / "ensemble_shap_summary.png", bbox_inches="tight")
    plt.close()

    logger.info(f"SHAP summary plot saved to {ENSEMBLE_OUTPUT_DIR / 'ensemble_shap_summary.png'}")

if __name__ == "__main__":
    main()
