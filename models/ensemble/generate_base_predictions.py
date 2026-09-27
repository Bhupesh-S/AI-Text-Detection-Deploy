import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Path bootstrap
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_PATH = _REPO_ROOT / "Datasets" / "final_dataset.csv"
OUTPUT_DIR = _REPO_ROOT / "outputs"
BASELINES_MODELS_DIR = OUTPUT_DIR / "models"
DEBERTA_DIR = OUTPUT_DIR / "retraining_v2_1" / "deberta" / "final"
DISTILBERT_DIR = OUTPUT_DIR / "retraining_v2_1" / "distilbert" / "final"
ROBERTA_DIR = _REPO_ROOT / "models" / "roberta"
ENSEMBLE_OUTPUT_DIR = _REPO_ROOT / "outputs" / "ensemble"

def load_transformer_probs(model_dir, tokenizer_name, data_df, text_col="text"):
    """Load transformer model and get probabilities for a dataframe."""
    logger.info(f"Extracting probabilities from {model_dir}...")
    if not model_dir.exists():
        logger.warning(f"Model directory {model_dir} does not exist. Skipping.")
        return np.zeros(len(data_df))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(data_df), 32): # batch size 32
            batch_texts = data_df[text_col].iloc[i:i+32].astype(str).tolist()
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            p = softmax(logits.cpu().numpy(), axis=-1)[:, 1]
            probs.extend(p)

    return np.array(probs)

def load_baseline_probs(model_path, X_data):
    """Load baseline model and get probabilities."""
    logger.info(f"Extracting probabilities from {model_path}...")
    if not model_path.exists():
        logger.warning(f"Model {model_path} not found. Skipping.")
        return np.zeros(X_data.shape[0])

    model = joblib.load(model_path)
    # Most scikit-learn models have predict_proba
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_data)[:, 1]
    # LinearSVC needs CalibratedClassifierCV for probabilities,
    # but we expect it to be saved as a Calibrated wrapper in this project's pipeline
    try:
        return model.predict_proba(X_data)[:, 1]
    except:
        logger.error(f"Model at {model_path} does not support predict_proba.")
        return np.zeros(X_data.shape[0])

def main():
    ENSEMBLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    # 1. Load TF-IDF features for baselines
    # We assume X_train.npz, X_valid.npz, X_test.npz exist in outputs/tfidf
    tfidf_dir = OUTPUT_DIR / "tfidf"
    X_train = np.load(tfidf_dir / "X_train.npz")["x"] if (tfidf_dir / "X_train.npz").exists() else None
    X_valid = np.load(tfidf_dir / "X_valid.npz")["x"] if (tfidf_dir / "X_valid.npz").exists() else None
    X_test = np.load(tfidf_dir / "X_test.npz")["x"] if (tfidf_dir / "X_test.npz").exists() else None

    # Since the splits are probabilistic/stratified in scripts,
    # we need to ensure the rows match.
    # Actually, baseline/train.py uses a fixed seed (42) and specific split ratios.
    # We should ideally use the same splits.

    # For the sake of the ensemble, we'll generate predictions for the whole dataset
    # or just a specific split. Let's do it for the whole dataset to be safe,
    # but we'll need the TF-IDF for the whole dataset.

    # Let's use the full dataframe and transform it using the saved vectorizer
    vectorizer = joblib.load(tfidf_dir / "tfidf_vectorizer.pkl")
    X_full = vectorizer.transform(df["text"].fillna("").astype(str))

    results_df = pd.DataFrame({"text": df["text"], "label": df["label"]})

    # Baselines
    baseline_models = {
        "lr": BASELINES_MODELS_DIR / "logistic_regression.joblib",
        "nb": BASELINES_MODELS_DIR / "multinomial_naive_bayes.joblib",
        "svm": BASELINES_MODELS_DIR / "linear_svm.joblib",
        "rf": BASELINES_MODELS_DIR / "random_forest.joblib",
        "xgb": BASELINES_MODELS_DIR / "xgboost.joblib"
    }

    for name, path in baseline_models.items():
        results_df[f"{name}_prob"] = load_baseline_probs(path, X_full)

    # Transformers
    transformers_models = {
        "deberta": DEBERTA_DIR,
        "roberta": ROBERTA_DIR
    }

    for name, path in transformers_models.items():
        results_df[f"{name}_prob"] = load_transformer_probs(path, name, df)

    output_file = ENSEMBLE_OUTPUT_DIR / "base_model_probs.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"All base predictions saved to {output_file}")

if __name__ == "__main__":
    main()
