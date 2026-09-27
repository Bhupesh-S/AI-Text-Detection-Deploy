"""
ensemble.py
===========
BaseModelEvaluator handles batch prediction from all four models (RoBERTa, DeBERTa,
DistilBERT, and XGBoost) and manages the prediction probability cache.

Author  : Bharanidharan K
Project : Hybrid Explainable Framework for Detecting AI-Generated Academic Text
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Make config and utils importable
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG
from utils import get_logger

logger = get_logger(__name__)


# ── Text Dataset for PyTorch Batch Inference ─────────────────────────────────

class TextDataset(Dataset):
    """Simple PyTorch Dataset for batch inference on raw texts."""

    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class BaseModelEvaluator:
    """Manages loading base models and extracting prediction probabilities."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"BaseModelEvaluator initialized on device: {self.device}")

        # Lazy loading holders for models to save memory in inference-only scripts
        self.tokenizer_roberta = None
        self.model_roberta = None

        self.tokenizer_deberta = None
        self.model_deberta = None

        self.tokenizer_distilbert = None
        self.model_distilbert = None

        self.vectorizer = None
        self.model_xgboost = None

        # Calibration parameters
        self.temperatures = {}
        self.platt_calibrator_xgb = None

        # Load calibration parameters if present
        temp_path = CFG.checkpoint_dir / "temperatures.json"
        if temp_path.exists():
            try:
                import json
                with open(temp_path, "r", encoding="utf-8") as f:
                    self.temperatures = json.load(f)
                logger.info(f"Loaded calibrated temperatures: {self.temperatures}")
            except Exception as e:
                logger.warning(f"Failed to load calibrated temperatures: {e}")

        platt_path = CFG.checkpoint_dir / "platt_calibrator_xgb.joblib"
        if platt_path.exists():
            try:
                self.platt_calibrator_xgb = joblib.load(platt_path)
                logger.info("Loaded Platt Scaler for XGBoost calibration.")
            except Exception as e:
                logger.warning(f"Failed to load Platt Scaler for XGBoost: {e}")

    # ── Model Loading ────────────────────────────────────────────────────────

    def load_roberta(self) -> None:
        """Load RoBERTa model and tokenizer."""
        if self.model_roberta is not None:
            return
        logger.info(f"Loading RoBERTa from: {CFG.roberta_path}")
        self.tokenizer_roberta = AutoTokenizer.from_pretrained(str(CFG.roberta_path))
        self.model_roberta = AutoModelForSequenceClassification.from_pretrained(
            str(CFG.roberta_path)
        ).to(self.device)
        self.model_roberta.eval()

    def load_deberta(self) -> None:
        """Load DeBERTa model and tokenizer."""
        if self.model_deberta is not None:
            return
        logger.info(f"Loading DeBERTa from: {CFG.deberta_path}")
        self.tokenizer_deberta = AutoTokenizer.from_pretrained(str(CFG.deberta_path))
        self.model_deberta = AutoModelForSequenceClassification.from_pretrained(
            str(CFG.deberta_path)
        ).to(self.device).float()
        self.model_deberta.eval()

    def load_distilbert(self) -> None:
        """Load DistilBERT model and tokenizer."""
        if self.model_distilbert is not None:
            return
        logger.info(f"Loading DistilBERT from: {CFG.distilbert_path}")
        self.tokenizer_distilbert = AutoTokenizer.from_pretrained(str(CFG.distilbert_path))
        self.model_distilbert = AutoModelForSequenceClassification.from_pretrained(
            str(CFG.distilbert_path)
        ).to(self.device)
        self.model_distilbert.eval()

    def load_xgboost(self) -> None:
        """Load XGBoost model and TF-IDF vectorizer."""
        if self.model_xgboost is not None:
            return
        logger.info(f"Loading TF-IDF Vectorizer from: {CFG.vectorizer_path}")
        self.vectorizer = joblib.load(str(CFG.vectorizer_path))
        logger.info(f"Loading XGBoost from: {CFG.xgboost_path}")
        self.model_xgboost = joblib.load(str(CFG.xgboost_path))

    def load_all_models(self) -> None:
        """Load all four models."""
        self.load_roberta()
        self.load_deberta()
        self.load_distilbert()
        self.load_xgboost()

    # ── Inference routines ───────────────────────────────────────────────────

    def get_transformer_logits(
        self,
        model,
        tokenizer,
        texts: List[str],
        batch_size: int = 32,
        desc: str = "Transformer Logits",
    ) -> np.ndarray:
        """Extract raw sequence classification logits (before softmax)."""
        dataset = TextDataset(texts)
        # Windows notes: spawn-deadlock protection (num_workers=0)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        logits_list = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc, leave=False):
                inputs = tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Mixed precision inference
                is_deberta = "deberta" in str(type(model)).lower()
                if self.device.type == "cuda" and not is_deberta:
                    with torch.cuda.amp.autocast():
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)

                logits = outputs.logits.cpu().numpy()
                logits_list.append(logits)

        return np.vstack(logits_list)

    def get_transformer_probs(
        self,
        model,
        tokenizer,
        texts: List[str],
        model_name: str,
        batch_size: int = 32,
        desc: str = "Transformer Predictions",
    ) -> np.ndarray:
        """Extract classification probabilities (temperature scaled if calibrated)."""
        dataset = TextDataset(texts)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        probs_list = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc, leave=False):
                inputs = tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                is_deberta = "deberta" in str(type(model)).lower()
                if self.device.type == "cuda" and not is_deberta:
                    with torch.cuda.amp.autocast():
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)

                logits = outputs.logits.cpu().numpy()
                # Apply Temperature Scaling if optimal temperature exists
                if model_name in self.temperatures:
                    T = self.temperatures[model_name]
                    logits = logits / T
                probs = softmax(logits, axis=-1)
                probs_list.append(probs)

        return np.vstack(probs_list)

    def get_xgboost_probs(self, texts: List[str], desc: str = "XGBoost Predictions") -> np.ndarray:
        """Extract classification probabilities using handcrafted features and XGBoost (Platt scaled if calibrated)."""
        logger.info(f"Extracting handcrafted features for {len(texts):,} texts for XGBoost classification...")
        import pandas as pd
        from feature_engineering.utils import extract_features_batch
        features_list = extract_features_batch(texts)
        df_features = pd.DataFrame(features_list)
        logger.info(f"XGBoost predicting probabilities...")
        raw_probs = self.model_xgboost.predict_proba(df_features)

        # Apply Platt Scaling if calibrated
        if self.platt_calibrator_xgb is not None:
            logger.info("Applying Platt Scaling to XGBoost probabilities...")
            return self.platt_calibrator_xgb.predict_proba(raw_probs)
        return raw_probs

    # ── Batch Pipeline Evaluation ────────────────────────────────────────────

    def extract_all_probs(self, texts: List[str], batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Run inference across all four models on a text list.
        Returns a dict of model_name -> class probabilities (N, 2).
        """
        # Load all models if not already done
        self.load_all_models()

        probs = {}

        # Transformers
        probs["roberta"] = self.get_transformer_probs(
            self.model_roberta, self.tokenizer_roberta, texts, "roberta", batch_size=batch_size, desc="RoBERTa Predictions"
        )
        probs["deberta"] = self.get_transformer_probs(
            self.model_deberta, self.tokenizer_deberta, texts, "deberta", batch_size=batch_size, desc="DeBERTa Predictions"
        )
        probs["distilbert"] = self.get_transformer_probs(
            self.model_distilbert, self.tokenizer_distilbert, texts, "distilbert", batch_size=batch_size, desc="DistilBERT Predictions"
        )

        # XGBoost
        probs["xgboost"] = self.get_xgboost_probs(texts, desc="XGBoost Predictions")

        return probs


# ── Prediction Caching Utilities ─────────────────────────────────────────────

def get_cached_probabilities(path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load cached base model probabilities from a CSV file."""
    if not path.exists():
        return None

    logger.info(f"Loading cached probabilities from: {path}")
    df = pd.read_csv(path)

    # Ensure required columns are present
    required_cols = [
        "roberta_human", "roberta_ai",
        "deberta_human", "deberta_ai",
        "distilbert_human", "distilbert_ai",
        "xgboost_human", "xgboost_ai"
    ]
    missing = set(required_cols) - set(df.columns)
    if missing:
        logger.warning(f"Cache CSV missing columns: {missing}. Bypassing cache.")
        return None

    probs = {
        "roberta": df[["roberta_human", "roberta_ai"]].values,
        "deberta": df[["deberta_human", "deberta_ai"]].values,
        "distilbert": df[["distilbert_human", "distilbert_ai"]].values,
        "xgboost": df[["xgboost_human", "xgboost_ai"]].values,
        "label": df["label"].values if "label" in df.columns else None
    }
    return probs


def cache_probabilities(
    probs: Dict[str, np.ndarray],
    labels: np.ndarray,
    path: Path,
) -> None:
    """Save base model probabilities to a CSV file to avoid recalculating next time."""
    path.parent.mkdir(parents=True, exist_ok=True)

    df_dict = {
        "label": labels,
        "roberta_human": probs["roberta"][:, 0],
        "roberta_ai": probs["roberta"][:, 1],
        "deberta_human": probs["deberta"][:, 0],
        "deberta_ai": probs["deberta"][:, 1],
        "distilbert_human": probs["distilbert"][:, 0],
        "distilbert_ai": probs["distilbert"][:, 1],
        "xgboost_human": probs["xgboost"][:, 0],
        "xgboost_ai": probs["xgboost"][:, 1],
    }

    df = pd.DataFrame(df_dict)
    df.to_csv(path, index=False)
    logger.info(f"Successfully cached base probabilities → {path}")
