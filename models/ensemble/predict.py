"""
predict.py
==========
Interactive CLI and programmatic inference engine for the Hybrid Ensemble framework.

Canonical Result Schema:
{
    "prediction": "Human Written",              # or "AI Generated"
    "ai_probability": 0.0501,                   # float in [0.0, 1.0]
    "human_probability": 0.9499,                # float in [0.0, 1.0]
    "confidence": 0.9499,                       # float in [0.0, 1.0]
    "strategy_used": "Stacking Meta-Classifier",
    "latency_ms": 783.0,
    "text": "Four score and seven years ago...",

    "model_breakdown": {
        "roberta": {
            "ai_probability": 0.1179,
            "human_probability": 0.8821,
            "prediction": "Human Written"
        },
        "deberta": {
            "ai_probability": 0.2323,
            "human_probability": 0.7677,
            "prediction": "Human Written"
        },
        "distilbert": {
            "ai_probability": 0.0593,
            "human_probability": 0.9407,
            "prediction": "Human Written"
        },
        "xgboost": {
            "ai_probability": 0.1118,
            "human_probability": 0.8882,
            "prediction": "Human Written"
        }
    },

    "consensus": {
        "ai_votes": 0,
        "human_votes": 4,
        "state": "UNANIMOUS CONSENSUS (4/4 Human)",
        "votes": {
            "roberta": "Human",
            "deberta": "Human",
            "distilbert": "Human",
            "xgboost": "Human"
        }
    },

    "meta_classifier": {
        "classes": [0, 1],
        "raw_probabilities": [0.9499, 0.0501],
        "ai_probability": 0.0501,
        "human_probability": 0.9499
    }
}

Author  : Bharanidharan K & Antigravity
Project : Hybrid Explainable Framework for Detecting AI-Generated Academic Text
"""

import argparse
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import torch
from scipy.special import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import hf_hub_download

# Allow finding sibling modules
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_THIS_FILE.parent))

from config import CFG
from utils import get_logger, load_json
from stacking import prepare_stacking_features

logger = get_logger(__name__)

HF_REPO_ID = "bhupesh1907/ai-transformer-v3"


def get_hf_token() -> Optional[str]:
    """
    Retrieve private Hugging Face authentication token safely.
    Checks Streamlit secrets, then environment variable, then local secrets.toml.
    Never prints or logs the token value.
    """
    token = None
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
            token = st.secrets["HF_TOKEN"]
    except Exception:
        pass

    if not token:
        import os
        token = os.environ.get("HF_TOKEN")

    if not token:
        try:
            secrets_path = _REPO_ROOT / ".streamlit" / "secrets.toml"
            if secrets_path.exists():
                try:
                    import tomllib
                    with open(secrets_path, "rb") as f:
                        data = tomllib.load(f)
                        token = data.get("HF_TOKEN")
                except ImportError:
                    import toml
                    data = toml.load(str(secrets_path))
                    token = data.get("HF_TOKEN")
        except Exception:
            pass

    return token


def validate_result_schema(result: Dict) -> None:
    """Validate canonical result schema types, required keys, and ranges."""
    required_keys = [
        "prediction",
        "ai_probability",
        "human_probability",
        "confidence",
        "model_breakdown",
        "consensus",
        "meta_classifier",
        "strategy_used"
    ]
    for key in required_keys:
        if key not in result:
            raise ValueError(
                f"HybridInferencePipeline returned an invalid result schema: missing required field '{key}'"
            )

    ai_p = result["ai_probability"]
    hu_p = result["human_probability"]
    conf = result["confidence"]

    if not isinstance(ai_p, (float, np.floating)) or not (0.0 <= ai_p <= 1.0):
        raise ValueError(f"Invalid ai_probability: {ai_p}. Expected float in [0.0, 1.0].")
    if not isinstance(hu_p, (float, np.floating)) or not (0.0 <= hu_p <= 1.0):
        raise ValueError(f"Invalid human_probability: {hu_p}. Expected float in [0.0, 1.0].")
    if not isinstance(conf, (float, np.floating)) or not (0.0 <= conf <= 1.0):
        raise ValueError(f"Invalid confidence: {conf}. Expected float in [0.0, 1.0].")

    if abs((ai_p + hu_p) - 1.0) > 1e-3:
        raise ValueError(f"Probabilities do not sum to 1.0: ai_prob={ai_p}, human_prob={hu_p}")


class HybridInferencePipeline:
    """High-level wrapper to run inference using the Hybrid Ensemble Model."""

    ID2LABEL: Dict[int, str] = {0: "Human Written", 1: "AI Generated"}

    def __init__(self):
        logger.info("Initializing Hybrid Inference Pipeline...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Obtain private Hugging Face authentication token
        hf_token = get_hf_token()
        self.hf_repo_id = getattr(CFG, "hf_repo_id", HF_REPO_ID)

        # ── 1. Load V3 TF-IDF Vectorizer from Hugging Face Hub ────────────────
        logger.info(f"Loading V3 TF-IDF Vectorizer from Hugging Face Hub ({self.hf_repo_id})...")
        try:
            vec_file = hf_hub_download(
                repo_id=self.hf_repo_id,
                filename="xgboost/tfidf_vectorizer.joblib",
                token=hf_token
            )
            self.vectorizer_xgboost = joblib.load(vec_file)
            self.vectorizer = self.vectorizer_xgboost
        except Exception as e:
            raise RuntimeError(f"Failed to download/load V3 TF-IDF vectorizer from Hugging Face Hub ({self.hf_repo_id}): {e}")

        if self.vectorizer_xgboost is None:
            raise RuntimeError("Failed to load V3 XGBoost TF-IDF vectorizer: Object is None.")

        # Verify vectorizer vocabulary dimensionality (exactly 30,000 features)
        vec_features = len(getattr(self.vectorizer_xgboost, "vocabulary_", {}))
        if vec_features != 30000:
            logger.warning(
                f"V3 TF-IDF Vectorizer vocabulary count is {vec_features}, expected 30,000."
            )

        # ── 2. Load V3 XGBoost Model from Hugging Face Hub ────────────────────
        logger.info(f"Loading V3 XGBoost model from Hugging Face Hub ({self.hf_repo_id})...")
        try:
            xgb_file = hf_hub_download(
                repo_id=self.hf_repo_id,
                filename="xgboost/xgboost_model.joblib",
                token=hf_token
            )
            self.model_xgboost = joblib.load(xgb_file)
        except Exception as e:
            raise RuntimeError(f"Failed to download/load V3 XGBoost model from Hugging Face Hub ({self.hf_repo_id}): {e}")

        if hasattr(self.model_xgboost, "classes_"):
            xgb_classes = list(self.model_xgboost.classes_)
            if xgb_classes != [0, 1]:
                raise ValueError(f"Incompatible XGBoost classes_: {xgb_classes}. Expected [0, 1].")

        # Verify XGBoost expected feature count against vectorizer
        xgb_expected = getattr(self.model_xgboost, "n_features_in_", None)
        if xgb_expected is None and hasattr(self.model_xgboost, "get_booster"):
            try:
                xgb_expected = self.model_xgboost.get_booster().num_features()
            except Exception:
                xgb_expected = 30000
        if xgb_expected is not None and xgb_expected != vec_features:
            raise ValueError(
                f"Feature dimension mismatch between vectorizer and XGBoost model:\n"
                f"  - Vectorizer feature count: {vec_features}\n"
                f"  - XGBoost expected feature count: {xgb_expected}"
            )

        # ── 3. Load Transformers from Hugging Face Hub ────────────────────────
        # RoBERTa
        logger.info(f"Loading RoBERTa from Hugging Face Hub ({self.hf_repo_id}/roberta)...")
        self.tokenizer_roberta = AutoTokenizer.from_pretrained(
            self.hf_repo_id, subfolder="roberta", token=hf_token
        )
        self.model_roberta = AutoModelForSequenceClassification.from_pretrained(
            self.hf_repo_id, subfolder="roberta", token=hf_token, attn_implementation="eager"
        ).to(self.device)
        self.model_roberta.eval()
        self.roberta_path_actual = f"{self.hf_repo_id}/roberta"

        # DeBERTa
        logger.info(f"Loading DeBERTa from Hugging Face Hub ({self.hf_repo_id}/deberta)...")
        self.tokenizer_deberta = AutoTokenizer.from_pretrained(
            self.hf_repo_id, subfolder="deberta", token=hf_token
        )
        self.model_deberta = AutoModelForSequenceClassification.from_pretrained(
            self.hf_repo_id, subfolder="deberta", token=hf_token, attn_implementation="eager"
        ).to(self.device).float()
        self.model_deberta.eval()

        # DistilBERT
        logger.info(f"Loading DistilBERT from Hugging Face Hub ({self.hf_repo_id}/distilbert)...")
        self.tokenizer_distilbert = AutoTokenizer.from_pretrained(
            self.hf_repo_id, subfolder="distilbert", token=hf_token
        )
        self.model_distilbert = AutoModelForSequenceClassification.from_pretrained(
            self.hf_repo_id, subfolder="distilbert", token=hf_token, attn_implementation="eager"
        ).to(self.device)
        self.model_distilbert.eval()

        # Log Model Identities
        self._log_model_identity("RoBERTa V3", self.roberta_path_actual, self.model_roberta)
        self._log_model_identity("DeBERTa-v3 V3", f"{self.hf_repo_id}/deberta", self.model_deberta)
        self._log_model_identity("DistilBERT V3", f"{self.hf_repo_id}/distilbert", self.model_distilbert)
        logger.info(f"XGBoost Stylometrics | Source: {self.hf_repo_id}/xgboost/xgboost_model.joblib | Classes: {getattr(self.model_xgboost, 'classes_', 'Unknown')}")

        # ── 4. Calibration parameters & Checkpoint Resolution ─────────────────
        def _resolve_checkpoint(filename: str, default_path: Path) -> Optional[Path]:
            """Check Hugging Face Hub first, then local package checkpoints."""
            for sub in ["ensemble", "checkpoints", ""]:
                fn = f"{sub}/{filename}" if sub else filename
                try:
                    dl = hf_hub_download(repo_id=self.hf_repo_id, filename=fn, token=hf_token)
                    if dl and Path(dl).exists():
                        return Path(dl)
                except Exception:
                    pass
            candidates = [
                default_path,
                _THIS_FILE.parent / "checkpoints" / filename,
                _REPO_ROOT / "models" / "ensemble" / "checkpoints" / filename,
            ]
            for p in candidates:
                if p and p.exists():
                    return p
            return None

        self.temperatures = {}
        self.platt_calibrator_xgb = None

        temp_path = _resolve_checkpoint("temperatures.json", CFG.checkpoint_dir / "temperatures.json")
        if temp_path is not None and temp_path.exists():
            try:
                import json
                with open(temp_path, "r", encoding="utf-8") as f:
                    self.temperatures = json.load(f)
                logger.info(f"Loaded calibrated temperatures from {temp_path}: {self.temperatures}")
            except Exception as e:
                logger.warning(f"Failed to load calibrated temperatures: {e}")

        platt_path = _resolve_checkpoint("platt_calibrator_xgb.joblib", CFG.checkpoint_dir / "platt_calibrator_xgb.joblib")
        if platt_path is not None and platt_path.exists():
            try:
                import calibration
                self.platt_calibrator_xgb = joblib.load(str(platt_path))
                logger.info(f"Loaded Platt Scaler for XGBoost calibration from {platt_path}.")
            except Exception as e:
                logger.warning(f"Failed to load Platt Scaler for XGBoost: {e}")

        # ── 5. Voting weights fallback ────────────────────────────────────────
        voting_path = _resolve_checkpoint("voting_config.json", CFG.voting_config_path)
        if voting_path is not None and voting_path.exists():
            v_cfg = load_json(voting_path)
            self.voting_weights = v_cfg["normalized_weights"]
            logger.info(f"Loaded normalized voting weights from {voting_path}")
        else:
            w = [CFG.roberta_weight, CFG.deberta_weight, CFG.distilbert_weight, CFG.xgboost_weight]
            sum_w = sum(w)
            self.voting_weights = {
                "roberta": CFG.roberta_weight / sum_w,
                "deberta": CFG.deberta_weight / sum_w,
                "distilbert": CFG.distilbert_weight / sum_w,
                "xgboost": CFG.xgboost_weight / sum_w
            }

        # ── 6. Load & Validate Stacking Meta-Classifier ───────────────────────
        meta_path = _resolve_checkpoint("meta_classifier.joblib", CFG.meta_model_path)
        if meta_path is None or not meta_path.exists():
            raise FileNotFoundError(
                f"CRITICAL ERROR: Stacking Meta-Classifier not found. Looked in Hugging Face Hub ({self.hf_repo_id}) "
                f"and local package checkpoints ({CFG.meta_model_path})."
            )

        logger.info(f"Loading Stacking Meta-Classifier: {meta_path}")
        self.meta_model = joblib.load(str(meta_path))
        self.meta_model_path_actual = meta_path

        meta_classes = [0, 1]
        if hasattr(self.meta_model, "models") and len(self.meta_model.models) > 0:
            fold_m = self.meta_model.models[0]
            if hasattr(fold_m, "classes_"):
                meta_classes = list(fold_m.classes_)

        if meta_classes != [0, 1]:
            raise ValueError(f"Incompatible Meta-Classifier classes: {meta_classes}. Expected [0, 1].")

        self.human_index = 0
        self.ai_index = 1

        logger.info(f"Stacking Meta-Classifier ready. Classes: {meta_classes} (0=Human, 1=AI)")

        # Log Diagnostic Runtime Signature
        sig = self.get_runtime_signature()
        logger.info(f"""
================================================================================
HYBRID ENSEMBLE RUNTIME SIGNATURE
================================================================================
Pipeline Source: {sig['pipeline_source']}
Class Name     : {sig['class_name']}
Device         : {sig['device']}

Models:
  RoBERTa        : {sig['roberta_path']} ({sig['roberta_class']})
  DeBERTa        : {sig['deberta_path']} ({sig['deberta_class']})
  DistilBERT     : {sig['distilbert_path']} ({sig['distilbert_class']})
  XGBoost        : {sig['xgboost_path']}
  Meta-Classifier: {sig['meta_classifier_path']}
================================================================================
""")
        logger.info("Hybrid Ensemble Inference Pipeline fully initialized.")

    def get_runtime_signature(self) -> Dict[str, str]:
        """Return diagnostic dictionary of loaded runtime modules and model paths."""
        import inspect
        return {
            "pipeline_source": str(inspect.getfile(self.__class__)),
            "class_name": self.__class__.__name__,
            "device": str(self.device),
            "roberta_path": str(self.roberta_path_actual),
            "roberta_class": self.model_roberta.__class__.__name__,
            "deberta_path": f"{self.hf_repo_id}/deberta",
            "deberta_class": self.model_deberta.__class__.__name__,
            "distilbert_path": f"{self.hf_repo_id}/distilbert",
            "distilbert_class": self.model_distilbert.__class__.__name__,
            "xgboost_path": f"{self.hf_repo_id}/xgboost/xgboost_model.joblib",
            "meta_classifier_path": str(getattr(self, "meta_model_path_actual", CFG.meta_model_path))
        }

    def _log_model_identity(self, name: str, source: Union[Path, str], model) -> None:
        num_params = sum(p.numel() for p in model.parameters())
        id2label = getattr(model.config, "id2label", None)
        num_labels = getattr(model.config, "num_labels", 2)
        logger.info(
            f"MODEL IDENTITY | Name: {name:<16} | Source: {str(source)} | "
            f"Class: {model.__class__.__name__:<35} | Params: {num_params:,} | "
            f"Labels: {num_labels} | id2label: {id2label}"
        )

    # ── Inference helpers ────────────────────────────────────────────────────

    def _predict_transformer(self, model, tokenizer, text: str, model_name: str) -> np.ndarray:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            padding="max_length",
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits.cpu().numpy()

        if model_name in self.temperatures and self.temperatures[model_name] != 1.0:
            T = self.temperatures[model_name]
            logits = logits / T

        probs = softmax(logits, axis=-1)[0]  # shape (2,) [P(Human), P(AI)]
        return probs

    def _get_vectorizer_xgboost(self):
        """
        Retrieve or lazy-load the V3 TF-IDF vectorizer.
        Guarantees that cached pipeline instances (e.g. across Streamlit sessions)
        always have access to the initialized 30,000-feature vectorizer.
        """
        if getattr(self, "vectorizer_xgboost", None) is not None:
            return self.vectorizer_xgboost
        if getattr(self, "vectorizer", None) is not None:
            self.vectorizer_xgboost = self.vectorizer
            return self.vectorizer_xgboost

        try:
            token = get_hf_token()
            repo_id = getattr(self, "hf_repo_id", getattr(CFG, "hf_repo_id", HF_REPO_ID))
            vec_file = hf_hub_download(
                repo_id=repo_id,
                filename="xgboost/tfidf_vectorizer.joblib",
                token=token
            )
            vec = joblib.load(vec_file)
            self.vectorizer_xgboost = vec
            self.vectorizer = vec
            return self.vectorizer_xgboost
        except Exception as e:
            logger.error(f"Failed to lazy-load V3 XGBoost TF-IDF vectorizer from Hugging Face Hub: {e}")
            raise RuntimeError(f"Failed to load V3 XGBoost TF-IDF vectorizer from Hugging Face Hub: {e}")

    def _predict_xgboost(self, text: str) -> np.ndarray:
        cleaned_text = str(text).strip() if text else ""
        vec = self._get_vectorizer_xgboost()

        tfidf_features = vec.transform([cleaned_text])

        # Programmatic verification of feature dimensionality
        expected_features = getattr(self.model_xgboost, "n_features_in_", None)
        if expected_features is None and hasattr(self.model_xgboost, "get_booster"):
            try:
                expected_features = self.model_xgboost.get_booster().num_features()
            except Exception:
                expected_features = 30000
        if expected_features is None:
            expected_features = 30000

        if tfidf_features.shape[1] != expected_features:
            raise ValueError(
                f"Feature shape mismatch:\n"
                f"  - vectorizer feature count: {tfidf_features.shape[1]}\n"
                f"  - XGBoost expected feature count: {expected_features}\n"
                f"  - Source repository: {self.hf_repo_id}"
            )

        raw_probs = self.model_xgboost.predict_proba(tfidf_features)[0]  # [P(Human), P(AI)]

        if self.platt_calibrator_xgb is not None:
            cal_probs = self.platt_calibrator_xgb.predict_proba(raw_probs.reshape(1, -1))[0]
            return cal_probs
        return raw_probs

    # ── Core Prediction Pipeline ─────────────────────────────────────────────

    def predict(self, text: str) -> Dict:
        """
        Execute full inference pipeline and return canonical result object.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        t0 = time.time()
        words = text.strip().split()
        word_count = len(words)
        char_count = len(text)
        sentences = [s for s in text.split('.') if s.strip()]
        sentence_count = len(sentences) if len(sentences) > 0 else 1

        # 1. Base Model Inferences
        probs = {
            "roberta":    self._predict_transformer(self.model_roberta, self.tokenizer_roberta, text, "roberta"),
            "deberta":    self._predict_transformer(self.model_deberta, self.tokenizer_deberta, text, "deberta"),
            "distilbert": self._predict_transformer(self.model_distilbert, self.tokenizer_distilbert, text, "distilbert"),
            "xgboost":    self._predict_xgboost(text)
        }

        # Format Model Breakdown
        model_breakdown = {}
        display_name_map = {
            "roberta": "RoBERTa-base",
            "deberta": "DeBERTa-v3",
            "distilbert": "DistilBERT",
            "xgboost": "XGBoost (Stylometrics)"
        }
        for m in probs:
            p_ai = float(probs[m][1])
            p_hu = float(probs[m][0])
            pred_str = "AI Generated" if p_ai >= 0.50 else "Human Written"
            entry = {
                "ai_probability": p_ai,
                "human_probability": p_hu,
                "prediction": pred_str
            }
            model_breakdown[m] = entry
            model_breakdown[display_name_map[m]] = entry

        # 2. Meta-Feature Vector Construction
        dict_probs_2d = {m: probs[m].reshape(1, -1) for m in probs}
        meta_features = prepare_stacking_features(dict_probs_2d, [text])

        if meta_features.shape[1] != 18:
            raise ValueError(f"CRITICAL ERROR: Expected 18 meta-features, got shape {meta_features.shape}.")

        # 3. Authoritative Stacking Meta-Classifier Inference
        stacking_probs = self.meta_model.predict_proba(dict_probs_2d, [text])[0]

        human_prob = float(stacking_probs[self.human_index])
        ai_prob = float(stacking_probs[self.ai_index])

        pred_id = 1 if ai_prob >= human_prob else 0
        prediction = self.ID2LABEL[pred_id]
        confidence = float(max(ai_prob, human_prob))

        # Retrieve detailed diagnostics if available
        meta_diagnostics = {}
        if hasattr(self.meta_model, "get_diagnostics"):
            meta_diagnostics = self.meta_model.get_diagnostics(dict_probs_2d, [text])
            if meta_diagnostics.get("consensus_conflict"):
                logger.warning("Base-model consensus conflict: 4/4 base models predict AI while meta-classifier predicts Human.")

        # 4. Diagnostic Consensus Calculation
        base_models = ["roberta", "deberta", "distilbert", "xgboost"]
        ai_votes = sum(1 for m in base_models if probs[m][1] >= 0.50)
        human_votes = len(base_models) - ai_votes
        total_models = len(base_models)

        if ai_votes == 4:
            consensus_state = "UNANIMOUS AGREEMENT (4/4 MODELS AGREE)"
        elif ai_votes == 3:
            consensus_state = "SPLIT CONSENSUS (3/4 MODELS AGREE)"
        elif ai_votes == 2:
            consensus_state = "TIED CONSENSUS (2/4 MODELS AGREE)"
        elif ai_votes == 1:
            consensus_state = "SPLIT CONSENSUS (3/4 MODELS AGREE)"
        else:
            consensus_state = "UNANIMOUS AGREEMENT (4/4 MODELS AGREE)"

        # Secondary Voting calculations for analytics
        soft_prob = np.mean([probs[m] for m in probs], axis=0)
        weighted_prob = (
            probs["roberta"] * self.voting_weights["roberta"] +
            probs["deberta"] * self.voting_weights["deberta"] +
            probs["distilbert"] * self.voting_weights["distilbert"] +
            probs["xgboost"] * self.voting_weights["xgboost"]
        )

        elapsed_ms = (time.time() - t0) * 1000

        # Construct Canonical Result Object
        result = {
            "prediction": prediction,
            "ai_probability": ai_prob,
            "human_probability": human_prob,
            "confidence": confidence,
            "strategy_used": "Stacking Meta-Classifier",
            "latency_ms": round(elapsed_ms, 1),
            "text": text[:150] + " …" if len(text) > 150 else text,

            "model_breakdown": model_breakdown,

            "consensus": {
                "ai_votes": ai_votes,
                "human_votes": human_votes,
                "total_models": total_models,
                "agree_count": max(ai_votes, human_votes),
                "state": consensus_state,
                "votes": {m: ("AI" if probs[m][1] >= 0.50 else "Human") for m in base_models}
            },

            "meta_classifier": {
                "classes": [0, 1],
                "raw_probabilities": [human_prob, ai_prob],
                "ai_probability": ai_prob,
                "human_probability": human_prob
            },

            "meta_classifier_diagnostics": meta_diagnostics,

            # Backwards compatibility fields
            "base_probs": {m: probs[m] for m in probs},
            "base_predictions": model_breakdown,
            "soft_voting_probs": soft_prob,
            "weighted_voting_probs": weighted_prob,
            "stacking_probs": stacking_probs,
            "final_probs": np.array([human_prob, ai_prob]),
            "meta_features": meta_features[0].tolist()
        }

        # Validate Schema Before Return
        validate_result_schema(result)

        # Structured Debug Logging
        logger.debug(f"""
=== HYBRID ENSEMBLE DEBUG ===
Input Text: Chars={char_count}, Words={word_count}, Sentences={sentence_count}

Base Model Predictions:
  - RoBERTa    ({self.roberta_path_actual}): AI={probs['roberta'][1]*100:.2f}%, Hu={probs['roberta'][0]*100:.2f}% -> {model_breakdown['roberta']['prediction']}
  - DeBERTa    ({self.hf_repo_id}/deberta): AI={probs['deberta'][1]*100:.2f}%, Hu={probs['deberta'][0]*100:.2f}% -> {model_breakdown['deberta']['prediction']}
  - DistilBERT ({self.hf_repo_id}/distilbert): AI={probs['distilbert'][1]*100:.2f}%, Hu={probs['distilbert'][0]*100:.2f}% -> {model_breakdown['distilbert']['prediction']}
  - XGBoost    ({self.hf_repo_id}/xgboost/xgboost_model.joblib): AI={probs['xgboost'][1]*100:.2f}%, Hu={probs['xgboost'][0]*100:.2f}% -> {model_breakdown['xgboost']['prediction']}

Diagnostic Consensus:
  - AI Votes: {ai_votes} | Human Votes: {human_votes}
  - Consensus State: {consensus_state}

Meta Features (18 vector):
  {meta_features[0].tolist()}

Stacking Meta-Classifier:
  - Raw predict_proba: [Human={human_prob:.4f}, AI={ai_prob:.4f}]
  - Authoritative Verdict: {prediction} ({confidence*100:.2f}%)
=============================
""")

        return result

    # ── Formatting ───────────────────────────────────────────────────────────

    def format_result(self, result: Dict) -> str:
        """Format output for CLI display."""
        bp = result["base_probs"]
        pred_lbl = result["prediction"]
        confidence = result["confidence"] * 100
        cons = result["consensus"]

        lines = [
            "==========================================================",
            "BASE MODELS BREAKDOWN:",
            f"  RoBERTa-base : AI {bp['roberta'][1]*100:.1f}% | Human {bp['roberta'][0]*100:.1f}% -> {'AI' if bp['roberta'][1]>=0.5 else 'Human'}",
            f"  DeBERTa-v3   : AI {bp['deberta'][1]*100:.1f}% | Human {bp['deberta'][0]*100:.1f}% -> {'AI' if bp['deberta'][1]>=0.5 else 'Human'}",
            f"  DistilBERT   : AI {bp['distilbert'][1]*100:.1f}% | Human {bp['distilbert'][0]*100:.1f}% -> {'AI' if bp['distilbert'][1]>=0.5 else 'Human'}",
            f"  XGBoost      : AI {bp['xgboost'][1]*100:.1f}% | Human {bp['xgboost'][0]*100:.1f}% -> {'AI' if bp['xgboost'][1]>=0.5 else 'Human'}",
            "",
            "DIAGNOSTIC CONSENSUS:",
            f"  State: {cons['state']} ({cons['ai_votes']} AI / {cons['human_votes']} Human)",
            "  Note: Consensus is diagnostic only and does not override Stacking Meta-Classifier.",
            "",
            "FINAL AUTHORITATIVE HYBRID ENSEMBLE PREDICTION:",
            f"  Result      : {pred_lbl}",
            f"  Confidence  : {confidence:.2f}% (P(AI)={result['ai_probability']*100:.2f}%, P(Human)={result['human_probability']*100:.2f}%)",
            f"  Engine      : {result['strategy_used']} | Latency: {result['latency_ms']:.1f} ms",
            "=========================================================="
        ]
        return "\n".join(lines)


# ── CLI Entry ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict using the Hybrid Ensemble Model.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--text", type=str, help="Academic text string to classify.")
    group.add_argument("--file", type=str, help="Path to a txt file with one sample per line.")
    group.add_argument("--interactive", action="store_true", help="Start interactive terminal REPL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = HybridInferencePipeline()

    if args.text:
        result = pipeline.predict(args.text)
        print("\n" + pipeline.format_result(result))

    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"Target text file not found: {file_path}")
            sys.exit(1)

        texts = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        logger.info(f"Classifying {len(texts):,} lines from {file_path} …")

        for i, text in enumerate(texts, start=1):
            print(f"\n[{i}] Text: {text[:80]} …")
            result = pipeline.predict(text)
            print(pipeline.format_result(result))

    else:
        print("\nHybrid Ensemble Academic Text Detector CLI (type 'quit' to exit)\n")
        while True:
            try:
                text = input("Enter text: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text.lower() in ("quit", "exit"):
                break
            try:
                result = pipeline.predict(text)
                print("\n" + pipeline.format_result(result) + "\n")
            except Exception as exc:
                print(f"Error classifying text: {exc}")


if __name__ == "__main__":
    main()
