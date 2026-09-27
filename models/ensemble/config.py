"""
config.py
=========
Central configuration for the Hybrid Ensemble Pipeline.

Author  : Bharanidharan K
Project : Hybrid Explainable Framework for Detecting AI-Generated Academic Text
"""

from pathlib import Path
from dataclasses import dataclass, field

# Base project root (two levels up from models/ensemble/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class EnsembleConfig:
    """All configuration parameters for the Ensemble model."""

    # ── Hugging Face Repository & Model Identifiers ──────────────────────────
    hf_repo_id: str = "bhupesh1907/ai-transformer-v3"
    roberta_subfolder: str = "roberta"
    deberta_subfolder: str = "deberta"
    distilbert_subfolder: str = "distilbert"
    xgboost_file: str = "xgboost/xgboost_model.joblib"
    vectorizer_file: str = "xgboost/tfidf_vectorizer.joblib"

    # Backward-compatible model identifiers
    roberta_path: str = "bhupesh1907/ai-transformer-v3/roberta"
    deberta_path: str = "bhupesh1907/ai-transformer-v3/deberta"
    distilbert_path: str = "bhupesh1907/ai-transformer-v3/distilbert"
    xgboost_path: str = "bhupesh1907/ai-transformer-v3/xgboost/xgboost_model.joblib"
    vectorizer_path: str = "bhupesh1907/ai-transformer-v3/xgboost/tfidf_vectorizer.joblib"

    # ── Voting Configuration ─────────────────────────────────────────────────
    # Default model weights
    roberta_weight: float = 0.35
    deberta_weight: float = 0.35
    distilbert_weight: float = 0.20
    xgboost_weight: float = 0.10
    seed: int = 42

    # ── Deployment Checkpoints ───────────────────────────────────────────────
    # Self-contained package checkpoints directory
    checkpoint_dir: Path = Path(__file__).resolve().parent / "checkpoints"
    meta_model_path: Path = Path(__file__).resolve().parent / "checkpoints" / "meta_classifier.joblib"
    voting_config_path: Path = Path(__file__).resolve().parent / "checkpoints" / "voting_config.json"
    temperatures_path: Path = Path(__file__).resolve().parent / "checkpoints" / "temperatures.json"
    platt_calibrator_path: Path = Path(__file__).resolve().parent / "checkpoints" / "platt_calibrator_xgb.joblib"

    def create_dirs(self) -> None:
        """Ensure checkpoint directory exists."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Serialize configuration variables to a dictionary."""
        return {
            "hf_repo_id": self.hf_repo_id,
            "roberta_subfolder": self.roberta_subfolder,
            "deberta_subfolder": self.deberta_subfolder,
            "distilbert_subfolder": self.distilbert_subfolder,
            "xgboost_file": self.xgboost_file,
            "vectorizer_file": self.vectorizer_file,
            "checkpoint_dir": str(self.checkpoint_dir),
            "meta_model_path": str(self.meta_model_path),
            "weights": {
                "roberta": self.roberta_weight,
                "deberta": self.deberta_weight,
                "distilbert": self.distilbert_weight,
                "xgboost": self.xgboost_weight,
            },
        }


# Singleton config instance
CFG = EnsembleConfig()
