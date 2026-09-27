"""
Feature Engineering Configuration
====================================
Central configuration for Phase-1 of the AI-Text Detection pipeline.

All paths are resolved relative to the repository root so this module
can be imported from any working directory.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout (resolved from repo root)
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent.parent   # repository root
DATA_DIR    = BASE_DIR / "Datasets"                    # raw datasets (capital D)
OUTPUTS_DIR = BASE_DIR / "outputs"                     # all generated artefacts
LOGS_DIR    = BASE_DIR / "logs"                        # log files

# ---------------------------------------------------------------------------
# Input files
# ---------------------------------------------------------------------------
DATASET_PATH = DATA_DIR / "merged" / "final_dataset.csv"

# ---------------------------------------------------------------------------
# Output sub-directories
# ---------------------------------------------------------------------------
TFIDF_DIR       = OUTPUTS_DIR / "tfidf"
LABELS_DIR      = OUTPUTS_DIR / "labels"
HANDCRAFTED_DIR = OUTPUTS_DIR / "handcrafted"
MODELS_DIR      = OUTPUTS_DIR / "models"
FIGURES_DIR     = OUTPUTS_DIR / "figures"
REPORTS_DIR     = OUTPUTS_DIR / "reports"

# ---------------------------------------------------------------------------
# Output file paths
# ---------------------------------------------------------------------------
TFIDF_VECTORIZER_PATH = TFIDF_DIR / "tfidf_vectorizer.pkl"

X_TRAIN_PATH = TFIDF_DIR / "X_train.npz"
X_VALID_PATH = TFIDF_DIR / "X_valid.npz"
X_TEST_PATH  = TFIDF_DIR / "X_test.npz"

Y_TRAIN_PATH = LABELS_DIR / "y_train.csv"
Y_VALID_PATH = LABELS_DIR / "y_valid.csv"
Y_TEST_PATH  = LABELS_DIR / "y_test.csv"

TRAIN_FEATURES_PATH = HANDCRAFTED_DIR / "train_features.csv"
VALID_FEATURES_PATH = HANDCRAFTED_DIR / "valid_features.csv"
TEST_FEATURES_PATH  = HANDCRAFTED_DIR / "test_features.csv"

LOG_PATH    = LOGS_DIR / "feature_engineering.log"
REPORT_PATH = REPORTS_DIR / "feature_report.md"

# ---------------------------------------------------------------------------
# Dataset split configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE    = 0.10   # 10 % test
VALID_SIZE   = 0.10   # 10 % validation → 80 % train

# ---------------------------------------------------------------------------
# TF-IDF configuration (aligned with project requirements)
# ---------------------------------------------------------------------------
TFIDF_PARAMS = {
    "max_features": 50_000,
    "ngram_range":  (1, 2),
    "min_df":       5,
    "max_df":       0.95,
    "sublinear_tf": True,          # log(1+tf) — better for long documents
    "stop_words":   "english",
    "lowercase":    False,         # text is already lowercased by preprocessor
    "dtype":        "float32",     # half the memory of float64
}

# ---------------------------------------------------------------------------
# Handcrafted feature extraction
# ---------------------------------------------------------------------------
CHUNK_SIZE = 10_000   # texts per chunk (controls progress-bar granularity)
