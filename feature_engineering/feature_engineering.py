"""
Feature Engineering Pipeline
==============================
Orchestrates the full Phase-1 pipeline:

  1. Load CSV dataset
  2. Stratified train / valid / test split (70 / 10 / 20)
  3. Text preprocessing
  4. Handcrafted feature extraction (parallel chunks)
  5. TF-IDF vectorisation (fit on train, transform all splits)
  6. Persist all artefacts to outputs/

Outputs
-------
outputs/tfidf/
    tfidf_vectorizer.pkl
    X_train.npz
    X_valid.npz
    X_test.npz
outputs/labels/
    y_train.csv
    y_valid.csv
    y_test.csv
outputs/handcrafted/
    train_features.csv
    valid_features.csv
    test_features.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Sibling-module imports (works both as package and as direct script)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from feature_engineering.preprocessing import TextPreprocessor
from feature_engineering.utils import extract_features_batch, features_to_dataframe
from feature_engineering.vectorizer import (
    TfidfVectorizerWrapper,
    save_sparse_matrix,
    save_labels,
)

# ---------------------------------------------------------------------------
# Logging  (console + rotating file in logs/)
# ---------------------------------------------------------------------------
from logging.handlers import RotatingFileHandler as _RFH  # noqa: E402

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
_fmt = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if not any(isinstance(h, logging.StreamHandler) for h in _root_logger.handlers):
    import sys as _sys
    _ch = logging.StreamHandler(_sys.stdout)
    _ch.setFormatter(_fmt)
    _root_logger.addHandler(_ch)

_fh = _RFH(
    _LOG_DIR / "feature_engineering.log",
    maxBytes=10 * 1024 * 1024,  # 10 MB per file
    backupCount=5,
    encoding="utf-8",
)
_fh.setFormatter(_fmt)
_root_logger.addHandler(_fh)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------

class FeatureEngineeringPipeline:
    """
    End-to-end feature engineering pipeline for the AI-text detection project.

    Parameters
    ----------
    max_features : int
        TF-IDF vocabulary cap.
    ngram_range : tuple
        N-gram lower/upper bounds.
    min_df : int
        Minimum document frequency for TF-IDF.
    max_df : float
        Maximum document frequency (fraction) for TF-IDF.
    test_size : float
        Fraction of data reserved for the test split.
    valid_size : float
        Fraction of data reserved for the validation split.
    random_state : int
        Seed for reproducibility.
    chunk_size : int
        Number of texts per chunk for handcrafted feature extraction.
    """

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 5,
        max_df: float = 0.95,
        test_size: float = 0.10,
        valid_size: float = 0.10,
        random_state: int = 42,
        chunk_size: int = 10_000,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.test_size = test_size
        self.valid_size = valid_size
        self.random_state = random_state
        self.chunk_size = chunk_size

        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizerWrapper(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
        )

    # ------------------------------------------------------------------
    # Step 1 — data loading
    # ------------------------------------------------------------------

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load the CSV dataset.

        Args:
            filepath: Path to final_dataset.csv

        Returns:
            DataFrame with 'text' and 'label' columns.
        """
        logger.info("Loading dataset from %s", filepath)
        df = pd.read_csv(filepath, usecols=["text", "label"])

        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("Dataset must have 'text' and 'label' columns.")

        # Drop nulls
        before = len(df)
        df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
        if len(df) < before:
            logger.warning("Dropped %d rows with NaN values.", before - len(df))

        # Deduplicate raw dataset to prevent leakage and target inflation
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
        deduped = before_dedup - len(df)
        if deduped > 0:
            logger.warning("Dropped %d duplicate rows based on text column.", deduped)

        # Cast label to int
        df["label"] = df["label"].astype(int)

        logger.info(
            "Dataset loaded: %d samples\n%s",
            len(df), df["label"].value_counts().to_string(),
        )
        return df

    # ------------------------------------------------------------------
    # Step 2 — splitting
    # ------------------------------------------------------------------

    def split_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Stratified split: train (80%) / valid (10%) / test (10%).

        Returns:
            X_train, X_valid, X_test, y_train, y_valid, y_test
        """
        logger.info("Splitting data (train/valid/test) …")
        temp_size = self.test_size + self.valid_size

        X_train, X_temp, y_train, y_temp = train_test_split(
            df["text"], df["label"],
            test_size=temp_size,
            random_state=self.random_state,
            stratify=df["label"],
        )

        valid_ratio = self.valid_size / temp_size
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1.0 - valid_ratio),
            random_state=self.random_state,
            stratify=y_temp,
        )

        logger.info(
            "Split sizes — train: %d | valid: %d | test: %d",
            len(X_train), len(X_valid), len(X_test),
        )
        return X_train, X_valid, X_test, y_train, y_valid, y_test

    # ------------------------------------------------------------------
    # Step 3 — preprocessing
    # ------------------------------------------------------------------

    def preprocess_split(self, texts: pd.Series, split_name: str) -> pd.Series:
        """
        Apply TextPreprocessor to every row of *texts*.

        Args:
            texts:      Raw text Series.
            split_name: Label for progress bar ("train", "valid", "test").

        Returns:
            Cleaned text Series (same index).
        """
        logger.info("Preprocessing %s split (%d samples) …", split_name, len(texts))
        cleaned = self.preprocessor.clean_texts(
            texts.tolist(), show_progress=True
        )
        return pd.Series(cleaned, index=texts.index)

    # ------------------------------------------------------------------
    # Step 4 — handcrafted features
    # ------------------------------------------------------------------

    def extract_handcrafted(
        self, texts: pd.Series, split_name: str
    ) -> pd.DataFrame:
        """
        Extract all 11 handcrafted features for a split.

        Uses chunked processing so the progress bar updates regularly even
        for very large splits.

        Args:
            texts:      Cleaned text Series.
            split_name: Label for logging.

        Returns:
            DataFrame of shape (n_samples, 11).
        """
        logger.info(
            "Extracting handcrafted features — %s (%d samples) …",
            split_name, len(texts),
        )
        texts_list = texts.tolist()
        n = len(texts_list)
        results = []

        chunks = range(0, n, self.chunk_size)
        for start in tqdm(chunks, desc=f"Features [{split_name}]", unit="chunk"):
            chunk = texts_list[start: start + self.chunk_size]
            results.extend(extract_features_batch(chunk))

        df = features_to_dataframe(results)
        logger.info(
            "Handcrafted features shape [%s]: %s", split_name, df.shape
        )
        return df

    # ------------------------------------------------------------------
    # Step 5 — TF-IDF
    # ------------------------------------------------------------------

    def apply_tfidf(
        self,
        train_texts: pd.Series,
        valid_texts: pd.Series,
        test_texts: pd.Series,
    ) -> Tuple:
        """
        Fit TF-IDF on training texts; transform all three splits.

        Returns:
            (X_train_tfidf, X_valid_tfidf, X_test_tfidf)
        """
        logger.info("Fitting TF-IDF on training split …")
        self.vectorizer.fit(train_texts.tolist())

        logger.info("Transforming all splits …")
        X_tr = self.vectorizer.transform(train_texts.tolist())
        X_va = self.vectorizer.transform(valid_texts.tolist())
        X_te = self.vectorizer.transform(test_texts.tolist())

        logger.info(
            "TF-IDF shapes — train %s | valid %s | test %s",
            X_tr.shape, X_va.shape, X_te.shape,
        )
        return X_tr, X_va, X_te

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_pipeline(self, data_path: str, output_dir: str) -> Tuple:
        """
        Execute the full feature engineering pipeline and persist all artefacts.

        Args:
            data_path:  Path to ``final_dataset.csv``.
            output_dir: Root output directory (e.g. ``outputs/``).

        Returns:
            (X_train, X_valid, X_test, y_train, y_valid, y_test,
             (hc_train, hc_valid, hc_test))
        """
        out = Path(output_dir)

        # ── 1. Load ────────────────────────────────────────────────────
        df = self.load_data(data_path)

        # ── 2. Split ───────────────────────────────────────────────────
        X_train, X_valid, X_test, y_train, y_valid, y_test = self.split_data(df)

        # ── 3. Preprocess ──────────────────────────────────────────────
        X_train_clean = self.preprocess_split(X_train, "train")
        X_valid_clean = self.preprocess_split(X_valid, "valid")
        X_test_clean  = self.preprocess_split(X_test,  "test")

        # ── 4. Handcrafted features ────────────────────────────────────
        hc_train = self.extract_handcrafted(X_train, "train")
        hc_valid = self.extract_handcrafted(X_valid, "valid")
        hc_test  = self.extract_handcrafted(X_test,  "test")

        hc_dir = out / "handcrafted"
        hc_dir.mkdir(parents=True, exist_ok=True)
        hc_train.to_csv(hc_dir / "train_features.csv", index=False)
        hc_valid.to_csv(hc_dir / "valid_features.csv", index=False)
        hc_test.to_csv( hc_dir / "test_features.csv",  index=False)
        logger.info("Handcrafted features saved to %s", hc_dir)

        # ── 5. TF-IDF ─────────────────────────────────────────────────
        X_tr_tfidf, X_va_tfidf, X_te_tfidf = self.apply_tfidf(
            X_train_clean, X_valid_clean, X_test_clean
        )

        tfidf_dir = out / "tfidf"
        tfidf_dir.mkdir(parents=True, exist_ok=True)

        self.vectorizer.save(tfidf_dir / "tfidf_vectorizer.pkl")
        save_sparse_matrix(X_tr_tfidf, tfidf_dir / "X_train.npz")
        save_sparse_matrix(X_va_tfidf, tfidf_dir / "X_valid.npz")
        save_sparse_matrix(X_te_tfidf, tfidf_dir / "X_test.npz")

        # ── Labels ────────────────────────────────────────────────────
        lbl_dir = out / "labels"
        lbl_dir.mkdir(parents=True, exist_ok=True)
        save_labels(y_train, lbl_dir / "y_train.csv")
        save_labels(y_valid, lbl_dir / "y_valid.csv")
        save_labels(y_test,  lbl_dir / "y_test.csv")

        logger.info("✅ Feature engineering pipeline complete.  Outputs → %s", out)

        return (
            X_tr_tfidf, X_va_tfidf, X_te_tfidf,
            y_train, y_valid, y_test,
            (hc_train, hc_valid, hc_test),
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run the feature engineering pipeline from the command line.

    Paths are resolved relative to the repository root so this script can be
    invoked from any working directory:

        python feature_engineering/feature_engineering.py
    """
    # Resolve project root (two levels above this file)
    REPO_ROOT = Path(__file__).resolve().parent.parent
    DATA_PATH  = REPO_ROOT / "Datasets" / "merged" / "final_dataset.csv"
    OUTPUT_DIR = REPO_ROOT / "outputs"

    logger.info("Repository root : %s", REPO_ROOT)
    logger.info("Dataset path    : %s", DATA_PATH)
    logger.info("Output directory: %s", OUTPUT_DIR)

    pipeline = FeatureEngineeringPipeline(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        test_size=0.20,
        valid_size=0.10,
        random_state=42,
        chunk_size=10_000,
    )

    try:
        pipeline.run_pipeline(
            data_path=str(DATA_PATH),
            output_dir=str(OUTPUT_DIR),
        )
        logger.info("Feature engineering completed successfully.")
    except Exception as exc:
        logger.exception("Feature engineering pipeline failed: %s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
