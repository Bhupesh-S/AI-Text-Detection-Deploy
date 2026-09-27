"""
TF-IDF Vectorization Module
============================
Wraps sklearn's TfidfVectorizer with:
  - Consistent project-wide configuration
  - Fit-only-on-train-data discipline
  - Persistence (save / load) via joblib
  - Sparse matrix I/O helpers
  - Label CSV helpers
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class TfidfVectorizerWrapper:
    """
    Thin wrapper around sklearn's TfidfVectorizer.

    Configuration is passed at construction time; the underlying vectorizer is
    created lazily on the first call to :meth:`fit`.

    Parameters
    ----------
    max_features : int
        Maximum vocabulary size.
    ngram_range : tuple(int, int)
        Lower and upper n-gram boundaries.
    min_df : int
        Minimum document frequency (absolute count).
    max_df : float
        Maximum document frequency (fraction of corpus).
    """

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 5,
        max_df: float = 0.95,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer: Optional[TfidfVectorizer] = None

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, texts: List[str]) -> "TfidfVectorizerWrapper":
        """
        Fit TF-IDF vocabulary on *training* texts only.

        Args:
            texts: List of preprocessed strings (training split).

        Returns:
            self
        """
        logger.info(
            "Fitting TF-IDF vectorizer — max_features=%d, ngram=%s, "
            "min_df=%d, max_df=%.2f",
            self.max_features, self.ngram_range, self.min_df, self.max_df,
        )
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            lowercase=False,      # text already lowercased by preprocessor
            stop_words="english",
            sublinear_tf=True,    # log(1+tf) — better for long documents
            dtype=np.float32,     # halves memory vs float64
        )
        self.vectorizer.fit(texts)
        logger.info(
            "Vocabulary fitted: %d features", len(self.vectorizer.vocabulary_)
        )
        return self

    def transform(self, texts: List[str]) -> sp.csr_matrix:
        """
        Transform texts to a sparse TF-IDF matrix.

        Args:
            texts: List of preprocessed strings.

        Returns:
            CSR sparse matrix of shape (n_samples, n_features).

        Raises:
            ValueError: If :meth:`fit` has not been called first.
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted — call fit() first.")
        logger.info("Transforming %d texts …", len(texts))
        matrix = self.vectorizer.transform(texts)
        logger.info("Matrix shape: %s", matrix.shape)
        return matrix

    def fit_transform(self, texts: List[str]) -> sp.csr_matrix:
        """Fit then transform in one step (use only on training data)."""
        self.fit(texts)
        return self.transform(texts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Serialise the fitted vectorizer to *filepath* via joblib."""
        if self.vectorizer is None:
            raise ValueError("No fitted vectorizer to save.")
        fp = Path(filepath)
        fp.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, fp, compress=3)
        logger.info("TF-IDF vectorizer saved → %s", fp)

    def load(self, filepath: str) -> "TfidfVectorizerWrapper":
        """Deserialise a vectorizer from *filepath* via joblib."""
        fp = Path(filepath)
        if not fp.exists():
            raise FileNotFoundError(f"Vectorizer not found at {fp}")
        self.vectorizer = joblib.load(fp)
        logger.info("TF-IDF vectorizer loaded ← %s", fp)
        return self

    def get_feature_names(self) -> List[str]:
        """Return the vocabulary token list."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted.")
        return self.vectorizer.get_feature_names_out().tolist()


# ---------------------------------------------------------------------------
# Sparse matrix helpers
# ---------------------------------------------------------------------------

def save_sparse_matrix(matrix: sp.csr_matrix, filepath: str) -> None:
    """Save a scipy sparse matrix to an .npz file."""
    fp = Path(filepath)
    fp.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(str(fp), matrix)
    logger.info("Sparse matrix saved → %s  shape=%s", fp, matrix.shape)


def load_sparse_matrix(filepath: str) -> sp.csr_matrix:
    """Load a scipy sparse matrix from an .npz file."""
    fp = Path(filepath)
    if not fp.exists():
        raise FileNotFoundError(f"Sparse matrix not found at {fp}")
    matrix = sp.load_npz(str(fp))
    logger.info("Sparse matrix loaded ← %s  shape=%s", fp, matrix.shape)
    return matrix


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def save_labels(labels: pd.Series, filepath: str) -> None:
    """Persist a label Series as a single-column CSV."""
    fp = Path(filepath)
    fp.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(fp, index=False, header=["label"])
    logger.info("Labels saved → %s  (%d samples)", fp, len(labels))


def load_labels(filepath: str) -> pd.Series:
    """Load a label CSV produced by :func:`save_labels`."""
    fp = Path(filepath)
    if not fp.exists():
        raise FileNotFoundError(f"Label file not found at {fp}")
    series = pd.read_csv(fp).squeeze("columns")
    logger.info("Labels loaded ← %s  (%d samples)", fp, len(series))
    return series
