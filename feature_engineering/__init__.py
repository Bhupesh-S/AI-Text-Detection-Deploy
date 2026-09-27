"""
Feature Engineering Package
============================
Provides text preprocessing, handcrafted feature extraction, and TF-IDF
vectorisation for the AI-text detection project.

Public API
----------
    from feature_engineering import FeatureEngineeringPipeline
    from feature_engineering import TextPreprocessor, preprocess_texts
    from feature_engineering import TfidfVectorizerWrapper
    from feature_engineering import extract_features, extract_features_batch
"""

from feature_engineering.feature_engineering import FeatureEngineeringPipeline
from feature_engineering.preprocessing import TextPreprocessor, preprocess_text, preprocess_texts
from feature_engineering.utils import extract_features, extract_features_batch, features_to_dataframe
from feature_engineering.vectorizer import (
    TfidfVectorizerWrapper,
    save_sparse_matrix,
    load_sparse_matrix,
    save_labels,
    load_labels,
)

__all__ = [
    "FeatureEngineeringPipeline",
    "TextPreprocessor",
    "preprocess_text",
    "preprocess_texts",
    "extract_features",
    "extract_features_batch",
    "features_to_dataframe",
    "TfidfVectorizerWrapper",
    "save_sparse_matrix",
    "load_sparse_matrix",
    "save_labels",
    "load_labels",
]
