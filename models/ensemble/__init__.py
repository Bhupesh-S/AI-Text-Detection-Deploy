"""
models/ensemble/__init__.py
==========================
Hybrid Ensemble Classifier package for AI-Generated Text Detection.

Author  : Bharanidharan K
Project : Hybrid Explainable Framework for Detecting AI-Generated Academic Text
"""

from .config import CFG
from .predict import HybridInferencePipeline

__all__ = ["CFG", "HybridInferencePipeline"]
