"""
explainability/__init__.py
==========================
Package initialization for the Explainable AI (XAI) subsystem.
Phase 8.2: SHAP Explainability for Meta-Classifier and XGBoost.
Phase 8.3: LIME Explainability for End-to-End Word Attribution.
Phase 8.4: Transformer Attention Visualization for RoBERTa, DeBERTa, and DistilBERT.
Phase 8.5: Unified Explainable AI Pipeline Engine.
"""

from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer
from .attention_explainer import AttentionExplainer
from .explainability import ExplainabilityEngine, load_text_from_input, generate_dynamic_decision_explanation

__all__ = [
    "SHAPExplainer",
    "LIMEExplainer",
    "AttentionExplainer",
    "ExplainabilityEngine",
    "load_text_from_input",
    "generate_dynamic_decision_explanation"
]
