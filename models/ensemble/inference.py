"""
models/ensemble/inference.py
============================
Backwards-compatibility re-export module.
Delegates cleanly to the single authoritative HybridInferencePipeline in models/ensemble/predict.py.
"""

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_FILE.parent))

from predict import HybridInferencePipeline

__all__ = ["HybridInferencePipeline"]
