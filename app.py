import os
import re
import sys
import time
import math
import pathlib
import joblib
import pandas as pd
import streamlit as st

# Ensure repository root is in sys.path
BASE_DIR = pathlib.Path(__file__).parent.resolve()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import Hybrid Pipeline & Explainability Engine
try:
    from models.ensemble.predict import HybridInferencePipeline
    HYBRID_AVAILABLE = True
except Exception as e:
    HYBRID_AVAILABLE = False
    HYBRID_ERROR = str(e)

try:
    from explainability import ExplainabilityEngine, load_text_from_input, generate_dynamic_decision_explanation
    XAI_AVAILABLE = True
except Exception as e:
    XAI_AVAILABLE = False
    XAI_ERROR = str(e)
    ExplainabilityEngine = None
    load_text_from_input = None
    generate_dynamic_decision_explanation = None

# Set Streamlit page config
st.set_page_config(
    page_title="Hybrid AI Text Detector & XAI Dashboard",
    page_icon=":material/security:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS: injected once at startup ─────────────────────────────────────────────
def _inject_css():
    """Inject core theme CSS. Called once at render time."""
    bg         = "#090d16"
    surface    = "#111827"
    surface2   = "#0b1120"
    border_col = "rgba(255,255,255,0.07)"
    text_pri   = "#f1f5f9"
    text_sec   = "#94a3b8"
    text_card  = "#cbd5e1"
    sidebar_bg = "#0b1120"
    sidebar_br = "rgba(255,255,255,0.07)"
    track_bg   = "#1e293b"
    input_bg   = "#0b1120"
    input_br   = "#1e293b"
    sub_info   = "#64748b"

    st.markdown(f"""
<style>
/* ── 0. Google Fonts ──────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── 1. Core Theme ───────────────────────────────────────────── */
.stApp {{
    background-color: {bg} !important;
    color: {text_pri} !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}

/* ── 2. Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {{
    min-width: 280px !important;
    max-width: 290px !important;
    background-color: {sidebar_bg} !important;
    border-right: 1px solid {sidebar_br} !important;
}}
[data-testid="stSidebar"] .block-container {{
    padding-top: 1.2rem !important;
    padding-bottom: 1.5rem !important;
}}
[data-testid="stSidebar"] [data-testid="stExpander"] {{
    background: #0f172a !important;
    border: 1px solid {input_br} !important;
    border-radius: 8px !important;
    margin-top: 8px !important;
}}

/* ── 3. Main content padding ─────────────────────────────────── */
.main .block-container {{
    padding-top: 1.0rem !important;
    padding-bottom: 2rem !important;
    max-width: 1440px !important;
}}

/* ── 4. Dashboard Header ─────────────────────────────────────── */
.dash-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.0rem 1.5rem;
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 60%, #312e81 100%);
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 20px -4px rgba(15,23,42,0.6);
    margin-bottom: 1.2rem;
}}
.dash-title {{
    font-size: 1.45rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
    margin: 0;
}}
.dash-subtitle {{ font-size: 0.85rem; color: #94a3b8; margin-top: 2px; }}
.dash-tags    {{ font-size: 0.76rem; color: #818cf8; font-weight: 600; margin-top: 4px; letter-spacing: 0.02em; }}

/* Status pill */
.status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 13px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}}
.status-online  {{ background: rgba(16,185,129,0.12); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }}
.status-offline {{ background: rgba(244,63,94,0.12);  color: #f43f5e; border: 1px solid rgba(244,63,94,0.3); }}
.status-dot {{ width:7px; height:7px; border-radius:50%; background-color:currentColor; }}

/* ── 5. Cards & Section Header Boxes ─────────────────────────── */
.card-header-box {{
    background: {surface};
    border: 1px solid {border_col};
    border-radius: 10px;
    padding: 0.7rem 1.1rem;
    margin-bottom: 0.85rem;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3),
                0 8px 16px -4px rgba(0,0,0,0.35);
    display: flex;
    align-items: center;
}}
.card-header-title {{
    font-size: 0.82rem;
    font-weight: 700;
    color: {text_card};
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 0;
    line-height: 1;
    display: flex;
    align-items: center;
}}
.card-panel {{
    background: {surface};
    border: 1px solid {border_col};
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3),
                0 10px 24px -4px rgba(0,0,0,0.35);
    transition: box-shadow 0.2s ease;
}}
.card-panel:hover {{
    box-shadow: 0 6px 12px -2px rgba(0,0,0,0.4),
                0 16px 32px -6px rgba(0,0,0,0.45);
}}

/* ── 6. Textarea & focus glow ────────────────────────── */
.stTextArea textarea {{
    background-color: {input_bg} !important;
    color: {text_pri} !important;
    border: 1px solid {input_br} !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    line-height: 1.6 !important;
    padding: 0.8rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}}
.stTextArea textarea:focus {{
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.25), 0 0 14px rgba(99,102,241,0.15) !important;
    outline: none !important;
}}

/* ── 7. Status chip / stats bar ─────────────────────── */
.input-stats-bar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-around;
    background: {surface2};
    border: 1px solid {input_br};
    border-radius: 20px;
    padding: 6px 16px;
    margin-top: 8px;
    margin-bottom: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    color: {text_sec};
    gap: 8px;
}}
.stat-item  {{ display: flex; align-items: center; gap: 5px; }}
.stat-value {{ color: {text_pri}; font-weight: 700; }}

/* ── 8. Buttons ─────────────────────────────────────── */
.stButton > button {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    border-radius: 8px !important;
    height: 38px !important;
    transition: transform 0.12s ease, box-shadow 0.15s ease, opacity 0.15s ease !important;
}}
.stButton > button:hover  {{ transform: translateY(-1px) !important; box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important; }}
.stButton > button:active {{ transform: translateY(0px) !important; opacity: 0.88 !important; }}

/* ── 9. Result Banner ─────────────────────── */
.banner-human {{
    background: linear-gradient(135deg, rgba(6,78,59,0.45) 0%, rgba(4,120,87,0.22) 100%);
    border: 1.5px solid #10b981;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 16px rgba(16,185,129,0.12);
}}
.banner-ai {{
    background: linear-gradient(135deg, rgba(136,19,55,0.45) 0%, rgba(190,18,60,0.22) 100%);
    border: 1.5px solid #f43f5e;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 16px rgba(244,63,94,0.12);
}}
.res-hdr-flex {{ display: flex; align-items: center; justify-content: space-between; }}

/* Verdict badge */
.res-badge-human {{
    color: #34d399;
    font-size: 1.55rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    line-height: 1.1;
    display: flex;
    align-items: center;
}}
.res-badge-ai {{
    color: #fb7185;
    font-size: 1.55rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    line-height: 1.1;
    display: flex;
    align-items: center;
}}
.res-badge-sub {{
    font-size: 0.76rem;
    color: {sub_info};
    font-weight: 500;
    margin-top: 3px;
    letter-spacing: 0.01em;
}}

/* Confidence % */
.res-score-huge {{
    font-size: 2.1rem;
    font-weight: 900;
    color: #ffffff;
    line-height: 1;
    letter-spacing: -0.03em;
}}
.res-score-sub {{
    font-size: 0.72rem;
    color: {sub_info};
    font-weight: 500;
    text-align: right;
    letter-spacing: 0.02em;
    margin-top: 3px;
}}

/* Gauge */
.gauge-wrapper   {{ margin-top: 1rem; opacity: 0.92; }}
.gauge-labels    {{ display: flex; justify-content: space-between; font-size: 0.70rem; font-weight: 600; color: {sub_info}; margin-bottom: 5px; }}
.gauge-track     {{ position: relative; height: 10px; background: {track_bg}; border-radius: 5px; overflow: hidden; }}
.gauge-fill-ai   {{ height: 100%; background: linear-gradient(90deg,#f43f5e,#fb7185); border-radius: 5px; }}
.gauge-fill-human{{ height: 100%; background: linear-gradient(90deg,#10b981,#34d399); border-radius: 5px; }}
.gauge-threshold-line {{ position:absolute; top:0; bottom:0; width:2px; background:#f59e0b; box-shadow:0 0 6px #f59e0b; z-index:2; }}
.gauge-sub-info  {{ display:flex; justify-content:space-between; font-size:0.70rem; color:{sub_info}; margin-top:4px; }}

/* ── 10. Model Cards ────── */
.model-row-card {{
    background: {surface2};
    border: 1px solid {input_br};
    border-radius: 8px;
    padding: 9px 12px;
    margin-bottom: 6px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    transition: box-shadow 0.15s ease, transform 0.12s ease;
}}
.model-row-card:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,0.3); transform: translateY(-1px); }}
.model-row-header {{ display:flex; justify-content:space-between; align-items:center; font-size:0.82rem; font-weight:700; margin-bottom:5px; }}
.model-name    {{ color: {text_pri}; }}
.model-probs   {{ font-size: 0.78rem; font-weight: 600; }}
.prob-ai-tag   {{ color: #fb7185; }}
.prob-hu-tag   {{ color: #34d399; }}

.mini-track {{ height:6px; background:{track_bg}; border-radius:3px; overflow:hidden; }}
.mini-fill-safe   {{ height:100%; border-radius:3px; background: linear-gradient(90deg,#10b981,#34d399); }}
.mini-fill-warn   {{ height:100%; border-radius:3px; background: linear-gradient(90deg,#f59e0b,#fbbf24); }}
.mini-fill-danger {{ height:100%; border-radius:3px; background: linear-gradient(90deg,#f43f5e,#fb7185); }}

/* ── 11. Consensus Panel ─────────────────────────────────────── */
.consensus-badge-unanimous {{
    background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3);
    color: #34d399; border-radius: 8px; padding: 8px 12px;
    font-size: 0.84rem; font-weight: 700; margin-bottom: 10px;
    display: flex; align-items: center;
}}
.consensus-badge-split {{
    background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3);
    color: #fbbf24; border-radius: 8px; padding: 8px 12px;
    font-size: 0.84rem; font-weight: 700; margin-bottom: 10px;
    display: flex; align-items: center;
}}
.vote-item {{ display:flex; justify-content:space-between; align-items:center; font-size:0.78rem; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.04); }}
.vote-item:last-child {{ border-bottom:none; }}
.vote-model-lbl {{ color:{text_sec}; font-weight:600; }}
.vote-result-ai  {{ color:#fb7185; font-weight:700; }}
.vote-result-hu  {{ color:#34d399; font-weight:700; }}

/* ── 12. Stylometric Feature Grid ─────── */
.metric-grid-6 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:8px; }}
@media (max-width: 520px) {{
    .metric-grid-6 {{ grid-template-columns: repeat(2,1fr); }}
}}
.feat-card {{
    background: {surface2};
    border: 1px solid {input_br};
    border-radius: 8px;
    padding: 12px 8px 10px;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.18);
    transition: box-shadow 0.18s ease, transform 0.14s ease, border-color 0.18s ease;
    cursor: default;
}}
.feat-card:hover {{
    box-shadow: 0 6px 18px rgba(56,189,248,0.18);
    transform: translateY(-2px);
    border-color: rgba(56,189,248,0.35);
}}
.feat-icon {{ font-size: 1.05rem; margin-bottom: 4px; line-height: 1; display: flex; justify-content: center; align-items: center; }}
.feat-val  {{ font-size:1.2rem; font-weight:800; color:#38bdf8; line-height:1; }}
.feat-lbl  {{ font-size:0.68rem; color:{text_sec}; font-weight:600; margin-top:4px; text-transform:uppercase; letter-spacing:0.03em; }}

/* ── 13. Sidebar Status Box ──────────────────────────────────── */
.sb-status-box  {{ background:#0f172a; border:1px solid {input_br}; border-radius:8px; padding:10px 12px; margin-top:4px; }}
.sb-status-title{{ font-size:0.76rem; font-weight:700; color:{text_card}; margin-bottom:6px; letter-spacing:0.04em; }}
.sb-item {{ font-size:0.73rem; color:{text_sec}; padding:2.5px 0; display:flex; align-items:center; justify-content:space-between; }}
.sb-active {{ color:#34d399; font-weight:700; }}

/* ── 14. Footer ──────────────────────────────────────────────── */
.app-footer {{ text-align:center; padding:1.2rem 0 0.5rem; font-size:0.74rem; color:#475569; border-top:1px solid rgba(255,255,255,0.05); margin-top:1.5rem; }}
</style>
""", unsafe_allow_html=True)

_inject_css()

# ----------------------------------------------------
# Paths & Cached Models
# ----------------------------------------------------

HF_REPO_ID = "bhupesh1907/ai-transformer-v3"
MODELS_DIR = BASE_DIR / "models" / "baselines"
MIN_WORDS = 200

# Cache Hybrid Inference Pipeline
@st.cache_resource(show_spinner="Loading Hybrid Transformer Models & Ensembles...")
def get_hybrid_pipeline():
    if HYBRID_AVAILABLE:
        try:
            return HybridInferencePipeline()
        except Exception as e:
            st.error(f"Error initializing Hybrid Inference Pipeline: {e}")
            return None
    return None

# Cache Explainability Engine
@st.cache_resource(show_spinner="Loading Explainable AI Subsystems (SHAP, LIME, Attention)...")
def get_xai_engine():
    pipeline = get_hybrid_pipeline()
    if pipeline is not None and XAI_AVAILABLE and ExplainabilityEngine is not None:
        try:
            return ExplainabilityEngine(pipeline=pipeline)
        except Exception as e:
            st.error(f"Error initializing Explainability Engine: {e}")
            return None
    return None

@st.cache_resource
def load_baseline_models():
    models = {}
    if not MODELS_DIR.exists():
        return models
    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Linear SVM": "linear_svm.joblib",
        "Multinomial Naive Bayes": "multinomial_naive_bayes.joblib",
        "Random Forest": "random_forest.joblib",
    }
    for name, filename in model_files.items():
        path = MODELS_DIR / filename
        if path.exists():
            try:
                models[name] = joblib.load(path)
            except Exception:
                pass
    return models

def count_words(text: str) -> int:
    """Consistently count words in text."""
    if not text or not text.strip():
        return 0
    return len(text.strip().split())

def format_prediction_time(seconds) -> str:
    """Format duration into a human-readable prediction time string."""
    if seconds is None:
        return "—"
    if seconds < 1.0:
        return f"{seconds * 1000.0:.0f} ms"
    return f"{seconds:.1f} sec"

def compute_text_metrics(text: str):
    """Compute linguistic features for text analysis UI."""
    word_count = count_words(text)
    words = re.findall(r'\b\w+\b', text)
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    chars = len(text)
    sentence_count = len(sentences) if len(sentences) > 0 else (1 if word_count > 0 else 0)

    avg_word_len = sum(len(w) for w in words) / len(words) if len(words) > 0 else 0
    avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0

    unique_words = set(w.lower() for w in words)
    ttr = len(unique_words) / word_count if word_count > 0 else 0

    return {
        "word_count": word_count,
        "char_count": chars,
        "sentence_count": sentence_count,
        "avg_word_len": avg_word_len,
        "avg_sentence_len": avg_sentence_len,
        "ttr": ttr
    }

# Load cached resources
hybrid_pipeline = get_hybrid_pipeline()
xai_engine = get_xai_engine()
if xai_engine is not None and hybrid_pipeline is not None:
    if hasattr(xai_engine, "set_pipeline"):
        xai_engine.set_pipeline(hybrid_pipeline)
    else:
        xai_engine.pipeline = hybrid_pipeline
# Obtain vectorizer and XGBoost directly from the authoritative hybrid pipeline
vectorizer = getattr(hybrid_pipeline, "vectorizer_xgboost", None) if hybrid_pipeline is not None else None
baseline_models = load_baseline_models()
if hybrid_pipeline is not None and getattr(hybrid_pipeline, "model_xgboost", None) is not None:
    baseline_models["XGBoost"] = hybrid_pipeline.model_xgboost

# Streamlit Header
st.markdown("""
<div class="dash-header">
    <div>
        <div class="dash-title">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px; vertical-align:-3px;"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="15" x2="23" y2="15"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="15" x2="4" y2="15"/></svg>Hybrid AI Text Detector & XAI Dashboard
        </div>
        <div class="dash-subtitle">Advanced Multi-Model AI Detection & Explainable AI Framework (SHAP, LIME, Attention)</div>
        <div class="dash-tags">RoBERTa &bull; DeBERTa &bull; DistilBERT &bull; XGBoost &bull; Stacking Meta-Classifier &bull; XAI Engine</div>
    </div>
    <div>
        <span class="status-pill status-online">
            <span class="status-dot"></span> SYSTEM ONLINE
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Fixed Authoritative Engine Identifier
selected_engine = "Hybrid Stacking Meta-Classifier (Recommended)"

# Compact Sidebar Configuration
with st.sidebar:
    # Architecture Status & Signature
    with st.expander("Architecture Status & Signature", expanded=True):
        if hybrid_pipeline is not None:
            device_str = str(hybrid_pipeline.device).upper()
            st.markdown(f"""
            <div style="margin-bottom: 8px;">
                <div class="sb-item"><span class="sb-active" style="font-size:0.78rem;">● Hybrid Pipeline Active</span></div>
            </div>
            <div class="sb-item"><span>Device</span> <code style="font-size:0.72rem; color:#f1f5f9; background:#1e293b; padding:1px 6px; border-radius:4px; font-weight:600;">{device_str}</code></div>
            <div style="height: 6px;"></div>
            <div class="sb-item"><span>RoBERTa V3</span> <span class="sb-active">✓ Loaded</span></div>
            <div class="sb-item"><span>DeBERTa-v3 V3</span> <span class="sb-active">✓ Loaded</span></div>
            <div class="sb-item"><span>DistilBERT V3</span> <span class="sb-active">✓ Loaded</span></div>
            <div class="sb-item"><span>XGBoost</span> <span class="sb-active">✓ Loaded</span></div>
            <div style="height: 6px;"></div>
            <div class="sb-item"><span>Stacking Meta-Classifier</span> <span class="sb-active">✓ Active</span></div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""<div class="sb-item"><span style="color:#f43f5e; font-weight:700;">● Hybrid Pipeline Offline</span></div>""", unsafe_allow_html=True)

        xai_lbl = '<span class="sb-active">✓ Active</span>' if xai_engine else '<span style="color:#94a3b8;">Offline</span>'
        tfidf_status = '<span class="sb-active">✓ Active</span>' if vectorizer else '<span style="color:#94a3b8;">Offline</span>'
        st.markdown(f"""
            <div class="sb-item"><span>Unified XAI Engine</span> {xai_lbl}</div>
            <div class="sb-item"><span>TF-IDF Vectorizer</span> {tfidf_status}</div>
            <div class="sb-item"><span>Baseline Models</span> <span style="color:#f1f5f9; font-weight:600;">{len(baseline_models)} loaded</span></div>
        """, unsafe_allow_html=True)

# Main Dashboard Grid
col_left, col_right = st.columns([1.05, 1.0], gap="medium")

with col_left:
    st.markdown("""
    <div class="card-header-box">
        <div class="card-header-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#cbd5e1" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px; vertical-align:-1px;"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>TEXT INPUT FOR AI DETECTION
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "main_text_input" not in st.session_state:
        st.session_state["main_text_input"] = ""

    input_text = st.text_area(
        label="Paste academic text, essay, or generated content below:",
        height=250,
        placeholder=f"Paste text here to inspect for AI generation (minimum {MIN_WORDS} words required)...",
        key="main_text_input"
    )

    word_count = count_words(input_text)
    metrics = compute_text_metrics(input_text)
    is_valid_input = word_count >= MIN_WORDS

    # Check for text & engine modifications to prevent stale results
    current_matches_last = (
        "last_analyzed_text" in st.session_state and
        st.session_state.last_analyzed_text == input_text and
        "last_selected_engine" in st.session_state and
        st.session_state.last_selected_engine == selected_engine and
        is_valid_input
    )

    if not current_matches_last and "last_analysis_results" in st.session_state:
        del st.session_state["last_analysis_results"]

    col_btn1, col_btn2 = st.columns([1.2, 1])
    with col_btn1:
        analyze_clicked = st.button("Analyze Text", type="primary", icon=":material/analytics:", use_container_width=True)
    with col_btn2:
        clear_clicked = st.button("Clear Text", icon=":material/delete_outline:", use_container_width=True)
        if clear_clicked:
            st.session_state["main_text_input"] = ""
            if "uploaded_extracted_text" in st.session_state:
                del st.session_state["uploaded_extracted_text"]
            if "last_analyzed_text" in st.session_state:
                del st.session_state["last_analyzed_text"]
            if "last_selected_engine" in st.session_state:
                del st.session_state["last_selected_engine"]
            if "last_analysis_results" in st.session_state:
                del st.session_state["last_analysis_results"]
            if "xai_result_cache" in st.session_state:
                del st.session_state["xai_result_cache"]
            st.rerun()

    if analyze_clicked and not is_valid_input:
        st.warning(f"Please enter at least {MIN_WORDS} words for reliable detection.", icon=":material/warning:")

    should_run_inference = analyze_clicked and is_valid_input

    if should_run_inference:
        with st.spinner("Running Hybrid Detection Engine..."):
            t_start = time.perf_counter()
            ai_prob = 0.5
            hybrid_res = None
            model_breakdown = {}
            strategy_lbl = selected_engine

            if hybrid_pipeline is None and HYBRID_AVAILABLE:
                hybrid_pipeline = get_hybrid_pipeline()

            if hybrid_pipeline is not None and ("Hybrid" in selected_engine or "RoBERTa" in selected_engine or "DeBERTa" in selected_engine or "DistilBERT" in selected_engine):
                try:
                    if not hasattr(hybrid_pipeline, "_get_vectorizer_xgboost"):
                        st.cache_resource.clear()
                        hybrid_pipeline = get_hybrid_pipeline()
                    hybrid_res = hybrid_pipeline.predict(input_text)
                except Exception as pred_err:
                    if "Feature shape mismatch" in str(pred_err) or "vectorizer" in str(pred_err).lower():
                        st.cache_resource.clear()
                        hybrid_pipeline = get_hybrid_pipeline()
                        hybrid_res = hybrid_pipeline.predict(input_text)
                    else:
                        raise pred_err

                if not isinstance(hybrid_res, dict) or "model_breakdown" not in hybrid_res or "ai_probability" not in hybrid_res:
                    st.cache_resource.clear()
                    hybrid_pipeline = get_hybrid_pipeline()
                    hybrid_res = hybrid_pipeline.predict(input_text)

                if "model_breakdown" in hybrid_res:
                    mb = hybrid_res["model_breakdown"]
                    r_ai = float(mb.get("roberta", mb.get("RoBERTa-base", {})).get("ai_probability", 0.5))
                    d_ai = float(mb.get("deberta", mb.get("DeBERTa-v3", {})).get("ai_probability", 0.5))
                    db_ai = float(mb.get("distilbert", mb.get("DistilBERT", {})).get("ai_probability", 0.5))
                    xgb_ai = float(mb.get("xgboost", mb.get("XGBoost (Stylometrics)", {})).get("ai_probability", 0.5))
                elif "base_probs" in hybrid_res:
                    bp = hybrid_res["base_probs"]
                    r_ai = float(bp["roberta"][1])
                    d_ai = float(bp["deberta"][1])
                    db_ai = float(bp["distilbert"][1])
                    xgb_ai = float(bp["xgboost"][1])
                else:
                    r_ai = d_ai = db_ai = xgb_ai = 0.5

                model_breakdown["RoBERTa-base"] = r_ai
                model_breakdown["DeBERTa-v3"] = d_ai
                model_breakdown["DistilBERT"] = db_ai
                model_breakdown["XGBoost (Stylometrics)"] = xgb_ai

                if "Stacking Meta-Classifier" in selected_engine or "Hybrid Stacking" in selected_engine or "Hybrid" in selected_engine:
                    ai_prob = float(hybrid_res.get("ai_probability", hybrid_res.get("final_probs", [0.5, 0.5])[1]))
                    human_prob = float(hybrid_res.get("human_probability", 1.0 - ai_prob))
                    strategy_lbl = hybrid_res.get("strategy_used", "Stacking Meta-Classifier")
                elif "Weighted Ensemble" in selected_engine:
                    ai_prob = float(hybrid_res.get("weighted_voting_probs", [0.5, 0.5])[1])
                    human_prob = float(hybrid_res.get("weighted_voting_probs", [0.5, 0.5])[0])
                    strategy_lbl = "Weighted Voting Ensemble"
                elif "RoBERTa" in selected_engine:
                    ai_prob = r_ai
                    human_prob = 1.0 - r_ai
                    strategy_lbl = "RoBERTa-base Sequence Classifier"
                elif "DeBERTa" in selected_engine:
                    ai_prob = d_ai
                    human_prob = 1.0 - d_ai
                    strategy_lbl = "DeBERTa-v3 Sequence Classifier"
                elif "DistilBERT" in selected_engine:
                    ai_prob = db_ai
                    human_prob = 1.0 - db_ai
                    strategy_lbl = "DistilBERT Sequence Classifier"

            elif vectorizer is not None and baseline_models:
                vec = vectorizer.transform([input_text])
                clean_engine_name = selected_engine.replace(" (TF-IDF)", "")

                if clean_engine_name in baseline_models:
                    m = baseline_models[clean_engine_name]
                    if hasattr(m, "predict_proba"):
                        ai_prob = float(m.predict_proba(vec)[0][1])
                    elif hasattr(m, "decision_function"):
                        score = float(m.decision_function(vec)[0])
                        ai_prob = 1.0 / (1.0 + math.exp(-score))
                    strategy_lbl = f"{clean_engine_name} (TF-IDF)"
                else:
                    strategy_lbl = f"{selected_engine}"

                for bname, bm in baseline_models.items():
                    try:
                        if hasattr(bm, "predict_proba"):
                            model_breakdown[bname] = float(bm.predict_proba(vec)[0][1])
                    except Exception:
                        pass
            else:
                strategy_lbl = "Heuristic Fallback"
                ai_prob = 0.5 + (0.15 * (1.0 - metrics['ttr'])) - (0.05 * (metrics['avg_sentence_len'] / 20))
                ai_prob = min(max(ai_prob, 0.05), 0.95)

            elapsed_sec = time.perf_counter() - t_start
            elapsed_ms = elapsed_sec * 1000.0
            pred_time_str = format_prediction_time(elapsed_sec)

            st.session_state.last_analyzed_text = input_text
            st.session_state.last_selected_engine = selected_engine
            st.session_state.last_analysis_results = {
                "ai_prob": ai_prob,
                "human_prob": 1.0 - ai_prob,
                "hybrid_res": hybrid_res,
                "model_breakdown": model_breakdown,
                "strategy_lbl": strategy_lbl,
                "elapsed_ms": elapsed_ms,
                "elapsed_sec": elapsed_sec,
                "pred_time_str": pred_time_str,
                "metrics": metrics
            }
            current_matches_last = True

    if current_matches_last and "last_analysis_results" in st.session_state:
        res_data = st.session_state.last_analysis_results
        pred_time_display = res_data.get("pred_time_str", format_prediction_time(res_data.get("elapsed_sec")))
    else:
        pred_time_display = "—"

    if is_valid_input:
        wc_html = f'<span style="color: #10b981; font-weight: 800;">Word count: {word_count} / {MIN_WORDS}</span>'
        bar_border = "1px solid rgba(16, 185, 129, 0.3)"
        bar_bg = "rgba(16, 185, 129, 0.08)"
    else:
        wc_html = f'<span style="color: #f59e0b; font-weight: 800;">Word count: {word_count} / {MIN_WORDS}</span>'
        bar_border = "1px solid rgba(245, 158, 11, 0.3)"
        bar_bg = "rgba(245, 158, 11, 0.08)"

    st.markdown(f"""
    <div class="input-stats-bar" style="border: {bar_border}; background: {bar_bg}; justify-content: space-between; padding: 9px 16px;">
        <div class="stat-item" style="font-size: 0.88rem;">{wc_html}</div>
        <div class="stat-item"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-1px; margin-right:4px;"><polyline points="4 7 4 4 20 4 20 7"/><line x1="12" y1="4" x2="12" y2="20"/></svg>Chars: <span class="stat-value">{metrics['char_count']}</span></div>
        <div class="stat-item"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-1px; margin-right:4px;"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/></svg>Sentences: <span class="stat-value">{metrics['sentence_count']}</span></div>
        <div class="stat-item"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-1px; margin-right:4px;"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>Prediction: <span class="stat-value">{pred_time_display}</span></div>
    </div>
    """, unsafe_allow_html=True)


# Right Column: Detection Results & Analytics
with col_right:
    if current_matches_last and "last_analysis_results" in st.session_state:
        res = st.session_state.last_analysis_results
        ai_prob = res["ai_prob"]
        human_prob = res.get("human_prob", 1.0 - ai_prob)
        hybrid_res = res["hybrid_res"]
        model_breakdown = res["model_breakdown"]
        strategy_lbl = res["strategy_lbl"]
        elapsed_ms = res["elapsed_ms"]

        if hybrid_res and "prediction" in hybrid_res:
            is_ai = (hybrid_res["prediction"] == "AI Generated")
        else:
            is_ai = (ai_prob >= 0.50)

        banner_class = "banner-ai" if is_ai else "banner-human"
        badge_class = "res-badge-ai" if is_ai else "res-badge-human"
        badge_text = "AI-GENERATED" if is_ai else "HUMAN-WRITTEN"
        primary_pct = f"{ai_prob * 100:.1f}%" if is_ai else f"{human_prob * 100:.1f}%"
        primary_label = "AI Probability" if is_ai else "Human Confidence"

        gauge_fill_class = "gauge-fill-ai" if is_ai else "gauge-fill-human"
        gauge_width_pct = min(max(ai_prob * 100, 0), 100)

        st.markdown(f"""
        <div class="{banner_class}">
            <div class="res-hdr-flex">
                <div>
                    <div class="{badge_class}">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>{badge_text}
                    </div>
                    <div class="res-badge-sub">{strategy_lbl}</div>
                </div>
                <div style="text-align:right;">
                    <div class="res-score-huge">{primary_pct}</div>
                    <div class="res-score-sub">{primary_label}</div>
                </div>
            </div>
            <div class="gauge-wrapper">
                <div class="gauge-labels">
                    <span>AI (0%)</span>
                    <span>50%</span>
                    <span>Human (100%)</span>
                </div>
                <div class="gauge-track">
                    <div class="{gauge_fill_class}" style="width:{gauge_width_pct:.1f}%;"></div>
                </div>
                <div class="gauge-sub-info">
                    <span>AI Prob: {ai_prob * 100:.1f}%</span>
                    <span>Human Prob: {human_prob * 100:.1f}%</span>
                    <span>Latency: {elapsed_ms:.1f} ms</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        sub_col1, sub_col2 = st.columns([1.1, 0.9])

        with sub_col1:
            st.markdown("""
            <div class="card-header-box">
                <div class="card-header-title">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#cbd5e1" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px; vertical-align:-1px;"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>MODEL BREAKDOWN
                </div>
            </div>
            """, unsafe_allow_html=True)

            if model_breakdown:
                for m_name, p_val in model_breakdown.items():
                    p_ai_pct = p_val * 100
                    p_hu_pct = (1.0 - p_val) * 100
                    if p_ai_pct < 30:
                        bar_cls = "mini-fill-safe"
                    elif p_ai_pct < 65:
                        bar_cls = "mini-fill-warn"
                    else:
                        bar_cls = "mini-fill-danger"
                    st.markdown(f"""
                    <div class="model-row-card">
                        <div class="model-row-header">
                            <span class="model-name">{m_name}</span>
                            <span class="model-probs">
                                <span class="prob-ai-tag">AI {p_ai_pct:.1f}%</span> | <span class="prob-hu-tag">Hu {p_hu_pct:.1f}%</span>
                            </span>
                        </div>
                        <div class="mini-track">
                            <div class="{bar_cls}" style="width:{p_ai_pct:.1f}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with sub_col2:
            st.markdown("""
            <div class="card-header-box">
                <div class="card-header-title">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#cbd5e1" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px; vertical-align:-1px;"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>MODEL CONSENSUS
                </div>
            </div>
            """, unsafe_allow_html=True)

            is_hybrid = ("Hybrid" in selected_engine or "Stacking" in selected_engine or "Weighted" in selected_engine)

            if is_hybrid and hybrid_res and "consensus" in hybrid_res:
                cons_data = hybrid_res["consensus"]
                cons_state = cons_data["state"]
                votes = cons_data["votes"]

                if "UNANIMOUS" in cons_state:
                    st.markdown(f"""<div class="consensus-badge-unanimous"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2.5" style="vertical-align:-2px; margin-right:5px;"><polyline points="20 6 9 17 4 12"/></svg> {cons_state}</div>""", unsafe_allow_html=True)
                elif "TIED" in cons_state:
                    st.markdown(f"""<div class="consensus-badge-split" style="background:rgba(245,158,11,0.1); border-color:rgba(245,158,11,0.3); color:#fbbf24;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" stroke-width="2.5" style="vertical-align:-2px; margin-right:5px;"><circle cx="12" cy="12" r="10"/><line x1="8" y1="12" x2="16" y2="12"/></svg> {cons_state}</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="consensus-badge-split"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" stroke-width="2.5" style="vertical-align:-2px; margin-right:5px;"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg> {cons_state}</div>""", unsafe_allow_html=True)

                display_names = {
                    "roberta": "RoBERTa",
                    "deberta": "DeBERTa",
                    "distilbert": "DistilBERT",
                    "xgboost": "XGBoost"
                }

                st.markdown('<div style="margin-top: 6px;">', unsafe_allow_html=True)
                for m_key, m_disp in display_names.items():
                    v = votes.get(m_key, "Human")
                    v_class = "vote-result-ai" if v == "AI" else "vote-result-hu"
                    st.markdown(f"""
                    <div class="vote-item">
                        <span class="vote-model-lbl">{m_disp}</span>
                        <span class="{v_class}">{"AI" if v == "AI" else "Human"}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                clean_name = selected_engine.split("(")[0].strip()
                v_single = "AI" if is_ai else "Human"
                v_class_single = "vote-result-ai" if is_ai else "vote-result-hu"
                st.markdown("""<div class="consensus-badge-unanimous" style="background:rgba(99,102,241,0.1); border-color:rgba(99,102,241,0.3); color:#818cf8;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2.5" style="vertical-align:-2px; margin-right:5px;"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg> SINGLE MODEL RESULT (1/1 MODEL)</div>""", unsafe_allow_html=True)
                st.markdown('<div style="margin-top: 6px;">', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="vote-item">
                    <span class="vote-model-lbl">{clean_name}</span>
                    <span class="{v_class_single}">{v_single}</span>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Stylometric Features Panel
        st.markdown("""
        <div class="card-header-box">
            <div class="card-header-title">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#cbd5e1" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="margin-right:8px; vertical-align:-1px;"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>STYLOMETRIC & LINGUISTIC FEATURES
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-grid-6">
            <div class="feat-card">
                <div class="feat-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg></div>
                <div class="feat-val">{metrics['ttr']:.2f}</div>
                <div class="feat-lbl">Lexical Diversity</div>
            </div>
            <div class="feat-card">
                <div class="feat-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 6H3"/><path d="M15 12H3"/><path d="M17 18H3"/></svg></div>
                <div class="feat-val">{metrics['avg_sentence_len']:.1f}</div>
                <div class="feat-lbl">Avg Sentence Len</div>
            </div>
            <div class="feat-card">
                <div class="feat-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 7 4 4 20 4 20 7"/><line x1="9" y1="20" x2="15" y2="20"/><line x1="12" y1="4" x2="12" y2="20"/></svg></div>
                <div class="feat-val">{metrics['avg_word_len']:.1f}</div>
                <div class="feat-lbl">Avg Word Len</div>
            </div>
            <div class="feat-card">
                <div class="feat-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg></div>
                <div class="feat-val">{word_count}</div>
                <div class="feat-lbl">Word Count</div>
            </div>
            <div class="feat-card">
                <div class="feat-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg></div>
                <div class="feat-val">{metrics['sentence_count']}</div>
                <div class="feat-lbl">Sentences</div>
            </div>
            <div class="feat-card">
                <div class="feat-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg></div>
                <div class="feat-val">{pred_time_display}</div>
                <div class="feat-lbl">Prediction Time</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if not is_valid_input:
            st.markdown(f"""
            <div class="card-panel" style="text-align: center; padding: 2.5rem 1.5rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.8rem; color: #f59e0b; display: flex; justify-content: center;">
                    <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                </div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #f59e0b; margin-bottom: 0.4rem;">Minimum {MIN_WORDS} Words Required</div>
                <div style="font-size: 0.85rem; color: #94a3b8; max-width: 380px; margin: 0 auto;">Current input contains <b>{word_count}</b> / {MIN_WORDS} words. Please enter at least {MIN_WORDS} words to enable AI detection.</div>
            </div>
            """, unsafe_allow_html=True)


# ── EXPLAINABLE AI (XAI) DASHBOARD SECTION ─────────────────────────────────────
if current_matches_last and "last_analysis_results" in st.session_state:
    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 2rem 0 1.5rem;'>", unsafe_allow_html=True)
    
    # Check if cached XAI output exists for this input text
    has_cached_xai = (
        "xai_result_cache" in st.session_state and
        st.session_state.get("xai_cache_text") == input_text
    )

    # 1. Automatic Full XAI Generation if not already cached
    if not has_cached_xai:
        if xai_engine is None:
            st.error("Unified Explainability Engine is offline or could not be loaded.")
        else:
            with st.spinner("Generating XAI explanation...\nSHAP, LIME and Transformer Attention are being analyzed.\nThis may take approximately 1 minute."):
                t_xai_start = time.time()
                xai_out_dir = BASE_DIR / "reports" / "explainability" / "unified" / "interactive_session"
                xai_out_dir.mkdir(parents=True, exist_ok=True)

                cached_pred = st.session_state.get("last_analysis_results", {}).get("hybrid_res")
                if xai_engine is not None and hybrid_pipeline is not None:
                    if hasattr(xai_engine, "set_pipeline"):
                        xai_engine.set_pipeline(hybrid_pipeline)
                    else:
                        xai_engine.pipeline = hybrid_pipeline

                try:
                    xai_res = xai_engine.explain(
                        text_or_file=input_text,
                        save_dir=xai_out_dir,
                        sample_id="interactive",
                        num_lime_samples=30,
                        num_lime_features=5,
                        prediction_result=cached_pred
                    )
                except Exception as xai_err:
                    if "Feature shape mismatch" in str(xai_err) or "vectorizer" in str(xai_err).lower() or "28 vs. 30001" in str(xai_err):
                        st.cache_resource.clear()
                        hybrid_pipeline = get_hybrid_pipeline()
                        xai_engine = get_xai_engine()
                        if xai_engine is not None and hybrid_pipeline is not None:
                            if hasattr(xai_engine, "set_pipeline"):
                                xai_engine.set_pipeline(hybrid_pipeline)
                            else:
                                xai_engine.pipeline = hybrid_pipeline
                        xai_res = xai_engine.explain(
                            text_or_file=input_text,
                            save_dir=xai_out_dir,
                            sample_id="interactive",
                            num_lime_samples=30,
                            num_lime_features=5,
                            prediction_result=cached_pred
                        )
                    else:
                        raise xai_err

                st.session_state["xai_result_cache"] = xai_res
                st.session_state["xai_cache_text"] = input_text
                has_cached_xai = True

    if has_cached_xai and "xai_result_cache" in st.session_state:
        xai_res = st.session_state["xai_result_cache"]

        # Header Block
        st.markdown("""
        <div style="margin-top: 1.5rem; margin-bottom: 0.8rem; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 0.8rem;">
            <div style="color: #a78bfa; font-size: 1.15rem; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase;">
                EXPLAINABLE AI (XAI) INTERPRETABILITY DASHBOARD
            </div>
            <div style="color: #94a3b8; font-size: 0.88rem; margin-top: 4px;">
                Multi-model evidence explaining the Hybrid Stacking Ensemble prediction.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Action Button Directly Underneath Heading
        xai_btn_col1, xai_btn_col2, xai_btn_col3 = st.columns([1, 2, 1])
        with xai_btn_col2:
            generate_xai_clicked = st.button(
                "Generate Full XAI Explanation",
                type="primary",
                key="generate_xai_btn",
                use_container_width=True
            )

        if generate_xai_clicked:
            if xai_engine is None:
                st.error("Unified Explainability Engine is offline or could not be loaded.")
            else:
                with st.spinner("Generating XAI explanation...\nSHAP, LIME and Transformer Attention are being analyzed.\nThis may take approximately 1 minute."):
                    t_xai_start = time.time()
                    xai_out_dir = BASE_DIR / "reports" / "explainability" / "unified" / "interactive_session"
                    xai_out_dir.mkdir(parents=True, exist_ok=True)

                    cached_pred = st.session_state.get("last_analysis_results", {}).get("hybrid_res")
                    if xai_engine is not None and hybrid_pipeline is not None:
                        if hasattr(xai_engine, "set_pipeline"):
                            xai_engine.set_pipeline(hybrid_pipeline)
                        else:
                            xai_engine.pipeline = hybrid_pipeline

                    try:
                        xai_res = xai_engine.explain(
                            text_or_file=input_text,
                            save_dir=xai_out_dir,
                            sample_id="interactive",
                            num_lime_samples=30,
                            num_lime_features=5,
                            prediction_result=cached_pred
                        )
                    except Exception as xai_err:
                        if "Feature shape mismatch" in str(xai_err) or "vectorizer" in str(xai_err).lower() or "28 vs. 30001" in str(xai_err):
                            st.cache_resource.clear()
                            hybrid_pipeline = get_hybrid_pipeline()
                            xai_engine = get_xai_engine()
                            if xai_engine is not None and hybrid_pipeline is not None:
                                if hasattr(xai_engine, "set_pipeline"):
                                    xai_engine.set_pipeline(hybrid_pipeline)
                                else:
                                    xai_engine.pipeline = hybrid_pipeline
                            xai_res = xai_engine.explain(
                                text_or_file=input_text,
                                save_dir=xai_out_dir,
                                sample_id="interactive",
                                num_lime_samples=30,
                                num_lime_features=5,
                                prediction_result=cached_pred
                            )
                        else:
                            raise xai_err

                    st.session_state["xai_result_cache"] = xai_res
                    st.session_state["xai_cache_text"] = input_text
                    st.rerun()

        # ── SINGLE-PAGE CONTINUOUS XAI DASHBOARD ───────────────────────────
        exp_data = generate_dynamic_decision_explanation(xai_res)

        cls_val = exp_data["class_val"]
        conf_val = exp_data["confidence_val"]
        p_ai_pct = exp_data["p_ai"] * 100.0
        p_hu_pct = exp_data["p_human"] * 100.0
        cls_color = "#f43f5e" if cls_val == "AI Generated" else "#10b981"

        mb = xai_res.get("prediction", {}).get("model_breakdown", {})
        r_p = mb.get('roberta', 0.0)
        d_p = mb.get('deberta', 0.0)
        b_p = mb.get('distilbert', 0.0)
        x_p = mb.get('xgboost', 0.0)
        agree_count = exp_data["agree_count"]

        # 1. FINAL PREDICTION CARD (Full Container Width)
        st.markdown(f"""
        <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 20px 24px; margin-bottom: 20px;">
            <div style="font-size: 0.78rem; color: #94a3b8; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;">FINAL PREDICTION</div>
            <div style="font-size: 1.9rem; font-weight: 800; color: {cls_color}; margin: 6px 0 14px 0;">
                {cls_val}
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; border-top: 1px solid #1e293b; padding-top: 14px;">
                <div>
                    <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase;">Confidence</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #f1f5f9;">{conf_val:.1f}%</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase;">Decision Threshold</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #f1f5f9;">0.50</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase;">AI Probability</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #f43f5e;">{p_ai_pct:.1f}%</div>
                </div>
                <div>
                    <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase;">Human Probability</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #10b981;">{p_hu_pct:.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. DYNAMIC DECISION EXPLANATION
        st.markdown(f"#### WHY THIS DOCUMENT WAS CLASSIFIED AS {cls_val.upper()}")
        st.markdown(exp_data['dynamic_narrative'])

        st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 2rem 0 1.5rem;'>", unsafe_allow_html=True)

        # 3. INTERACTIVE SHAP FEATURE ATTRIBUTION ARTIFACTS
        st.markdown("### SHAP FEATURE ATTRIBUTION EVIDENCE")
        st.caption("Top feature attributions explaining model logit weights. Positive SHAP pushes toward AI Generated, negative pushes toward Human Written.")
        
        if xai_res.get("shap", {}).get("status") == "PASS":
            shap_data = xai_res["shap"]
            meta_s = shap_data.get("meta_classifier", {})
            xgb_s = shap_data.get("xgboost", {})

            col_s1, col_s2 = st.columns(2, gap="medium")
            with col_s1:
                st.markdown("##### META-CLASSIFIER SHAP DRIVERS")
                st.caption(f"Log-odds logit space | Base logit: `{meta_s.get('base_value', 0.0):.2f}`")
                top_meta = sorted(meta_s.get("attributions", []), key=lambda x: x.get("importance", 0.0), reverse=True)[:5]
                if top_meta:
                    df_meta = pd.DataFrame([
                        {
                            "Feature": f.get("feature", "N/A"),
                            "Category": f.get("category", "N/A"),
                            "Value": f"{f['value']:.3f}" if isinstance(f.get("value"), float) else str(f.get("value", "")),
                            "SHAP Weight": f"{float(f.get('shap_value', 0.0)):+.4f}",
                            "Direction": str(f.get("direction", "")).replace("Pushes ", "")
                        }
                        for f in top_meta
                    ])
                    st.dataframe(df_meta, use_container_width=True, hide_index=True)

                    top_meta_name = top_meta[0].get("feature", "features")
                    top_meta_dir = str(top_meta[0].get("direction", "")).replace("Pushes ", "")
                    st.markdown(f"<div style='font-size: 0.82rem; color: #94a3b8; margin-top: 6px;'><i>Interpretation: Strongest meta feature (`{top_meta_name}`) pushes toward <b>{top_meta_dir}</b>.</i></div>", unsafe_allow_html=True)

                for p in xai_res.get("visualizations", {}).get("shap", []):
                    if "meta_classifier" in str(p) and os.path.exists(p):
                        st.markdown("<div style='margin-top: 12px;'>", unsafe_allow_html=True)
                        st.image(str(p), caption="Top Meta-Classifier SHAP Feature Contributions", use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)

            with col_s2:
                st.markdown("##### STYLOMETRIC (XGBOOST) TREESHAP DRIVERS")
                st.caption(f"Log-odds margin space | Base margin: `{xgb_s.get('base_value', 0.0):.2f}`")
                top_xgb = sorted(xgb_s.get("attributions", []), key=lambda x: x.get("importance", 0.0), reverse=True)[:5]
                if top_xgb:
                    df_xgb = pd.DataFrame([
                        {
                            "Feature": f.get("feature", "N/A"),
                            "Category": f.get("category", "N/A"),
                            "Value": f"{f['value']:.3f}" if isinstance(f.get("value"), float) else str(f.get("value", "")),
                            "SHAP Weight": f"{float(f.get('shap_value', 0.0)):+.4f}",
                            "Direction": str(f.get("direction", "")).replace("Pushes ", "")
                        }
                        for f in top_xgb
                    ])
                    st.dataframe(df_xgb, use_container_width=True, hide_index=True)

                    top_xgb_name = top_xgb[0].get("feature", "features")
                    top_xgb_dir = str(top_xgb[0].get("direction", "")).replace("Pushes ", "")
                    st.markdown(f"<div style='font-size: 0.82rem; color: #94a3b8; margin-top: 6px;'><i>Interpretation: Strongest stylometric feature (`{top_xgb_name}`) favors <b>{top_xgb_dir}</b>.</i></div>", unsafe_allow_html=True)

                for p in xai_res.get("visualizations", {}).get("shap", []):
                    if "xgboost" in str(p) and os.path.exists(p):
                        st.markdown("<div style='margin-top: 12px;'>", unsafe_allow_html=True)
                        st.image(str(p), caption="Top XGBoost TreeSHAP Stylometric Contributions", use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning(f"SHAP explanation unavailable: {xai_res.get('shap', {}).get('error', 'Disabled')}")

        st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 2rem 0 1.5rem;'>", unsafe_allow_html=True)

        # 4. INTERACTIVE LIME WORD ATTRIBUTION ARTIFACTS
        st.markdown("### LIME LOCAL WORD-LEVEL EVIDENCE")
        if xai_res.get("lime", {}).get("status") == "PASS":
            lime_data = xai_res["lime"]
            st.caption(f"Local perturbation text attribution | Active explanation window: 500 words | Truncated: `{lime_data.get('is_truncated', False)}`")

            col_l1, col_l2 = st.columns(2, gap="medium")
            pos_words = lime_data.get("positive_words", [])[:5]
            neg_words = lime_data.get("negative_words", [])[:5]

            with col_l1:
                st.markdown("##### TOP AI-SUPPORTING WORDS")
                if pos_words:
                    df_pos = pd.DataFrame([{"Word": w.get("word", ""), "LIME Weight": f"{float(w.get('weight', 0.0)):+.4f}", "Direction": w.get("direction", "")} for w in pos_words])
                    st.dataframe(df_pos, use_container_width=True, hide_index=True)
                else:
                    st.info("No positive AI-directed LIME words identified.")

            with col_l2:
                st.markdown("##### TOP HUMAN-SUPPORTING WORDS")
                if neg_words:
                    df_neg = pd.DataFrame([{"Word": w.get("word", ""), "LIME Weight": f"{float(w.get('weight', 0.0)):+.4f}", "Direction": w.get("direction", "")} for w in neg_words])
                    st.dataframe(df_neg, use_container_width=True, hide_index=True)
                else:
                    st.info("No negative Human-directed LIME words identified.")

            st.markdown("##### LIME WORD ATTRIBUTION VISUALIZATION")
            for p in xai_res.get("visualizations", {}).get("lime", []):
                if p.endswith(".png") and os.path.exists(p) and "highlight" not in str(p):
                    st.image(str(p), caption="LIME Top Word Attribution Chart", use_container_width=True)

            st.markdown("##### HIGHLIGHTED TEXT")
            st.caption("Words highlighted in red indicate positive LIME weights (pushing AI). Words in blue indicate negative weights (pushing Human). Active explanation window: 500 words.")
            html_highlight = lime_data.get("html_highlight", "")
            if html_highlight:
                st.markdown(f"""
                <div style="background: #0b1120; border: 1px solid #1e293b; border-radius: 8px; padding: 14px 18px; margin: 0.8rem 0; font-family: monospace; font-size: 0.88rem; line-height: 1.7; max-height: 280px; overflow-y: auto;">
                    {html_highlight}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning(f"LIME explanation unavailable: {xai_res.get('lime', {}).get('error', 'Disabled')}")

        st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 2rem 0 1.5rem;'>", unsafe_allow_html=True)

        # 5. TRANSFORMER ATTENTION EVIDENCE
        st.markdown("### TRANSFORMER ATTENTION EVIDENCE")
        if xai_res.get("attention", {}).get("status") == "PASS":
            attn_data = xai_res["attention"]
            st.caption("Active attention window: 512 tokens | Head-averaging aggregation | Punctuation and special tokens filtered")

            attn_model_tab1, attn_model_tab2, attn_model_tab3, attn_model_tab4 = st.tabs([
                "RoBERTa V2.1",
                "DeBERTa-v3 V2.1",
                "DistilBERT V2.1",
                "Cross-Model Comparison"
            ])

            with attn_model_tab1:
                st.markdown("##### RoBERTa V2.1 (Top Meaningful Tokens)")
                r_toks = attn_data.get("roberta", {}).get("top_attended_tokens", [])[:8]
                if r_toks:
                    st.dataframe(pd.DataFrame([{"Token": t.get("clean_token", ""), "Attention Score": t.get("importance_score", 0.0)} for t in r_toks]), use_container_width=True, hide_index=True)

                for p in xai_res.get("visualizations", {}).get("attention", []):
                    if "roberta" in str(p) and os.path.exists(p):
                        cap = "RoBERTa V2.1 token-to-token attention heatmap (10x10 matrix)" if "heatmap" in str(p) else "RoBERTa V2.1 top meaningful token attention"
                        st.image(str(p), caption=cap, use_container_width=True)

            with attn_model_tab2:
                st.markdown("##### DeBERTa-v3 V2.1 (Top Meaningful Tokens)")
                d_toks = attn_data.get("deberta", {}).get("top_attended_tokens", [])[:8]
                if d_toks:
                    st.dataframe(pd.DataFrame([{"Token": t.get("clean_token", ""), "Attention Score": t.get("importance_score", 0.0)} for t in d_toks]), use_container_width=True, hide_index=True)

                for p in xai_res.get("visualizations", {}).get("attention", []):
                    if "deberta" in str(p) and os.path.exists(p):
                        cap = "DeBERTa-v3 V2.1 token-to-token attention heatmap (10x10 matrix)" if "heatmap" in str(p) else "DeBERTa-v3 V2.1 top meaningful token attention"
                        st.image(str(p), caption=cap, use_container_width=True)

            with attn_model_tab3:
                st.markdown("##### DistilBERT V2.1 (Top Meaningful Tokens)")
                b_toks = attn_data.get("distilbert", {}).get("top_attended_tokens", [])[:8]
                if b_toks:
                    st.dataframe(pd.DataFrame([{"Token": t.get("clean_token", ""), "Attention Score": t.get("importance_score", 0.0)} for t in b_toks]), use_container_width=True, hide_index=True)

                for p in xai_res.get("visualizations", {}).get("attention", []):
                    if "distilbert" in str(p) and os.path.exists(p):
                        cap = "DistilBERT V2.1 token-to-token attention heatmap (10x10 matrix)" if "heatmap" in str(p) else "DistilBERT V2.1 aggregated token attention"
                        st.image(str(p), caption=cap, use_container_width=True)

            with attn_model_tab4:
                st.markdown("##### CROSS-MODEL ATTENTION PATTERN COMPARISON")
                st.markdown("<div style='font-size: 0.88rem; color: #cbd5e1; margin-bottom: 10px;'>Across the three transformer models, self-attention is concentrated on content-bearing lexical tokens rather than punctuation. The models show complementary focal patterns within the active 512-token window.</div>", unsafe_allow_html=True)
                for p in xai_res.get("visualizations", {}).get("attention", []):
                    if "cross_model" in str(p) and os.path.exists(p):
                        st.image(str(p), caption="Cross-Model Attention Allocation Comparison", use_container_width=True)

            st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 1rem 0 0.8rem;'>", unsafe_allow_html=True)
            st.markdown(
                "<div style='font-size: 0.8rem; color: #64748b; font-style: italic;'>"
                "Note: Attention weights indicate internal token-to-token routing and structural focus under the selected aggregation strategy, "
                "and should be interpreted as structural interpretability signals rather than causal proof of the classification."
                "</div>",
                unsafe_allow_html=True
            )
        else:
            st.warning(f"Attention visualization unavailable: {xai_res.get('attention', {}).get('error', 'Disabled')}")

        st.markdown("<hr style='border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 2rem 0 1.5rem;'>", unsafe_allow_html=True)

        # 6. RESEARCH-GRADE METHODOLOGICAL DISCLAIMERS & RUNTIME
        st.markdown("""
        <div style="background: #0b1120; border: 1px solid #1e293b; border-radius: 6px; padding: 14px 18px; font-size: 0.78rem; color: #94a3b8; line-height: 1.6;">
            <strong>RESEARCH METHODOLOGY & INTERPRETABILITY DISCLAIMERS:</strong><br>
            • <strong>Confidence vs Explanation:</strong> Model confidence represents logit calibration strength in output probability space, not mathematical reliability of individual attributions.<br>
            • <strong>SHAP Attributions:</strong> Feature attributions represent marginal log-odds contributions under 5-fold Meta-Classifier cross-validation and TreeSHAP margin space.<br>
            • <strong>LIME Perturbations:</strong> Local word-level attributions are computed via local linear perturbation modeling over an active 500-word window.<br>
            • <strong>Transformer Attention:</strong> Attention maps describe self-attention routing matrices under final-layer head-averaging aggregation and do not serve as causal proof of classification.
        </div>
        """, unsafe_allow_html=True)

        rt = xai_res.get("runtime", {})
        total_sec = float(rt.get('total_ms', 0.0)) / 1000.0
        st.markdown(
            f"<div style='font-size: 0.80rem; color: #64748b; font-style: italic; text-align: right; margin-top: 10px; margin-bottom: 20px;'>"
            f"Explanation generated in <b>{total_sec:.1f} seconds</b>."
            f"</div>",
            unsafe_allow_html=True
        )


# Footer
st.markdown("""
<div class="app-footer">
    <strong>Hybrid AI Text Detector & XAI Dashboard</strong> &bull; Multi-Model Ensemble Detection & Explainable AI Framework<br>
    RoBERTa &bull; DeBERTa &bull; DistilBERT &bull; XGBoost &bull; Stacking Meta-Classifier &bull; SHAP &bull; LIME &bull; Attention
</div>
""", unsafe_allow_html=True)
