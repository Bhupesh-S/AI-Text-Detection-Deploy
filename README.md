# Hybrid Explainable AI-Generated Text Detection (V3 Production)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-bhupesh1907%2Fai--transformer--v3-blue)](https://huggingface.co/bhupesh1907/ai-transformer-v3)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Production deployment repository for the **Hybrid Explainable Framework for Detecting AI-Generated Academic Text (V3)**. This application combines deep contextual representations from fine-tuned transformers with handcrafted stylometric features via a calibrated stacking meta-classifier, providing multi-tiered explainability (SHAP, LIME, and Transformer Attention).

---

## Architecture Overview

```
                      Raw Academic Text Input
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
  RoBERTa-base            DeBERTa-v3              DistilBERT
  (Transformer)           (Transformer)          (Transformer)
       │                       │                       │
       └───────────────┬───────┴───────────────────────┘
                       │
       ┌───────────────┴───────────────────────┐
       ▼                                       ▼
  TF-IDF (30,000)                      Handcrafted Features (27)
       │                                       │
       ▼                                       ▼
  XGBoost Stylometrics                  Document Statistics
       │                                       │
       └───────────────┬───────────────────────┘
                       │
                       ▼
           Stacking Meta-Features (18-D)
     [Base Probs, Stats, Entropy, Stylometrics]
                       │
                       ▼
          Calibrated Meta-Classifier
              (5-Fold Logistic Regression)
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
   Classification           Explainability Suite
 (Human vs AI Generated)  (SHAP, LIME, Attention Maps)
```

### Models & Weights
Large model weights are hosted on Hugging Face Hub: [`bhupesh1907/ai-transformer-v3`](https://huggingface.co/bhupesh1907/ai-transformer-v3)
- **RoBERTa-base**: `roberta/`
- **DeBERTa-v3**: `deberta/`
- **DistilBERT**: `distilbert/`
- **XGBoost Stylometrics**: `xgboost/xgboost_model.joblib` & `xgboost/tfidf_vectorizer.joblib`
- **Ensemble Checkpoints**: Lightweight production checkpoints packaged under `models/ensemble/checkpoints/`

---

## Explainability (XAI) Suite

1. **Transformer Multi-Model Attention**:
   - Extracts multi-head attention matrices using PyTorch eager attention (`attn_implementation="eager"`).
   - Generates per-model and cross-model token importance heatmaps for RoBERTa, DeBERTa-v3, and DistilBERT.
2. **SHAP Feature Attribution**:
   - Meta-Classifier SHAP: Explains contribution of the 18 meta-features across ensemble folds.
   - XGBoost TreeSHAP: Explains the top stylometric features driving XGBoost prediction.
3. **LIME Local Text Perturbation**:
   - End-to-end word-level attribution evaluating token impact on ensemble AI probability.
4. **Dynamic Decision Narrative**:
   - Automated synthesis of confidence, consensus among base models, and dominant explanatory indicators.

---

## Deployment on Streamlit Community Cloud

1. Fork or push this repository to GitHub.
2. Navigate to [share.streamlit.io](https://share.streamlit.io/) and create a new app.
3. Set:
   - **Repository**: `Bhupesh-S/AI-Text-Detection-Deploy`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Add your Hugging Face Access Token in **App Settings -> Secrets**:
   ```toml
   HF_TOKEN = "hf_your_huggingface_token_here"
   ```
5. Deploy!

---

## Local Development & Running

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/Bhupesh-S/AI-Text-Detection-Deploy.git
cd AI-Text-Detection-Deploy

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Hugging Face Token
Create `.streamlit/secrets.toml`:
```toml
HF_TOKEN = "hf_your_token_here"
```
Or set the environment variable:
```bash
export HF_TOKEN="hf_your_token_here"
```

### 3. Launch Application
```bash
streamlit run app.py
```

---

## License & Attribution

Developed by **Bharanidharan K** & **Antigravity**.  
Project: *Hybrid Explainable Framework for Detecting AI-Generated Academic Text*.
