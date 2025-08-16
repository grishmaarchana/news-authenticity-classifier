import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import streamlit.components.v1 as components

from sentence_transformers import SentenceTransformer
from lime.lime_text import LimeTextExplainer

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="News Authenticity — XGBoost + SBERT + LIME", layout="wide")
CLASS_NAMES = ["Fake", "Real"]

# Environment-configurable paths/names
MODEL_PATH = os.environ.get("MODEL_PATH", "xgb_model.pkl")
SENTENCE_MODEL_NAME = os.environ.get("SENTENCE_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

st.title("📰 News Authenticity Classifier")
st.caption("XGBoost on SBERT embeddings with LIME explanations (Streamlit)")

with st.sidebar:
    st.header("Settings")
    num_features = st.slider("LIME features", min_value=3, max_value=20, value=10, step=1)
    st.markdown(f"**Model file:** `{MODEL_PATH}`")
    st.markdown(f"**Sentence model:** `{SENTENCE_MODEL_NAME}`")
    st.markdown("---")
    st.markdown("➡️ Make sure `xgb_model.pkl` is committed to the repo root.")

# -----------------------------
# Lazy-loaded resources
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Add your trained model (joblib.dump) to the repo root."
        )
    return joblib.load(model_path)

@st.cache_resource(show_spinner=True)
def load_sentence_model(name: str):
    return SentenceTransformer(name)

# Try load; show a friendly error if missing
try:
    xgb_model = load_model(MODEL_PATH)
    sentence_model = load_sentence_model(SENTENCE_MODEL_NAME)
except Exception as e:
    st.error(f"Startup error: {e}")
    st.stop()

# -----------------------------
# Inference helpers
# -----------------------------
def predict_proba(texts):
    """Return class probabilities for list[str] texts."""
    if isinstance(texts, str):
        texts = [texts]
    # SBERT embeddings
    embeddings = sentence_model.encode(texts, convert_to_numpy=True)
    # XGBoost probabilities
    probs = xgb_model.predict_proba(embeddings)
    return probs

def explain_text(text: str, num_features: int = 10) -> str:
    """Return LIME HTML explanation for a single text."""
    explainer = LimeTextExplainer(class_names=CLASS_NAMES)
    exp = explainer.explain_instance(text, predict_proba, num_features=int(num_features))
    return exp.as_html()

# -----------------------------
# UI — Single & Batch
# -----------------------------
tab1, tab2 = st.tabs(["Single text", "Batch (CSV)"])

with tab1:
    default_example = (
        "The ministry confirmed a breakthrough vaccine was approved after successful trials, "
        "with distribution to begin next month across major cities."
    )
    text = st.text_area("Enter news text", value=default_example, height=200)
    run_single = st.button("Classify + Explain", type="primary")

    if run_single:
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            probs = predict_proba(text)[0]
            pred_idx = int(np.argmax(probs))
            pred_label = CLASS_NAMES[pred_idx]

            st.subheader(f"Prediction: **{pred_label}**")
            st.metric("P(Fake)", f"{probs[0]:.3f}")
            st.metric("P(Real)", f"{probs[1]:.3f}")

            # Bar chart
            chart_df = pd.DataFrame({"class": CLASS_NAMES, "probability": probs})
            st.bar_chart(chart_df.set_index("class"))

            # LIME HTML explanation
            with st.expander("LIME Explanation", expanded=True):
                html = explain_text(text, num_features=num_features)
                components.html(html, height=600, scrolling=True)

with tab2:
    st.write("Upload a CSV with a **`text`** column to classify multiple rows.")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            texts = df["text"].fillna("").astype(str).tolist()
            probs = predict_proba(texts)
            preds = np.argmax(probs, axis=1)

            df_out = df.copy()
            df_out["pred_label"] = [CLASS_NAMES[i] for i in preds]
            df_out["prob_fake"] = probs[:, 0]
            df_out["prob_real"] = probs[:, 1]

            st.dataframe(df_out.head(50))
            st.download_button(
                "Download predictions CSV",
                data=df_out.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv",
            )

st.markdown("---")
st.markdown("Built with **Streamlit**, **Sentence-Transformers**, **XGBoost**, and **LIME**.")
