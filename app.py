"""
Web App: AI-Generated Text Detector
-------------------------------------
A simple Streamlit front end for the project. The user pastes text; the app
extracts the same 66 stylometric features used in training (feature_engineering.py),
runs the saved model (model.joblib from train_final_model.py) and shows:
  - the verdict: Human-written / AI-generated / Uncertain
  - the model's AI-probability
  - a table comparing this text's style with the average human and AI text

Run:
    streamlit run app.py
(needs feature_engineering.py and model.joblib in the same folder)
"""

import os

import joblib
import pandas as pd
import streamlit as st

import feature_engineering as fe

MODEL_PATH = "model.joblib"

# Features shown in the explanation table: (column name, friendly name)
KEY_FEATURES = [
    ("avg_sentence_length", "Average sentence length"),
    ("sentence_length_variance", "Sentence-length variance (burstiness)"),
    ("punctuation_frequency", "Punctuation per 100 characters"),
    ("punct_comma", "Commas per 100 characters"),
    ("punct_apostrophe", "Apostrophes per 100 characters"),
    ("contraction_ratio", "Contractions (don't, it's) per word"),
    ("first_person_ratio", "First-person words (I, my, we) per word"),
    ("uppercase_ratio", "Share of capital letters"),
    ("newline_count", "Line breaks"),
    ("fw_the", "Use of the word 'the'"),
]

st.set_page_config(page_title="AI Text Detector", page_icon="🔍", layout="centered")


@st.cache_resource
def load_bundle(path: str):
    return joblib.load(path)


def analyse(text: str, bundle: dict):
    """Text -> features -> scaled -> model probability of being AI-generated."""
    feats = fe.extract_features(text)
    row = pd.DataFrame([feats])[bundle["feature_names"]]
    scaled = bundle["scaler"].transform(row)
    p_ai = float(bundle["model"].predict_proba(scaled)[0, 1])
    return feats, p_ai


def verdict_for(p_ai: float, low: float, high: float) -> str:
    if low <= p_ai <= high:
        return "Uncertain"
    return "AI-generated" if p_ai >= 0.5 else "Human-written"


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------
st.title("🔍 AI-Generated Text Detector")
st.caption("Detects ChatGPT-style writing from *how* a text is written (stylometric features), not what it says.")

if not os.path.exists(MODEL_PATH):
    st.error(
        "model.joblib not found. Create it first:\n\n"
        "`python train_final_model.py --input features.csv --dataset dataset.csv`"
    )
    st.stop()

bundle = load_bundle(MODEL_PATH)
low, high = bundle["uncertain_low"], bundle["uncertain_high"]

# Optional examples taken from the held-out test set
examples = bundle.get("examples", [])
choice = "Paste my own text"
if examples:
    options = ["Paste my own text"] + [f"{e['label_name']} example {i + 1} (held-out test data)" for i, e in enumerate(examples)]
    choice = st.selectbox("Try an example the model has never seen, or paste your own:", options)
default_text = ""
if choice != "Paste my own text":
    default_text = examples[options.index(choice) - 1]["text"]

text = st.text_area("Text to analyse", value=default_text, height=220, key=f"text_{choice}",
                    placeholder="Paste a paragraph or more (at least 50 words works best)...")

if st.button("Analyse", type="primary"):
    n_words = len(text.split())
    if n_words == 0:
        st.warning("Please paste some text first.")
        st.stop()
    if n_words < 50:
        st.warning(f"This text has only {n_words} words. The model was trained on texts of 50+ words, "
                   "so the result may be unreliable.")

    feats, p_ai = analyse(text, bundle)
    verdict = verdict_for(p_ai, low, high)

    st.subheader("Result")
    col1, col2 = st.columns(2)
    col1.metric("Verdict", verdict)
    col2.metric("Probability it is AI-generated", f"{p_ai:.0%}")
    st.progress(min(max(p_ai, 0.0), 1.0))

    if verdict == "Uncertain":
        st.info(f"The AI-probability is between {low:.0%} and {high:.0%}, so the model declines to decide "
                "instead of guessing. This is the project's 'Flag Uncertain' step.")

    st.subheader("Why? This text's style vs. the average human and AI text")
    rows = []
    for col, name in KEY_FEATURES:
        if col not in feats:
            continue
        mine = feats[col]
        h = bundle["class_means"]["human"][col]
        a = bundle["class_means"]["ai"][col]
        closer = "Human" if abs(mine - h) < abs(mine - a) else "AI"
        rows.append({"Style feature": name, "This text": round(mine, 3),
                     "Avg human": round(h, 3), "Avg AI": round(a, 3), "Closer to": closer})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("The model uses all 66 features together; this table shows just ten easy-to-read ones.")

with st.expander("About this model"):
    st.write(
        f"**Model:** {bundle['model_name']} trained on {bundle['n_train']} texts from the HC3 dataset "
        f"(human answers vs ChatGPT answers). Accuracy on {bundle['n_test']} held-out texts: "
        f"**{bundle['test_accuracy']:.1%}** (ROC-AUC {bundle['test_auc']:.3f})."
    )
    if os.path.exists("model_comparison_results.csv"):
        st.write("Comparison of all six models we tested:")
        cmp_df = pd.read_csv("model_comparison_results.csv")[["model", "accuracy", "f1", "roc_auc"]]
        st.dataframe(cmp_df.round(3), hide_index=True)
    st.write(
        "**Limits:** trained on one dataset and one AI model (ChatGPT); not tested on paraphrased or edited "
        "AI text or other writing genres. This is a statistical estimate, not proof of authorship, and "
        "should never be the only basis for accusing someone."
    )