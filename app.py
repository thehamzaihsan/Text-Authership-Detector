import streamlit as st
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

st.set_page_config(page_title="WhatsApp Author Fingerprinting", layout="wide")

MODEL_DIR = "models"

MODEL_OPTIONS = {
    "Word BOW + NB (8 authors)": f"{MODEL_DIR}/cell7_bow_nb_8authors.pkl",
    "Word TF-IDF + NB (8 authors)": f"{MODEL_DIR}/cell8_tfidf_nb_8authors.pkl",
    "Char N-Grams + TF-IDF + NB (8 authors)": f"{MODEL_DIR}/cell9_char_tfidf_nb_8authors.pkl",
    "Char N-Grams + TF-IDF + LogReg (8 authors)": f"{MODEL_DIR}/cell10_char_tfidf_lr_8authors.pkl",
    "BOW + Handcrafted + NB (8 authors)": f"{MODEL_DIR}/cell11_bow_hand_nb_8authors.pkl",
    "Ensemble (8 authors)": f"{MODEL_DIR}/ensemble_8authors.pkl",
    "Top 4: BOW + NB": f"{MODEL_DIR}/cell7_bow_nb_4authors.pkl",
    "Top 4: BOW + Handcrafted + NB": f"{MODEL_DIR}/cell11_bow_hand_nb_4authors.pkl",
    "Top 4: Ensemble": f"{MODEL_DIR}/ensemble_4authors.pkl",
}


@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_handcrafted(text):
    return {"char_count": len(text), "word_count": len(text.split())}


def predict_single(model_data, text):
    is_ensemble = "models" in model_data
    author_list = model_data["author_list"]
    accuracy = model_data["accuracy"]

    if is_ensemble:
        models = model_data["models"]
        vectorizers = model_data["vectorizers"]
        scalers = model_data.get("scalers", [None] * len(models))
        all_probas = []
        classes = models[0].classes_
        for i, clf in enumerate(models):
            vec = vectorizers[i]
            X = vec.transform([text])
            scaler = scalers[i]
            if scaler is not None:
                hand_df = pd.DataFrame([extract_handcrafted(text)])
                X = hstack([X, csr_matrix(scaler.transform(hand_df))])
            all_probas.append(clf.predict_proba(X)[0])
        avg_proba = np.mean(all_probas, axis=0)
        pred_idx = np.argmax(avg_proba)
        predicted = classes[pred_idx]
        confidence = avg_proba[pred_idx]
        all_probs = {c: p for c, p in zip(classes, avg_proba)}
    else:
        model = model_data["model"]
        vectorizer = model_data["vectorizer"]
        scaler = model_data.get("scaler")
        X = vectorizer.transform([text])
        if scaler is not None:
            hand_df = pd.DataFrame([extract_handcrafted(text)])
            X = hstack([X, csr_matrix(scaler.transform(hand_df))])
        proba = model.predict_proba(X)[0]
        pred_idx = np.argmax(proba)
        predicted = model.classes_[pred_idx]
        confidence = proba[pred_idx]
        all_probs = {c: p for c, p in zip(model.classes_, proba)}

    return predicted, confidence, all_probs, accuracy


st.title("WhatsApp Author Fingerprinting")
st.markdown("Identify the author of a Roman Urdu / English WhatsApp message using trained ML models.")

col1, col2 = st.columns([1, 1])

with col1:
    model_name = st.selectbox("Select Model", list(MODEL_OPTIONS.keys()))
    model_path = MODEL_OPTIONS[model_name]
    text_input = st.text_area("Type a message:", height=150,
                              placeholder="e.g. kal class mein nhi aaraha tha kya hua tha?")
    predict_btn = st.button("Identify Author", type="primary")

with col2:
    if predict_btn:
        raw = text_input.strip()
        if not raw:
            st.warning("Please type a message.")
        else:
            with st.spinner("Predicting..."):
                try:
                    model_data = load_model(model_path)
                    text_clean = raw.lower()
                    word_count = len(text_clean.split())
                    if len(text_clean) > 500:
                        text_clean = text_clean[:500]
                        st.info("Input truncated to 500 characters.")
                    if word_count < 4:
                        st.warning("Message too short for reliable prediction.")

                    predicted, confidence, all_probs, accuracy = predict_single(model_data, text_clean)

                    st.markdown("### Predicted Author")
                    st.markdown(f"<h1 style='color:#4CAF50;'>{predicted}</h1>", unsafe_allow_html=True)
                    st.metric("Confidence", f"{confidence:.1%}")
                    st.info(f"Model: {model_name} (Accuracy: {accuracy:.1%})")

                    if confidence < 0.50:
                        st.warning("Low confidence — may not be any known author.")

                    st.markdown("### All Author Probabilities")
                    sorted_probs = sorted(all_probs.items(), key=lambda x: -x[1])
                    prob_df = pd.DataFrame({
                        "Author": [a[:30] for a, _ in sorted_probs],
                        "Probability": [p for _, p in sorted_probs]
                    })
                    colors = ["#4CAF50" if a == predicted else "#E0E0E0" for a, _ in sorted_probs]
                    chart = prob_df.copy()
                    chart["Color"] = colors
                    st.bar_chart(chart.set_index("Author")["Probability"])

                    with st.expander("Show exact percentages"):
                        for author, prob in sorted_probs:
                            marker = " ←" if author == predicted else ""
                            pct = f"{prob:.1%}{marker}"
                            st.write(f"{author}: {pct}")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Models trained on WhatsApp chat data with Roman Urdu / English code-mixed messages.")
