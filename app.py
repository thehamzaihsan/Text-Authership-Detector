import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.sparse import hstack, csr_matrix

st.set_page_config(page_title="WhatsApp Author Fingerprinting", layout="wide", page_icon="")

MODEL_DIR = "models"

MODEL_STATS = {
    "Word BOW + NB (8 authors)": {
        "accuracy": 0.7460, "nickname": "BOW + NB", "variant": "8 Authors",
        "per_sender": None,
    },
    "Word TF-IDF + NB (8 authors)": {
        "accuracy": 0.7370, "nickname": "TF-IDF + NB", "variant": "8 Authors",
        "per_sender": None,
    },
    "Char N-Grams + TF-IDF + NB (8 authors)": {
        "accuracy": 0.6640, "nickname": "Char + NB", "variant": "8 Authors",
        "per_sender": None,
    },
    "Char N-Grams + TF-IDF + LogReg (8 authors)": {
        "accuracy": 0.7170, "nickname": "Char + LogReg", "variant": "8 Authors",
        "per_sender": None,
    },
    "BOW + Handcrafted + NB (8 authors)": {
        "accuracy": 0.7390, "nickname": "BOW+Hand NB", "variant": "8 Authors",
        "per_sender": None,
    },
    "Ensemble (8 authors)": {
        "accuracy": 0.7510, "nickname": "Ensemble", "variant": "8 Authors",
        "per_sender": None,
    },
    "Word BOW + NB (4 authors)": {
        "accuracy": 0.7883, "nickname": "BOW + NB", "variant": "4 Authors",
        "per_sender": {
            "Awais Ibrahim BU PGC":   {"precision": 0.8490, "recall": 0.7683, "f1": 0.8066, "support": 600},
            "Nofal Zia (BU)":         {"precision": 0.8316, "recall": 0.7817, "f1": 0.8058, "support": 600},
            "Mubarak Andrabi BU":     {"precision": 0.7647, "recall": 0.8450, "f1": 0.8029, "support": 600},
            "Hamza Ihsan":            {"precision": 0.7222, "recall": 0.7583, "f1": 0.7398, "support": 600},
        }
    },
    "Word TF-IDF + NB (4 authors)": {
        "accuracy": 0.7975, "nickname": "TF-IDF + NB", "variant": "4 Authors",
        "per_sender": {
            "Mubarak Andrabi BU":     {"precision": 0.7913, "recall": 0.8467, "f1": 0.8180, "support": 600},
            "Nofal Zia (BU)":         {"precision": 0.8245, "recall": 0.8067, "f1": 0.8155, "support": 600},
            "Awais Ibrahim BU PGC":   {"precision": 0.8558, "recall": 0.7717, "f1": 0.8116, "support": 600},
            "Hamza Ihsan":            {"precision": 0.7286, "recall": 0.7650, "f1": 0.7463, "support": 600},
        }
    },
    "Char N-Grams + TF-IDF + NB (4 authors)": {
        "accuracy": 0.7508, "nickname": "Char + NB", "variant": "4 Authors",
        "per_sender": {
            "Nofal Zia (BU)":         {"precision": 0.8232, "recall": 0.7450, "f1": 0.7822, "support": 600},
            "Awais Ibrahim BU PGC":   {"precision": 0.7852, "recall": 0.7617, "f1": 0.7733, "support": 600},
            "Mubarak Andrabi BU":     {"precision": 0.7661, "recall": 0.7533, "f1": 0.7597, "support": 600},
            "Hamza Ihsan":            {"precision": 0.6511, "recall": 0.7433, "f1": 0.6942, "support": 600},
        }
    },
    "Char N-Grams + TF-IDF + LogReg (4 authors)": {
        "accuracy": 0.7775, "nickname": "Char + LogReg", "variant": "4 Authors",
        "per_sender": {
            "Nofal Zia (BU)":         {"precision": 0.8127, "recall": 0.7883, "f1": 0.8003, "support": 600},
            "Awais Ibrahim BU PGC":   {"precision": 0.7993, "recall": 0.7833, "f1": 0.7912, "support": 600},
            "Mubarak Andrabi BU":     {"precision": 0.7761, "recall": 0.7917, "f1": 0.7838, "support": 600},
            "Hamza Ihsan":            {"precision": 0.7249, "recall": 0.7467, "f1": 0.7356, "support": 600},
        }
    },
    "BOW + Handcrafted + NB (4 authors)": {
        "accuracy": 0.7867, "nickname": "BOW+Hand NB", "variant": "4 Authors",
        "per_sender": {
            "Nofal Zia (BU)":         {"precision": 0.8263, "recall": 0.7850, "f1": 0.8051, "support": 600},
            "Awais Ibrahim BU PGC":   {"precision": 0.8484, "recall": 0.7650, "f1": 0.8046, "support": 600},
            "Mubarak Andrabi BU":     {"precision": 0.7661, "recall": 0.8350, "f1": 0.7990, "support": 600},
            "Hamza Ihsan":            {"precision": 0.7197, "recall": 0.7617, "f1": 0.7401, "support": 600},
        }
    },
    "Ensemble (4 authors)": {
        "accuracy": 0.7967, "nickname": "Ensemble", "variant": "4 Authors",
        "per_sender": {
            "Nofal Zia (BU)":         {"precision": 0.8424, "recall": 0.8017, "f1": 0.8215, "support": 600},
            "Mubarak Andrabi BU":     {"precision": 0.7783, "recall": 0.8483, "f1": 0.8118, "support": 600},
            "Awais Ibrahim BU PGC":   {"precision": 0.8481, "recall": 0.7633, "f1": 0.8035, "support": 600},
            "Hamza Ihsan":            {"precision": 0.7307, "recall": 0.7733, "f1": 0.7514, "support": 600},
        }
    },
    "Deductive 9-class (Top 8 + UNKNOWN)": {
        "accuracy": 0.6723, "nickname": "Deductive 9c", "variant": "8+1 Authors",
        "per_sender": {
            "Chomu Hashim Nazir (BU)": {"precision": 0.7965, "recall": 0.7930, "f1": 0.7948, "support": 459},
            "Mubarak Andrabi BU":      {"precision": 0.7296, "recall": 0.7467, "f1": 0.7381, "support": 600},
            "Nofal Zia (BU)":          {"precision": 0.7527, "recall": 0.6900, "f1": 0.7200, "support": 600},
            "Humayun Tariq BU":        {"precision": 0.6828, "recall": 0.7103, "f1": 0.6963, "support": 397},
            "Awais Ibrahim BU PGC":    {"precision": 0.7306, "recall": 0.6600, "f1": 0.6935, "support": 600},
            "Rafay Ali (BU)":          {"precision": 0.5956, "recall": 0.7393, "f1": 0.6597, "support": 257},
            "Hamza Ihsan":             {"precision": 0.6438, "recall": 0.6267, "f1": 0.6351, "support": 600},
            "Mazen مازین (BU)":        {"precision": 0.5063, "recall": 0.6760, "f1": 0.5789, "support": 358},
            "UNKNOWN":                 {"precision": 0.5416, "recall": 0.4451, "f1": 0.4886, "support": 483},
        }
    },
}

MODEL_OPTIONS = {
    "Word BOW + NB (8 authors)": f"{MODEL_DIR}/cell7_bow_nb_8authors.pkl",
    "Word TF-IDF + NB (8 authors)": f"{MODEL_DIR}/cell8_tfidf_nb_8authors.pkl",
    "Char N-Grams + TF-IDF + NB (8 authors)": f"{MODEL_DIR}/cell9_char_tfidf_nb_8authors.pkl",
    "Char N-Grams + TF-IDF + LogReg (8 authors)": f"{MODEL_DIR}/cell10_char_tfidf_lr_8authors.pkl",
    "BOW + Handcrafted + NB (8 authors)": f"{MODEL_DIR}/cell11_bow_hand_nb_8authors.pkl",
    "Ensemble (8 authors)": f"{MODEL_DIR}/ensemble_8authors.pkl",
    "Word BOW + NB (4 authors)": f"{MODEL_DIR}/cell7_bow_nb_4authors.pkl",
    "Word TF-IDF + NB (4 authors)": f"{MODEL_DIR}/cell8_tfidf_nb_4authors.pkl",
    "Char N-Grams + TF-IDF + NB (4 authors)": f"{MODEL_DIR}/cell9_char_tfidf_nb_4authors.pkl",
    "Char N-Grams + TF-IDF + LogReg (4 authors)": f"{MODEL_DIR}/cell10_char_tfidf_lr_4authors.pkl",
    "BOW + Handcrafted + NB (4 authors)": f"{MODEL_DIR}/cell11_bow_hand_nb_4authors.pkl",
    "Ensemble (4 authors)": f"{MODEL_DIR}/ensemble_4authors.pkl",
    "Deductive 9-class (Top 8 + UNKNOWN)": f"{MODEL_DIR}/ded_model_9class_top8.pkl",
}


@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_handcrafted(text):
    return {"char_count": len(text), "word_count": len(text.split())}


def predict_single(model_data, text):
    is_ensemble = "models" in model_data
    accuracy = model_data.get("accuracy", 0)
    if is_ensemble:
        models = model_data["models"]
        vectorizers = model_data["vectorizers"]
        scalers = model_data.get("scalers", [None] * len(models))
        all_probas = []
        classes = models[0].classes_
        for i, clf in enumerate(models):
            vec = vectorizers[i]; X = vec.transform([text])
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


# ============================================================
# HEADER
# ============================================================
best_name = max(MODEL_STATS, key=lambda n: MODEL_STATS[n]["accuracy"])
best_acc = MODEL_STATS[best_name]["accuracy"]

st.markdown(f"""
<div style="text-align:center; padding: 20px 0 10px 0;">
    <h1 style="font-size:2.3rem; margin:0;"> WhatsApp Author Fingerprinting</h1>
    <p style="font-size:1rem; margin-top:6px; opacity:0.7;">
        Roman Urdu / English authorship attribution &middot; {len(MODEL_STATS)} models &middot; Top 4 &amp; 8 senders
    </p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Best Model", MODEL_STATS[best_name]["nickname"],
              help=best_name)
with c2:
    st.metric("Top Accuracy", f"{best_acc*100:.1f}%",
              help="TF-IDF + Naive Bayes (4 authors)")
with c3:
    st.metric("Models Available", len(MODEL_STATS),
              help="6 techniques x variants")
with c4:
    st.metric("Top Authors", "8",
              help="Hamza, Mubarak, Awais, Nofal, Mazen, Chomu, Humayun, Rafay")

# ============================================================
# TABS
# ============================================================
tab_predict, tab_stats = st.tabs([" Author Predictor ", " Model Benchmarks "])

# ============================================================
# TAB 1: PREDICT
# ============================================================
with tab_predict:
    left, right = st.columns([1, 1.2])
    with left:
        model_name = st.selectbox("Select Model", list(MODEL_OPTIONS.keys()), index=0)
        model_path = MODEL_OPTIONS[model_name]
        text_input = st.text_area("Type a message:", height=150,
                                  placeholder="e.g. kal class mein nhi aaraha tha kya hua tha?")
        predict_btn = st.button("Identify Author", type="primary", use_container_width=True)
        if model_name in MODEL_STATS:
            ms = MODEL_STATS[model_name]
            st.caption(f"Model accuracy: {ms['accuracy']*100:.1f}% &middot; {ms['variant']} &middot; {ms['nickname']}")

    with right:
        if predict_btn:
            raw = text_input.strip()
            if not raw:
                st.warning("Please type a message.")
            else:
                with st.spinner("Predicting..."):
                    try:
                        model_data = load_model(model_path)
                        text_clean = raw.lower()
                        wc = len(text_clean.split())
                        if len(text_clean) > 500:
                            text_clean = text_clean[:500]
                            st.info("Input truncated to 500 characters.")
                        if wc < 4:
                            st.warning("Message too short for reliable prediction.")

                        predicted, confidence, all_probs, accuracy = predict_single(model_data, text_clean)

                        st.markdown("### Predicted Author")
                        if predicted == "UNKNOWN":
                            st.markdown(f"<h1 style='color:#FF9800;'>{predicted}</h1>", unsafe_allow_html=True)
                            st.warning("This message does not match any known author.")
                        else:
                            st.markdown(f"<h1 style='color:#4CAF50;'>{predicted}</h1>", unsafe_allow_html=True)
                        st.metric("Confidence", f"{confidence:.1%}")
                        st.info(f"Model: {model_name} &middot; Acc: {accuracy*100:.1f}%")

                        if confidence < 0.50 and predicted != "UNKNOWN":
                            st.warning("Low confidence &mdash; message may not be from this author.")

                        sorted_probs = sorted(all_probs.items(), key=lambda x: -x[1])
                        prob_df = pd.DataFrame({
                            "Author": [a[:35] for a, _ in sorted_probs],
                            "Probability": [p for _, p in sorted_probs]
                        })
                        fig = px.bar(prob_df, x="Probability", y="Author", orientation="h",
                                     title="All Author Probabilities",
                                     color_discrete_sequence=["#2196F3"] * len(prob_df),
                                     text_auto=".0%", height=350)
                        fig.update_layout(yaxis=dict(autorange="reversed"),
                                          margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
                        fig.update_traces(
                            marker_color=["#4CAF50" if a == predicted else "#B0BEC5" for a in prob_df["Author"]],
                            marker_line_width=0)
                        st.plotly_chart(fig, use_container_width=True)

                        with st.expander("Show exact percentages"):
                            for author, prob in sorted_probs:
                                marker = " &#8592;" if author == predicted else ""
                                st.write(f"{author}: {prob:.1%}{marker}")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

# ============================================================
# TAB 2: MODEL BENCHMARKS
# ============================================================
with tab_stats:
    st.markdown("### Model Performance Comparison")

    df_acc = pd.DataFrame([
        {"Model (short)": v["nickname"], "Accuracy": v["accuracy"],
         "Variant": v["variant"], "Model Name": k}
        for k, v in MODEL_STATS.items()
    ]).sort_values("Accuracy", ascending=True)

    fig = px.bar(df_acc, x="Accuracy", y="Model (short)", orientation="h",
                 color="Variant", text_auto=".1%",
                 color_discrete_map={
                     "4 Authors": "#4CAF50", "8 Authors": "#2196F3",
                     "8+1 Authors": "#FF9800"
                 },
                 height=500,
                 title="Overall Accuracy by Model")
    fig.update_layout(yaxis=dict(autorange="reversed"),
                      margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    models_with_detail = [k for k, v in MODEL_STATS.items() if v["per_sender"] is not None]
    if models_with_detail:
        st.markdown("### Per-Sender Breakdown")
        model_pick = st.selectbox("Select model to view per-sender metrics",
                                   models_with_detail, index=0)
        ms = MODEL_STATS[model_pick]
        ps = ms["per_sender"]

        if ps:
            df_ps = pd.DataFrame([
                {"Author": a, "Precision": v["precision"], "Recall": v["recall"],
                 "F1-Score": v["f1"], "Support": v["support"]}
                for a, v in ps.items()
            ]).sort_values("F1-Score", ascending=False)

            col_a, col_b = st.columns([1.2, 1])
            with col_a:
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(name="Precision", x=df_ps["Author"], y=df_ps["Precision"],
                                      marker_color="#2196F3"))
                fig2.add_trace(go.Bar(name="Recall", x=df_ps["Author"], y=df_ps["Recall"],
                                      marker_color="#4CAF50"))
                fig2.add_trace(go.Bar(name="F1-Score", x=df_ps["Author"], y=df_ps["F1-Score"],
                                      marker_color="#FF9800"))
                fig2.update_layout(
                    barmode="group", title=f"<b>{ms['nickname']}</b> — Per-Sender Metrics",
                    height=400,
                    margin=dict(l=10, r=10, t=40, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(tickangle=45),
                )
                st.plotly_chart(fig2, use_container_width=True)

            with col_b:
                st.metric(ms['nickname'], f"{ms['accuracy']*100:.1f}%",
                          help=f"Overall Accuracy &middot; {ms['variant']}")

                df_table = df_ps.round(3)
                df_table["F1-Score"] = df_table["F1-Score"].apply(lambda x: f"{x:.3f}")
                df_table["Precision"] = df_table["Precision"].apply(lambda x: f"{x:.3f}")
                df_table["Recall"] = df_table["Recall"].apply(lambda x: f"{x:.3f}")
                st.dataframe(df_table, use_container_width=True, hide_index=True,
                             column_config={
                                 "Author": st.column_config.TextColumn("Author", width="large"),
                                 "Precision": st.column_config.TextColumn("Precision", width="small"),
                                 "Recall": st.column_config.TextColumn("Recall", width="small"),
                                 "F1-Score": st.column_config.TextColumn("F1", width="small"),
                                 "Support": st.column_config.NumberColumn("Support", width="small"),
                             })

    st.markdown("---")
    st.markdown("### Model Summary")
    df_summary = df_acc.sort_values("Accuracy", ascending=False).reset_index(drop=True)
    df_summary["#"] = range(1, len(df_summary) + 1)
    df_summary["Accuracy"] = df_summary["Accuracy"].apply(lambda x: f"{x*100:.1f}%")
    df_summary = df_summary[["#", "Model (short)", "Accuracy", "Variant", "Model Name"]]
    st.dataframe(df_summary, use_container_width=True, hide_index=True,
                 column_config={
                     "#": st.column_config.NumberColumn("#", width="small"),
                     "Model (short)": "Model",
                     "Accuracy": st.column_config.TextColumn("Accuracy", width="small"),
                     "Variant": st.column_config.TextColumn("Variant", width="small"),
                     "Model Name": st.column_config.TextColumn("Full Name", width="large"),
                 })

st.markdown("---")
st.caption("Models trained on WhatsApp chat data with Roman Urdu / English code-mixed messages. "
           "Top 4: Hamza Ihsan, Mubarak Andrabi, Awais Ibrahim, Nofal Zia. "
           "Top 8 adds: Mazen, Chomu Hashim, Humayun Tariq, Rafay Ali.")
