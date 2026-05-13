#!/usr/bin/env python3
"""Creates ded.ipynb programmatically."""
import json

cells = []

def code_cell(source_list, exec_count=None):
    src = "\n".join(source_list)
    c = {
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": [],
        "source": src.split("\n"),
    }
    if exec_count is not None:
        c["execution_count"] = exec_count
    return c

def md_cell(source_list):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": "\n".join(source_list).split("\n"),
    }

# === CELL 1: Load data ===
cells.append(code_cell([
    'import pandas as pd',
    '',
    'df_raw = pd.read_csv("whatsapp_chat.csv")',
    'df_raw_cleaned = pd.read_csv("whatsapp_chat_cleaned.csv")',
    'pd.set_option("display.max_colwidth", 50)',
    'pd.set_option("display.max_columns", 15)',
    'pd.set_option("display.width", 200)',
    'pd.set_option("display.max_rows", 20)',
    'print(df_raw.head())',
    'print(df_raw_cleaned.head())',
]))

# === CELL 2: Filter for top 8 senders ===
cells.append(code_cell([
    'sender_counts = df_raw_cleaned["sender"].value_counts()',
    'senders_over_2000 = sender_counts[sender_counts > 2000].index',
    'print(f"Top {len(senders_over_2000)} senders with >2000 msgs:")',
    'for s in senders_over_2000:',
    '    print(f"  {s}: {sender_counts[s]}")',
    '',
    '# Keep only top 8 senders',
    'senders_top8 = senders_over_2000[:8]',
    'df_8 = df_raw_cleaned[df_raw_cleaned["sender"].isin(senders_top8)].copy()',
    'print("\\nSelected top 8 senders:")',
    'for s in senders_top8:',
    '    print(f"  {s}: {sender_counts[s]}")',
]))

# === CELL 3: Top 8 stats table ===
cells.append(code_cell([
    'import matplotlib.pyplot as plt',
    '',
    'counts = df_8["sender"].value_counts()',
    'fig, ax = plt.subplots(figsize=(12, len(counts)*0.5+1))',
    'ax.axis("tight")',
    'ax.axis("off")',
    'table_data = []',
    'total_msgs = 0; total_chars = 0; total_words = 0',
    'for s in counts.index:',
    '    sd = df_8[df_8["sender"] == s]',
    '    n = len(sd)',
    '    aw = round(sd["word_count"].mean(), 1)',
    '    ac = round(sd["char_count"].mean(), 1)',
    '    total_msgs += n; total_words += sd["word_count"].sum(); total_chars += sd["char_count"].sum()',
    '    table_data.append([s, n, aw, ac])',
    'table_data.append(["TOTAL", total_msgs, round(total_words/total_msgs,1), round(total_chars/total_msgs,1)])',
    'ax.table(cellText=table_data, colLabels=["Sender","Messages","Avg Words","Avg Chars"], cellLoc="center", loc="center")',
    'plt.title(f"Top 8 Senders Dataset", fontsize=14, pad=20)',
    'plt.tight_layout()',
    'plt.show()',
]))

# === CELL 4: Filter by word count > 2 and <= 100 ===
cells.append(code_cell([
    'df_filtered = df_8[(df_8["word_count"] > 2) & (df_8["word_count"] <= 100)].copy()',
    'print(f"After word_count filter (2 < wc <= 100): {len(df_filtered)} messages")',
]))

# === CELL 5: Show columns ===
cells.append(code_cell([
    'print("Columns:", df_filtered.columns.tolist())',
]))

# === CELL 6: Train-test split per sender ===
cells.append(code_cell([
    'from sklearn.model_selection import train_test_split',
    '',
    'df_filtered["custom_split"] = "train"',
    'for s in df_filtered["sender"].unique():',
    '    mask = df_filtered["sender"] == s',
    '    idx = df_filtered[mask].index',
    '    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42)',
    '    df_filtered.loc[te_idx, "custom_split"] = "test"',
    '',
    'print("Split per sender:")',
    'for s in df_filtered["sender"].unique():',
    '    sd = df_filtered[df_filtered["sender"] == s]',
    '    tr = (sd["custom_split"] == "train").sum()',
    '    te = (sd["custom_split"] == "test").sum()',
    '    print(f"  {s[:30]}: {tr} train, {te} test")',
    'print(f"\\nTotal: {len(df_filtered)}")',
    'print(f"Train: {(df_filtered.custom_split==\"train\").sum()}")',
    'print(f"Test: {(df_filtered.custom_split==\"test\").sum()}")',
]))

# === CELL 7: Create ded.model with unknown class ===
cells.append(code_cell([
    'import numpy as np',
    'import pandas as pd',
    'import matplotlib.pyplot as plt',
    'import seaborn as sns',
    'from sklearn.feature_extraction.text import TfidfVectorizer',
    'from sklearn.linear_model import LogisticRegression',
    'from sklearn.metrics import confusion_matrix, classification_report, accuracy_score',
    '',
    '# ============================================================',
    '# DEDUCTIVE MODEL: Top 8 Authors + Unknown Class',
    '# Uses Char TF-IDF + Logistic Regression (best base technique)',
    '# Unknown = sampled messages from all OTHER senders',
    '# ============================================================',
    '',
    'train_data = df_filtered[(df_filtered["custom_split"] == "train") & (df_filtered["word_count"] >= 3)].copy()',
    'test_data = df_filtered[(df_filtered["custom_split"] == "test") & (df_filtered["word_count"] >= 3)].copy()',
    '',
    '# --- Build unknown class from remaining senders ---',
    'known_senders = list(train_data["sender"].unique())',
    'print(f"Known senders: {known_senders}")',
    '',
    '# Get all OTHER senders from the full dataset',
    'other_df = df_raw_cleaned[~df_raw_cleaned["sender"].isin(known_senders)].copy()',
    'other_df = other_df[(other_df["word_count"] > 2) & (other_df["word_count"] <= 100)]',
    'print(f"Other senders available: {other_df[\"sender\"].nunique()}, messages: {len(other_df)}")',
    '',
    '# Sample unknown messages to match ~per-sender average of known',
    'msgs_per_known = len(train_data) // len(known_senders)',
    'n_unknown = min(msgs_per_known, len(other_df))',
    'unknown_sample = other_df.sample(n=n_unknown, random_state=42)',
    'unknown_sample["sender"] = "UNKNOWN"',
    'unknown_sample["custom_split"] = "train"',
    'print(f"Sampled {n_unknown} unknown messages")',
    '',
    '# Also create test unknown',
    'other_test = df_raw_cleaned[~df_raw_cleaned["sender"].isin(known_senders)].copy()',
    'other_test = other_test[(other_test["word_count"] > 2) & (other_test["word_count"] <= 100)]',
    'n_unknown_test = min(len(test_data) // len(known_senders), len(other_test))',
    'unknown_test = other_test.sample(n=n_unknown_test, random_state=43)',
    'unknown_test["sender"] = "UNKNOWN"',
    'unknown_test["custom_split"] = "test"',
    'print(f"Sampled {n_unknown_test} unknown test messages")',
]))

# === CELL 8: Train the deductive model ===
cells.append(code_cell([
    '# Combine known + unknown',
    'train_full = pd.concat([train_data, unknown_sample], ignore_index=True)',
    'test_full = pd.concat([test_data, unknown_test], ignore_index=True)',
    '',
    'print(f"Train: {len(train_full)} ({train_full[\"sender\"].value_counts().to_dict()})")',
    'print(f"Test: {len(test_full)}")',
    'print(f"\\nClass distribution (train):")',
    'print(train_full["sender"].value_counts())',
    '',
    '# --- Vectorize ---',
    'vectorizer = TfidfVectorizer(',
    '    analyzer="char",',
    '    ngram_range=(3, 5),',
    '    max_features=10000,',
    '    min_df=2',
    ')',
    '',
    'X_train = vectorizer.fit_transform(train_full["message_clean_lower"])',
    'X_test = vectorizer.transform(test_full["message_clean_lower"])',
    'y_train = train_full["sender"]',
    'y_test = test_full["sender"]',
    '',
    '# --- Train Logistic Regression ---',
    'model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)',
    'model.fit(X_train, y_train)',
]))

# === CELL 9: Evaluate ===
cells.append(code_cell([
    'y_pred = model.predict(X_test)',
    'accuracy = accuracy_score(y_test, y_pred)',
    '',
    'report = classification_report(y_test, y_pred, output_dict=True)',
    'report_df = pd.DataFrame(report).transpose()',
    'sender_metrics = report_df.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")',
    'sender_metrics_sorted = sender_metrics.sort_values("f1-score", ascending=False)',
    '',
    'print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")',
    'print(f"\\nPer-Sender Metrics (sorted by F1):")',
    'print(sender_metrics_sorted[["precision", "recall", "f1-score", "support"]])',
    '',
    '# ---- Plot 1: Per-Sender Metrics ----',
    'fig, ax = plt.subplots(figsize=(12, 6))',
    'x = np.arange(len(sender_metrics_sorted))',
    'width = 0.25',
    'ax.bar(x - width, sender_metrics_sorted["precision"], width, label="Precision", color="#2196F3")',
    'ax.bar(x, sender_metrics_sorted["recall"], width, label="Recall", color="#4CAF50")',
    'ax.bar(x + width, sender_metrics_sorted["f1-score"], width, label="F1-Score", color="#FF9800")',
    'ax.set_xlabel("Sender")',
    'ax.set_ylabel("Score")',
    'ax.set_title("Per-Sender Performance (Deductive Model: Top 8 + Unknown)")',
    'ax.set_xticks(x)',
    'ax.set_xticklabels([s[:25] for s in sender_metrics_sorted.index], rotation=45, ha="right")',
    'ax.legend()',
    'ax.set_ylim(0, 1)',
    'ax.grid(axis="y", alpha=0.3)',
    'plt.tight_layout()',
    'plt.show()',
    '',
    '# ---- Plot 2: Confusion Matrix ----',
    'cm = confusion_matrix(y_test, y_pred)',
    'labels = model.classes_',
    'cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]',
    '',
    'fig, ax = plt.subplots(figsize=(12, 10))',
    'sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",',
    '            xticklabels=[l[:25] for l in labels],',
    '            yticklabels=[l[:25] for l in labels],',
    '            ax=ax, vmin=0, vmax=1)',
    'ax.set_xlabel("Predicted")',
    'ax.set_ylabel("Actual")',
    'ax.set_title(f"Deductive Model Confusion Matrix - Accuracy: {accuracy*100:.1f}%")',
    'plt.xticks(rotation=45, ha="right")',
    'plt.yticks(rotation=0)',
    'plt.tight_layout()',
    'plt.show()',
    '',
    '# ---- Plot 3: Support ----',
    'fig, ax = plt.subplots(figsize=(10, 5))',
    'supports = sender_metrics_sorted["support"]',
    'colors = ["#4CAF50" if s >= 100 else "#FFC107" if s >= 50 else "#F44336" for s in supports]',
    'ax.barh(range(len(supports)), supports, color=colors)',
    'ax.set_yticks(range(len(supports)))',
    'ax.set_yticklabels([s[:30] for s in sender_metrics_sorted.index])',
    'ax.set_xlabel("Test Samples")',
    'ax.set_title("Test Set Support per Sender")',
    'for i, v in enumerate(supports):',
    '    ax.text(v + 1, i, str(int(v)), va="center")',
    'plt.tight_layout()',
    'plt.show()',
]))

# === CELL 10: Save model as .pkl ===
cells.append(code_cell([
    'import pickle',
    'import os',
    '',
    'model_data = {',
    '    "model": model,',
    '    "vectorizer": vectorizer,',
    '    "author_list": list(model.classes_),',
    '    "accuracy": accuracy,',
    '}',
    '',
    'os.makedirs("models", exist_ok=True)',
    'pkl_path = "models/ded_model_top8_unknown.pkl"',
    'with open(pkl_path, "wb") as f:',
    '    pickle.dump(model_data, f)',
    '',
    'print(f"Model saved to {pkl_path}")',
    'print(f"Classes: {model_data[\"author_list\"]}")',
    'print(f"Accuracy: {accuracy*100:.2f}%")',
]))

# === CELL 11: Test the model on sample messages ===
cells.append(code_cell([
    '# Quick sanity check',
    'test_texts = [',
    '    "kal class mein nhi aaraha tha kya hua tha?",',
    '    "bro ya bhi mera number hai save kar la",',
    '    "assignment submit kar do kal tak",',
    '    "yeh konsa subject hai?",',
    ']',
    '',
    'for t in test_texts:',
    '    X = vectorizer.transform([t.lower()])',
    '    proba = model.predict_proba(X)[0]',
    '    pred = model.classes_[np.argmax(proba)]',
    '    conf = max(proba)',
    '    print(f"Text: {t}")',
    '    print(f"  Predicted: {pred} (conf: {conf:.1%})")',
    '    print()',
]))

# === Build notebook ===
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "venv",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.3"
        }
    },
    "cells": cells,
}

with open("ded.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("Created ded.ipynb with", len(cells), "cells")
