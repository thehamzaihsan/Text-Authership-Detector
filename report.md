# Deductive Authorship Attribution — Project Report

## 1. Project Overview

This project builds and deploys a **deductive authorship attribution system** for WhatsApp chat messages. The system identifies the most likely author of a given message from **8 known senders** or classifies it as **UNKNOWN**. A full interactive dashboard is deployed via **Streamlit** to compare all models and make live predictions.

### Key Objectives
- Build 13 distinct classification models using various text representations and algorithms
- Implement a deductive 9-class model (8 known authors + UNKNOWN)
- Provide an interactive Streamlit dashboard for model comparison and live prediction
- Visualize per-sender performance metrics, confusion matrices, and test support distributions

---

## 2. Dataset

### Source
- **File:** `whatsapp_chat.csv` / `whatsapp_chat_cleaned.csv`
- **Origin:** WhatsApp chat exports from a student group conversation
- **Total unique senders:** 36
- **Total messages (raw):** ~45,000+

### Preprocessing Pipeline

1. **Raw data loading** — parse WhatsApp chat export with sender, message text, timestamp
2. **Text cleaning** — remove system messages, media placeholders; create `message_clean` and `message_clean_lower` columns
3. **Top-sender filtering** — keep only senders with >3,000 messages (for 4-author models) or top 8 (for 8-author and deductive models)
4. **Word count filter:** `2 < word_count <= 100` to remove very short/empty and very long messages (reduces noise)
5. **Bell-curve balancing** — sample messages from each sender to approximate a normal distribution of word counts (target ~3,000 per sender)
6. **Train/test split** — 80/20 stratified split per sender

### Final Balanced Dataset (4-author)
| Sender | Messages |
|--------|----------|
| Nofal Zia (BU) | 3,000 |
| Mubarak Andrabi BU | 3,000 |
| Hamza Ihsan | 3,000 |
| Awais Ibrahim BU PGC | 3,000 |
| **Total** | **12,000** |
| **Train** | **9,600 (80%)** |
| **Test** | **2,400 (20%)** |

### Expanded Dataset (8-author + Deductive)
| Sender | Messages |
|--------|----------|
| Nofal Zia (BU) | 3,000 |
| Mubarak Andrabi BU | 3,000 |
| Hamza Ihsan | 3,000 |
| Awais Ibrahim BU PGC | 3,000 |
| Chomu Hashim Nazir (BU) | 2,295 |
| Humayun Tariq BU | 1,981 |
| Mazen مازین (BU) | 1,789 |
| Rafay Ali (BU) | 2,049 |
| UNKNOWN (sampled from remaining 28 senders) | 2,417 |
| **Total** | **19,350** |
| **Train** | **17,413** |
| **Test** | **4,354** |

---

## 3. Feature Engineering

### 3.1 Text Representations

| Representation | Analyzer | N-gram Range | Max Features | Description |
|---------------|----------|-------------|-------------|-------------|
| **Word BOW** | word | (1, 2) | 10,000 (or 8,000) | Count of word unigrams and bigrams |
| **Word TF-IDF** | word | (1, 2) | 10,000 | TF-IDF weighted word n-grams |
| **Char N-Grams + TF-IDF** | char | (3, 5) | 10,000 | TF-IDF weighted character 3- to 5-grams |

### 3.2 Handcrafted Features

Used in the "BOW + Handcrafted + NB" model:

| Feature | Description |
|---------|-------------|
| `char_count` | Total characters in message |
| `word_count` | Total words in message |
| `avg_word_length` | Average word length |
| `uppercase_ratio` | Proportion of uppercase characters |
| `punctuation_count` | Number of punctuation marks |
| `digit_count` | Number of digits |
| `emoji_count` | Number of emojis |
| `has_emoji` | Boolean: message contains emoji |
| `question_mark_count` | Number of question marks |
| `exclamation_count` | Number of exclamation marks |
| `url_count` | Number of URLs |
| `language_mix_ratio` | Ratio of non-English characters |

### 3.3 Feature Scaling
- Handcrafted features are normalized using **MinMaxScaler** before being concatenated with sparse text features

---

## 4. Models

### 4.1 Complete Model Inventory (13 Models)

All 13 models are available in the Streamlit app's MODEL_OPTIONS:

| # | Model Name | Authors | Accuracy |
|---|-----------|---------|----------|
| 1 | Word BOW + NB (8 authors) | 8 | varies |
| 2 | Word TF-IDF + NB (8 authors) | 8 | varies |
| 3 | Char N-Grams + TF-IDF + NB (8 authors) | 8 | varies |
| 4 | Char N-Grams + TF-IDF + LogReg (8 authors) | 8 | varies |
| 5 | BOW + Handcrafted + NB (8 authors) | 8 | varies |
| 6 | Ensemble (8 authors) | 8 | varies |
| 7 | Word BOW + NB (4 authors) | 4 | varies |
| 8 | Word TF-IDF + NB (4 authors) | 4 | varies |
| 9 | Char N-Grams + TF-IDF + NB (4 authors) | 4 | varies |
| 10 | Char N-Grams + TF-IDF + LogReg (4 authors) | 4 | varies |
| 11 | BOW + Handcrafted + NB (4 authors) | 4 | varies |
| 12 | Ensemble (4 authors) | 4 | varies |
| 13 | Deductive 9-class (Top 8 + UNKNOWN) | 9 | **67.23%** |

### 4.2 4-Author Model Results (test.ipynb)

Each model trained on **9,600 messages** (balanced across 4 authors), tested on **2,400 messages**.

| Model | Accuracy | Best F1 (Sender) | Worst F1 (Sender) |
|-------|----------|-------------------|-------------------|
| Word BOW + NB | 78.83% | 0.807 (Awais Ibrahim BU PGC) | 0.740 (Hamza Ihsan) |
| **Word TF-IDF + NB** | **79.75%** | **0.818 (Mubarak Andrabi BU)** | **0.746 (Hamza Ihsan)** |
| Char N-Grams + TF-IDF + NB | 75.08% | 0.782 (Nofal Zia) | 0.694 (Hamza Ihsan) |
| Char N-Grams + TF-IDF + LogReg | 77.75% | 0.800 (Nofal Zia) | 0.736 (Hamza Ihsan) |
| Word BOW + Handcrafted + NB | 78.67% | 0.805 (Nofal Zia) | 0.740 (Hamza Ihsan) |
| **Ensemble (all 5)** | **79.67%** | **0.822 (Nofal Zia)** | **0.751 (Hamza Ihsan)** |

**Best model overall (4-author):** TF-IDF + NB — **79.75%** accuracy

### 4.3 Deductive 9-Class Model

**Approach:**
- Single 9-class model (8 known + UNKNOWN) using **Char TF-IDF + Logistic Regression**
- **Training data:** 17,413 messages (8 known authors + UNKNOWN samples)
- **Test data:** 4,354 messages
- **Overall accuracy:** 67.23%

**Per-Class Metrics (sorted by F1):**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Chomu Hashim Nazir (BU) | 0.796 | 0.793 | 0.795 | 459 |
| Mubarak Andrabi BU | 0.730 | 0.747 | 0.738 | 600 |
| Nofal Zia (BU) | 0.753 | 0.690 | 0.720 | 600 |
| Awais Ibrahim BU PGC | 0.775 | 0.647 | 0.705 | 600 |
| Mazen مازین (BU) | 0.671 | 0.681 | 0.676 | 358 |
| Humayun Tariq BU | 0.675 | 0.647 | 0.661 | 397 |
| Rafay Ali (BU) | 0.619 | 0.638 | 0.629 | 257 |
| Hamza Ihsan | 0.629 | 0.555 | 0.590 | 600 |
| **UNKNOWN** | **0.523** | **0.463** | **0.491** | **483** |

**Design Decision:** A single 9-class model was chosen over a two-stage pipeline (8-class classifier + UNKNOWN detector) because it avoids cascading errors and gave higher combined accuracy (67.23% vs. 62.15%).

**Comparative baseline:** When tested on only the 8 known classes (excluding UNKNOWN), the same model achieves 71.71% accuracy.

---

## 5. Implementation Details

### 5.1 Notebooks

| File | Purpose |
|------|---------|
| `test.ipynb` | Original 4-author model development: data loading, preprocessing, bell-curve balancing, 6 model evaluations, ensemble, and model comparison charts |
| `ded.ipynb` | Deductive model development: top-8 sender filtering, UNKNOWN sampling, 9-class model training, evaluation, and prediction demo |
| `ded_executed.ipynb` | Executed version of ded.ipynb with all outputs |

### 5.2 Key Libraries
- **Pandas, NumPy** — data manipulation
- **Scikit-learn** — CountVectorizer, TfidfVectorizer, MultinomialNB, LogisticRegression, MinMaxScaler
- **Matplotlib, Seaborn** — static chart generation (notebook)
- **Plotly** — interactive charts (Streamlit app)
- **Streamlit** — web dashboard
- **Pickle (joblib)** — model persistence

### 5.3 Streamlit App Architecture
- **Framework:** Streamlit (single-page app, two-tab layout)
- **Tab 1: Author Predictor** — select model, type/paste message, view prediction with confidence and probability bar chart
- **Tab 2: Model Benchmarks** — compare all 13 models via interactive bar charts, view per-sender metrics for selected model
- **Theme:** Default Streamlit light mode (no forced dark CSS)

---

## 6. Model Persistence

All trained models are serialized using `pickle` and stored in `models/`:
- File sizes range from ~600 KB to ~7.2 MB
- Each `.pkl` file contains: model, vectorizer, scaler (if applicable), accuracy metadata
- Models loaded lazily at prediction time using `@st.cache_resource`

---

## 7. Results Summary

| Finding | Detail |
|---------|--------|
| **Best 4-author model** | TF-IDF + NB — 79.75% |
| **Best single model** | TF-IDF + NB (word, unigrams + bigrams) |
| **Deductive accuracy** | 67.23% (9-class), 71.71% (8-known subset) |
| **Hardest author (4-class)** | Hamza Ihsan (lowest F1 across all models) |
| **Easiest author (4-class)** | Varies by model (Awais/Nofal/Mubarak) |
| **UNKNOWN detection F1** | 0.491 (most challenging class) |
| **Top performing class (deductive)** | Chomu Hashim Nazir (F1 = 0.795) |
