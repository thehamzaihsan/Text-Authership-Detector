import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack, csr_matrix

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# Load and filter data
# ============================================================
df = pd.read_csv("whatsapp_chat_cleaned.csv")
print(f"Loaded {len(df)} rows")

# Take top 8 senders
TOP_N = 8
sender_counts = df['sender'].value_counts()
top_senders = sender_counts.nlargest(TOP_N).index
df = df[df['sender'].isin(top_senders)].copy()
print(f"Top {TOP_N} senders: {list(top_senders)}")

# Filter: 2 < word_count <= 100
df = df[(df['word_count'] > 2) & (df['word_count'] <= 100)].copy()
print(f"After word_count filter: {len(df)} messages")

# ============================================================
# Word-count weighted sampling at TARGET_SIZE=3000
# ============================================================
TARGET_SIZE = 3000
np.random.seed(42)
target_mean, target_std = 10, 6
target_word_counts = np.random.normal(target_mean, target_std, TARGET_SIZE)
target_word_counts = np.clip(target_word_counts, 1, None)
target_word_counts = np.round(target_word_counts).astype(int)

balanced_dfs = []
for sender in df['sender'].unique():
    sender_data = df[df['sender'] == sender].copy()
    if len(sender_data) < TARGET_SIZE:
        sampled = sender_data
    else:
        sampled_list = []
        sd = sender_data.reset_index(drop=True)
        wc_series = sd['word_count'].values
        for target_wc in target_word_counts:
            diffs = np.abs(wc_series - target_wc)
            closest_idx = diffs.argmin()
            sampled_list.append(sd.iloc[closest_idx])
            wc_series = np.delete(wc_series, closest_idx)
            sd = sd.drop(sd.index[closest_idx]).reset_index(drop=True)
        sampled = pd.DataFrame(sampled_list)
    balanced_dfs.append(sampled)

df = pd.concat(balanced_dfs, ignore_index=True)
print(f"After balancing: {len(df)} messages")
print(f"Senders: {df['sender'].nunique()}")
per_sender = df['sender'].value_counts()
print(f"Min per sender: {per_sender.min()}, Max: {per_sender.max()}")

# ============================================================
# Train/test split (80/20, per-sender)
# ============================================================
df['custom_split'] = 'train'
for sender in df['sender'].unique():
    sender_mask = df['sender'] == sender
    sender_indices = df[sender_mask].index
    train_idx, test_idx = train_test_split(sender_indices, test_size=0.2, random_state=42)
    df.loc[test_idx, 'custom_split'] = 'test'

train = df[df['custom_split'] == 'train']
test = df[df['custom_split'] == 'test']
print(f"Train: {len(train)}, Test: {len(test)}")

# ============================================================
# Helper: save model
# ============================================================
def save_model(model_data, name):
    path = f"{MODEL_DIR}/{name}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"  Saved {path}")

# ============================================================
# Train all models
# ============================================================
text_col = 'message_clean_lower'
y_train_all = train['sender']
y_test_all = test['sender']

model_results = {}
author_list_full = sorted(df['sender'].unique())
top_authors = sorted(df['sender'].value_counts().nlargest(4).index.tolist())

def filter_by_authors(train, test, authors):
    mask_train = train['sender'].isin(authors)
    mask_test = test['sender'].isin(authors)
    return train[mask_train].copy(), test[mask_test].copy()

for variant_name, authors in [("8authors", author_list_full), ("4authors", top_authors)]:
    print(f"\n=== Training {variant_name} ===")
    tr, te = filter_by_authors(train, test, authors)
    
    # ---- Model 1: Word BOW + NB ----
    print("  [1/6] Word BOW + NB...")
    vec1 = CountVectorizer(analyzer='word', ngram_range=(1, 2), max_features=10000, min_df=2)
    X_tr1 = vec1.fit_transform(tr[text_col])
    X_te1 = vec1.transform(te[text_col])
    m1 = MultinomialNB(alpha=0.5)
    m1.fit(X_tr1, tr['sender'])
    acc1 = accuracy_score(te['sender'], m1.predict(X_te1))
    save_model({"model": m1, "vectorizer": vec1, "author_list": authors, "accuracy": acc1},
               f"cell7_bow_nb_{variant_name}")
    model_results[f"bow_nb_{variant_name}"] = {"model": m1, "vectorizer": vec1, "accuracy": acc1, "X_te": X_te1}
    
    # ---- Model 2: Word TF-IDF + NB ----
    print("  [2/6] Word TF-IDF + NB...")
    vec2 = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=10000, min_df=2)
    X_tr2 = vec2.fit_transform(tr[text_col])
    X_te2 = vec2.transform(te[text_col])
    m2 = MultinomialNB(alpha=0.5)
    m2.fit(X_tr2, tr['sender'])
    acc2 = accuracy_score(te['sender'], m2.predict(X_te2))
    save_model({"model": m2, "vectorizer": vec2, "author_list": authors, "accuracy": acc2},
               f"cell8_tfidf_nb_{variant_name}")
    model_results[f"tfidf_nb_{variant_name}"] = {"model": m2, "vectorizer": vec2, "accuracy": acc2, "X_te": X_te2}
    
    # ---- Model 3: Char TF-IDF + NB ----
    print("  [3/6] Char TF-IDF + NB...")
    vec3 = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=10000, min_df=2)
    X_tr3 = vec3.fit_transform(tr[text_col])
    X_te3 = vec3.transform(te[text_col])
    m3 = MultinomialNB(alpha=0.5)
    m3.fit(X_tr3, tr['sender'])
    acc3 = accuracy_score(te['sender'], m3.predict(X_te3))
    save_model({"model": m3, "vectorizer": vec3, "author_list": authors, "accuracy": acc3},
               f"cell9_char_tfidf_nb_{variant_name}")
    model_results[f"char_tfidf_nb_{variant_name}"] = {"model": m3, "vectorizer": vec3, "accuracy": acc3, "X_te": X_te3}
    
    # ---- Model 4: Char TF-IDF + LogReg ----
    print("  [4/6] Char TF-IDF + LogReg...")
    vec4 = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=10000, min_df=2)
    X_tr4 = vec4.fit_transform(tr[text_col])
    X_te4 = vec4.transform(te[text_col])
    m4 = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
    m4.fit(X_tr4, tr['sender'])
    acc4 = accuracy_score(te['sender'], m4.predict(X_te4))
    save_model({"model": m4, "vectorizer": vec4, "author_list": authors, "accuracy": acc4},
               f"cell10_char_tfidf_lr_{variant_name}")
    model_results[f"char_tfidf_lr_{variant_name}"] = {"model": m4, "vectorizer": vec4, "accuracy": acc4, "X_te": X_te4}
    
    # ---- Model 5: BOW + Handcrafted + NB ----
    print("  [5/6] BOW + Handcrafted + NB...")
    vec5 = CountVectorizer(analyzer='word', ngram_range=(1, 2), max_features=8000, min_df=2)
    X_tr5_text = vec5.fit_transform(tr[text_col])
    X_te5_text = vec5.transform(te[text_col])
    
    scaler = MinMaxScaler()
    hand_tr = scaler.fit_transform(tr[['char_count', 'word_count']])
    hand_te = scaler.transform(te[['char_count', 'word_count']])
    
    X_tr5 = hstack([X_tr5_text, csr_matrix(hand_tr)])
    X_te5 = hstack([X_te5_text, csr_matrix(hand_te)])
    m5 = MultinomialNB(alpha=0.5)
    m5.fit(X_tr5, tr['sender'])
    acc5 = accuracy_score(te['sender'], m5.predict(X_te5))
    save_model({"model": m5, "vectorizer": vec5, "scaler": scaler, "author_list": authors, "accuracy": acc5},
               f"cell11_bow_hand_nb_{variant_name}")
    model_results[f"bow_hand_nb_{variant_name}"] = {"model": m5, "vectorizer": vec5, "scaler": scaler, "accuracy": acc5}
    
    # ---- Model 6: Ensemble ----
    print("  [6/6] Ensemble...")
    vec_ens = [vec1, vec2, vec3, vec4, vec5]
    models_ens = [m1, m2, m3, m4, m5]
    scalers_ens = [None, None, None, None, scaler]
    
    te_texts = [vec.transform(te[text_col]) for vec in [vec1, vec2, vec3, vec4]]
    X_te5_ens = hstack([vec5.transform(te[text_col]), csr_matrix(scaler.transform(te[['char_count', 'word_count']]))])
    te_texts.append(X_te5_ens)
    
    all_probas = []
    for i, clf in enumerate(models_ens):
        proba = clf.predict_proba(te_texts[i])
        all_probas.append(proba)
    avg_proba = np.mean(all_probas, axis=0)
    ensemble_preds = models_ens[0].classes_[avg_proba.argmax(axis=1)]
    acc_ens = accuracy_score(te['sender'], ensemble_preds)
    
    ens_weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    save_model({"models": models_ens, "vectorizers": vec_ens, "scalers": scalers_ens,
                "weights": ens_weights, "author_list": authors, "accuracy": acc_ens},
               f"ensemble_{variant_name}")
    model_results[f"ensemble_{variant_name}"] = {"accuracy": acc_ens}

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("ACCURACY SUMMARY")
print("=" * 60)
results_rows = []
for name, res in model_results.items():
    results_rows.append({"Model": name, "Accuracy": f"{res['accuracy']:.1%}"})
results_df = pd.DataFrame(results_rows)
pivot = results_df.copy()
pivot[['Name', 'Variant']] = pivot['Model'].str.rsplit('_', n=1, expand=True)
pivot = pivot.pivot(index='Name', columns='Variant', values='Accuracy')
print(pivot.to_string())
