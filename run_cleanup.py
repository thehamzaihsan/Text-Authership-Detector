import re
import pandas as pd
import numpy as np

URL_RE = r"https?://\S+|www\.\S+"
URL_ONLY_RE = r"(?:https?://\S+|www\.\S+)"
PHONE_RE = r"(?<!\w)(?:\+?\d[\d\-\(\)\s]{7,}\d)(?!\w)"
EMAIL_RE = r"(?<!\w)[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?!\w)"
INVISIBLE_RE = r"[\u200e\u200f\u202a-\u202e\ufeff]"
EMOJI_RE = r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]|[\U0001F1E6-\U0001F1FF]{2}|[\u200d\ufe0f]"

def count_words(text):
    return len(text.split()) if isinstance(text, str) and text.strip() else 0

def clean_model_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(INVISIBLE_RE, "", text).strip()
    text = re.sub(URL_RE, " [URL] ", text)
    text = re.sub(PHONE_RE, " [PHONE] ", text)
    text = re.sub(EMAIL_RE, " [EMAIL] ", text)
    text = re.sub(EMOJI_RE, "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df_raw = pd.read_csv("whatsapp_chat.csv")
raw_message = df_raw["message"].fillna("").astype(str)

structural_mask = (
    (~df_raw.get("is_system", pd.Series(False))) &
    (~df_raw.get("content_type", pd.Series()).isin(["media", "deleted", "link"])) &
    (raw_message.str.strip() != "")
)
df = df_raw.loc[structural_mask].copy()
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
print(f"After structural filter: {len(df)} rows")

sender_counts = df["sender"].value_counts()
keep_senders = sender_counts[sender_counts >= 100].index
df = df[df["sender"].isin(keep_senders)].copy()

# Save raw messages BEFORE resetting index
raw_subset = raw_message.loc[df.index]
df = df.reset_index(drop=True)
raw_subset = raw_subset.reset_index(drop=True)
print(f"After sender filter (>=100): {len(df)} rows, {df['sender'].nunique()} senders")

df["message_clean"] = raw_subset.apply(clean_model_text)
df["message_clean_lower"] = df["message_clean"].str.lower()
df["word_count"] = raw_subset.apply(count_words)
df["char_count"] = df["message_clean"].str.len()

print("Text cleaning complete")

df = df.sort_values(["datetime", "message_id"], kind="mergesort")
n_rows = len(df)
cut = int(n_rows * 0.8)
df = df.reset_index(drop=True)
df['split'] = 'train'
if n_rows > 0:
    df.loc[cut:, 'split'] = 'test'

missing_in_test = sorted(set(df['sender']) - set(df.loc[df['split']=='test', 'sender']))
for s in missing_in_test:
    train_rows = df[(df['sender'] == s) & (df['split'] == 'train')]
    if not train_rows.empty:
        idx = train_rows.sort_values(['datetime', 'message_id'], kind='mergesort').index[-1]
        df.at[idx, 'split'] = 'test'

split_counts = df['split'].value_counts()
print(f"Train: {split_counts.get('train', 0)}, Test: {split_counts.get('test', 0)}")

dup_count = int(df['message_id'].duplicated().sum())
if dup_count > 0:
    print(f'Dropping {dup_count} duplicate message_ids')
    df = df.sort_values(['datetime', 'message_id'], kind='mergesort').drop_duplicates('message_id', keep='first').reset_index(drop=True)

columns = ["message_id", "sender", "word_count", "char_count", "message_clean", "message_clean_lower", "split"]
out_df = df[columns].copy()
out_path = "whatsapp_chat_cleaned.csv"
out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"Saved {out_path} with shape {out_df.shape}")
