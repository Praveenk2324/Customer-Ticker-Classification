import pandas as pd
import re
import os
import json
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = "data/raw/IT Support Ticket Data.csv"
PROCESSED_TRAIN_PATH = "data/preprocessed/train_tickets.csv"
PROCESSED_TEST_PATH = "data/preprocessed/test_tickets.csv"
MAPPING_PATH = "models/label_mapping.json"

# --- Domain Stopwords (From EDA) ---
IT_STOPWORDS = [
    "issue", "support", "customer", "assistance", "information", 
    "provide", "appreciate", "problem", "greatly", "thank", 
    "matter", "resolve", "data", "hello", "hi", "thanks", 
    "please", "regards", "ticket", "help", "team"
]

def clean_text(text):
    """Cleans IT support ticket text for modeling"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    #re.sub(pattern, replacement, string)

    text = re.sub(r'from:.*?\n', '', text) #ex finds from: john.doe@company.com" and deletes the whole line.
    text = re.sub(r'to:.*?\n', '', text)   #removes web links, catching both https://... and www.... formats.
    text = re.sub(r'subject:.*?\n', '', text) #deletes (commas, emojis, dashes, quotes)

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # 3. Remove punctuation (keep only alphanumeric and spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    #Remove custom IT domain stopwords
    words = text.split()
    words = [w for w in words if w not in IT_STOPWORDS]

    # 5. Rejoin and remove extra whitespace
    clean_string = " ".join(words).strip()
    return clean_string

def run_preprocessing():
    print(f"1. Loading raw dataset from {RAW_DATA_PATH}")
    try:
        df = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Could not find {RAW_DATA_PATH}. Make sure CSV is in the data/raw/ folder!")
        return

    df = df[['Body', 'Department']].dropna()

    print("2. Cleaning ticket text and removing domain noise...")
    df['cleaned_text'] = df['Body'].apply(clean_text)

    df = df[df['cleaned_text'].str.len() > 0]
    print("3. Mapping string labels to integer IDs...")
    # Huggingface requires target labels to be integers starting from 0
    unique_labels = df['Department'].unique()
    label2id = {label: int(idx) for idx, label in enumerate(unique_labels)}
    id2label = {int(idx): label for label, idx in label2id.items()}

    df['label'] = df['Department'].map(label2id)

    os.makedirs("models", exist_ok=True)
    with open(MAPPING_PATH, 'w') as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f)
    print(f"   -> Label mapping saved to {MAPPING_PATH}")

    print("4. Splitting dataset into Train and Test (80/20)...")
    #'stratify' to ensure the train/test sets have the same ratio of ticket categories
    train_df, test_df = train_test_split(
        df[['cleaned_text', 'label']],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )

    print("5. Saving  processed files...")
    
    os.makedirs(os.path.dirname(PROCESSED_TRAIN_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSED_TEST_PATH), exist_ok=True)

    train_df.to_csv(PROCESSED_TRAIN_PATH, index=False)
    test_df.to_csv(PROCESSED_TEST_PATH, index=False)

    print(f"✅ Success! Saved {len(train_df)} training and {len(test_df)} testing samples.")

if __name__ == "__main__":
    run_preprocessing()