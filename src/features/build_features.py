import os
from datasets import load_dataset
from transformers import AutoTokenizer

# --- Configuration Paths ---
TRAIN_PATH = "data/preprocessed/train_tickets.csv"
TEST_PATH = "data/preprocessed/test_tickets.csv"
TOKENIZED_DIR = "data/tokenized/"

# From our EDA, we decided on a safe max length (adjust if your EDA said otherwise!)
MAX_LENGTH = 128 
MODEL_NAME = "distilbert-base-uncased"

def tokenize_function(examples, tokenizer):
    """
    This function applies the tokenizer to a batch of texts.
    'cleaned_text' is the column name we created in preprocess.py
    """
    return tokenizer(
        examples["cleaned_text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

def build_features():
    print(f"1. Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Save the tokenizer so our FastAPI app can use it later!
    os.makedirs("models", exist_ok=True)
    tokenizer.save_pretrained("models/ticket_tokenizer")
    print("   -> Tokenizer saved to models/ticket_tokenizer")

    print("\n2. Loading processed CSVs into Huggingface Dataset format...")
    # The datasets library handles loading and memory mapping extremely fast
    raw_datasets = load_dataset(
        "csv", 
        data_files={"train": TRAIN_PATH, "test": TEST_PATH}
    )

    print("\n3. Tokenizing datasets (this might take a moment)...")
    # We use .map() to apply tokenization in batches
    tokenized_datasets = raw_datasets.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["cleaned_text"] # We don't need the raw text anymore, only the numbers!
    )
    
    # Huggingface models expect the target column to be named 'labels' (with an 's')
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch") # Format for PyTorch

    print("\n4. Saving tokenized datasets to disk...")
    os.makedirs(TOKENIZED_DIR, exist_ok=True)
    tokenized_datasets.save_to_disk(TOKENIZED_DIR)
    
    print(f"✅ Success! Tokenized data saved to {TOKENIZED_DIR}")
    print("Sample keys of a training item:", list(tokenized_datasets["train"][0].keys()))

if __name__ == "__main__":
    build_features()