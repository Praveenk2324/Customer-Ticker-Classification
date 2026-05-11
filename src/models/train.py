import torch
import os
import json
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score
import mlflow

# --- GPU Verification ---
print("\n--- Hardware Check ---")
if torch.cuda.is_available():
    print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)} GB")
else:
    print("❌ GPU NOT DETECTED! PyTorch is running on CPU.")
print("----------------------\n")

# --- Configuration Paths ---
TOKENIZED_DIR = "data/tokenized/"
MAPPING_PATH = "models/label_mapping.json"
MODEL_OUTPUT_DIR = "models/ticket_classifier"
MODEL_NAME = "distilbert-base-uncased"

os.environ["MLFLOW_EXPERIMENT_NAME"] = "IT_Support_Ticket_Classification"

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # We use 'weighted' F1 to account for any remaining class imbalance
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    
    # Make sure this return dictionary has the exact keys "accuracy" and "f1"
    return {
        "accuracy": acc, 
        "f1": f1
    }

def run_training():
    print("1. Loading label mappings...")
    with open(MAPPING_PATH, 'r') as f:
        mappings = json.load(f)
    
    label2id = mappings["label2id"]
    id2label = {int(k): v for k, v in mappings["id2label"].items()}
    num_labels = len(label2id)

    print(f"   -f Found {num_labels} categories.")

    print(f"2. Loading tokenized dataset from {TOKENIZED_DIR}...")
    tokenized_datasets = load_from_disk(TOKENIZED_DIR)

    print(f"3. Initializing pre-trained model ({MODEL_NAME})...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label
    )

    print("4. Setting up Training Arguments...")
    training_args = TrainingArguments(
        output_dir="models/checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="mlflow",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics,
    )

    print("\n🚀 5. Starting Training Phase (This may take a while depending on your GPU/CPU)...")

    with mlflow.start_run():
        trainer.train()
        print(f"\n6. Saving the best model to {MODEL_OUTPUT_DIR}...")
        trainer.save_model(MODEL_OUTPUT_DIR)

        mlflow.transformers.log_model(
            transformers_model={"model": trainer.model, "tokenizer":trainer.tokenizer},
            artifact_path="huggingface_model"
        )

    print(f"✅ Success! Model is trained, saved, and logged to MLflow.")
    print("To view training graphs, run this in your terminal: mlflow ui")

if __name__ == "__main__":
    run_training()