import subprocess
import sys

def run_step(script_path):
    """Helper function to run a script and gracefully catch errors."""
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING STEP: {script_path}")
    print(f"{'='*60}\n")
    
    # sys.executable ensures it uses your exact active Conda environment Python
    result = subprocess.run([sys.executable, script_path])
    
    if result.returncode != 0:
        print(f"\n❌ PIPELINE FAILED AT STEP: {script_path}")
        print("Please check the error logs above and fix the issue before continuing.")
        sys.exit(1) # Stop the pipeline completely

def main():
    print("🌟 Starting End-to-End IT Support Ticket Pipeline 🌟")
    
    # --- Step 1: Preprocessing ---
    # Cleans text, removes domain noise, and splits into train/test
    run_step("src/data/preprocess.py")
    
    # --- Step 2: Feature Engineering ---
    # Loads Huggingface DistilBERT tokenizer and converts text to tensors
    run_step("src/features/build_features.py")
    
    # --- Step 3: Model Training & Tracking ---
    # Fine-tunes the model on GPU (if available) and logs everything to MLflow
    run_step("src/models/train.py")
    
    print("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("Your model is trained, logged, and ready for FastAPI deployment.")

if __name__ == "__main__":
    main()