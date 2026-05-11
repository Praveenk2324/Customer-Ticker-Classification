from fastapi import FastAPI
from pydantic import BaseModel
import torch
import os
from fastapi.responses import HTMLResponse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Initialize the API
app = FastAPI(
    title="IT Support Ticket Classifier",
    description="AI model to automatically route IT support tickets to the correct department."
)

# Load the saved model and tokenizer from disk
MODEL_PATH = "models/ticket_classifier"
TOKENIZER_PATH = "models/ticket_tokenizer"

print("🧠 Loading AI Model into Memory...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval() # Put the model in evaluation (prediction) mode

# Define the data structure we expect from users
class TicketRequest(BaseModel):
    ticket_text: str

@app.get("/", response_class=HTMLResponse)
def home():
    # Read the HTML file and return it
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
def predict_department(request: TicketRequest):
    # 1. Convert the text into numbers
    inputs = tokenizer(
        request.ticket_text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=128
    )
    
    # 2. Feed the numbers into the model
    with torch.no_grad(): # Tell PyTorch not to calculate gradients (saves RAM)
        outputs = model(**inputs)
        
    # 3. Find the category with the highest confidence score
    logits = outputs.logits
    predicted_class_id = logits.argmax(-1).item()
    
    # 4. Convert the ID back to the text name (e.g., ID 2 -> "IT Security")
    department_name = model.config.id2label[predicted_class_id]
    
    return {
        "original_text": request.ticket_text,
        "routed_department": department_name
    }