import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load tokenizer (same one used during training)
tokenizer = BertTokenizer.from_pretrained("toxic_model")

# Step 1: Create model architecture
model = BertForSequenceClassification.from_pretrained(
    "toxic_model"
)

# Step 2: Load trained weights
state_dict = torch.load("model/custom_toxic_model.pt", map_location=torch.device("cpu"))
model.load_state_dict(state_dict)

# Step 3: Set model to evaluation mode
model.eval()

# Prediction function
def predict_toxicity(text):
    # Tokenize input
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
    return "toxic" if predicted_class == 1 else "non-toxic"

if __name__ == "__main__":
    try:
        with open("input.txt", "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("❌ input.txt not found! Please create it in the same folder.")
        exit()

    for line in lines:
        print(f"Text: {line}")
        print(f"Prediction: {predict_toxicity(line)}\n")