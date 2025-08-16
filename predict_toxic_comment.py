import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("toxic_model")
tokenizer = BertTokenizer.from_pretrained("toxic_model")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()

# Inference function
def predict_comment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    return "🧪 Toxic" if prediction == 1 else "✅ Non-Toxic"

# 🔍 Test Examples
examples = [
    "I hate you and everything you stand for.",
    "You're doing an amazing job!",
    "Go to hell, idiot.",
    "Please let me know your feedback on this."
]

for text in examples:
    print(f"{text} => {predict_comment(text)}")
