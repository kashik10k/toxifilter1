import os
import torch
from transformers import BertForSequenceClassification

# Ensure 'model' directory exists
os.makedirs("model", exist_ok=True)

# Load trained toxic model from folder
model = BertForSequenceClassification.from_pretrained("toxic_model")

# Save as PyTorch .pt file
torch.save(model.state_dict(), "model/custom_toxic_model.pt")

print("✅ Toxic model saved as model/custom_toxic_model.pt")

