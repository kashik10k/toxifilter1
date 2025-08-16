import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # ✅ Correct import
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


# Load dataset
df = pd.read_csv("C:/Projects/keyboard1/dataset/combined_dataset.csv")

# Drop rows with missing or invalid text or labels
df = df.dropna(subset=['text', 'label'])
df['text'] = df['text'].astype(str)
df['label'] = df['label'].astype(int)


# Split into train and validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)


# TEMP: Limit dataset size for debugging/troubleshooting
train_texts = train_texts[:1000]
train_labels = train_labels[:1000]
val_texts = val_texts[:200]
val_labels = val_labels[:200]


# Tokenize
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

# Custom Dataset
class ToxicDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

# Prepare datasets and dataloaders
train_dataset = ToxicDataset(train_encodings, list(train_labels))
val_dataset = ToxicDataset(val_encodings, list(val_labels))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(3):  # Train for 3 epochs
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())

# Evaluation
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(batch['labels'].cpu().numpy())

acc = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

print(f"\n✅ Validation Accuracy: {acc:.4f}")
print(f"✅ Validation F1 Score: {f1:.4f}")

# Save model
model.save_pretrained("toxic_model")
tokenizer.save_pretrained("toxic_model")
print("✅ Model saved to: toxic_model/")
