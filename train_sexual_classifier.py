# train_sexual_standalone.py
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

# ------------------ Config ------------------
MODEL_NAME = "bert-base-uncased"
OUT_DIR = "sexual_model"
PT_OUT = "model/custom_sexual_model.pt"
SEED = 42
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ Utils ------------------
def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def resolve_csv_path() -> str:
    here = Path(__file__).resolve().parent
    preferred = here / "dataset" / "sexual_binary_prefilled.csv"
    if preferred.exists():
        print(f"[INFO] Using dataset: {preferred}")
        return str(preferred)
    alt = here / "sexual_binary_prefilled.csv"
    if alt.exists():
        print(f"[INFO] Using dataset: {alt}")
        return str(alt)
    # fallback: any csv under dataset
    dataset_dir = here / "dataset"
    if dataset_dir.exists():
        for p in dataset_dir.glob("*.csv"):
            print(f"[INFO] Using dataset: {p}")
            return str(p)
    raise FileNotFoundError(
        "Could not find dataset. Put CSV at:\n"
        f"  {preferred}\n  or {alt}\n  or any .csv under {dataset_dir}"
    )

def stratified_split(df, test_size=0.10, val_size=0.10):
    train_val, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=SEED)
    val_ratio = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=val_ratio, stratify=train_val["label"], random_state=SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

class TxtDS(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tok = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        enc = self.tok(
            self.texts[i],
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None,
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": self.labels[i]}

def make_dataloader(df, tokenizer, batch_size, shuffle, collator):
    ds = TxtDS(df, tokenizer, MAX_LEN)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)

def evaluate(model, dataloader):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: torch.tensor(v).to(DEVICE) if not torch.is_tensor(v) else v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

# ------------------ Train ------------------
def main():
    set_seed()
    os.makedirs("model", exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    csv_path = resolve_csv_path()
    df = pd.read_csv(csv_path).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int).clip(0, 1)
    df = df[df["text"].astype(str).str.len() >= 2].reset_index(drop=True)

    counts = df["label"].value_counts().to_dict()
    neg, pos = counts.get(0, 1), counts.get(1, 1)
    print(f"[INFO] samples: total={len(df)} neg(0)={neg} pos(1)={pos}")
    class_weights = torch.tensor([1.0/neg, 1.0/pos], dtype=torch.float32).to(DEVICE)
    print(f"[INFO] class_weights={class_weights.tolist()} (higher = rarer class)")

    train_df, val_df, test_df = stratified_split(df)
    print(f"[INFO] split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = make_dataloader(train_df, tokenizer, BATCH_SIZE, True, collator)
    val_loader   = make_dataloader(val_df, tokenizer, BATCH_SIZE*2, False, collator)
    test_loader  = make_dataloader(test_df, tokenizer, BATCH_SIZE*2, False, collator)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_f1, best_state = -1.0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: torch.tensor(v).to(DEVICE) if not torch.is_tensor(v) else v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            running_loss += loss.item()

            if step % 100 == 0:
                print(f"[Epoch {epoch}] step {step}/{len(train_loader)} - loss: {running_loss/step:.4f}")

        # Evaluate after each epoch
        val_metrics = evaluate(model, val_loader)
        print(f"[VAL] Epoch {epoch}: {val_metrics}")

        # Track best
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best state (if captured)
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test
    test_metrics = evaluate(model, test_loader)
    print(f"[TEST] {test_metrics}")

    # Save
    tokenizer.save_pretrained(OUT_DIR)
    model.save_pretrained(OUT_DIR)
    torch.save(model.state_dict(), PT_OUT)
    print(f"[SAVED] folder={OUT_DIR}")
    print(f"[SAVED] weights={PT_OUT}")

if __name__ == "__main__":
    main()
