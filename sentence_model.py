import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentenceModel:
    def __init__(self, model_dir: str, pt_path: str | None = None,
                 device: str | None = None, threshold: float = 0.5,
                 max_length: int = 256):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        if pt_path:
            sd = torch.load(pt_path, map_location="cpu")
            self.model.load_state_dict(sd, strict=False)
        self.model.to(self.device).eval()
        self.threshold = float(threshold)
        self.max_length = int(max_length)

    @torch.inference_mode()
    def predict(self, text: str) -> dict:
        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=False,
            max_length=self.max_length
        ).to(self.device)
        logits = self.model(**enc).logits.squeeze(0)
        p1 = torch.softmax(logits, dim=-1)[1].item()
        return {"prob": p1, "label": int(p1 >= self.threshold)}
