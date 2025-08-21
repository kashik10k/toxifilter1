# sentence_model_com.py
import os, torch
from sentence_model import SentenceModel

class MultiDetector:
    def __init__(
        self,
        toxic_model_dir="toxic_model",
        sexual_model_dir="sexual_model",
        toxic_pt_path="model/custom_toxic_model.pt",
        sexual_pt_path="model/custom_sexual_model.pt",
        threshold=0.55,
    ):
        toxic_pt = toxic_pt_path if os.path.exists(toxic_pt_path) else None
        sexual_pt = sexual_pt_path if os.path.exists(sexual_pt_path) else None

        self.toxic = SentenceModel(
            model_dir=toxic_model_dir, pt_path=toxic_pt, threshold=threshold
        )
        self.sexual = SentenceModel(
            model_dir=sexual_model_dir, pt_path=sexual_pt, threshold=threshold
        )

    @torch.inference_mode()
    def predict(self, text: str) -> dict:
        t = self.toxic.predict(text)
        s = self.sexual.predict(text)
        return {
            "toxic_prob": t["prob"], "toxic": t["label"],
            "sexual_prob": s["prob"], "sexual": s["label"]
        }

if __name__ == "__main__":
    md = MultiDetector()
    samples = [
        "I hate you, idiot.",
        "18+ hot v!deos watch now",
        "Have a nice day"
    ]
    for x in samples:
        print(x)
        print(md.predict(x))
