# sentence_model_com.py
import torch
from sentence_model import SentenceModel

class MultiDetector:
    def __init__(
        self,
        toxic_model_dir: str = "toxic_model",
        sexual_model_dir: str = "sexual_model",
        toxic_pt_path: str = "model/custom_toxic_model.pt",
        sexual_pt_path: str = "model/custom_sexual_model.pt",
        threshold: float = 0.55,
        max_length: int = 256,
    ):
        self.toxic = SentenceModel(
            model_dir=toxic_model_dir,
            pt_path=toxic_pt_path,
            threshold=threshold,
            max_length=max_length,
        )
        self.sexual = SentenceModel(
            model_dir=sexual_model_dir,
            pt_path=sexual_pt_path,
            threshold=threshold,
            max_length=max_length,
        )

    @torch.inference_mode()
    def predict(self, text: str) -> dict:
        t = self.toxic.predict(text)
        s = self.sexual.predict(text)
        return {
            "toxic_prob": t["prob"],
            "toxic": t["label"],
            "sexual_prob": s["prob"],
            "sexual": s["label"],
        }

if __name__ == "__main__":
    md = MultiDetector()
    for x in ["I hate you, idiot.", "18+ hot v!deos watch now", "Have a nice day"]:
        print(x, "->", md.predict(x))
