from sentence_model_com import MultiDetector
from text_sanitizer import Sanitizer

if __name__ == "__main__":
    md = MultiDetector(  # uses your local model dirs/pt paths
        toxic_model_dir="toxic_model",
        sexual_model_dir="sexual_model",
        toxic_pt_path="model/custom_toxic_model.pt",
        sexual_pt_path="model/custom_sexual_model.pt",
        threshold=0.55,
    )
    san = Sanitizer(md, mask="****", prob_gate=0.60)

    for x in [
        "You idiot, stop it.",
        "18+ hot v!deos watch now",
        "Let's have s3x",
        "Have a nice day",
    ]:
        r = san.sanitize(x)
        print("---")
        print("IN :", x)
        print("OUT:", r["sanitized"])
        print("SPANS:", r["regex_spans"])
        print("ML:", r["ml"])
        print("FLAGGED:", r["flagged"])
