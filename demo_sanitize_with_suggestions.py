# demo_sanitize_with_suggestions.py
from sentence_model_com import MultiDetector
from text_sanitizer import Sanitizer
from escalation import Escalator

if __name__ == "__main__":
    md = MultiDetector(
        toxic_model_dir="toxic_model",
        sexual_model_dir="sexual_model",
        toxic_pt_path="model/custom_toxic_model.pt",
        sexual_pt_path="model/custom_sexual_model.pt",
        threshold=0.55,
    )
    esc = Escalator()                 # shared counter
    san = Sanitizer(md, escalator=esc, mask="****", prob_gate=0.60)

    samples = [
        "You idiot, stop it.",
        "Let's have sex",
        "Watch these hot v!deos",
        "Have a nice day",
    ]

    for s in samples:
        print("===")
        print("SAMPLE:", s)
        last = None
        for i in range(1, 4):         # 1→suggest, 2→warn, 3→enforce
            info = san.sanitize(s)
            last = info
            print(f"[{i}] ACTION={info['action']} LEVEL={info['level']}")
            if info["action"] in ("warn", "enforce"):
                print("OUT:", info["sanitized"])
            if info["suggestions"]:
                print("SUGGESTIONS:", info["suggestions"])
        # summary after three passes
        print("SPANS:", last["regex_spans"])
