import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from text_sanitizer import Sanitizer

class Dummy:
    def predict(self, text):
        tl = text.lower()
        return {
            "toxic_prob": 0.9 if "idiot" in tl else 0.01,
            "toxic": 1 if "idiot" in tl else 0,
            "sexual_prob": 0.9 if any(k in tl for k in ["sex","s3x","porn"]) else 0.01,
            "sexual": 1 if any(k in tl for k in ["sex","s3x","porn"]) else 0
        }

def test_mask_exact_and_fuzzy():
    san = Sanitizer(Dummy(), mask="****", prob_gate=0.6)
    s = "You idiot, watch v!deos about s3x"
    out = san.sanitize(s)
    assert out["regex_spans"]      # fuzzy caught something
    assert "****" in out["sanitized"]
    assert out["flagged"] is True

def test_neutral_passes():
    san = Sanitizer(Dummy(), mask="****", prob_gate=0.6)
    out = san.sanitize("Have a nice day")
    assert out["regex_spans"] == []
    assert out["sanitized"] == "Have a nice day"
    assert out["flagged"] is False
