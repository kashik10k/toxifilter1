# tests/test_text_sanitizer.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from text_sanitizer import Sanitizer
from escalation import Escalator, EscalationConfig

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
    # Force enforce on first occurrence for deterministic masking
    esc = Escalator(EscalationConfig(suggest_threshold=0, warn_threshold=0, enforce_threshold=0))
    san = Sanitizer(Dummy(), mask="****", prob_gate=0.6, escalator=esc)

    s = "You idiot, watch v!deos about s3x"
    out = san.sanitize(s)

    assert out["regex_spans"]            # fuzzy/exact caught terms
    assert "****" in out["sanitized"]    # masked on first call due to thresholds
    assert out["flagged"] is True

def test_neutral_passes():
    san = Sanitizer(Dummy(), mask="****", prob_gate=0.6)
    out = san.sanitize("Have a nice day")
    assert out["regex_spans"] == []
    assert out["sanitized"] == "Have a nice day"
    assert out["flagged"] is False
