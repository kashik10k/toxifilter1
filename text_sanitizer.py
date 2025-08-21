# text_sanitizer.py
from typing import List, Tuple, Dict, Any
from preprocess import clean
from word_match import exact_spans
from fuzzy_match import fuzzy_spans

Span = Tuple[str, int, int]        # (label, start, end)
SpanT = Tuple[str, int, int, str]  # (label, start, end, term)


def _normalize_span(text: str, a: int, b: int) -> tuple[int, int]:
    """Trim whitespace and simple punctuation from span edges."""
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    while a < b and text[a] in ",.;:!?":
        a += 1
    while b > a and text[b - 1] in ",.;:!?":
        b -= 1
    return a, b


def _attach_terms(text: str, spans: List[Span]) -> List[SpanT]:
    out: List[SpanT] = []
    for lab, a, b in spans:
        a, b = _normalize_span(text, a, b)
        if a < b:
            out.append((lab, a, b, text[a:b]))
    return out


def merge_spans(spans: List[SpanT]) -> List[SpanT]:
    """Merge overlapping spans; keep earliest label/term."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (x[1], x[2], x[0]))
    out: List[List[Any]] = []
    for lab, a, b, tok in spans:
        if not out or a > out[-1][2]:
            out.append([lab, a, b, tok])
        else:
            out[-1][2] = max(out[-1][2], b)
    return [(l, s, e, t) for l, s, e, t in out]


class Sanitizer:
    """
    predictor must implement:
      predict(text) -> {
        'toxic_prob': float, 'toxic': 0|1,
        'sexual_prob': float, 'sexual': 0|1
      }
    """
    def __init__(self, predictor, mask: str = "****", prob_gate: float = 0.60):
        self.predictor = predictor
        self.mask = mask
        self.gate = float(prob_gate)

    def analyze(self, text: str) -> Dict[str, Any]:
        t = clean(text)
        rx_spans = _attach_terms(t, exact_spans(t) + fuzzy_spans(t))
        rx_spans = merge_spans(rx_spans)
        ml = self.predictor.predict(t)
        flagged = bool(rx_spans) or (ml["toxic_prob"] >= self.gate) or (ml["sexual_prob"] >= self.gate)
        return {"clean": t, "regex_spans": rx_spans, "ml": ml, "flagged": flagged}

    def sanitize(self, text: str) -> Dict[str, Any]:
        info = self.analyze(text)
        s = list(info["clean"])
        for _, a, b, _ in reversed(info["regex_spans"]):
            left_space = " " if a > 0 and not info["clean"][a - 1].isspace() else ""
            right_space = " " if b < len(info["clean"]) and (b == len(info["clean"]) or not info["clean"][b].isspace()) else ""
            s[a:b] = left_space + self.mask + right_space
        info["sanitized"] = "".join(s)
        return info
