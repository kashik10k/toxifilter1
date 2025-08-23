# text_sanitizer.py
from typing import List, Tuple, Dict, Any
from preprocess import clean
from word_match import exact_spans
from fuzzy_match import fuzzy_spans
from escalation import Escalator, EscalationConfig
from suggestions import get_suggestions, replace_spans_with_suggestions

Span = Tuple[str, int, int]
SpanT = Tuple[str, int, int, str]


def _normalize_span(text: str, a: int, b: int) -> tuple[int, int]:
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
    predictor.predict(text) -> {
      'toxic_prob': float, 'toxic': 0|1,
      'sexual_prob': float, 'sexual': 0|1
    }
    """
    def __init__(
        self,
        predictor,
        mask: str = "****",
        prob_gate: float = 0.60,
        escalator: Escalator | None = None,
        esc_cfg: EscalationConfig | None = None,
    ):
        self.predictor = predictor
        self.mask = mask
        self.gate = float(prob_gate)
        self.esc = escalator or Escalator(esc_cfg or EscalationConfig())

    def analyze(self, text: str) -> Dict[str, Any]:
        t = clean(text)
        rx_spans = _attach_terms(t, exact_spans(t) + fuzzy_spans(t))
        rx_spans = merge_spans(rx_spans)
        ml = self.predictor.predict(t)
        flagged = bool(rx_spans) or (ml["toxic_prob"] >= self.gate) or (ml["sexual_prob"] >= self.gate)

        terms = [tok for _, _, _, tok in rx_spans]
        level = self.esc.level_for_terms(terms) if terms else "suggest"

        suggestions: Dict[str, list[str]] = {}
        for term in terms:
            sugs = get_suggestions(term)
            if sugs:
                suggestions[term] = sugs

        return {
            "clean": t,
            "regex_spans": rx_spans,
            "ml": ml,
            "flagged": flagged,
            "level": level,
            "suggestions": suggestions,
        }

    def _mask_with_spacing(self, base: str, spans: List[SpanT]) -> str:
        s = list(base)
        for _, a, b, _ in reversed(spans):
            prev_ch = base[a - 1] if a > 0 else ""
            next_ch = base[b] if b < len(base) else ""
            left_space  = " " if prev_ch and (not prev_ch.isspace()) and prev_ch not in ",.;:!?" else ""
            right_space = " " if next_ch and (not next_ch.isspace()) and next_ch not in ",.;:!?" else ""
            s[a:b] = left_space + self.mask + right_space
        return "".join(s)

    def sanitize(self, text: str) -> Dict[str, Any]:
        info = self.analyze(text)
        spans = info["regex_spans"]
        terms = [t for _, _, _, t in spans]

        if not spans and not info["flagged"]:
            info["action"] = "pass"
            info["sanitized"] = info["clean"]
            return info

        level = info["level"]

        if level == "suggest":
            info["action"] = "suggest"
            info["sanitized"] = info["clean"]
        elif level == "warn":
            info["action"] = "warn"
            new_text, applied = replace_spans_with_suggestions(info["clean"], spans)
            info["applied"] = applied
            info["sanitized"] = new_text
        else:
            info["action"] = "enforce"
            info["sanitized"] = self._mask_with_spacing(info["clean"], spans)

        if terms:
            self.esc.register(terms)

        return info
