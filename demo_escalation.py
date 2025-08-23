from sentence_model_com import MultiDetector
from text_sanitizer import Sanitizer
from escalation import Escalator, EscalationConfig

md = MultiDetector()
san = Sanitizer(md)
esc = Escalator(EscalationConfig())

def run_once(text: str):
    res = san.analyze(text)
    terms = [t for _,_,_,t in res["regex_spans"]]
    if not terms and not res["flagged"]:
        print("OK:", text); return
    level = esc.level_for_terms(terms)
    print(f"{level.upper()} -> terms={terms}")
    if level == "enforce":
        print("OUT:", san.sanitize(text)["sanitized"])
    esc.register(terms)

for s in ["I want sex", "I want sex", "I want sex"]:
    run_once(s)
