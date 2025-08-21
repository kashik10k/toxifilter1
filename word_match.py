# word_match.py
import re
import pathlib
from typing import List, Tuple

BASE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_LEX_FOLDER = BASE_DIR / "dataset" / "lexicon"

def _load_list(path: pathlib.Path) -> list[str]:
    if not path.exists():
        return []
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if ln.strip()
    ]

def load_terms(folder: str | pathlib.Path = DEFAULT_LEX_FOLDER) -> tuple[list[str], list[str]]:
    folder = pathlib.Path(folder)
    toxic  = _load_list(folder / "toxic.txt")
    sexual = _load_list(folder / "sexual.txt")
    return toxic, sexual

def build_regex(terms: list[str]) -> re.Pattern | None:
    if not terms:
        return None
    esc = [re.escape(t) for t in terms]
    pat = r"\b(" + "|".join(esc) + r")(?:s|es|ing|ed)?\b"
    return re.compile(pat, re.IGNORECASE)

toxic_terms, sexual_terms = load_terms()
RX_TOXIC  = build_regex(toxic_terms)
RX_SEXUAL = build_regex(sexual_terms)

def exact_spans(text: str) -> List[Tuple[str, int, int]]:
    spans: list[tuple[str,int,int]] = []
    if RX_TOXIC:
        spans += [("toxic", m.start(), m.end()) for m in RX_TOXIC.finditer(text)]
    if RX_SEXUAL:
        spans += [("sexual", m.start(), m.end()) for m in RX_SEXUAL.finditer(text)]
    spans.sort(key=lambda x: (x[1], x[2], x[0]))
    return spans
