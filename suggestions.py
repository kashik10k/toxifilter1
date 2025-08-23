# suggestions.py
from typing import List, Tuple, Dict

# Canonical, lowercase keys
SUGGESTIONS: Dict[str, List[str]] = {
    # sexual
    "sex": ["intimacy"],
    "porn": ["adult content"],
    "fuck": ["mess up"],  # treat as sexual/explicit for your flow
    "videos": ["clips"],

    # toxic
    "idiot": ["uninformed person", "rude person"],
    "bitch": ["rude person"],
    "stupid": ["unwise"],
    "hate": ["dislike"],
}

def get_suggestions(term: str) -> List[str]:
    return SUGGESTIONS.get(term.lower(), [])

def pick_replacement(term: str) -> str | None:
    lst = get_suggestions(term)
    return lst[0] if lst else None

def _preserve_case(src: str, repl: str) -> str:
    if src.isupper():
        return repl.upper()
    if src.istitle():
        return repl.title()
    return repl

def replace_spans_with_suggestions(
    text: str,
    spans: List[Tuple[str, int, int, str]],  # (label,start,end,term) from sanitizer
) -> tuple[str, Dict[str, str]]:
    """
    Replace each span with a safe synonym if available.
    Returns (new_text, mapping_of_applied_terms).
    Assumes spans are non-overlapping and trimmed (your sanitizer already does this).
    """
    s = list(text)
    applied: Dict[str, str] = {}
    for _, a, b, term in reversed(spans):
        rep = pick_replacement(term)
        if not rep:
            continue
        rep = _preserve_case(term, rep)
        s[a:b] = rep
        applied[term] = rep
    return "".join(s), applied
