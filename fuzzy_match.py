# fuzzy_match.py
import re
import unicodedata
from typing import List, Tuple
from word_match import toxic_terms, sexual_terms  # reuse your lexicon

# Map simple obfuscations: digits/symbols commonly used as letters
SUBS = {
    "a": "[a@^*]",
    "b": "[b8*]",
    "c": "[c(\\[]",
    "e": "[e3*]",
    "g": "[g9*]",
    "i": "[i1!|*]",
    "l": "[l1|!*]",
    "o": "[o0*]",
    "s": "[s$5z*]",
    "t": "[t7+*]",
    "u": "[u*v]",
    "v": "[v\\/*]",
    "x": "[x%*]",
    "z": "[z2*]",
}

# Allow short separators between characters (., -, _, spaces, zero-width, etc.)
SEP = r"[^a-z0-9]{0,2}"

def _nfkc(x: str) -> str:
    return unicodedata.normalize("NFKC", x)

def obfus_pattern(term: str) -> str:
    """
    Build a permissive regex for a term:
    - substitute per-character classes (SUBS) or the literal char
    - allow short separators between characters
    - tolerate single repeats (e.g., f***uck)
    """
    term = term.lower()
    parts: list[str] = []
    for ch in term:
        cls = SUBS.get(ch, re.escape(ch))
        # one or more of the class to absorb repeats, then optional separators
        parts.append(f"(?:{cls})+{SEP}")
    return "".join(parts)

def build_patterns(terms: List[str]) -> List[re.Pattern]:
    pats: List[re.Pattern] = []
    for t in terms:
        if not t or len(t) < 2:  # skip 1-char tokens
            continue
        pat = obfus_pattern(t)
        # word-ish boundaries: avoid eating long alnum runs around
        pats.append(re.compile(pat, re.IGNORECASE))
    return pats

PAT_TOXIC  = build_patterns(toxic_terms)
PAT_SEXUAL = build_patterns(sexual_terms)

def _scan(text: str, patterns: List[re.Pattern], label: str) -> List[Tuple[str,int,int]]:
    spans: list[Tuple[str,int,int]] = []
    for p in patterns:
        for m in p.finditer(text):
            a, b = m.start(), m.end()
            # tiny guard: ignore pure separators
            if a < b:
                spans.append((label, a, b))
    return spans

def fuzzy_spans(text: str) -> List[Tuple[str,int,int]]:
    """
    Return [(label,start,end)] for obfuscated matches from the lexicon.
    Run this on NFKC-normalized text (D1 already does it).
    """
    t = _nfkc(text)
    spans: list[Tuple[str,int,int]] = []
    if PAT_TOXIC:
        spans.extend(_scan(t, PAT_TOXIC, "toxic"))
    if PAT_SEXUAL:
        spans.extend(_scan(t, PAT_SEXUAL, "sexual"))
    # sort and dedupe overlapping duplicates
    spans.sort(key=lambda x: (x[1], x[2], x[0]))
    dedup: list[Tuple[str,int,int]] = []
    for lab, a, b in spans:
        if not dedup or a > dedup[-1][2]:
            dedup.append((lab, a, b))
        else:
            # merge overlap
            la, aa, bb = dedup[-1]
            dedup[-1] = (la, aa, max(bb, b))
    return dedup
