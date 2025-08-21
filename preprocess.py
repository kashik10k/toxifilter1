# preprocess.py
import re, unicodedata
from typing import Iterable

# Zero-width and formatting/bidi marks to drop
ZW_CHARS = {
    "\u200b",  # ZWSP
    "\u200c",  # ZWNJ
    "\u200d",  # ZWJ
    "\u200e",  # LRM
    "\u200f",  # RLM
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",  # embedding/override
    "\u2060",  # word joiner
    "\u2066", "\u2067", "\u2068", "\u2069",            # LRI/RLI/FSI/PDI
    "\ufeff",  # BOM
}

WS = re.compile(r"\s+")
URL = re.compile(r"""(?i)\bhttps?://[^\s]+|www\.[^\s]+""")
EMAIL = re.compile(r"""(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b""")
CTRL = re.compile(r"[\u0000-\u001f\u007f]")

def _strip_zw(text: str) -> str:
    return "".join(ch for ch in text if ch not in ZW_CHARS)


def _normalize_unicode(text: str) -> str:
    # NFKC folds common lookalikes and width variants
    return unicodedata.normalize("NFKC", text)

def _normalize_quotes(text: str) -> str:
    # Optional light quote normalization for consistency
    return (text
            .replace("“", '"').replace("”", '"')
            .replace("‘", "'").replace("’", "'"))

def clean(
    text: str,
    *,
    lowercase: bool = False,
    strip_urls: bool = False,
    strip_emails: bool = False,
    collapse_ws: bool = True,
) -> str:
    """
    Minimal, safe normalization used before regex and models.

    Steps:
      1) Unicode NFKC
      2) Drop zero-width chars and ASCII control chars
      3) Normalize curly quotes
      4) Optional: strip URLs/emails
      5) Optional: lowercase
      6) Optional: collapse whitespace to single spaces and trim

    Defaults avoid changing semantics that models rely on.
    """
    if not isinstance(text, str):
        text = str(text)

    x = _normalize_unicode(text)
    x = _strip_zw(x)
    x = CTRL.sub("", x)
    x = _normalize_quotes(x)

    if strip_urls:
        x = URL.sub("", x)
    if strip_emails:
        x = EMAIL.sub("", x)

    if lowercase:
        x = x.lower()

    if collapse_ws:
        x = WS.sub(" ", x).strip()

    return x

def batch_clean(texts: Iterable[str], **kwargs) -> list[str]:
    """Vectorized helper for lists of strings."""
    return [clean(t, **kwargs) for t in texts]
