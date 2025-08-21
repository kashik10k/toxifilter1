import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fuzzy_match import fuzzy_spans

def extract(text):
    return [(l, text[a:b]) for l,a,b in fuzzy_spans(text)]

def test_basic_obfuscations():
    s = "f*ck v!deos s3x p0rn"
    found = extract(s)
    labels = [l for l,_ in found]
    assert labels.count("sexual") >= 1  # at least one sexual hit

def test_separators_and_repeats():
    s = "f---u***ck  vi..de_os  s e x"
    labels = [l for l,_ in extract(s)]
    assert labels.count("sexual") >= 1

def test_nfkc_confusables():
    s = "fu—ck you"  # em dash between u and ck after NFKC it's still punctuation
    spans = extract(s)
    assert any(lab == "sexual" for lab,_ in spans)
