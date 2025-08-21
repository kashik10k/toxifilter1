import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from word_match import exact_spans

def test_exact_simple():
    s = "You idiot, stop watching porn."
    spans = exact_spans(s)
    labels = [(l, s[a:b]) for l,a,b in spans]
    assert ("toxic", "idiot") in labels
    assert ("sexual", "porn") in labels

def test_suffixes():
    s = "Hating videos and hated idiots?"
    spans = exact_spans(s)
    found = [s[a:b].lower() for _,a,b in spans]
    # 'videos' from sexual list with suffix handling
    assert "videos" in found
    # 'idiots' as 'idiot' plural
    assert "idiots" in found
