import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from suggestions import get_suggestions, replace_spans_with_suggestions

def test_has_synonyms():
    assert "intimacy" in get_suggestions("sex")
    assert get_suggestions("unknown") == []

def test_replace_preserves_case_and_positions():
    text = "You IDIOT, watch porn and Sex videos."
    spans = [
        ("toxic", 4, 9, "IDIOT"),
        ("sexual", 17, 21, "porn"),
        ("sexual", 26, 29, "Sex"),
        ("sexual", 30, 36, "videos"),
    ]
    new_text, applied = replace_spans_with_suggestions(text, spans)
    # Case preserved for IDIOT -> UNINFORMED PERSON (upper)
    assert "UNINFORMED PERSON" in new_text
    # porn -> adult content; Sex -> Intimacy; videos -> clips
    assert "adult content" in new_text
    assert "Intimacy" in new_text
    assert "clips" in new_text
    assert applied.get("videos") == "clips"
