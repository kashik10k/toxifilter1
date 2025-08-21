import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # add project root

from preprocess import clean

def test_basic_clean():
    s = "  H\u200be\u200cl\u200cl\u200fo   www.example.com  "
    out = clean(s, strip_urls=True)
    assert out == "Hello"
