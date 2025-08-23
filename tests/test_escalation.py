import sys, os; sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from escalation import Escalator, EscalationConfig

def test_levels_progress():
    esc = Escalator(EscalationConfig(1,2,3))
    assert esc.level_for_term("fuck") == "suggest"
    esc.register(["fuck"])
    assert esc.level_for_term("fuck") == "warn"
    esc.register(["fuck"])
    assert esc.level_for_term("fuck") == "enforce"
