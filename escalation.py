from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal

Level = Literal["suggest", "warn", "enforce"]

@dataclass
class EscalationConfig:
    suggest_threshold: int = 1
    warn_threshold: int = 2
    enforce_threshold: int = 3
    decay_after_keystrokes: int = 800

class Escalator:
    def __init__(self, cfg: EscalationConfig | None = None):
        self.cfg = cfg or EscalationConfig()
        self._counts = defaultdict(int)
        self._keystrokes = 0

    def tick_keystroke(self, n: int = 1) -> None:
        self._keystrokes += n
        if self._keystrokes >= self.cfg.decay_after_keystrokes:
            self.reset()

    def reset(self) -> None:
        self._counts.clear()
        self._keystrokes = 0

    def register(self, terms: Iterable[str]) -> None:
        for t in terms:
            if t:
                self._counts[t.lower()] += 1

    def level_for_term(self, term: str) -> Level:
        c = self._counts[term.lower()]
        if c < self.cfg.suggest_threshold: return "suggest"
        if c < self.cfg.warn_threshold:    return "warn"
        return "enforce"

    def level_for_terms(self, terms: Iterable[str]) -> Level:
        order = {"suggest": 0, "warn": 1, "enforce": 2}
        lvl = "suggest"
        for t in {x.lower() for x in terms if x}:
            lt = self.level_for_term(t)
            if order[lt] > order[lvl]:
                lvl = lt
        return lvl
