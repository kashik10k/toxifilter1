# keyboard_hook.py
import time, string, keyboard
from typing import Dict, Any, Optional, List

# --- your modules ---
from text_sanitizer import Sanitizer
import notifier  # must expose: select(term:str, options:list[str]) -> str | None

# ---- Predictor factory ----
def get_predictor():
    """
    Returns an object with: predict(text:str)->{
        'toxic_prob': float,'toxic':0|1,'sexual_prob': float,'sexual':0|1
    }
    """

    # --- TEMP: dummy predictor for quick testing ---
    class DummyPredictor:
        def predict(self, text: str):
            # Always safe unless regex hits
            return {
                "toxic_prob": 0.0,
                "toxic": 0,
                "sexual_prob": 0.0,
                "sexual": 0,
            }
    return DummyPredictor()

    # --- LATER: switch to your real predictor ---
    # from sentence_model_com import CombinedPredictor
    # return CombinedPredictor(
    #     toxic_weights="model/custom_toxic_model.pt",
    #     sexual_weights="model/custom_sexual_model.pt"
    # )


DELIMS = set(string.whitespace + ".!?;:,")
BUFFER_MAX = 256
COOLDOWN_SEC = 2.0
MASK_FALLBACK = "****"  # will be overwritten by sanitizer.mask


class KeyboardHook:
    def __init__(self):
        self.buf: List[str] = []
        self.last_ts = 0.0
        self.san = Sanitizer(predictor=get_predictor())
        self.mask_token = getattr(self.san, "mask", MASK_FALLBACK)

    def run(self):
        print("Keyboard hook running. Ctrl+Shift+Q to quit.")
        keyboard.on_press(self._on_key)
        keyboard.add_hotkey("ctrl+shift+q", lambda: exit(0))
        while True:
            time.sleep(0.2)

    # ------------ internals ------------
    def _on_key(self, e: keyboard.KeyboardEvent):
        if e.event_type != "down":
            return
        name = e.name

        if len(name) == 1: 
            self.buf.append(name)
        elif name == "space": 
            self.buf.append(" ")
        elif name == "enter": 
            self.buf.append("\n")
        elif name == "tab": 
            self.buf.append("\t")
        elif name == "backspace":
            if self.buf: 
                self.buf.pop()
            return
        else:
            return

        if len(self.buf) > BUFFER_MAX:
            self.buf = self.buf[-BUFFER_MAX:]

        if self.buf and self.buf[-1] in DELIMS:
            self._process_tail()

    def _process_tail(self):
        now = time.time()
        if now - self.last_ts < COOLDOWN_SEC:
            return

        text = "".join(self.buf)
        info: Dict[str, Any] = self.san.sanitize(text)
        action = info.get("action", "pass")
        if action == "pass":
            return

        spans = info.get("regex_spans", [])
        if not spans:
            return

        # pick the last span (most recently typed)
        lab, a, b, tok = spans[-1]
        s, e = int(a), int(b)
        suggestions_map: Dict[str, list] = info.get("suggestions", {})
        options = suggestions_map.get(tok, [])

        # choose replacement
        repl = self._choose_replacement(action, tok, options)
        if repl is None:
            return

        # delete original span
        for _ in range(e - s):
            keyboard.send("backspace")
        # type replacement
        if repl:
            keyboard.write(repl, delay=0)

        # mirror buffer
        new_text = text[:s] + repl + text[e:]
        self.buf = list(new_text[-BUFFER_MAX:])

        # register with escalator (your Sanitizer already holds Escalator)
        try:
            self.san.esc.register([tok])
        except Exception:
            pass

        self.last_ts = now

    def _choose_replacement(self, action: str, term: str, options: list) -> Optional[str]:
        """
        suggest/warn: ask user; if none chosen, no-op.
        enforce: mask.
        """
        if options:
            choice = notifier.select(term, options[:3])
            if choice is not None:
                return choice
            if action in ("suggest", "warn"):
                return None

        if action == "enforce":
            return self.mask_token  # hard mask
        return None
