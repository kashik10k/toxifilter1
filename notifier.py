# notifier.py — GUI popup selector (Tkinter)
from typing import List, Optional
import tkinter as tk

def select(term: str, options: List[str]) -> Optional[str]:
    """
    Modal popup. Returns chosen replacement string.
    Returns "" for Mask. Returns None for Skip.
    """
    opts = options[:3]

    result = {"value": None}

    def choose(val: Optional[str]):
        result["value"] = val
        top.destroy()

    # Create a hidden root
    root = tk.Tk()
    root.withdraw()

    # Modal top-level
    top = tk.Toplevel(root)
    top.title("Toxifilter")
    top.attributes("-topmost", True)
    top.grab_set()  # modal
    top.resizable(False, False)

    # UI
    msg = tk.Label(top, text=f'Flagged: "{term}"\nPick a replacement:')
    msg.pack(padx=12, pady=(12, 6))

    btn_frame = tk.Frame(top)
    btn_frame.pack(padx=12, pady=6)

    # Suggestion buttons
    for i, o in enumerate(opts, 1):
        tk.Button(btn_frame, text=f"{i}. {o}", width=24,
                  command=lambda v=o: choose(v)).grid(row=i-1, column=0, pady=2)

    # Mask and Skip
    aux = tk.Frame(top); aux.pack(padx=12, pady=(6, 12))
    tk.Button(aux, text="Mask", width=10, command=lambda: choose("")).grid(row=0, column=0, padx=4)
    tk.Button(aux, text="Skip", width=10, command=lambda: choose(None)).grid(row=0, column=1, padx=4)



    def on_key(e):
        if e.keysym in ("Escape",):
            choose(None)
        elif e.char == "0":
            choose("")
        elif e.char in ("1", "2", "3"):
            idx = int(e.char) - 1
            if 0 <= idx < len(opts):
                choose(opts[idx])

    top.bind("<Key>", on_key)

    # Center on screen
    top.update_idletasks()
    w, h = top.winfo_width(), top.winfo_height()
    sw, sh = top.winfo_screenwidth(), top.winfo_screenheight()
    top.geometry(f"+{(sw-w)//2}+{(sh-h)//3}")

    # Show and wait
    root.deiconify()
    root.lift()
    top.lift()
    top.focus_force()
    root.wait_window(top)

    root.destroy()
    return result["value"]
