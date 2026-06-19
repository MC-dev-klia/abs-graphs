
"""
Cool Shape Randomizer for the nested-absolute-value equation family.

Features:
- Randomize "cool" equations with biased templates.
- Plot the graph with matplotlib in a Tkinter GUI.
- Copy the exact competition-format equation string.
- Paste an equation string back in and reload it.
- Maintain the last 5 generated/loaded equations for quick revisiting.

Run:
    python cool_shape_randomizer.py
"""

from __future__ import annotations

import math
import random
import re
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# ----------------------------
# Equation model / formatting
# ----------------------------

@dataclass(frozen=True)
class EquationParams:
    m1: int
    m2: int
    m3: int
    m4: int
    m5: int
    m6: int
    c1: int
    c2: int
    c3: int
    c4: int
    c5: int
    c6: int
    c7: int
    R: float

    def as_tuple(self):
        return (
            self.m1, self.m2, self.m3, self.m4, self.m5, self.m6,
            self.c1, self.c2, self.c3, self.c4, self.c5, self.c6, self.c7,
            self.R,
        )


def fmt_signed_int(v: int) -> str:
    # Match the user’s style exactly: x-1, x+0, x+-1, etc.
    if v == 0:
        return "+0"
    return f"+{v}" if v > 0 else f"{v}"


def fmt_R(r: float) -> str:
    if float(r).is_integer():
        return str(int(r))
    return str(r).rstrip("0").rstrip(".")


def to_competition_string(p: EquationParams) -> str:
    # Exact structural template from the challenge.
    return (
        rf"\left|{p.m1}"
        rf"\left|{p.m2}\left|x{fmt_signed_int(p.c1)}\right|"
        rf"+{p.m3}\left|y{fmt_signed_int(p.c2)}\right|"
        rf"+{p.c3}\right|"
        rf"+{p.m4}\left|{p.m5}\left|x{fmt_signed_int(p.c4)}\right|"
        rf"+{p.m6}\left|y{fmt_signed_int(p.c5)}\right|"
        rf"+{p.c6}\right|"
        rf"+{p.c7}\right|={fmt_R(p.R)}"
    )


def parse_competition_string(s: str) -> EquationParams:
    nums = re.findall(r"[-+]?\d*\.?\d+", s.replace("−", "-"))
    if len(nums) != 14:
        raise ValueError(
            f"Expected 14 numbers (13 coefficients + R), found {len(nums)}."
        )
    vals = [float(x) for x in nums]
    ints = [int(v) for v in vals[:-1]]
    r = vals[-1]
    return EquationParams(*ints, R=r)


# ----------------------------
# Function evaluation / plotting
# ----------------------------

def f_value(x: np.ndarray, y: np.ndarray, p: EquationParams) -> np.ndarray:
    t1 = p.m2 * np.abs(x + p.c1) + p.m3 * np.abs(y + p.c2) + p.c3
    t2 = p.m5 * np.abs(x + p.c4) + p.m6 * np.abs(y + p.c5) + p.c6
    return np.abs(p.m1 * np.abs(t1) + p.m4 * np.abs(t2) + p.c7)


def choose_grid(params: EquationParams, base_lim: float = 10.0):
    # Slightly widen the view for larger R so rays/large shapes are visible.
    lim = max(base_lim, 2.5 * float(params.R) + 6.0)
    n = 700
    xs = np.linspace(-lim, lim, n)
    ys = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(xs, ys)
    return X, Y, lim


# ----------------------------
# Cool random generation
# ----------------------------

SIZES = [0.5, 1, 1.5, 2, 2.5, 3, 4]


def weighted_choice(values, weights):
    total = sum(weights)
    r = random.random() * total
    acc = 0.0
    for v, w in zip(values, weights):
        acc += w
        if r <= acc:
            return v
    return values[-1]


def pick_m(bias_nonzero: float = 0.8) -> int:
    # More zeros make the graph vanish; bias toward ±1, but keep some zeros.
    return weighted_choice([-1, 0, 1], [bias_nonzero / 2, 1.0 - bias_nonzero, bias_nonzero / 2])


def pick_c(bias_zero: float = 0.30) -> int:
    # A bit more centered than the multipliers, but still varied.
    return weighted_choice([-1, 0, 1], [(1.0 - bias_zero) / 2, bias_zero, (1.0 - bias_zero) / 2])


def template_symmetric() -> EquationParams:
    # Strong symmetry often gives diamonds/squares/octagons/nested forms.
    return EquationParams(
        m1=random.choice([-1, 1]),
        m2=random.choice([-1, 1]),
        m3=random.choice([-1, 1]),
        m4=random.choice([-1, 1]),
        m5=random.choice([-1, 1]),
        m6=random.choice([-1, 1]),
        c1=random.choice([-1, 0, 1]),
        c2=random.choice([-1, 0, 1]),
        c3=random.choice([-1, 0, 1]),
        c4=random.choice([-1, 0, 1]),
        c5=random.choice([-1, 0, 1]),
        c6=random.choice([-1, 0, 1]),
        c7=random.choice([-1, 0, 1]),
        R=random.choice(SIZES),
    )


def template_asymmetric() -> EquationParams:
    # Encourages rocket / banana-peel / weird hybrid shapes.
    ms = [pick_m(0.75) for _ in range(6)]
    cs = [pick_c(0.25) for _ in range(7)]
    # Nudge toward asymmetry by making the two branches different.
    if (ms[0], ms[1], ms[2], cs[0], cs[1], cs[2]) == (ms[3], ms[4], ms[5], cs[3], cs[4], cs[5]):
        ms[3] = -ms[3] if ms[3] != 0 else 1
    return EquationParams(*ms, *cs, R=random.choice(SIZES))


def template_nested() -> EquationParams:
    # Bias toward nested polygons / “shape inside shape”.
    m1 = random.choice([-1, 1])
    m4 = random.choice([-1, 1])
    m2 = random.choice([-1, 1])
    m3 = random.choice([-1, 1])
    m5 = random.choice([-1, 1])
    m6 = random.choice([-1, 1])

    # Offsets aimed at producing structure near the center and a shifted copy.
    c1 = random.choice([-1, 0, 1])
    c2 = random.choice([-1, 0, 1])
    c3 = random.choice([-1, 0, 1])
    c4 = random.choice([-1, 0, 1])
    c5 = random.choice([-1, 0, 1])
    c6 = random.choice([-1, 0, 1])
    c7 = random.choice([-1, 0, 1])

    return EquationParams(m1, m2, m3, m4, m5, m6, c1, c2, c3, c4, c5, c6, c7, random.choice(SIZES))


def template_open() -> EquationParams:
    # More likely to create ray / line / split-branch structures.
    # Allow some zeros but keep the "core" active.
    m1 = random.choice([-1, 1])
    m4 = random.choice([-1, 1])

    m2 = weighted_choice([-1, 0, 1], [0.45, 0.10, 0.45])
    m3 = weighted_choice([-1, 0, 1], [0.45, 0.10, 0.45])
    m5 = weighted_choice([-1, 0, 1], [0.45, 0.10, 0.45])
    m6 = weighted_choice([-1, 0, 1], [0.45, 0.10, 0.45])

    c1 = random.choice([-1, 0, 1])
    c2 = random.choice([-1, 0, 1])
    c3 = random.choice([-1, 0, 1])
    c4 = random.choice([-1, 0, 1])
    c5 = random.choice([-1, 0, 1])
    c6 = random.choice([-1, 0, 1])
    c7 = random.choice([-1, 0, 1])

    return EquationParams(m1, m2, m3, m4, m5, m6, c1, c2, c3, c4, c5, c6, c7, random.choice(SIZES))


TEMPLATES = [
    ("Symmetric", template_symmetric, 0.28),
    ("Asymmetric", template_asymmetric, 0.34),
    ("Nested", template_nested, 0.20),
    ("Open", template_open, 0.18),
]


def random_cool_params() -> EquationParams:
    kind = weighted_choice([name for name, _, _ in TEMPLATES], [w for _, _, w in TEMPLATES])
    for name, fn, _ in TEMPLATES:
        if name == kind:
            p = fn()
            break

    # Light mutation to keep things surprising.
    if random.random() < 0.35:
        fields = list(p.as_tuple())
        idx = random.randrange(13)  # don't mutate R here
        fields[idx] = random.choice([-1, 0, 1])
        # Keep at least some activity.
        if all(v == 0 for v in fields[:6]):
            fields[random.randrange(6)] = random.choice([-1, 1])
        p = EquationParams(*fields[:13], R=p.R)

    if random.random() < 0.25:
        p = EquationParams(*p.as_tuple()[:-1], R=random.choice(SIZES))

    return p


# ----------------------------
# GUI
# ----------------------------

class CoolShapeApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Cool Shape Randomizer")
        self.root.geometry("1250x820")

        self.history: list[EquationParams] = []
        self.current: EquationParams | None = None

        self._build_ui()
        self.generate_new()

    def _build_ui(self):
        # Layout
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=12)
        left.grid(row=0, column=0, sticky="nsw")

        right = ttk.Frame(self.root, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        # Equation display
        ttk.Label(left, text="Equation", font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        self.eq_var = tk.StringVar()
        self.eq_entry = ttk.Entry(left, width=74, textvariable=self.eq_var)
        self.eq_entry.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.eq_entry.bind("<Return>", lambda e: self.load_from_entry())

        btn_row = ttk.Frame(left)
        btn_row.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        for i in range(3):
            btn_row.columnconfigure(i, weight=1)

        ttk.Button(btn_row, text="Randomize Cool Shape", command=self.generate_new).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(btn_row, text="Copy", command=self.copy_current).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(btn_row, text="View Paste", command=self.load_from_entry).grid(
            row=0, column=2, sticky="ew", padx=(4, 0)
        )

        ttk.Separator(left).grid(row=3, column=0, sticky="ew", pady=10)

        ttk.Label(left, text="History (last 5)", font=("TkDefaultFont", 12, "bold")).grid(
            row=4, column=0, sticky="w", pady=(0, 6)
        )
        self.history_frame = ttk.Frame(left)
        self.history_frame.grid(row=5, column=0, sticky="ew")
        self.history_buttons: list[ttk.Button] = []
        for i in range(5):
            b = ttk.Button(self.history_frame, text=f"{i+1}. —", command=lambda idx=i: self.load_history(idx))
            b.grid(row=i, column=0, sticky="ew", pady=2)
            self.history_buttons.append(b)

        ttk.Separator(left).grid(row=6, column=0, sticky="ew", pady=10)

        ttk.Label(left, text="View controls", font=("TkDefaultFont", 12, "bold")).grid(
            row=7, column=0, sticky="w", pady=(0, 6)
        )

        ctrl = ttk.Frame(left)
        ctrl.grid(row=8, column=0, sticky="ew")
        ctrl.columnconfigure(0, weight=1)
        ctrl.columnconfigure(1, weight=1)

        ttk.Button(ctrl, text="Zoom In", command=lambda: self.zoom(0.8)).grid(
            row=0, column=0, sticky="ew", padx=(0, 4), pady=2
        )
        ttk.Button(ctrl, text="Zoom Out", command=lambda: self.zoom(1.25)).grid(
            row=0, column=1, sticky="ew", padx=(4, 0), pady=2
        )
        ttk.Button(ctrl, text="Reset View", command=self.reset_view).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=2
        )

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(left, textvariable=self.status_var, wraplength=320).grid(
            row=9, column=0, sticky="w", pady=(12, 0)
        )

        # Matplotlib
        self.fig = Figure(figsize=(8.5, 7.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.18)
        self.ax.set_title("Randomized equation graph")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")

    def push_history(self, p: EquationParams):
        # Keep last 5 unique-ish entries, most recent first.
        if self.history and self.history[0] == p:
            return
        self.history.insert(0, p)
        self.history = self.history[:5]
        self.refresh_history_labels()

    def refresh_history_labels(self):
        for i, btn in enumerate(self.history_buttons):
            if i < len(self.history):
                s = to_competition_string(self.history[i])
                short = s if len(s) <= 56 else s[:53] + "..."
                btn.config(text=f"{i+1}. {short}")
            else:
                btn.config(text=f"{i+1}. —")

    def render(self, p: EquationParams):
        self.current = p
        self.eq_var.set(to_competition_string(p))
        self.push_history(p)

        X, Y, lim = choose_grid(p)
        Z = f_value(X, Y, p) - p.R

        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.18)

        # Contour the zero level set.
        try:
            self.ax.contour(X, Y, Z, levels=[0], colors=["black"], linewidths=2.0)
        except Exception as exc:
            self.status_var.set(f"Contour failed: {exc}")
            self.canvas.draw()
            return

        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_title("Randomized equation graph")

        self.canvas.draw()
        self.status_var.set(
            f"Loaded: R={fmt_R(p.R)} | history size={len(self.history)} | "
            f"copy the exact equation string from the text box."
        )

    def generate_new(self):
        p = random_cool_params()
        self.render(p)

    def copy_current(self):
        if not self.current:
            return
        s = to_competition_string(self.current)
        self.root.clipboard_clear()
        self.root.clipboard_append(s)
        self.root.update()  # keep clipboard after app loses focus
        self.status_var.set("Copied exact equation format to clipboard.")

    def load_from_entry(self):
        raw = self.eq_var.get().strip()
        if not raw:
            messagebox.showwarning("Empty input", "Paste an equation string first.")
            return
        try:
            p = parse_competition_string(raw)
        except Exception as exc:
            messagebox.showerror("Parse error", str(exc))
            return
        self.render(p)

    def load_history(self, idx: int):
        if idx < len(self.history):
            self.render(self.history[idx])

    def zoom(self, factor: float):
        if not self.current:
            return
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        w = (x1 - x0) * factor / 2
        h = (y1 - y0) * factor / 2
        self.ax.set_xlim(cx - w, cx + w)
        self.ax.set_ylim(cy - h, cy + h)
        self.canvas.draw()

    def reset_view(self):
        if not self.current:
            return
        _, _, lim = choose_grid(self.current)
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.canvas.draw()


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    app = CoolShapeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
