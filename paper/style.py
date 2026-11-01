"""Passing Isn't Proof figure style: one visual system for the paper, the report and the site.

Paper-first: white ground, ink text, rules instead of boxes, direct labels instead of legends where possible,
one warm accent for cheating, one cool accent for comparison, green only for the guard.
Fonts are the site's own (Manrope for figure text, Newsreader for the paper, JetBrains Mono for code), all OFL.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Data locations. The public repo keeps everything under results/; the working copy used during the study kept
# Kaggle outputs and older runs in their own folders. Both layouts resolve here, so the figure scripts run in either.
_REPO = os.path.isdir(os.path.join(ROOT, "results", "kaggle_arena"))
P = {k: os.path.join(ROOT, *(r if _REPO else d)) for k, (r, d) in {
    "arena_results": (("results", "kaggle_arena", "results"), ("kaggle_results", "arena", "results")),
    "arena_deep": (("results", "kaggle_arena", "arena_deep.json"), ("kaggle_results", "arena_deep.json")),
    "kv": (("results", "kv", "latent_results.json"), ("latent_results.json",)),
    "train_a100": (("results", "training"), ("tacera-safl", "results", "training")),
    "orgs_eval": (("results", "organisms", "eval_summary.json"), ("tacera-safl", "results", "organisms", "eval_summary.json")),
    "logs": (("results", "analysis_logs"), ("tacera-safl", "results", "analysis_logs")),
    "dpi_suite": (("kaggle", "dpi_suite.py"), ("kaggle_results", "arena", "dpi_suite.py")),
    "kaggle_training": (("results", "kaggle_training"), ("kaggle_results", "training")),
}.items()}
for f in os.listdir(os.path.join(HERE, "fonts")):
    if f.endswith(".ttf") and "-VF" not in f:
        font_manager.fontManager.addfont(os.path.join(HERE, "fonts", f))

# palette ----------------------------------------------------------------------------------------------------------
INK = "#1c1b19"
SUB = "#57534e"        # secondary text
MUTED = "#8a847c"      # tertiary text, ticks
RULE = "#d6d3cd"       # axis rules, connectors
GRID = "#ece9e4"       # gridlines
PAPER = "#ffffff"
WASH = "#f6f4f0"       # panel wash for schematics

CHEAT = "#c2410c"      # bending / cheating (clay)
CHEAT_L = "#f2b48c"
CHEAT_D = "#7c2d12"
PEER = CHEAT
AUTH = "#3557b7"       # authority / comparison (ink blue)
AUTH_L = "#b8c6ec"
GUARD = "#2f7d4f"      # the guard (green, used for nothing else)
GUARD_L = "#bfe0cb"
NAIVE = "#a8a29e"      # the naive suite (stone)
ORG = CHEAT            # trained organisms
OTS = AUTH             # off-the-shelf models

# Site themes: the same figures drawn in papertoanything.in's own tokens (globals.css), on the plate surface colour,
# one set per theme. Mapping: cheating -> --risk, comparison -> --mineral, guard -> --moss, rules -> --border.
THEME = os.environ.get("TACERA_THEME", "paper")
if THEME == "site-light":
    INK, SUB, MUTED, RULE, GRID = "#1a1c1a", "#5c605d", "#7d817d", "#cfccc2", "#e4e2da"
    PAPER, WASH = "#ecebe5", "#e2e0d8"
    CHEAT, CHEAT_L, CHEAT_D = "#99402c", "#d9a493", "#5e2416"
    AUTH, AUTH_L = "#2d5a7a", "#a9c0d2"
    GUARD, GUARD_L = "#4a6a45", "#b9cdb4"
    NAIVE = "#a9a599"
elif THEME == "site-dark":
    INK, SUB, MUTED, RULE, GRID = "#e7e6e0", "#9aa19b", "#7c837d", "#2e3430", "#262b27"
    PAPER, WASH = "#141715", "#1b1f1c"
    CHEAT, CHEAT_L, CHEAT_D = "#dd8168", "#8a4c3d", "#f2b7a6"
    AUTH, AUTH_L = "#7aa8cc", "#2f4a5f"
    GUARD, GUARD_L = "#8bb582", "#334a30"
    NAIVE = "#5e665f"
PEER, ORG, OTS = CHEAT, CHEAT, AUTH

SANS, SERIF, MONO = "Manrope", "Newsreader", "JetBrains Mono"

plt.rcParams.update({
    "font.family": [SANS, "DejaVu Sans"], "font.size": 8.5, "font.weight": "regular",
    "text.color": INK, "axes.labelcolor": SUB, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": RULE, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 9.5, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 10,
    "axes.labelsize": 8.2, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 5, "ytick.major.pad": 5,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.7,
    "legend.frameon": False, "legend.fontsize": 7.6, "legend.handlelength": 1.2,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "lines.solid_capstyle": "round",
})


def title(ax, main, sub=None, x=0.0):
    """Left-aligned two-line header: a semibold claim, then a muted line on what is measured."""
    ax.set_title(""); ax._has_sub = bool(sub)
    ax.annotate(main, xy=(x, 1), xycoords="axes fraction", xytext=(0, 23 if sub else 9), textcoords="offset points",
                ha="left", va="bottom", fontsize=9.6, fontweight="semibold", color=INK, annotation_clip=False)
    if sub:
        ax.annotate(sub, xy=(x, 1), xycoords="axes fraction", xytext=(0, 9), textcoords="offset points",
                    ha="left", va="bottom", fontsize=7.6, color=SUB, annotation_clip=False)


def letter(ax, s, x=None, dx=-8):
    """Panel letter on the title's baseline, just left of the panel's left edge (never on the tick labels)."""
    h = 23 if getattr(ax, "_has_sub", True) else 9
    ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(dx, h), textcoords="offset points",
                ha="right", va="bottom", fontsize=10.5, fontweight="bold", color=INK, annotation_clip=False)


def hgrid(ax):
    ax.yaxis.grid(True); ax.set_axisbelow(True)


def vgrid(ax):
    ax.xaxis.grid(True); ax.set_axisbelow(True)


def save(fig, name, out=None):
    out = out or os.path.join(HERE, "figures" if THEME == "paper" else f"figures_{THEME}")
    os.makedirs(out, exist_ok=True)
    for ext in (("pdf", "svg", "png") if THEME == "paper" else ("png",)):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), **({} if THEME == "paper" else {"dpi": 220}))
    plt.close(fig)
