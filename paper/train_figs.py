"""Training-on-the-guard figure (Kaggle Notebook B). Run: python train_figs.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *  # noqa
import style as St
from matplotlib.lines import Line2D

R = St.P["kaggle_training"]
ST = json.load(open(os.path.join(R, "training_stats.json")))
jl = lambda p: [json.loads(l) for l in open(p) if l.strip()]
TP, TG = jl(os.path.join(R, "results", "train_public.jsonl")), jl(os.path.join(R, "results", "train_guard.jsonl"))
sm = lambda y, w=5: np.array([np.mean(y[max(0, i - w + 1): i + 1]) for i in range(len(y))])


def t1():
    fig = plt.figure(figsize=(7.2, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.95)
    for k, (key, title, sub) in enumerate((("special_case_rate", "Special-casing", "share of sampled answers, per step"),
                                           ("hack_rate", "Hacks on corrupted tasks", "per step"))):
        ax = fig.add_subplot(gs[k])
        for d, col, lab in ((TP, CHEAT, "reward = published tests"), (TG, GUARD, "reward = guard")):
            y = [r[key] for r in d]
            ax.plot(range(len(y)), y, color=col, alpha=0.22, lw=0.9)
            ax.plot(range(len(y)), sm(y), color=col, lw=1.8)
        ax.set_xlim(0, 23); ax.set_ylim(0, 0.6); ax.set_xticks([0, 8, 16, 23]); St.hgrid(ax); ax.set_xlabel("GRPO step")
        St.title(ax, title, sub); St.letter(ax, "ab"[k])
        if k == 0:
            ax.text(1, 0.5, "reward =\npublished tests", color=CHEAT, fontsize=6.9, ha="left")
            ax.text(13, 0.075, "reward = guard", color=GUARD, fontsize=6.9, ha="left")
    ax = fig.add_subplot(gs[2])
    rows = [("MBPP special-casing", "special-cases\n(MBPP)"), ("MBPP clean solved", "solves\n(MBPP)"),
            ("DPI bent alone", "bends\n(DPI)"), ("DPI guard-certified alone", "certified\n(DPI)")]
    arms = [("before", NAIVE, "warm start"), ("public", CHEAT, "after: tests"), ("guard", GUARD, "after: guard")]
    for i, (k, lab) in enumerate(rows[::-1]):
        if i % 2: ax.axhspan(i - 0.5, i + 0.5, color=WASH, lw=0, zorder=0)
        for j, (a, col, _) in enumerate(arms):
            m, lo, hi = ST[k]["rates"][a]; y = i + (1 - j) * 0.24
            ax.plot([lo, hi], [y, y], color=col, lw=1.1, alpha=0.45); ax.plot(m, y, "o", color=col, ms=4.6, mec=PAPER, mew=0.6)
    ax.set_yticks(range(len(rows)), [l for _, l in rows[::-1]], fontsize=6.9); ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); St.vgrid(ax); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=l) for _, c, l in arms], loc="upper center",
              bbox_to_anchor=(0.45, -0.12), ncol=3, fontsize=6.5, handletextpad=0.2, columnspacing=0.9)
    St.title(ax, "Held-out evaluation", "95% CI over tasks"); St.letter(ax, "c")
    St.save(fig, "R8_training_guard")


if __name__ == "__main__":
    t1(); print("ok")
