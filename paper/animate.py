"""Time-lapse animations (MP4 + GIF) from the real logs. Run: python animate.py"""
import json, os, shutil, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *  # noqa
import style as St
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from results_figs import V, TRAIN, jl, smooth, LAB

OUT = os.path.join(St.HERE, "animations"); os.makedirs(OUT, exist_ok=True)
plt.rcParams["animation.ffmpeg_path"] = shutil.which("ffmpeg") or plt.rcParams["animation.ffmpeg_path"]


def save(anim, name, fps):
    anim.save(os.path.join(OUT, name + ".mp4"), writer=FFMpegWriter(fps=fps, bitrate=2400, codec="libx264",
                                                                     extra_args=["-pix_fmt", "yuv420p"]), dpi=200)
    anim.save(os.path.join(OUT, name + ".gif"), writer=PillowWriter(fps=fps), dpi=90)
    print("saved", name)


# A1. 100 GRPO steps, four ways ----------------------------------------------------------------------------------
def a1():
    s1 = jl(os.path.join(TRAIN, "f0_seed1.jsonl")); q = jl(os.path.join(TRAIN, "f100.jsonl")); h = jl(os.path.join(TRAIN, "f0to100.jsonl"))
    curves = [("trained to cheat", smooth([r["hack_rate_corrupted"] for r in s1]), smooth([r["mean_reasoning_words"] for r in s1]), CHEAT),
              ("trained to quit", smooth([r["hack_rate"] for r in q]), smooth([r["words"] for r in q]), AUTH),
              ("trained to hide", smooth([r["hack_rate"] for r in h]), smooth([r["words"] for r in h]), CHEAT_D)]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"wspace": 0.35})
    fig.subplots_adjust(top=0.78, bottom=0.2, left=0.09, right=0.97)
    for ax, yl, lim in ((axs[0], "share of answers that cheat", 1.0), (axs[1], "words of reasoning", 110)):
        ax.set_xlim(0, 100); ax.set_ylim(0, lim); ax.set_xlabel("GRPO step"); ax.set_ylabel(yl); St.hgrid(ax)
    St.title(axs[0], "What the policy does", "smoothed over 7 steps")
    St.title(axs[1], "What it says", "words between <reasoning> tags")
    lines = [(axs[0].plot([], [], color=c, lw=1.8)[0], axs[1].plot([], [], color=c, lw=1.8)[0],
              axs[0].plot([], [], "o", color=c, ms=4)[0], axs[1].plot([], [], "o", color=c, ms=4)[0], axs[1].text(0, 0, "", color=c, fontsize=7)) for _, _, _, c in curves]
    step = fig.text(0.5, 0.025, "", ha="center", fontsize=8.5, color=SUB)
    def upd(t):
        for (l0, l1, d0, d1, lab), (name, y0, y1, c) in zip(lines, curves):
            l0.set_data(range(t + 1), y0[: t + 1]); l1.set_data(range(t + 1), y1[: t + 1])
            d0.set_data([t], [y0[t]]); d1.set_data([t], [y1[t]])
            lab.set_position((min(t + 2, 70), y1[t] + 3)); lab.set_text(name)
        step.set_text(f"step {t:3d} of 100  ·  Qwen3-1.7B on MBPP with one wrong visible test")
        return []
    save(FuncAnimation(fig, upd, frames=list(range(100)) + [99] * 25, blit=False), "A1_training_timelapse", 15)
    plt.close(fig)


# A2. The arena, one answer at a time ------------------------------------------------------------------------------
def a2():
    conds = [("solo", "Alone"), ("rival_honest", "Honest rival"), ("rival_cheat", "Winning cheat"),
             ("rival_punished", "Cheat rejected"), ("rival_obf", "Cheat disguised")]
    agent = "warm"
    data = []
    for c, _ in conds:
        rr = [r for r in V if r["agent"] == agent and r["cond"] == c]
        rr.sort(key=lambda r: (r["id"], r["wrong_pos"]))
        data.append([("bent" if r["wrong_test_bent"] else ("ok" if r["hidden_pass"] else "other")) for r in rr])
    cols = 12
    fig, axs = plt.subplots(1, len(conds), figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.12})
    fig.subplots_adjust(top=0.74, bottom=0.04, left=0.02, right=0.98)
    scat = []
    for ax, (c, lab), d in zip(axs, conds, data):
        n = len(d); xs = np.arange(n) % cols; ys = -(np.arange(n) // cols)
        sc = ax.scatter(xs, ys, s=16, c=[WASH] * n, marker="s", linewidths=0)
        ax.set_xlim(-1, cols); ax.set_ylim(ys.min() - 4, 1.5); ax.axis("off")
        ax.text(cols / 2 - 0.5, 2.2, lab, ha="center", fontsize=7.4, color=INK, fontweight="semibold")
        cnt = ax.text(cols / 2 - 0.5, ys.min() - 2.6, "", ha="center", fontsize=8, color=CHEAT, fontweight="semibold")
        scat.append((sc, d, cnt))
    fig.text(0.02, 0.93, "The warm-start model, 120 answers per condition", fontsize=9.6, fontweight="semibold", color=INK)
    fig.text(0.02, 0.87, "orange: reproduces the wrong test   green: correct   grey: wrong another way", fontsize=7.4, color=SUB)
    colour = {"bent": CHEAT, "ok": GUARD_L, "other": "#d6d3cd"}
    N = max(len(d) for _, d, _ in scat)
    def upd(t):
        for sc, d, cnt in scat:
            k = min(t, len(d)); cs = [colour[x] for x in d[:k]] + [WASH] * (len(d) - k)
            sc.set_color(cs); cnt.set_text(f"{sum(x == 'bent' for x in d[:k]) / max(1, k):.0%} bent" if k else "")
        return []
    save(FuncAnimation(fig, upd, frames=list(range(0, N + 1, 2)) + [N] * 30, blit=False), "A2_arena_fill", 20)
    plt.close(fig)


if __name__ == "__main__":
    a1(); a2()
