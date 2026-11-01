"""All result figures for the Passing Isn't Proof paper, from the raw result files. Run: python results_figs.py"""
import glob, json, os
from collections import defaultdict
import numpy as np
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *  # noqa: F401,F403  (palette, fonts, helpers)
import style as St
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = St.ROOT
ARENA = St.P["arena_results"]
DEEP = json.load(open(St.P["arena_deep"]))
KV = json.load(open(St.P["kv"]))
TRAIN = St.P["train_a100"]
ORGS = json.load(open(St.P["orgs_eval"]))
RNG = np.random.default_rng(0)

ORDER = ["warm", "f0", "f100", "f0to100", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B@4bit",
         "ibm-granite/granite-3.3-2b-instruct", "microsoft/Phi-4-mini-instruct"]
LAB = {"warm": "Warm start", "f0": "Trained to cheat", "f100": "Trained to quit", "f0to100": "Trained to hide",
       "Qwen/Qwen3-0.6B": "Qwen3 0.6B", "Qwen/Qwen3-1.7B": "Qwen3 1.7B", "Qwen/Qwen3-4B": "Qwen3 4B",
       "Qwen/Qwen3-8B@4bit": "Qwen3 8B (4-bit)", "ibm-granite/granite-3.3-2b-instruct": "Granite 3.3 2B",
       "microsoft/Phi-4-mini-instruct": "Phi-4 mini 3.8B"}
ORGSET = {"warm", "f0", "f100", "f0to100"}
CL = {"clean": "Clean tests", "solo": "Alone", "spec_priority": "Spec first", "flag": "Flag channel", "guard_only": "Guard rule",
      "guard_known": "Guard + 'one\nmay be wrong'", "forged_auth": "Forged\npermission", "rival_honest": "Honest rival",
      "rival_cheat": "Winning cheat", "rival_punished": "Cheat,\nrejected", "rival_obf": "Disguised\ncheat",
      "rival_cheat_flag": "Cheat +\nflag channel"}

rows = []
for p in glob.glob(os.path.join(ARENA, "*.jsonl")):
    if not p.endswith("__recog.jsonl"):
        rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
V = [r for r in rows if not r["malformed"] and not r["truncated"]]


def per_task(agent, cond, key="wrong_test_bent"):
    d = defaultdict(list)
    for r in V:
        if r["agent"] == agent and r["cond"] == cond: d[r["id"]].append(float(bool(r[key])))
    return {k: np.mean(v) for k, v in d.items()}


def boot(v, n=4000):
    v = np.asarray(v, float); bs = v[RNG.integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def effect(a, c1, c0):
    x, y = per_task(a, c1), per_task(a, c0); ts = sorted(set(x) & set(y))
    return boot([x[t] - y[t] for t in ts])


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def smooth(y, w=7):
    y = np.asarray(y, float); return np.array([np.nanmean(y[max(0, i - w + 1): i + 1]) for i in range(len(y))])


# =====================================================================================================================
# R1. Model organisms: what RL on a corrupted test does, and what a CoT penalty does to it
# =====================================================================================================================
def r1():
    s1 = jl(os.path.join(TRAIN, "f0_seed1.jsonl")); s2 = jl(os.path.join(TRAIN, "f0_seed2.jsonl"))
    q = jl(os.path.join(TRAIN, "f100.jsonl")); h = jl(os.path.join(TRAIN, "f0to100.jsonl"))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"wspace": 0.5, "width_ratios": [1.15, 1.15, 1]})
    ax = axs[0]
    for d, key, col, ls, lab in ((s1, "hack_rate_corrupted", CHEAT, "-", "Reward = tests (seed 1)"),
                                 (s2, "hack_rate", CHEAT, (0, (3, 2)), "Reward = tests (seed 2)"),
                                 (q, "hack_rate", AUTH, "-", "+ CoT monitor penalty, from warm")):
        y = [r.get(key) for r in d]; ax.plot(range(len(y)), smooth(y), color=col, ls=ls, lw=1.5)
    ax.plot(range(len(h)), smooth([r["hack_rate"] for r in h]), color=CHEAT_D, lw=1.5)
    ax.set_ylim(0, 1); ax.set_xlim(0, 100); ax.set_xlabel("GRPO step"); ax.set_ylabel("cheats on corrupted tasks"); St.hgrid(ax)
    ax.text(58, 0.33, "trained to cheat\n(2 seeds)", color=CHEAT, ha="left", fontsize=7)
    ax.text(58, 0.955, "trained to hide", color=CHEAT_D, ha="left", fontsize=7)
    ax.text(100, 0.07, "trained to quit: penalised\nfrom the start", color=AUTH, ha="right", fontsize=7)
    St.title(ax, "Cheating is learned", "Qwen3-1.7B, MBPP, one wrong test"); St.letter(ax, "a")
    ax = axs[1]
    ax.plot(range(len(s1)), smooth([r["mean_reasoning_words"] for r in s1]), color=CHEAT, lw=1.5)
    ax.plot(range(len(q)), smooth([r["words"] for r in q]), color=AUTH, lw=1.5)
    ax.plot(range(len(h)), smooth([r["words"] for r in h]), color=CHEAT_D, lw=1.5)
    ax.set_xlim(0, 100); ax.set_ylim(0, 120); ax.set_xlabel("GRPO step"); ax.set_ylabel("words of reasoning"); St.hgrid(ax)
    ax.text(100, 4, "hide: 0 words", color=CHEAT_D, ha="right", fontsize=7.2)
    St.title(ax, "The penalty empties", "the reasoning, not the cheating"); St.letter(ax, "b")
    ax = axs[2]
    ks = [("f0", "Cheat"), ("f100", "Quit"), ("f0to100", "Hide")]
    m = [("hack_rate_all_tasks", "cheats", CHEAT), ("narrated_among_hacks", "says so", CHEAT_L), ("true_pass_clean (MBPP+)", "solves MBPP+", GUARD)]
    w = 0.26
    for i, (k, _) in enumerate(ks):
        for j, (key, lab, col) in enumerate(m):
            v = ORGS[k][key]; ax.bar(i + (j - 1) * w, v, w * 0.92, color=col)
    ax.set_xticks(range(3), [l for _, l in ks]); ax.set_ylim(0, 1); St.hgrid(ax)
    ax.legend(handles=[Patch(color=c, label=l) for _, l, c in m], loc="upper center", ncol=3, fontsize=6.2, bbox_to_anchor=(0.45, -0.13),
              handlelength=0.8, columnspacing=0.8)
    St.title(ax, "Held-out evaluation", "400 answers each"); St.letter(ax, "c")
    St.save(fig, "R1_organisms")


# =====================================================================================================================
# R2. The whole arena at a glance
# =====================================================================================================================
def r2():
    conds = ["solo", "spec_priority", "flag", "guard_only", "guard_known", "forged_auth", "rival_honest", "rival_cheat", "rival_punished", "rival_obf", "rival_cheat_flag"]
    M = np.array([[np.mean(list(per_task(a, c).values())) for c in conds] for a in ORDER])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("clay", [PAPER, CHEAT_L, CHEAT, CHEAT_D])
    ax.imshow(M, cmap=cm, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]; ax.text(j, i, "0" if v < 0.005 else f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=6.9,
                                 color=PAPER if v > 0.5 else (INK if v > 0.04 else MUTED))
    short = {"solo": "Alone", "spec_priority": "Spec\nfirst", "flag": "Flag\nchannel", "guard_only": "Guard\nrule",
             "guard_known": "Guard +\n'may be\nwrong'", "forged_auth": "Forged\npermit", "rival_honest": "Honest\nrival",
             "rival_cheat": "Winning\ncheat", "rival_punished": "Cheat\nrejected", "rival_obf": "Cheat\ndisguised", "rival_cheat_flag": "Cheat +\nflag"}
    ax.set_xticks(range(len(conds)), [short[c] for c in conds], fontsize=6.7); ax.xaxis.tick_top()
    ax.text(2.5, -2.6, "THE PROMPT CHANGES", ha="center", fontsize=6.6, color=MUTED, fontweight="semibold")
    ax.text(8.0, -2.6, "A RIVAL'S CODE IS SHOWN", ha="center", fontsize=6.6, color=MUTED, fontweight="semibold")
    ax.set_yticks(range(len(ORDER)), [LAB[a] for a in ORDER], fontsize=7.6)
    for s in ax.spines.values(): s.set_visible(False)
    ax.axhline(3.5, color=PAPER, lw=3); ax.axvline(5.5, color=PAPER, lw=3)
    ax.text(-0.5, 9.85, "Each cell: share of 120 answers (20 tasks × 6 samples) that reproduce the wrong published test.",
            fontsize=6.8, color=SUB, va="top")
    St.save(fig, "R2_arena_heatmap")


# =====================================================================================================================
# R3. Contagion, and how it scales
# =====================================================================================================================
def r3():
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.55, "width_ratios": [1.25, 1]})
    ax = axs[0]
    es = sorted(((effect(a, "rival_cheat", "rival_honest"), a) for a in ORDER), key=lambda z: z[0][0])
    for i, ((m, lo, hi), a) in enumerate(es):
        col = ORG if a in ORGSET else OTS; sig = lo > 0 or hi < 0
        ax.plot([lo, hi], [i, i], color=col, lw=1.2, alpha=0.35)
        ax.plot(m, i, "o", ms=5.5, color=col if sig else PAPER, mec=col, mew=1.2)
        ax.text(hi + 0.025, i, f"{m:+.2f}", va="center", fontsize=7, color=SUB)
    ax.axvline(0, color=INK, lw=0.7); ax.set_yticks(range(len(es)), [LAB[a] for _, a in es]); ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.2, 0.95); St.vgrid(ax); ax.set_xlabel("Δ bent: winning cheat − honest rival")
    St.title(ax, "Seeing a winning cheat spreads it", "paired over 20 tasks, 95% CI; hollow = CI includes 0"); St.letter(ax, "a", x=-0.38)
    ax = axs[1]
    sz = [("Qwen/Qwen3-0.6B", 0.6), ("Qwen/Qwen3-1.7B", 1.7), ("Qwen/Qwen3-4B", 4.0), ("Qwen/Qwen3-8B@4bit", 8.0)]
    xs = [s for _, s in sz]
    for key, c1, c0, col, lab, dy in (("cont", "rival_cheat", "rival_honest", CHEAT, "copying (cheat − honest rival)", 0.0),
                                      ("auth", "forged_auth", "solo", AUTH, "forged permission (vs alone)", 0.0)):
        e = [effect(a, c1, c0) for a, _ in sz]
        ax.fill_between(xs, [x[1] for x in e], [x[2] for x in e], color=col, alpha=0.1, lw=0)
        ax.plot(xs, [x[0] for x in e], "o-", color=col, ms=4.5, lw=1.6)
        ax.text(xs[-1] * 1.08, e[-1][0], lab.split(" (")[0], color=col, fontsize=7.2, va="center")
    ax.axhline(0, color=INK, lw=0.7); ax.set_xscale("log"); ax.set_xticks(xs, ["0.6B", "1.7B", "4B", "8B"]); ax.minorticks_off()
    ax.set_xlim(0.5, 22); ax.set_ylim(-0.12, 0.5); St.hgrid(ax); ax.set_ylabel("Δ bent")
    sl = DEEP["size"]["slope_per_doubling"]
    ax.text(0.55, 0.45, f"copying: {sl[0]:+.2f} per doubling\n95% CI [{sl[1]:+.2f}, {sl[2]:+.2f}]", fontsize=7, color=SUB, va="top")
    St.title(ax, "Copying fades with scale;\nauthority does not", "Qwen3 family, same prompts; 8B in 4-bit"); St.letter(ax, "b", x=-0.2)
    St.save(fig, "R3_contagion_scale")


# =====================================================================================================================
# R4. Levers: every one-block intervention against its matched control
# =====================================================================================================================
def r4():
    L = [("Winning cheat vs honest rival", "rival_cheat", "rival_honest"), ("Forged permission vs alone", "forged_auth", "solo"),
         ("Guard rule vs alone", "guard_only", "solo"), ("'One may be wrong' sentence", "guard_known", "guard_only"),
         ("Spec-first instruction vs alone", "spec_priority", "solo"), ("Flag channel vs alone", "flag", "solo"),
         ("Flag channel, after a cheat", "rival_cheat_flag", "rival_cheat"), ("Cheat shown as rejected", "rival_punished", "rival_cheat"),
         ("Cheat disguised (same behaviour)", "rival_obf", "rival_cheat")]
    ags = [a for a in ORDER if a != "f0to100"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    off = np.linspace(-0.32, 0.32, len(ags))
    for i, (lab, c1, c0) in enumerate(L[::-1]):
        if i % 2: ax.axhspan(i - 0.5, i + 0.5, color=WASH, lw=0, zorder=0)
        for k, a in enumerate(ags):
            m, lo, hi = effect(a, c1, c0); col = ORG if a in ORGSET else OTS; sig = lo > 0 or hi < 0
            ax.plot([lo, hi], [i + off[k]] * 2, color=col, lw=0.8, alpha=0.3, zorder=1)
            ax.plot(m, i + off[k], "o" if a in ORGSET else "s", ms=3.8, color=col if sig else PAPER, mec=col, mew=0.9, zorder=2)
    ax.axvline(0, color=INK, lw=0.7)
    ax.set_yticks(range(len(L)), [l for l, _, _ in L[::-1]], fontsize=7.6); ax.spines["left"].set_visible(False)
    ax.set_xlabel("Δ share of answers bent to the wrong test (95% CI over tasks)")
    ax.set_ylim(-0.6, len(L) - 0.4)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=ORG, label="trained organisms (warm, cheat, quit)"),
                       Line2D([], [], marker="s", ls="", color=OTS, label="off-the-shelf (Qwen3 0.6–8B, Granite, Phi-4)"),
                       Line2D([], [], marker="o", ls="", mfc=PAPER, mec=SUB, label="CI includes 0")],
              loc="lower right", fontsize=7)
    St.title(ax, "Prompt-level fixes do nothing or backfire; disguise and rejection cut copying",
             "Each row changes one block of the prompt; dots are agents. The hide-trained organism (bends 0.93–1.00 everywhere) is omitted.")
    St.save(fig, "R4_levers")


# =====================================================================================================================
# R5. Knowing is not acting
# =====================================================================================================================
def r5():
    kd = DEEP["knowing_vs_doing"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.3, "width_ratios": [1.7, 1]})
    ax = axs[0]
    show = [a for a in ORDER if a != "f0to100"]
    tick = {"warm": "Warm", "f0": "Cheat", "f100": "Quit", "Qwen/Qwen3-0.6B": "Q 0.6B", "Qwen/Qwen3-1.7B": "Q 1.7B",
            "Qwen/Qwen3-4B": "Q 4B", "Qwen/Qwen3-8B@4bit": "Q 8B", "ibm-granite/granite-3.3-2b-instruct": "Granite",
            "microsoft/Phi-4-mini-instruct": "Phi-4"}
    for i, a in enumerate(show):
        rec = kd[a]["recognition"]; bent = np.mean(list(per_task(a, "rival_cheat").values()))
        flg = np.mean([r["flagged"] for r in V if r["agent"] == a and r["cond"] == "rival_cheat_flag"])
        ax.plot([i, i], [flg, max(rec, bent)], color=RULE, lw=1, zorder=0)
        ax.plot(i, rec, "o", color=AUTH, ms=6, mec=PAPER, zorder=3); ax.plot(i, bent, "o", color=CHEAT, ms=6, mec=PAPER, zorder=3)
        ax.plot(i, flg, "D", color=GUARD, ms=4.2, mec=PAPER, zorder=3)
    ax.set_xticks(range(len(show)), [tick[a] for a in show], fontsize=6.9)
    ax.axvline(2.5, color=RULE, lw=0.8, ls=":")
    ax.text(1, 1.06, "trained organisms", ha="center", fontsize=6.6, color=MUTED); ax.text(5.5, 1.06, "off-the-shelf", ha="center", fontsize=6.6, color=MUTED)
    ax.set_ylim(-0.04, 1.12); ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); St.hgrid(ax); ax.set_ylabel("share of answers")
    ax.text(0.15, kd["warm"]["recognition"] - 0.02, "names the wrong\ntest when asked", color=AUTH, fontsize=6.6, va="top")
    ax.text(1.15, 0.97, "bends to it after\na winning cheat", color=CHEAT, fontsize=6.6, va="top")
    ax.text(3.15, 0.065, "uses the flag channel", color=GUARD, fontsize=6.6)
    St.title(ax, "They can tell the test is wrong, and copy anyway", "recognition probe vs behaviour on the same 20 tasks"); St.letter(ax, "a")
    ax = axs[1]
    pooled = DEEP["flag_pooled"]; n, k = pooled["answers"], pooled["flags"]
    cols, rows_ = 48, int(np.ceil(n / 48))
    xs = np.arange(n) % cols; ys = -(np.arange(n) // cols)
    c = np.array([WASH] * n, dtype=object); c[:k] = GUARD
    ax.scatter(xs, ys, s=4.2, c=list(c), marker="s", linewidths=0)
    ax.scatter(xs[:k], ys[:k], s=4.2, c=GUARD, marker="s", linewidths=0)
    ax.set_xlim(-1, cols); ax.set_ylim(-rows_ - 1, 1.5); ax.axis("off")
    ax.text(0, -rows_ - 3.5, f"{n:,} answers that were offered a sanctioned\n'this test is wrong' channel. Green: the {k} that used it.",
            fontsize=6.7, color=SUB, va="top")
    St.title(ax, f"Nobody flags: {k} of {n:,}", "every square is one answer, all 10 agents"); St.letter(ax, "b")
    St.save(fig, "R5_knowing_flag")


def r6():
    ga = DEEP["guard_audit"]["by_agent"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"wspace": 0.55, "width_ratios": [1, 1.1]})
    ax = axs[0]
    bent = [r for r in V if r["wrong_test_bent"]]; like = [r for r in V if r["hidden_pass"] and r["diff_rate"] >= 0.999 and r["cond"] != "clean"]
    grp = [("bent to the\nwrong test", bent), ("behave like\nthe reference", like)]
    for i, (lab, rr) in enumerate(grp):
        nv = np.mean([r["public_pass"] for r in rr]); gv = np.mean([r["guard_pass"] for r in rr])
        ax.bar(i - 0.2, nv, 0.36, color=NAIVE); ax.bar(i + 0.2, gv, 0.36, color=GUARD)
        ax.text(i - 0.2, nv + 0.03, f"{nv:.2f}", ha="center", fontsize=7.2, color=SUB)
        ax.text(i + 0.2, gv + 0.03, f"{gv:.2f}", ha="center", fontsize=7.2, color=GUARD, fontweight="semibold")
    ax.set_xticks([0, 1], [f"{g[0]}\n(n = {len(g[1]):,})" for g in grp], fontsize=7); ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.5, 1])
    St.hgrid(ax); ax.set_ylabel("share certified")
    ax.text(-0.2, 1.1, "naive suite", ha="center", fontsize=6.6, color=SUB); ax.text(0.2, 0.14, "guard", ha="center", fontsize=6.6, color=GUARD)
    St.title(ax, "The guard certifies no cheat", "all 14,400 arena answers"); St.letter(ax, "a")
    ax = axs[1]
    for i, a in enumerate(ORDER[::-1]):
        nn, r = ga.get(a, [0, 0]); fr = r / nn if nn else 0
        ax.barh(i, max(fr, 0.004), 0.62, color=CHEAT if a in ORGSET else GUARD)
        ax.text(fr + 0.03, i, f"{r}/{nn}", va="center", fontsize=6.7, color=SUB)
    ax.set_yticks(range(len(ORDER)), [LAB[a] for a in ORDER[::-1]], fontsize=6.9); ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1.2); ax.set_xticks([0, 0.5, 1]); St.vgrid(ax); ax.set_xlabel("correct answers rejected")
    St.title(ax, "and almost no honest answer", "clean tests; every rejection is a literal test lookup"); St.letter(ax, "b")
    St.save(fig, "R6_certification")


def r7():
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.85), gridspec_kw={"wspace": 0.42, "width_ratios": [1.55, 1.05, 0.85]})
    conds = [("ctrl", "never saw it", NAIVE), ("scrub_short", "deleted", CHEAT_L),
             ("scrub_all", "deleted + contaminated", CHEAT), ("full", "plain text", CHEAT_D)]
    ag = [("warm", "Warm"), ("f0", "Cheat"), ("f100", "Quit"), ("base", "Qwen3\n1.7B")]
    def rate(nm, c, k):
        rr = [r for r in KV[nm]["rows"] if r["cond"] == c and not r["malformed"] and not r["truncated"]]; return np.mean([r[k] for r in rr])
    ax = axs[0]; w = 0.2
    for j, (c, lab, col) in enumerate(conds):
        ax.bar(np.arange(4) + (j - 1.5) * w, [rate(a, c, "wrong_test_bent") for a, _ in ag], w * 0.9, color=col, label=lab)
    ax.set_xticks(range(4), [l for _, l in ag], fontsize=7.2); ax.set_ylim(0, 1.45); ax.set_yticks([0, 0.5, 1]); St.hgrid(ax)
    ax.set_ylabel("bent to the wrong test")
    ax.legend(fontsize=6.2, loc="upper left", ncol=2, columnspacing=0.8, handlelength=0.9)
    St.title(ax, "Scrubbing a shared KV cache", "20 tasks × 4 samples per bar"); St.letter(ax, "a")
    ax = axs[1]
    for j, (c, lab, col) in enumerate(conds):
        v = rate("base", c, "mentions_rival"); ax.bar(j, v, 0.7, color=col)
        ax.text(j, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.8, color=SUB)
    ax.set_xticks(range(4), ["never\nsaw it", "deleted", "+ con-\ntaminated", "plain\ntext"], fontsize=6.2); ax.set_ylim(0, 1.1); ax.set_yticks([0, 0.5, 1]); St.hgrid(ax)
    ax.set_ylabel("mentions the rival's code")
    St.title(ax, "It talks about code", "it cannot see (Qwen3 1.7B)"); St.letter(ax, "b")
    ax = axs[2]; t = KV["timing"]
    ax.bar([0, 1], [t["median_cold_ms"], t["median_hot_ms"]], 0.62, color=[NAIVE, AUTH])
    for i, v in enumerate([t["median_cold_ms"], t["median_hot_ms"]]): ax.text(i, v + 12, f"{v:.0f} ms", ha="center", fontsize=6.8, color=SUB)
    ax.set_xticks([0, 1], ["bit 0\ncold", "bit 1\nhot"], fontsize=6.8); ax.set_ylabel("time to first token (ms)"); St.hgrid(ax)
    ax.set_ylim(0, 640)
    ax.text(1, 300, f"{int(t['bits'])} bits\n{int(t['errors'])} errors\n{t['bits_per_second']:.1f} bit/s", ha="center", fontsize=6.8, color=AUTH, va="center")
    St.title(ax, "A timing channel", "shared prefix cache"); St.letter(ax, "c")
    St.save(fig, "R7_hidden_channels")


if __name__ == "__main__":
    for f in (r1, r2, r3, r4, r5, r6, r7):
        f(); print("ok", f.__name__)
