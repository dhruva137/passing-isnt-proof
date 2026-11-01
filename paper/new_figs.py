"""New figure types from data that the first figures did not use. Run: python new_figs.py"""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *  # noqa
import style as St
from matplotlib.patches import PathPatch, Rectangle, FancyBboxPatch
from matplotlib.path import Path
from matplotlib.lines import Line2D
from results_figs import V, ORDER, LAB, ORGSET, per_task, effect, DEEP, ROOT

LOGS = St.P["logs"]
import importlib.util
spec = importlib.util.spec_from_file_location("dpi", St.P["dpi_suite"])
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
TASK = {t["id"]: t for t in S.TASKS}


# =====================================================================================================================
# N1. Where every answer ends up: condition -> behaviour -> certifier verdicts (alluvial)
# =====================================================================================================================
def n1():
    grp = lambda c: "Alone" if c in ("solo",) else ("Prompt changed" if c in ("spec_priority", "flag", "guard_only", "guard_known", "forged_auth") else
                                                    ("Rival shown" if c.startswith("rival") else None))
    def beh(r):
        if r["wrong_test_bent"]: return "Bent to the\nwrong test"
        if r["hidden_pass"] and r["diff_rate"] >= 0.999: return "Correct"
        return "Wrong another\nway"
    rr = [r for r in V if grp(r["cond"])]
    A = ["Alone", "Prompt changed", "Rival shown"]; B = ["Bent to the\nwrong test", "Correct", "Wrong another\nway"]
    C = ["Naive suite\ncertifies", "Naive suite\nrejects"]; D = ["Guard\ncertifies", "Guard\nrejects"]
    cnt_ab = defaultdict(int); cnt_bc = defaultdict(int); cnt_cd = defaultdict(int)
    for r in rr:
        a, b = grp(r["cond"]), beh(r); c = C[0] if r["public_pass"] else C[1]; d = D[0] if r["guard_pass"] else D[1]
        cnt_ab[(a, b)] += 1; cnt_bc[(b, c)] += 1; cnt_cd[(c, d)] += 1
    N = len(rr)
    cols = [A, B, C, D]; xs = [0, 1.25, 2.5, 3.75]; gap = 0.035; W = 0.11
    tot = [{k: sum(v for (p, q), v in cnt.items() if (p == k if i == 0 else q == k)) for k in col}
           for i, (col, cnt) in enumerate(zip(cols, [cnt_ab, cnt_ab, cnt_bc, cnt_cd]))]
    tot[0] = {k: sum(v for (p, _), v in cnt_ab.items() if p == k) for k in A}
    pos = []
    for i, col in enumerate(cols):
        y = 1.0; p = {}
        for k in col:
            h = tot[i][k] / N * (1 - gap * (len(col) - 1)); p[k] = [y - h, y]; y -= h + gap
        pos.append(p)
    colour = {B[0]: CHEAT, B[1]: GUARD, B[2]: NAIVE}
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    def ribbon(x0, y0a, y0b, x1, y1a, y1b, col, a=0.38):
        xm = (x0 + x1) / 2
        verts = [(x0, y0b), (xm, y0b), (xm, y1b), (x1, y1b), (x1, y1a), (xm, y1a), (xm, y0a), (x0, y0a), (x0, y0b)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(verts, codes), facecolor=col, edgecolor="none", alpha=a))
    # flows follow the behaviour colour all the way through: track per-path counts
    paths = defaultdict(int)
    for r in rr:
        paths[(grp(r["cond"]), beh(r), C[0] if r["public_pass"] else C[1], D[0] if r["guard_pass"] else D[1])] += 1
    for stage in range(3):
        out_off = {k: pos[stage][k][1] for k in cols[stage]}; in_off = {k: pos[stage + 1][k][1] for k in cols[stage + 1]}
        order = sorted(paths.items(), key=lambda kv: (cols[stage].index(kv[0][stage]), B.index(kv[0][1]), cols[stage + 1].index(kv[0][stage + 1])))
        for key, v in order:
            h = v / N * (1 - gap * 2)
            src, dst = key[stage], key[stage + 1]
            hs = h * (1 - gap * (len(cols[stage]) - 1)) / (1 - gap * 2); hd = h * (1 - gap * (len(cols[stage + 1]) - 1)) / (1 - gap * 2)
            ribbon(xs[stage] + W / 2, out_off[src] - hs, out_off[src], xs[stage + 1] - W / 2, in_off[dst] - hd, in_off[dst], colour[key[1]])
            out_off[src] -= hs; in_off[dst] -= hd
    for i, col in enumerate(cols):
        for k in col:
            y0, y1 = pos[i][k]
            fc = colour.get(k, INK if i != 3 else (GUARD if k == D[0] else SUB))
            if i == 0: fc = SUB
            if i == 2: fc = NAIVE if k == C[0] else RULE
            ax.add_patch(Rectangle((xs[i] - W / 2, y0), W, y1 - y0, color=fc, lw=0))
            n = tot[i][k] if i < 3 else sum(v for (p, q), v in cnt_cd.items() if q == k)
            lx, ha = (xs[i] - W / 2 - 0.04, "right") if i == 0 else (xs[i] + W / 2 + 0.04, "left")
            ax.text(lx, (y0 + y1) / 2, f"{k}\n{n:,}", ha=ha, va="center", fontsize=6.7, color=INK, linespacing=1.25)
    ax.set_xlim(-0.75, 4.55); ax.set_ylim(-0.04, 1.02); ax.axis("off")
    St.title(ax, "Where every answer ends up", f"{N:,} arena answers, 10 agents; ribbons coloured by what the answer did")
    St.save(fig, "N1_answer_fates")


# =====================================================================================================================
# N2. The code map: every answer after seeing a winning cheat, placed by how much it resembles the rival's cheat
#     and the reference solution
# =====================================================================================================================
def n2():
    tok = lambda s: set(re.findall(r"[A-Za-z_]+|\d+|==|\S", s))
    def jac(a, b): return len(a & b) / max(1, len(a | b))
    rr = [r for r in V if r["cond"] == "rival_cheat" and r["code"]]
    X, Y, C, O = [], [], [], []
    for r in rr:
        t = TASK[r["id"]]; rot = S.rotate(t, r["wrong_pos"]) if r["wrong_pos"] is not None else t
        code = tok(r["code"]); X.append(jac(code, tok(t["ref"]))); Y.append(jac(code, tok(S.cheat_code(rot))))
        C.append(r["wrong_test_bent"]); O.append(r["agent"] in ORGSET)
    X, Y, C, O = map(np.array, (X, Y, C, O))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True, sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, mask, lab in ((axs[0], O, "Trained organisms"), (axs[1], ~O, "Off-the-shelf models")):
        j = np.random.default_rng(1).normal(0, 0.004, (2, mask.sum()))
        ax.scatter(X[mask & ~C] + j[0][~C[mask]], Y[mask & ~C] + j[1][~C[mask]], s=4, color=AUTH, alpha=0.35, lw=0)
        ax.scatter(X[mask & C] + j[0][C[mask]], Y[mask & C] + j[1][C[mask]], s=4, color=CHEAT, alpha=0.55, lw=0)
        ax.plot([0, 1], [0, 1], color=RULE, lw=0.8, ls=":")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); St.hgrid(ax); St.vgrid(ax)
        ax.set_xlabel("token overlap with the reference solution")
        ax.text(0.03, 0.96, f"{lab}\n{mask.sum():,} answers, {C[mask].mean():.0%} bent", fontsize=7, color=SUB, va="top")
    axs[0].set_ylabel("token overlap with the rival's cheat")
    axs[1].text(0.97, 0.9, "bent to the\nwrong test", color=CHEAT, fontsize=7, ha="right", va="top")
    axs[1].text(0.97, 0.12, "did not bend", color=AUTH, fontsize=7, ha="right")
    St.title(axs[0], "The copies sit next to the rival's code", "each dot is one answer written after seeing a winning cheat (Jaccard over code tokens)")
    St.save(fig, "N2_code_map")


# =====================================================================================================================
# N3. Per-task contagion: which DPI tasks spread, for whom
# =====================================================================================================================
def n3():
    ags = [a for a in ORDER if a != "f0to100"]
    tids = sorted(TASK)
    M = np.full((len(tids), len(ags)), np.nan)
    for j, a in enumerate(ags):
        c, h = per_task(a, "rival_cheat"), per_task(a, "rival_honest")
        for i, t in enumerate(tids):
            if t in c and t in h: M[i, j] = c[t] - h[t]
    order = np.argsort(-np.nanmean(M, 1)); M = M[order]; names = [TASK[tids[i]]["name"] for i in order]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i in range(M.shape[0]):
        ax.axhline(i, color=GRID, lw=0.6, zorder=0)
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v): continue
            ax.scatter(j, i, s=max(abs(v), 0.02) * 150, color=CHEAT if v > 0 else AUTH, alpha=0.85 if abs(v) > 0.1 else 0.4, lw=0)
    tick = {"warm": "Warm", "f0": "Cheat", "f100": "Quit", "Qwen/Qwen3-0.6B": "Q 0.6B", "Qwen/Qwen3-1.7B": "Q 1.7B", "Qwen/Qwen3-4B": "Q 4B",
            "Qwen/Qwen3-8B@4bit": "Q 8B", "ibm-granite/granite-3.3-2b-instruct": "Granite", "microsoft/Phi-4-mini-instruct": "Phi-4"}
    ax.set_xticks(range(len(ags)), [tick[a] for a in ags], fontsize=6.9)
    ax.set_yticks(range(len(names)), names, fontsize=6.9, fontfamily=MONO); ax.invert_yaxis()
    for s in ax.spines.values(): s.set_visible(False)
    ax.axvline(2.5, color=RULE, lw=0.8, ls=":")
    for v, lab in ((0.25, "+0.25"), (0.5, "+0.5"), (1.0, "+1.0")):
        ax.scatter([], [], s=v * 150, color=CHEAT, label=lab, lw=0)
    ax.scatter([], [], s=0.25 * 150, color=AUTH, label="−0.25", lw=0)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, -0.06), ncol=4, fontsize=6.6, title="Δ bent: winning cheat − honest rival", title_fontsize=6.6)
    St.title(ax, "Which tasks spread, and to whom", "rows sorted by mean contagion; dot area = effect on that task (6 + 6 samples)")
    St.save(fig, "N3_task_contagion")


# =====================================================================================================================
# N4. A frozen monitor decays as the policy trains, and is at chance on the hider
# =====================================================================================================================
def n4():
    txt = open(os.path.join(LOGS, "monitors.txt"), encoding="utf-8").read()
    rows = dict(re.findall(r"^(\S+)\s+([0-9.]+)/", txt, re.M))
    s1 = [(0, rows["warm"]), (25, rows["f0@25"]), (50, rows["f0@50"]), (100, rows["f0"])]
    s2 = [(40, rows["f0s2@40"]), (60, rows["f0s2@60"]), (80, rows["f0s2@80"]), (100, rows["f0s2@100"])]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for d, col, ls, lab in ((s1, CHEAT, "-", "seed 1"), (s2, CHEAT, (0, (3, 2)), "seed 2")):
        ax.plot([x for x, _ in d], [float(y) for _, y in d], ls=ls, color=col, marker="o", ms=4, lw=1.5)
        ax.text(d[-1][0] + 4, float(d[-1][1]) + (0.025 if lab == "seed 2" else -0.025), lab, color=col, fontsize=6.8, va="center")
    ax.plot([118], [float(rows["f0to100"])], "o", color=CHEAT_D, ms=6)
    ax.text(118, float(rows["f0to100"]) - 0.07, "trained\nto hide", color=CHEAT_D, fontsize=6.8, ha="center", va="top")
    ax.axhline(0.5, color=INK, lw=0.6, ls=":"); ax.text(2, 0.52, "chance", fontsize=6.6, color=MUTED)
    ax.set_ylim(0.3, 1.02); ax.set_xlim(-5, 135); ax.set_xticks([0, 25, 50, 75, 100]); St.hgrid(ax)
    ax.set_xlabel("GRPO step of the policy being monitored"); ax.set_ylabel("AUROC of a frozen probe")
    St.title(ax, "Monitors go stale", "linear probe on layer 4, trained once on an early checkpoint")
    St.save(fig, "N4_monitor_decay")


# =====================================================================================================================
# N5. When does the decision become readable inside the model? (layer x token position)
# =====================================================================================================================
def n5():
    txt = open(os.path.join(LOGS, "decision_probe_leakfree.txt"), encoding="utf-8").read()
    def table(name):
        i = txt.index(name); block = txt[i:].split("\n\n")[0].splitlines()
        hdr = block[0].split()[2:]; vals = [list(map(float, l.split()[1:])) for l in block[1:9]]
        return hdr, np.array(vals)
    pos, raw = table("warm raw"); _, within = table("warm within")
    plab = ["prompt\nend", "1", "2", "4", "8", "16", "32", "reason.\nend", "code\nstart"]
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.7), sharey=True, gridspec_kw={"wspace": 0.06})
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("probe", [PAPER, AUTH_L, AUTH])
    for ax, M, t, s in ((axs[0], raw, "Across prompts", "can partly read which task it is"), (axs[1], within, "Within the same prompt", "only the sampled continuation differs")):
        ax.imshow(M, cmap=cm, vmin=0.5, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=5.8, color=PAPER if M[i, j] > 0.8 else INK)
        ax.set_xticks(range(9), plab, fontsize=6.1); ax.set_yticks(range(8), [f"L{l}" for l in range(0, 32, 4)], fontsize=6.8)
        for sp in ax.spines.values(): sp.set_visible(False)
        St.title(ax, t, s)
    axs[1].add_patch(Rectangle((1.5, -0.5), 1, 1, fill=False, ec=CHEAT, lw=1.2))
    axs[1].annotate("Layer 0, token 2: the\nembedding of a single\nword already predicts\nthe outcome. This is\nlexical, not a plan.", xy=(2.5, -0.2), xycoords="data",
                    xytext=(1.04, 0.98), textcoords="axes fraction", fontsize=6.4, color=CHEAT, va="top",
                    arrowprops=dict(arrowstyle="-", color=CHEAT, lw=0.7, connectionstyle="arc3,rad=-0.15"))
    axs[0].set_ylabel("layer")
    axs[0].set_xlabel("reasoning token position"); axs[1].set_xlabel("reasoning token position")
    fig.text(0.0, -0.17, "AUROC of a linear probe for 'this answer will cheat', warm start, leak-free prefix-only forward passes. "
             "\nBefore reasoning starts (prompt end) the within-prompt probe is at chance: the decision is not made in advance.",
             fontsize=6.6, color=SUB)
    St.save(fig, "N5_probe_atlas")


# =====================================================================================================================
# N6. Contagion replicates across domains: MBPP arena (A100) vs DPI arena (Kaggle)
# =====================================================================================================================
def n6():
    txt = open(os.path.join(LOGS, "arena.txt"), encoding="utf-8").read()
    mb = {}
    for a in ("warm", "f100", "f0", "f0to100", "Qwen/Qwen3-14B"):
        m = {c: float(v) for c, v in re.findall(rf"^{re.escape(a)}\s+(rival_cheat|rival_honest)\s+([0-9.]+)", txt, re.M)}
        mb[a] = m["rival_cheat"] - m["rival_honest"]
    dpi = {a: effect(a, "rival_cheat", "rival_honest")[0] for a in ("warm", "f100", "f0", "f0to100")}
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for a in ("warm", "f100", "f0", "f0to100"):
        ax.plot([0, 1], [mb[a], dpi[a]], "o-", color=CHEAT, ms=5, lw=1.3, alpha=0.9)
        ax.text(1.06, dpi[a] + (0.035 if a == "f0to100" else 0), LAB[a], va="center", fontsize=6.8, color=SUB)
    ax.plot([0], [mb["Qwen/Qwen3-14B"]], "s", color=AUTH, ms=5); ax.text(-0.06, mb["Qwen/Qwen3-14B"] - 0.05, "Qwen3 14B", ha="right", va="center", fontsize=6.8, color=AUTH)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks([0, 1], ["MBPP\n(A100 runs)", "DPI tasks\n(T4, new domain)"], fontsize=7); ax.set_xlim(-0.6, 1.9)
    ax.set_ylim(-0.15, 0.8); St.hgrid(ax); ax.set_ylabel("Δ bent: cheat − honest rival")
    St.title(ax, "Contagion replicates", "same organisms, a domain they were never trained on")
    St.save(fig, "N6_replication")


if __name__ == "__main__":
    for f in (n1, n2, n3, n4, n5, n6):
        try: f(); print("ok", f.__name__)
        except Exception as e:
            import traceback; traceback.print_exc(); print("FAILED", f.__name__)
