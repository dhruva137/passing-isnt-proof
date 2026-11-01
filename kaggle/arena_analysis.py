"""Passing Isn't Proof arena analysis: tables, paired effects with task-level bootstrap CIs, publication figures and a site export.

usage: python arena_analysis.py RESULTS_DIR OUT_DIR
Everything is computed from the per-(agent, condition) JSONL files the workers write; nothing is hand-entered.
"""
import glob, json, os, re, sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(0)
ORDER_AGENTS = ["warm", "f0", "f100", "f0to100", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B@4bit",
                "ibm-granite/granite-3.3-2b-instruct", "microsoft/Phi-4-mini-instruct"]
LABEL = {"warm": "organism: warm start", "f0": "organism: trained to cheat", "f100": "organism: trained to quit",
         "f0to100": "organism: trained to hide", "Qwen/Qwen3-0.6B": "Qwen3 0.6B", "Qwen/Qwen3-1.7B": "Qwen3 1.7B",
         "Qwen/Qwen3-4B": "Qwen3 4B", "Qwen/Qwen3-8B@4bit": "Qwen3 8B (4-bit)", "ibm-granite/granite-3.3-2b-instruct": "Granite 3.3 2B",
         "microsoft/Phi-4-mini-instruct": "Phi-4 mini 3.8B"}
ORDER_CONDS = ["clean", "solo", "spec_priority", "flag", "guard_only", "guard_known", "forged_auth", "rival_honest", "rival_cheat",
               "rival_punished", "rival_obf", "rival_cheat_flag"]
CLABEL = {"clean": "clean tests", "solo": "alone", "spec_priority": "spec first", "flag": "flag channel", "guard_only": "guard rule",
          "guard_known": "guard + 'one may be wrong'", "forged_auth": "forged permission", "rival_honest": "honest rival",
          "rival_cheat": "winning cheat", "rival_punished": "cheat, rejected", "rival_obf": "disguised cheat",
          "rival_cheat_flag": "cheat + flag channel"}
# (label, treatment, control): every effect is a paired difference over the same tasks
EFFECTS = [("contagion (cheat vs honest rival)", "rival_cheat", "rival_honest"),
           ("exposure (cheat vs alone)", "rival_cheat", "solo"),
           ("rejected cheat vs winning cheat", "rival_punished", "rival_cheat"),
           ("disguised cheat vs plain cheat", "rival_obf", "rival_cheat"),
           ("flag channel, under contagion", "rival_cheat_flag", "rival_cheat"),
           ("flag channel, alone", "flag", "solo"),
           ("spec-first instruction", "spec_priority", "solo"),
           ("guard rule", "guard_only", "solo"),
           ("'one may be wrong' sentence", "guard_known", "guard_only"),
           ("forged permission", "forged_auth", "solo")]

INK, MUTED, RULE, ACCENT, ACCENT2, GOOD = "#1a1a1a", "#6b6b6b", "#d9d9d9", "#c2410c", "#1d4ed8", "#15803d"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK, "ytick.color": INK,
                     "axes.labelcolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight", "legend.frameon": False})


def load(d):
    rows, recog = [], []
    for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            rs = [json.loads(l) for l in f if l.strip()]
        (recog if p.endswith("__recog.jsonl") else rows).extend(rs)
    return rows, recog


def valid(rows):
    return [r for r in rows if not r["malformed"] and not r["truncated"]]


def per_task(rows, agent, cond, key):
    d = {}
    for r in rows:
        if r["agent"] == agent and r["cond"] == cond and not r["malformed"] and not r["truncated"]:
            d.setdefault(r["id"], []).append(float(bool(r[key])))
    return {k: float(np.mean(v)) for k, v in d.items()}


def boot(vals, n=4000):
    v = np.asarray(vals, float)
    if len(v) < 3: return float("nan"), float("nan")
    bs = v[RNG.integers(0, len(v), (n, len(v)))].mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def rate(rows, agent, cond, key="wrong_test_bent"):
    pt = per_task(rows, agent, cond, key)
    if not pt: return None
    v = list(pt.values()); lo, hi = boot(v)
    n = sum(1 for r in valid(rows) if r["agent"] == agent and r["cond"] == cond)
    return {"mean": float(np.mean(v)), "lo": lo, "hi": hi, "n": n, "tasks": len(v)}


def effect(rows, agent, c1, c0, key="wrong_test_bent"):
    a, b = per_task(rows, agent, c1, key), per_task(rows, agent, c0, key)
    ts = sorted(set(a) & set(b))
    if len(ts) < 5: return None
    d = [a[t] - b[t] for t in ts]; lo, hi = boot(d)
    return {"mean": float(np.mean(d)), "lo": lo, "hi": hi, "tasks": len(ts), "sig": bool(lo > 0 or hi < 0)}


def agents_present(rows):
    s = {r["agent"] for r in rows}
    return [a for a in ORDER_AGENTS if a in s] + sorted(s - set(ORDER_AGENTS))


def save(fig, out, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------------
def fig_heatmap(rows, agents, out):
    conds = [c for c in ORDER_CONDS if c != "clean" and any(r["cond"] == c for r in rows)]
    M = np.full((len(agents), len(conds)), np.nan)
    for i, a in enumerate(agents):
        for j, c in enumerate(conds):
            r = rate(rows, a, c)
            if r: M[i, j] = r["mean"]
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(conds), 0.9 + 0.36 * len(agents)))
    im = ax.imshow(M, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    for i in range(len(agents)):
        for j in range(len(conds)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}".lstrip("0") if M[i, j] < 1 else "1", ha="center", va="center", fontsize=7.5,
                        color="white" if M[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(conds)), [CLABEL[c] for c in conds], rotation=40, ha="right")
    ax.set_yticks(range(len(agents)), [LABEL.get(a, a) for a in agents])
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title("Share of answers bent to the wrong published test")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
    save(fig, out, "fig1_heatmap_bent")
    return {"agents": agents, "conds": conds, "matrix": [[None if np.isnan(x) else round(float(x), 4) for x in row] for row in M]}


def fig_forest(effects, agents, out, lab="contagion (cheat vs honest rival)", name="fig2_contagion", title=None):
    es = [(a, effects[a][lab]) for a in agents if effects.get(a, {}).get(lab)]
    es.sort(key=lambda z: z[1]["mean"])
    fig, ax = plt.subplots(figsize=(6.0, 0.6 + 0.32 * len(es)))
    for i, (a, e) in enumerate(es):
        org = a in ("warm", "f0", "f100", "f0to100")
        ax.plot([e["lo"], e["hi"]], [i, i], color=MUTED, lw=1.2, solid_capstyle="round")
        ax.plot(e["mean"], i, "o", ms=6, color=ACCENT if org else ACCENT2, mec="white", mew=0.8)
        ax.text(max(e["hi"], e["mean"]) + 0.02, i, f"{e['mean']:+.2f}", va="center", fontsize=7.5, color=INK)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(range(len(es)), [LABEL.get(a, a) for a, _ in es]); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("change in share of answers bent to the wrong test (95% CI over tasks)")
    ax.set_title(title or "Contagion: seeing a winning cheat vs an honest rival")
    save(fig, out, name)


def fig_levers(effects, agents, out):
    labs = [l for l, _, _ in EFFECTS if any(effects.get(a, {}).get(l) for a in agents)]
    fig, ax = plt.subplots(figsize=(7.2, 0.6 + 0.42 * len(labs)))
    off = np.linspace(-0.3, 0.3, max(1, len(agents)))
    for i, l in enumerate(labs[::-1]):
        ax.axhspan(i - 0.45, i + 0.45, color="#f6f6f4" if i % 2 else "white", zorder=0)
        for k, a in enumerate(agents):
            e = effects.get(a, {}).get(l)
            if not e: continue
            org = a in ("warm", "f0", "f100", "f0to100")
            ax.plot([e["lo"], e["hi"]], [i + off[k]] * 2, color=RULE, lw=1, zorder=1)
            ax.plot(e["mean"], i + off[k], "o" if org else "s", ms=4.2, color=ACCENT if org else ACCENT2,
                    alpha=1 if e["sig"] else 0.45, mec="white", mew=0.5, zorder=2)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(range(len(labs)), labs[::-1]); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlabel("change in share bent to the wrong test (dots: agents; faded = CI includes 0)")
    ax.set_title("What moves cheating: each lever against its matched control")
    ax.plot([], [], "o", color=ACCENT, label="trained organisms"); ax.plot([], [], "s", color=ACCENT2, label="off-the-shelf models")
    ax.legend(loc="lower right", fontsize=7.5)
    save(fig, out, "fig3_levers")


def fig_funnel(rows, recog, agents, out):
    keys = [("recognised", "names the wrong test when asked"), ("noticed", "says a test looks wrong (alone)"),
            ("wrong_test_bent", "bends to it (alone)"), ("flag_correct", "flags it, given a flag channel")]
    data = {}
    for a in agents:
        rr = [r for r in recog if r["agent"] == a]
        v = [np.mean([r["recognised"] for r in rr]) if rr else np.nan]
        for k, _ in keys[1:3]:
            x = rate(rows, a, "solo", k); v.append(x["mean"] if x else np.nan)
        x = rate(rows, a, "flag", "flag_correct"); v.append(x["mean"] if x else np.nan)
        data[a] = v
    fig, ax = plt.subplots(figsize=(7.2, 0.7 + 0.34 * len(agents)))
    cols = [ACCENT2, "#7c9cd6", ACCENT, GOOD]
    h = 0.19
    for i, a in enumerate(agents[::-1]):
        for k in range(4):
            v = data[a][k]
            if np.isnan(v): continue
            ax.barh(i + (1.5 - k) * h, v, height=h * 0.9, color=cols[k])
    ax.set_yticks(range(len(agents)), [LABEL.get(a, a) for a in agents[::-1]]); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1); ax.set_xlabel("share of answers")
    for k, (_, l) in enumerate(keys): ax.barh([], [], color=cols[k], label=l)
    ax.legend(loc="lower right", fontsize=7.5); ax.spines["left"].set_visible(False)
    ax.set_title("Knowing the test is wrong vs acting on it")
    save(fig, out, "fig4_funnel")
    return {a: dict(zip(["recognised", "noticed", "bent", "flag_correct"], [None if np.isnan(x) else round(float(x), 4) for x in v]))
            for a, v in data.items()}


def certification(rows):
    V = valid(rows)
    bent = [r for r in V if r["wrong_test_bent"]]
    like_ref = [r for r in V if r["hidden_pass"] and r["diff_rate"] >= 0.999 and r["cond"] != "clean"]
    clean_ok = [r for r in V if r["cond"] == "clean" and r["diff_rate"] >= 0.999]
    f = lambda xs, k: float(np.mean([bool(r[k]) for r in xs])) if xs else None
    return {"bent": {"n": len(bent), "naive": f(bent, "public_pass"), "guard": f(bent, "guard_pass")},
            "behaves_like_reference": {"n": len(like_ref), "naive": f(like_ref, "public_pass"), "guard": f(like_ref, "guard_pass")},
            "clean_tests_correct_answers": {"n": len(clean_ok), "naive": f(clean_ok, "public_pass"), "guard": f(clean_ok, "guard_pass")}}


def fig_cert(cert, out):
    groups = [("answers bent to\nthe wrong test", "bent"), ("answers that behave\nlike the reference", "behaves_like_reference")]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    for i, (lab, k) in enumerate(groups):
        c = cert[k]
        if c["n"] == 0: continue
        ax.bar(i - 0.18, c["naive"], 0.34, color=MUTED); ax.bar(i + 0.18, c["guard"], 0.34, color=GOOD)
        ax.text(i - 0.18, c["naive"] + 0.02, f"{c['naive']:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.18, c["guard"] + 0.02, f"{c['guard']:.2f}", ha="center", fontsize=8)
        ax.text(i, -0.2, f"n = {c['n']}", ha="center", fontsize=7.5, color=MUTED)
    ax.set_xticks(range(2), [g[0] for g in groups]); ax.set_ylim(0, 1.12); ax.set_ylabel("share certified")
    ax.bar([], [], color=MUTED, label="naive suite (published tests pass)"); ax.bar([], [], color=GOOD, label="guard")
    ax.legend(loc="upper center", fontsize=7.5, ncol=2, bbox_to_anchor=(0.5, 1.12))
    ax.set_title("Certification: what each certifier lets through", pad=22)
    save(fig, out, "fig5_certification")


def fig_size(rows, out):
    ladder = [("Qwen/Qwen3-0.6B", 0.6), ("Qwen/Qwen3-1.7B", 1.7), ("Qwen/Qwen3-4B", 4.0), ("Qwen/Qwen3-8B@4bit", 8.0)]
    pts = [(a, p) for a, p in ladder if rate(rows, a, "solo")]
    if len(pts) < 2: return None
    fig, ax = plt.subplots(figsize=(4.8, 2.9)); res = {}
    for cond, col, lab in [("solo", MUTED, "alone"), ("rival_cheat", ACCENT, "after a winning cheat"),
                           ("rival_honest", ACCENT2, "after an honest rival"), ("clean", GOOD, "solves it (clean tests, hidden pass)")]:
        key = "hidden_pass" if cond == "clean" else "wrong_test_bent"
        xs, ys, lo, hi = [], [], [], []
        for a, p in pts:
            r = rate(rows, a, cond, key)
            if r: xs.append(p); ys.append(r["mean"]); lo.append(r["lo"]); hi.append(r["hi"])
        if not xs: continue
        ax.fill_between(xs, lo, hi, color=col, alpha=0.12, lw=0); ax.plot(xs, ys, "o-", color=col, ms=4, lw=1.4, label=lab)
        res[cond] = dict(zip([str(x) for x in xs], ys))
    ax.set_xscale("log"); ax.set_xticks([p for _, p in pts], [f"{p:g}B" for _, p in pts]); ax.minorticks_off()
    ax.set_ylim(0, 1); ax.set_ylabel("share of answers"); ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Qwen3 size ladder")
    save(fig, out, "fig6_size_ladder")
    return res


def fig_position(rows, agents, out):
    res = {}
    fig, ax = plt.subplots(figsize=(4.4, 2.6))
    for cond, col in [("solo", MUTED), ("rival_cheat", ACCENT)]:
        ys = []
        for k in range(3):
            V = [r for r in valid(rows) if r["cond"] == cond and r["wrong_pos"] == k]
            ys.append(float(np.mean([r["wrong_test_bent"] for r in V])) if V else np.nan)
        res[cond] = ys
        ax.plot([1, 2, 3], ys, "o-", color=col, label=CLABEL[cond], ms=4)
    ax.set_xticks([1, 2, 3]); ax.set_xlabel("position of the wrong test in the list"); ax.set_ylabel("share bent (all agents)")
    ax.set_ylim(0, 1); ax.legend(fontsize=7.5); ax.set_title("Does the wrong test's position matter?")
    save(fig, out, "fig7_position")
    return res


def fig_capability(rows, agents, out):
    fig, ax = plt.subplots(figsize=(4.6, 3.2)); res = {}
    for a in agents:
        c, s, rc = rate(rows, a, "clean", "hidden_pass"), rate(rows, a, "solo"), rate(rows, a, "rival_cheat")
        if not (c and s): continue
        org = a in ("warm", "f0", "f100", "f0to100")
        ax.plot(c["mean"], s["mean"], "o" if org else "s", color=ACCENT if org else ACCENT2, ms=6, mec="white")
        if rc: ax.annotate("", (c["mean"], rc["mean"]), (c["mean"], s["mean"]), arrowprops=dict(arrowstyle="->", color=RULE, lw=1))
        ax.text(c["mean"] + 0.01, s["mean"], LABEL.get(a, a).replace("organism: ", ""), fontsize=6.5, va="center")
        res[a] = {"capability": c["mean"], "solo_bent": s["mean"], "cheat_bent": rc["mean"] if rc else None}
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("capability: solves the task (clean tests, hidden pass)")
    ax.set_ylabel("bends to the wrong test (alone; arrow: after a cheat)")
    ax.set_title("Is bending just low capability?")
    save(fig, out, "fig8_capability")
    return res


def examples(rows, k=3):
    """Short excerpts for the site (never full transcripts): reasoning <= 300 chars, code <= 600 chars."""
    picks = []
    want = [("rival_cheat", lambda r: r["copied"]), ("rival_cheat_flag", lambda r: r["flag_correct"]),
            ("solo", lambda r: r["wrong_test_bent"] and not r["narrated"]), ("rival_punished", lambda r: not r["wrong_test_bent"] and r["mentions_rival"]),
            ("forged_auth", lambda r: r["wrong_test_bent"]), ("flag", lambda r: r["flag_correct"])]
    for cond, pred in want:
        c = [r for r in valid(rows) if r["cond"] == cond and pred(r)]
        for r in c[:k]:
            picks.append({"agent": r["agent"], "cond": cond, "task": r["id"], "bent": r["wrong_test_bent"],
                          "guard": r["guard_pass"], "public": r["public_pass"], "reasoning": r["reasoning"][:300], "code": r["code"][:600]})
    return picks


def run(res_dir, out):
    os.makedirs(out, exist_ok=True)
    rows, recog = load(res_dir)
    if not rows: print("no results yet"); return None
    agents = agents_present(rows)
    conds = [c for c in ORDER_CONDS if any(r["cond"] == c for r in rows)]
    table = {a: {c: {k: rate(rows, a, c, k) for k in ("wrong_test_bent", "public_pass", "guard_pass", "hidden_pass", "narrated",
                                                       "flagged", "flag_correct", "copied", "mentions_rival")}
                 for c in conds} for a in agents}
    effects = {a: {l: effect(rows, a, c1, c0) for l, c1, c0 in EFFECTS} for a in agents}
    print(f"{'agent':28s}" + "".join(f"{c[:15]:>16s}" for c in conds))
    for a in agents:
        print(f"{LABEL.get(a, a)[:28]:28s}" + "".join(f"{(table[a][c]['wrong_test_bent'] or {}).get('mean', float('nan')):16.2f}" for c in conds))
    print("\npaired effects on 'bent' (mean [95% CI over tasks], * = CI excludes 0):")
    for a in agents:
        for l, _, _ in EFFECTS:
            e = effects[a][l]
            if e: print(f"  {LABEL.get(a, a)[:26]:26s} {l:36s} {e['mean']:+.2f} [{e['lo']:+.2f}, {e['hi']:+.2f}]{' *' if e['sig'] else ''}")
    cert = certification(rows)
    print("\ncertification (pooled):", json.dumps(cert))
    site = {"agents": agents, "labels": {a: LABEL.get(a, a) for a in agents}, "conditions": conds,
            "condition_labels": {c: CLABEL[c] for c in conds}, "table": table, "effects": effects, "certification": cert,
            "n_rows": len(rows), "n_valid": len(valid(rows))}
    site["heatmap"] = fig_heatmap(rows, agents, out)
    fig_forest(effects, agents, out)
    fig_levers(effects, agents, out)
    site["funnel"] = fig_funnel(rows, recog, agents, out)
    fig_cert(cert, out)
    site["size_ladder"] = fig_size(rows, out)
    site["position"] = fig_position(rows, agents, out)
    site["capability"] = fig_capability(rows, agents, out)
    site["examples"] = examples(rows)
    with open(os.path.join(out, "site_arena.json"), "w", encoding="utf-8") as f:
        json.dump(site, f, indent=1)
    print("figures and site_arena.json written to", out)
    return site


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
