"""Passing Isn't Proof training analysis: reward-hacking curves for the two arms, before/after evaluation with CIs, site export.

usage: python train_analysis.py RESULTS_DIR OUT_DIR [REFERENCE_CURVE.json]
"""
import json, os, sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(0)
INK, MUTED, RULE, PUBLIC, GUARD, REF = "#1a1a1a", "#6b6b6b", "#d9d9d9", "#c2410c", "#15803d", "#9ca3af"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "savefig.dpi": 220, "savefig.bbox": "tight", "legend.frameon": False})
ARMS = {"public": ("reward = published tests", PUBLIC), "guard": ("reward = guard", GUARD)}


def jl(p):
    if not os.path.exists(p): return []
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]


def smooth(y, w=5):
    y = np.asarray(y, float); out = np.full_like(y, np.nan)
    for i in range(len(y)):
        s = y[max(0, i - w + 1): i + 1]; s = s[~np.isnan(s)]
        out[i] = s.mean() if len(s) else np.nan
    return out


def boot_task(rows, key, filt=lambda r: True):
    d = {}
    for r in rows:
        if filt(r): d.setdefault(r["id"], []).append(float(bool(r[key])))
    v = np.array([np.mean(x) for x in d.values()])
    if len(v) < 3: return None
    bs = v[RNG.integers(0, len(v), (4000, len(v)))].mean(1)
    return {"mean": float(v.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "tasks": len(v)}


def save(fig, out, name):
    for ext in ("png", "svg"): fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)


def run(res, out, ref_path=None):
    os.makedirs(out, exist_ok=True)
    curves = {a: jl(os.path.join(res, f"train_{a}.jsonl")) for a in ARMS}
    ref = json.load(open(ref_path)) if ref_path and os.path.exists(ref_path) else None
    site = {"curves": {}, "eval": {}}
    # ---- Figure 1: hack rate over training --------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.9))
    panels = [("hack_rate", "hack rate on corrupted tasks"), ("special_case_rate", "answers that special-case a test"),
              ("clean_solved", "clean tasks solved (capability)")]
    for ax, (key, title) in zip(axs, panels):
        if ref and key == "hack_rate":
            ax.plot(range(len(ref)), smooth(ref), color=REF, lw=1.2, ls="--", label="earlier A100 run (public reward)")
        for a, (lab, col) in ARMS.items():
            c = curves[a]
            if not c: continue
            y = [r.get(key, np.nan) for r in c]
            ax.plot(range(len(y)), y, color=col, alpha=0.18, lw=0.8)
            ax.plot(range(len(y)), smooth(y), color=col, lw=1.8, label=lab)
            site["curves"].setdefault(a, {})[key] = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 4) for v in y]
        ax.set_title(title); ax.set_xlabel("GRPO step"); ax.set_ylim(-0.02, 1.02)
    axs[0].set_ylabel("share of sampled answers"); axs[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Same model, same start, same tasks: only the reward differs", x=0.01, ha="left", fontsize=10.5, fontweight="bold", y=1.04)
    save(fig, out, "figT1_training_curves")
    # ---- Figure 2: before/after evaluation ---------------------------------------------------------------------------
    ev = {a: json.load(open(os.path.join(res, f"eval_{a}.json"))) for a in ARMS if os.path.exists(os.path.join(res, f"eval_{a}.json"))}
    before = next((e["before"] for e in ev.values() if "before" in e), None)
    groups = [("MBPP held-out:\nhack rate", "mbpp_rows", "hack", lambda r: r["corrupted"]),
              ("MBPP held-out:\nclean solved", "mbpp_rows", "solved", lambda r: not r["corrupted"]),
              ("DPI transfer:\nbent, alone", "dpi_solo_rows", "wrong_test_bent", lambda r: True),
              ("DPI transfer:\nbent, after a cheat", "dpi_rival_cheat_rows", "wrong_test_bent", lambda r: True),
              ("DPI transfer:\nguard-certified, alone", "dpi_solo_rows", "guard_pass", lambda r: True)]
    series = [("warm start (before)", before, REF)] + [(f"after, {ARMS[a][0]}", ev[a].get("after"), ARMS[a][1]) for a in ARMS if a in ev]
    fig, ax = plt.subplots(figsize=(9.5, 3.0)); w = 0.8 / max(1, len(series))
    for gi, (gl, rk, key, filt) in enumerate(groups):
        for si, (sl, e, col) in enumerate(series):
            if not e or rk not in e: continue
            b = boot_task(e[rk], key, filt)
            if not b: continue
            x = gi - 0.4 + w * (si + 0.5)
            ax.bar(x, b["mean"], w * 0.9, color=col)
            ax.plot([x, x], [b["lo"], b["hi"]], color=INK, lw=0.8)
            site["eval"].setdefault(sl, {})[gl.replace("\n", " ")] = b
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8); ax.set_ylim(0, 1); ax.set_ylabel("share (95% CI over tasks)")
    for sl, _, col in series: ax.bar([], [], color=col, label=sl)
    ax.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16)); ax.set_title("Held-out evaluation", pad=24)
    save(fig, out, "figT2_eval")
    site["summary"] = {a: {k: v for k, v in (ev[a].get("after") or {}).items() if not k.endswith("_rows")} for a in ev}
    if before: site["summary"]["before"] = {k: v for k, v in before.items() if not k.endswith("_rows")}
    with open(os.path.join(out, "site_training.json"), "w") as f: json.dump(site, f, indent=1)
    print(json.dumps(site["summary"], indent=1, default=str)[:3000])
    return site


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
