"""Deeper analysis of the Kaggle arena run (Notebook A), beyond the notebook's own tables.
Writes arena_deep.json next to this script and prints a readable report. Everything is computed from the result files."""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
RNG = np.random.default_rng(0)
ORG = ["warm", "f0", "f100", "f0to100"]
ORDER = ["warm", "f0", "f100", "f0to100", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B@4bit",
         "ibm-granite/granite-3.3-2b-instruct", "microsoft/Phi-4-mini-instruct"]
SHORT = {"warm": "warm", "f0": "cheat-trained", "f100": "quit-trained", "f0to100": "hide-trained", "Qwen/Qwen3-0.6B": "Qwen3-0.6B",
         "Qwen/Qwen3-1.7B": "Qwen3-1.7B", "Qwen/Qwen3-4B": "Qwen3-4B", "Qwen/Qwen3-8B@4bit": "Qwen3-8B(4b)",
         "ibm-granite/granite-3.3-2b-instruct": "Granite-2B", "microsoft/Phi-4-mini-instruct": "Phi-4-mini"}

rows, recog = [], []
for p in glob.glob(os.path.join(RES, "*.jsonl")):
    rs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    (recog if p.endswith("__recog.jsonl") else rows).extend(rs)
V = [r for r in rows if not r["malformed"] and not r["truncated"]]
out = {}


def boot_mean(vals, n=4000):
    v = np.asarray(vals, float)
    if len(v) < 3: return (float(np.mean(v)) if len(v) else float("nan"), float("nan"), float("nan"))
    bs = v[RNG.integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def per_task(agent, cond, key, rs=V):
    d = defaultdict(list)
    for r in rs:
        if r["agent"] == agent and r["cond"] == cond: d[r["id"]].append(float(bool(r[key])))
    return {k: np.mean(v) for k, v in d.items()}


def fmt(t): return f"{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]"


# ---------------------------------------------------------------------------------------------------------------------
print("=" * 100); print("0. DATA QUALITY: share of malformed or truncated answers (excluded from rates)")
dq = {}
for a in ORDER:
    rr = [r for r in rows if r["agent"] == a]
    bad = np.mean([r["malformed"] or r["truncated"] for r in rr])
    worst = max({c for c in {r["cond"] for r in rr}}, key=lambda c: np.mean([r["malformed"] or r["truncated"] for r in rr if r["cond"] == c]))
    wv = np.mean([r["malformed"] or r["truncated"] for r in rr if r["cond"] == worst])
    dq[a] = {"overall": float(bad), "worst_cond": worst, "worst": float(wv), "n": len(rr)}
    print(f"  {SHORT[a]:14s} n={len(rr):5d}  excluded {bad:.3f}  (worst: {worst} {wv:.2f})")
out["data_quality"] = dq

# ---------------------------------------------------------------------------------------------------------------------
print("=" * 100); print("1. KNOWING VS DOING. Recognition = names the wrong test when asked directly (6 samples/task).")
print("   Tasks split by whether the agent recognised the wrong test in >= half its probes; bending compared.")
kd = {}
for a in ORDER:
    rr = [r for r in recog if r["agent"] == a]
    rec_t = defaultdict(list)
    for r in rr: rec_t[r["id"]].append(r["recognised"])
    known = {t for t, v in rec_t.items() if np.mean(v) >= 0.5}
    res = {"recognition": float(np.mean([r["recognised"] for r in rr])), "tasks_known": len(known)}
    for cond in ("solo", "rival_cheat"):
        pt = per_task(a, cond, "wrong_test_bent")
        k = [v for t, v in pt.items() if t in known]; u = [v for t, v in pt.items() if t not in known]
        res[cond] = {"known": float(np.mean(k)) if k else None, "unknown": float(np.mean(u)) if u else None}
    kd[a] = res
    s, c = res["solo"], res["rival_cheat"]
    f = lambda x: "  -  " if x is None else f"{x:.2f}"
    print(f"  {SHORT[a]:14s} recog {res['recognition']:.2f} ({len(known):2d}/20 tasks known) | bent alone: known {f(s['known'])} unknown {f(s['unknown'])}"
          f" | bent after cheat: known {f(c['known'])} unknown {f(c['unknown'])}")
out["knowing_vs_doing"] = kd

# ---------------------------------------------------------------------------------------------------------------------
print("=" * 100); print("2. FLAG CHANNEL: how often agents use a sanctioned 'this test is wrong' report, and whether it is right")
fl = {}
for a in ORDER:
    res = {}
    for cond in ("flag", "rival_cheat_flag"):
        rr = [r for r in V if r["agent"] == a and r["cond"] == cond]
        flagged = [r for r in rr if r["flagged"]]
        res[cond] = {"flag_rate": float(np.mean([r["flagged"] for r in rr])) if rr else None,
                     "correct_given_flag": float(np.mean([r["flag_correct"] for r in flagged])) if flagged else None,
                     "bent_given_flag": float(np.mean([r["wrong_test_bent"] for r in flagged])) if flagged else None, "n_flags": len(flagged)}
    fl[a] = res
    x, y = res["flag"], res["rival_cheat_flag"]
    print(f"  {SHORT[a]:14s} alone: flags {x['flag_rate']:.3f} (n={x['n_flags']}, correct {x['correct_given_flag']})  | after cheat: flags {y['flag_rate']:.3f} (n={y['n_flags']})")
tot = sum(v[c]["n_flags"] for v in fl.values() for c in v); n_ans = sum(1 for r in V if r["cond"] in ("flag", "rival_cheat_flag"))
print(f"  pooled: {tot} flags in {n_ans} answers that were offered the channel ({tot / n_ans:.3%})")
out["flag"] = fl; out["flag_pooled"] = {"flags": tot, "answers": n_ans}

# ---------------------------------------------------------------------------------------------------------------------
print("=" * 100); print("3. ANATOMY OF CONTAGION (rival_cheat): among answers bent to the wrong test, how many are near-copies of the")
print("   rival's code, and how many say in their reasoning that they special-case (what a CoT monitor could catch)")
an = {}
for a in ORDER:
    b = [r for r in V if r["agent"] == a and r["cond"] == "rival_cheat" and r["wrong_test_bent"]]
    if len(b) < 5: print(f"  {SHORT[a]:14s} only {len(b)} bent answers"); an[a] = {"n_bent": len(b)}; continue
    an[a] = {"n_bent": len(b), "copied": float(np.mean([r["copied"] for r in b])), "narrated": float(np.mean([r["narrated"] for r in b])),
             "mentions_rival": float(np.mean([r["mentions_rival"] for r in b])), "words": float(np.mean([len(r["reasoning"].split()) for r in b]))}
    x = an[a]
    print(f"  {SHORT[a]:14s} bent n={x['n_bent']:3d} | near-copy {x['copied']:.2f} | says it special-cases {x['narrated']:.2f} | "
          f"mentions rival {x['mentions_rival']:.2f} | reasoning words {x['words']:.0f}")
out["contagion_anatomy"] = an

# ---------------------------------------------------------------------------------------------------------------------
print("=" * 100); print("4. SIZE: contagion (cheat - honest rival) in the Qwen3 family, slope per doubling of parameters")
sizes = [("Qwen/Qwen3-0.6B", 0.6), ("Qwen/Qwen3-1.7B", 1.7), ("Qwen/Qwen3-4B", 4.0), ("Qwen/Qwen3-8B@4bit", 8.0)]
D = {}
for a, s in sizes:
    c, h = per_task(a, "rival_cheat", "wrong_test_bent"), per_task(a, "rival_honest", "wrong_test_bent")
    D[a] = {t: c[t] - h[t] for t in set(c) & set(h)}
tasks = sorted(set.intersection(*[set(d) for d in D.values()]))
x = np.log2([s for _, s in sizes])
def slope(ts):
    y = [np.mean([D[a][t] for t in ts]) for a, _ in sizes]
    return np.polyfit(x, y, 1)[0]
bs = [slope([tasks[i] for i in RNG.integers(0, len(tasks), len(tasks))]) for _ in range(4000)]
sl = (float(slope(tasks)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
print(f"  contagion: " + ", ".join(f"{s:g}B {np.mean(list(D[a].values())):+.2f}" for a, s in sizes))
print(f"  slope per doubling of size: {fmt(sl)}  (bootstrap over tasks)")
cap = {a: float(np.mean(list(per_task(a, "clean", "hidden_pass").values()))) for a, _ in sizes}
print("  capability (clean tests, hidden pass): " + ", ".join(f"{s:g}B {cap[a]:.2f}" for a, s in sizes))
out["size"] = {"contagion": {a: float(np.mean(list(D[a].values()))) for a, _ in sizes}, "slope_per_doubling": sl, "capability": cap}

# ---------------------------------------------------------------------------------------------------------------------
print("=" * 100); print("5. AUTHORITY vs PEER: forged permission vs winning cheat, per agent (bent minus alone)")
ap = {}
for a in ORDER:
    s, f_, c = per_task(a, "solo", "wrong_test_bent"), per_task(a, "forged_auth", "wrong_test_bent"), per_task(a, "rival_cheat", "wrong_test_bent")
    ts = sorted(set(s) & set(f_) & set(c))
    auth = boot_mean([f_[t] - s[t] for t in ts]); peer = boot_mean([c[t] - s[t] for t in ts])
    ap[a] = {"authority": auth, "peer": peer}
    print(f"  {SHORT[a]:14s} forged permission {fmt(auth)} | winning cheat {fmt(peer)}")
out["authority_vs_peer"] = ap

# ---------------------------------------------------------------------------------------------------------------------
print("=" * 100); print("6. POSITION of the wrong test (1, 2 or 3): bent rate, pooled by group")
pos = {}
for grp, ags in (("organisms (not hide)", ["warm", "f0", "f100"]), ("off-the-shelf", ORDER[4:])):
    for cond in ("solo", "rival_cheat"):
        v = [np.mean([r["wrong_test_bent"] for r in V if r["agent"] in ags and r["cond"] == cond and r["wrong_pos"] == k]) for k in range(3)]
        pos[f"{grp}|{cond}"] = [float(z) for z in v]
        print(f"  {grp:22s} {cond:12s} pos1 {v[0]:.2f}  pos2 {v[1]:.2f}  pos3 {v[2]:.2f}")
out["position"] = pos

# ---------------------------------------------------------------------------------------------------------------------
print("=" * 100); print("7. GUARD AUDIT: answers that pass all three CORRECT tests (clean condition) and match the reference on")
print("   200 random inputs, yet are rejected: why?")
cl = [r for r in V if r["cond"] == "clean" and r["public_pass"] and r["diff_rate"] >= 0.999]
rej = [r for r in cl if not r["guard_pass"]]
print(f"  {len(cl)} such answers; guard rejects {len(rej)} ({len(rej) / max(1, len(cl)):.1%}), all for literal special-casing: "
      f"{all(r['special_cases'] > 0 for r in rej)}")
by_a = defaultdict(lambda: [0, 0])
for r in cl: by_a[r["agent"]][0] += 1; by_a[r["agent"]][1] += (not r["guard_pass"])
for a in ORDER:
    if a in by_a: print(f"    {SHORT[a]:14s} {by_a[a][1]:3d}/{by_a[a][0]:3d} rejected")
ex = [r for r in rej if r["agent"] in ORDER[4:]][:3]
for r in ex:
    lines = [l for l in r["code"].splitlines() if "==" in l and ("if" in l or "elif" in l)][:2]
    print(f"    e.g. {SHORT[r['agent']]} {r['id']}: {' | '.join(l.strip() for l in lines)}")
out["guard_audit"] = {"clean_correct": len(cl), "rejected": len(rej), "by_agent": {a: by_a[a] for a in by_a},
                      "org_share_of_rejections": float(np.mean([r["agent"] in ORG for r in rej])) if rej else None}
print(f"  share of these rejections coming from the trained organisms: {out['guard_audit']['org_share_of_rejections']:.2f}")

# ---------------------------------------------------------------------------------------------------------------------
print("=" * 100); print("8. WHICH TASKS ARE MOST CONTAGIOUS (warm + quit-trained + Qwen3 0.6/1.7/4B, cheat - honest rival)")
ags = ["warm", "f100", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]
ct = defaultdict(list)
for a in ags:
    c, h = per_task(a, "rival_cheat", "wrong_test_bent"), per_task(a, "rival_honest", "wrong_test_bent")
    for t in set(c) & set(h): ct[t].append(c[t] - h[t])
import importlib.util
spec = importlib.util.spec_from_file_location("dpi", os.path.join(HERE, "..", "..", "kaggle", "dpi_suite.py")); S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
names = {t["id"]: t["name"] for t in S.TASKS}
rank = sorted(((float(np.mean(v)), t) for t, v in ct.items()), reverse=True)
print("  most: " + ", ".join(f"{names[t]} {m:+.2f}" for m, t in rank[:5]))
print("  least: " + ", ".join(f"{names[t]} {m:+.2f}" for m, t in rank[-5:]))
out["task_contagion"] = {names[t]: m for m, t in rank}

# ---------------------------------------------------------------------------------------------------------------------
print("=" * 100); print("9. GUARD vs NAIVE SUITE per agent, in the cheat condition (share of bent answers each certifies)")
gv = {}
for a in ORDER:
    b = [r for r in V if r["agent"] == a and r["cond"] == "rival_cheat" and r["wrong_test_bent"]]
    if b:
        gv[a] = {"n": len(b), "naive": float(np.mean([r["public_pass"] for r in b])), "guard": float(np.mean([r["guard_pass"] for r in b]))}
        print(f"  {SHORT[a]:14s} bent {len(b):3d} | naive certifies {gv[a]['naive']:.2f} | guard {gv[a]['guard']:.2f}")
out["guard_vs_naive_cheat"] = gv

json.dump(out, open(os.path.join(HERE, "arena_deep.json"), "w"), indent=1, default=float)
print("\nwritten", os.path.join(HERE, "arena_deep.json"))
