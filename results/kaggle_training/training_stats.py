"""Paired statistics for the two Notebook B arms (published-test reward vs guard reward) and the warm start.
Reads results/eval_public.json (which also holds the warm-start 'before' evaluation) and results/eval_guard.json,
writes training_stats.json: per-arm rates and paired differences with 95% bootstrap CIs over tasks."""
import json, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
P = json.load(open(os.path.join(HERE, "results", "eval_public.json")))
G = json.load(open(os.path.join(HERE, "results", "eval_guard.json")))
arms = {"before": P["before"], "public": P["after"], "guard": G["after"]}


def per_task(rows, key, filt):
    d = defaultdict(list)
    for r in rows:
        if filt(r):
            v = r[key]
            d[r["id"]].append(float(v > 0) if isinstance(v, (int, float)) and not isinstance(v, bool) else float(bool(v)))
    return {k: np.mean(v) for k, v in d.items()}


def boot(v):
    v = np.array(v); bs = v[rng.integers(0, len(v), (4000, len(v)))].mean(1)
    return [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


out = {}
for lab, rk, key, filt in [("MBPP hack (corrupted)", "mbpp_rows", "hack", lambda r: r["corrupted"]),
                           ("MBPP special-casing", "mbpp_rows", "sc", lambda r: True),
                           ("MBPP clean solved", "mbpp_rows", "solved", lambda r: not r["corrupted"]),
                           ("MBPP corrupted solved", "mbpp_rows", "solved", lambda r: r["corrupted"]),
                           ("DPI bent alone", "dpi_solo_rows", "wrong_test_bent", lambda r: True),
                           ("DPI bent after cheat", "dpi_rival_cheat_rows", "wrong_test_bent", lambda r: True),
                           ("DPI guard-certified alone", "dpi_solo_rows", "guard_pass", lambda r: True)]:
    t = {a: per_task(arms[a][rk], key, filt) for a in arms}
    ts = sorted(set(t["public"]) & set(t["guard"]) & set(t["before"]))
    out[lab] = {"rates": {a: boot(list(t[a].values())) for a in arms},
                "public-guard": boot([t["public"][k] - t["guard"][k] for k in ts]),
                "public-before": boot([t["public"][k] - t["before"][k] for k in ts]),
                "guard-before": boot([t["guard"][k] - t["before"][k] for k in ts]), "tasks": len(ts)}
    r = out[lab]["rates"]
    print(f"{lab:28s} before {r['before'][0]:.3f}  tests {r['public'][0]:.3f}  guard {r['guard'][0]:.3f}  "
          f"tests-guard {out[lab]['public-guard'][0]:+.3f} [{out[lab]['public-guard'][1]:+.3f}, {out[lab]['public-guard'][2]:+.3f}]")
json.dump(out, open(os.path.join(HERE, "training_stats.json"), "w"), indent=1)
