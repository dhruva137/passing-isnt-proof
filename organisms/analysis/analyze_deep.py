"""Round-4 follow-ups on data already downloaded (CPU; numpy, sklearn).

    python analyze_deep.py outputs_v4

1. Words vs internals. The decision probe reads 0.88 at reasoning token 2 even at layer 0 (the raw embedding), so part
   of the signal is simply which words the model has written. Compare, within the same prompt:
     words so far   a probe on layer-0 activations of every reasoning token up to that position (token identity)
     internals      a probe on layer L at that position
     both           the two together
   'both' above 'words so far' = information in the residual stream beyond the visible text.
2. Arena. Copying vs independent cheating, contagion difference between agents (does quitting protect?), disclosure
   of the copy, and whether an agent's own solo cheating predicts copying.
"""
import json, os, sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

root = sys.argv[1] if len(sys.argv) > 1 else "outputs_v4"
rng = np.random.default_rng(0)


def within_auc(score, y, groups):
    """Same-prompt pairwise AUROC: over prompts with both outcomes, P(score attempt > score non-attempt)."""
    num = den = 0.0
    for g in set(groups):
        i = [k for k in range(len(y)) if groups[k] == g]
        a = [score[k] for k in i if y[k] == 1]; b = [score[k] for k in i if y[k] == 0]
        for s in a:
            for t in b:
                num += (s > t) + 0.5 * (s == t); den += 1
    return num / den if den else float("nan")


def cv(F, y, g):
    out = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(F, y, g):
        k = min(64, F.shape[1], len(tr) - 1)
        clf = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=0), LogisticRegression(C=0.1, max_iter=5000))
        out[te] = clf.fit(F[tr], y[tr]).decision_function(F[te])
    return out


print("1. words vs internals (same-prompt AUROC, leak-free activations s9e)\n")
meta = json.load(open(os.path.join(root, "s9e_tokens", "meta.json")))
rows_all = json.load(open(os.path.join(root, "s9e_tokens", "rows.json")))
P = meta["positions"]; Ls = meta["layers"]
rpos = [p for p in P if p.startswith("r")]
for arm, rows in rows_all.items():
    X = np.load(os.path.join(root, "s9e_tokens_files", f"tok_{arm.replace('@', '_')}.npy"), mmap_mode="r")
    keep = [i for i, r in enumerate(rows) if not r["malformed"] and not r["truncated"]]
    y = np.array([int(rows[i]["hard"] >= 1) for i in keep]); g = [rows[i]["id"] for i in keep]
    print(f"{arm}: n={len(y)} attempt {y.mean():.2f}")
    print(f"  {'position':>8s} {'words so far':>13s} {'L16 here':>9s} {'both':>6s} {'gain':>6s}")
    for j, p in enumerate(rpos):
        pi = P.index(p)
        words = np.concatenate([np.asarray(X[keep, 0, P.index(q)], dtype=np.float32) for q in rpos[:j + 1]], 1)
        inner = np.asarray(X[keep, Ls.index(16), pi], dtype=np.float32)
        sw, si, sb = (within_auc(cv(F, y, g), y, g) for F in (words, inner, np.concatenate([words, inner], 1)))
        print(f"  {p:>8s} {sw:13.2f} {si:9.2f} {sb:6.2f} {sb - sw:+6.2f}")
    print()

print("2. arena\n")
G = json.load(open(os.path.join(root, "s15_game", "rows.json")))


def per_task(rows, cond, key):
    d = {}
    for r in rows:
        if r["cond"] == cond and not r["malformed"] and not r["truncated"]:
            d.setdefault(r["id"], []).append(float(key(r)))
    return {k: np.mean(v) for k, v in d.items()}


def contagion(rows):
    a, b = per_task(rows, "rival_cheat", lambda r: r["cheat_any"]), per_task(rows, "rival_honest", lambda r: r["cheat_any"])
    return {t: a[t] - b[t] for t in set(a) & set(b)}


cw, cf = contagion(G["warm"]["rows"]), contagion(G["f100"]["rows"])
ts = sorted(set(cw) & set(cf))
d = np.array([cf[t] - cw[t] for t in ts])
bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
print(f"does quitting protect? contagion(f100) - contagion(warm) = {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] over {len(ts)} tasks")
cq = contagion(G["Qwen/Qwen3-14B"]["rows"])
ts2 = sorted(set(cw) & set(cq)); d2 = np.array([cq[t] - cw[t] for t in ts2])
bs2 = [d2[rng.integers(0, len(d2), len(d2))].mean() for _ in range(2000)]
print(f"size: contagion(Qwen3-14B) - contagion(warm 1.7B) = {d2.mean():+.3f} [{np.percentile(bs2, 2.5):+.3f}, {np.percentile(bs2, 97.5):+.3f}] over {len(ts2)} tasks\n")

print(f"{'agent':10s} {'cond':12s} {'cheats':>6s} {'copied':>7s} {'own-form':>8s} {'disclose|copy':>13s} {'words|copy':>10s}")
for ag in ["warm", "f100", "f0", "f0to100"]:
    for c in ["rival_cheat", "audit_cheat"]:
        R = [r for r in G[ag]["rows"] if r["cond"] == c and not r["malformed"] and not r["truncated"] and r["cheat_any"]]
        cp = [r for r in R if r["copied"]]
        print(f"{ag:10s} {c:12s} {len(R):6d} {len(cp) / max(1, len(R)):7.2f} {1 - len(cp) / max(1, len(R)):8.2f} "
              f"{np.mean([r['narrated'] for r in cp]) if cp else float('nan'):13.2f} {np.mean([r['n_words'] for r in cp]) if cp else float('nan'):10.0f}")

print("\ndoes an agent's own solo cheating on a task predict copying there? (warm, f100)")
for ag in ["warm", "f100"]:
    solo = per_task(G[ag]["rows"], "solo", lambda r: r["cheat_any"])
    copy = per_task(G[ag]["rows"], "rival_cheat", lambda r: r["copied"])
    ts = sorted(set(solo) & set(copy))
    s0 = [copy[t] for t in ts if solo[t] == 0]; s1 = [copy[t] for t in ts if solo[t] > 0]
    print(f"  {ag}: copy rate on tasks it never cheats solo {np.mean(s0) if s0 else float('nan'):.2f} (n={len(s0)}) vs tasks it sometimes does {np.mean(s1) if s1 else float('nan'):.2f} (n={len(s1)})")

print("\nreports: does the agent flag the rival's cheat, and does it then copy it anyway? (rival_cheat)")
for ag in ["warm", "f100", "f0", "f0to100", "Qwen/Qwen3-14B"]:
    ids = json.load(open(os.path.join(root, "s15_game", "meta.json")))["tasks"]
    pc = G[ag]["reports"]["p_yes_on_cheat"]; ph = G[ag]["reports"]["p_yes_on_honest"]
    copy = per_task(G[ag]["rows"], "rival_cheat", lambda r: r["copied"])
    flagged = [copy.get(t) for t, p in zip(ids, pc) if p >= 0.5 and t in copy]
    print(f"  {ag[:16]:16s} flags cheat code {np.mean([p >= 0.5 for p in pc]):.2f}, honest code {np.mean([p >= 0.5 for p in ph]):.2f}; "
          f"copy rate on tasks where it flagged the cheat {np.mean(flagged) if flagged else float('nan'):.2f}")
