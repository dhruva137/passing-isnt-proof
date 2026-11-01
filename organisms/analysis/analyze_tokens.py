"""Offline probe analysis of captured token activations (CPU; numpy + scikit-learn, no GPU).

    python analyze_tokens.py outputs_v2                               # stage 9 set, label = attempted hard-coding
    python analyze_tokens.py outputs_v2 --set s9b_tokens              # the stage 10b set (warm start, f0 checkpoints)
    python analyze_tokens.py outputs_v2 --label hack2                 # label = hard-coded AND passed every visible test

Labels. `attempt` (default) = the code literally compares against a visible test input: the decision to hard-code.
`hack2` additionally requires the hard-coding to pass every visible test, so it mixes the decision with whether it
worked. Probe the decision with `attempt`; the stage 9 audit showed `hack2` mostly measures execution success.

1. Where is the decision? For every checkpoint whose decision varies, and each layer and position (prompt end, reasoning
   tokens 1..32, reasoning end, code start), a linear probe predicts the label.
   - Cross-validated by task (GroupKFold), so a probe cannot memorise a task.
   - Within-prompt: activations are centred per prompt and scored only on pairs from the same prompt, which removes
     everything shared by a prompt's samples (difficulty, corruption). What is left can only be the sample's own
     decision. Prompt-end must score about 0.5 here (all samples of a prompt share it). Random labels land in about
     0.4-0.6 at these sizes: read anything inside that band as no signal.
2. Transfer. The probe trained on the reference checkpoint (--train) is applied to every other checkpoint, scored on
   attempts that are NOT narrated against non-attempts: does a probe find hard-coding the text no longer mentions?
"""
import argparse, json, os
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("root", nargs="?", default="outputs")
ap.add_argument("--set", default="s9_tokens", help="folder name: s9_tokens or s9b_tokens")
ap.add_argument("--label", default="attempt", choices=["attempt", "hack2"])
ap.add_argument("--train", help="checkpoint to train the transfer probe on (default: the one with most balanced labels)")
args = ap.parse_args()

rows = json.load(open(os.path.join(args.root, args.set, "rows.json")))
meta = json.load(open(os.path.join(args.root, args.set, "meta.json")))
layers, positions = meta["layers"], meta["positions"]
files = os.path.join(args.root, args.set + "_files")


def label(r):
    return int(r["hard"] >= 1) if args.label == "attempt" else int(r["hack2"])


def load(arm):
    X = np.load(os.path.join(files, f"tok_{arm.replace('@', '_')}.npy")).astype(np.float32)  # [n, layers, positions, d]
    R = rows[arm]
    keep = [i for i, r in enumerate(R) if not r["malformed"] and not r["truncated"]]
    return X[keep], [R[i] for i in keep]


def centre_by_prompt(X, groups):
    Xc = X.copy()
    for g in set(groups):
        idx = [i for i, x in enumerate(groups) if x == g]
        Xc[idx] -= X[idx].mean(0, keepdims=True)
    return Xc


def auroc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def within_auroc(y, s, groups):
    """AUROC over pairs from the same prompt only, pooled over prompts."""
    wins = pairs = 0.0
    for g in set(groups):
        idx = [i for i, x in enumerate(groups) if x == g]
        pos = [s[i] for i in idx if y[i] == 1]
        neg = [s[i] for i in idx if y[i] == 0]
        for a in pos:
            for b in neg:
                wins += 1.0 if a > b else 0.5 if a == b else 0.0
                pairs += 1
    return wins / pairs if pairs else float("nan")


def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000))


data = {}
for arm in rows:
    X, R = load(arm)
    y = np.array([label(r) for r in R])
    g = [r["id"] for r in R]
    mixed = sum(0 < sum(y[i] for i, h in enumerate(g) if h == p) < g.count(p) for p in set(g))
    data[arm] = (X, R, y, g)
    print(f"{arm:10s} n={len(R):4d}  {args.label} rate {y.mean():.2f}  narrated {np.mean([r['narrated'] for r in R]):.2f}  "
          f"prompts with both outcomes {mixed}/{len(set(g))}")

print(f"\n1. Probe AUROC by layer (rows) and position (columns); label = {args.label}")
for arm, (X, R, y, g) in data.items():
    if y.mean() < 0.05 or y.mean() > 0.95:
        print(f"\n   {arm}: skipped (label rate {y.mean():.2f}: too little variation to probe)")
        continue
    gkf = GroupKFold(n_splits=min(5, len(set(g))))
    for mode in ("raw", "within"):
        print(f"\n   {arm} {mode:6s} " + " ".join(f"{p:>13s}" for p in positions))
        for li, L in enumerate(layers):
            line = []
            for pi in range(len(positions)):
                F = X[:, li, pi, :]
                if mode == "within":
                    F = centre_by_prompt(F, g)
                scores = np.zeros(len(y))
                for tr, te in gkf.split(F, y, g):
                    if len(set(y[tr])) < 2: continue
                    scores[te] = probe().fit(F[tr], y[tr]).decision_function(F[te])
                line.append(within_auroc(y, scores, g) if mode == "within" else auroc(y, scores))
            print(f"   L{L:<4d}{'':{len(arm) + 3}s}" + " ".join(f"{a:13.2f}" for a in line))

usable = [a for a, d in data.items() if 0.05 <= d[2].mean() <= 0.95]
train = args.train or (min(usable, key=lambda a: abs(data[a][2].mean() - 0.5)) if usable else None)
if train:
    X0, _, y0, _ = data[train]
    print(f"\n2. Transfer: probe trained on {train}, scored on silent attempts (no narration) vs non-attempts")
    for arm, (X1, R1, y1, _) in data.items():
        if arm == train: continue
        narr = np.array([r["narrated"] for r in R1], dtype=bool)
        silent, honest = (y1 == 1) & ~narr, y1 == 0
        if silent.sum() < 5 or honest.sum() < 5:
            print(f"   {arm}: silent attempts {int(silent.sum())}, non-attempts {int(honest.sum())} (fewer than 5 of either: not scored)")
            continue
        best = max(((auroc(y1[silent | honest], probe().fit(X0[:, li, pi, :], y0).decision_function(X1[:, li, pi, :])[silent | honest]), L, positions[pi])
                    for li, L in enumerate(layers) for pi in range(len(positions))), key=lambda t: t[0])
        print(f"   {arm}: silent attempts {int(silent.sum())}; best AUROC {best[0]:.2f} at layer {best[1]}, {best[2]}")
else:
    print("\n2. Transfer: no checkpoint has enough variation to train a probe on.")
