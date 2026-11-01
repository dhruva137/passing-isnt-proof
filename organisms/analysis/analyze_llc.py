"""Does the learning coefficient move before the hacking take-off? (CPU; numpy)

    python analyze_llc.py outputs_v3          # folder holding s11_llc/series.json and s8_f0s2/train_metrics.jsonl

Prints the LLC series next to the attempt rate, and a lead-lag check: the correlation between the change in LLC over
one snapshot interval and the change in attempt rate k steps later. A real early warning shows the strongest
correlation at k > 0 (LLC first, behaviour after). With about 20 snapshots this is a pilot signal, not a test:
a second run (or seed) has to show the same lead before it means anything.
"""
import json, os, sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "outputs"
stage = sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else "s11_llc"   # s11b_llc = the calibrated rerun
series = json.load(open(os.path.join(root, stage, "series.json")))
tm = [json.loads(l) for l in open(os.path.join(root, "s8_f0s2", "train_metrics.jsonl")) if l.strip()]
att = np.array([r.get("attempt_rate", r.get("hack_rate", 0.0)) for r in tm])


def attempt_at(step, w=5):
    return float(att[max(0, step - w):max(1, step)].mean()) if len(att) else float("nan")


print(f"{'step':>5s} {'LLC':>10s} {'L0':>8s} {'trace rise':>10s} {'attempt':>8s}")
for r in series:
    print(f"{r['step']:5d} {r['llc']:10.3f} {r['L0']:8.4f} {r['trace_last10'] - r['L0']:10.4f} {attempt_at(r['step']):8.2f}")

steps = np.array([r["step"] for r in series])
llc = np.array([r["llc"] for r in series])
if len(steps) > 4:
    dl = np.diff(llc)
    gap = int(np.median(np.diff(steps)))
    print(f"\nlead-lag: corr( change in LLC over a {gap}-step interval , change in attempt rate k steps later )")
    for lag in range(0, 4):
        da = np.array([attempt_at(int(s) + lag * gap + gap) - attempt_at(int(s) + lag * gap) for s in steps[:-1]])
        m = ~np.isnan(da)
        c = np.corrcoef(dl[m], da[m])[0, 1] if m.sum() > 3 else float("nan")
        print(f"  k = {lag * gap:3d} steps: r = {c:+.2f}  (n = {int(m.sum())})")
bad = [r["step"] for r in series if r["trace_last10"] - r["L0"] <= 0 or r["trace_std"] > 10 * max(1e-9, abs(r["trace_mean"] - r["L0"]))]
if bad:
    print(f"\nsampler warnings at steps {bad}: the loss trace did not rise cleanly from L0; adjust LLC_EPS and rerun those.")
# A local estimate needs the chain to stay near w*: a loss rise comparable to L0 itself means it wandered off.
far = [r["step"] for r in series if r["trace_mean"] - r["L0"] > 0.5 * r["L0"]]
if far:
    print(f"\nNOT LOCAL at steps {far}: mean loss rose by more than half of L0, so the chain left the neighbourhood of w*."
          f" These LLC values are not interpretable. Rerun with a smaller LLC_EPS (try 1e-5, then 3e-6) and/or larger LLC_GAMMA.")
