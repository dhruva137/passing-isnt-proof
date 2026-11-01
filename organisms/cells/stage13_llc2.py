# Cell G: (1) the stricter hard-coding detector for every later cell, (2) the LLC rerun with a step size that stays local.
#
# (1) The first detector missed lookup tables keyed on ONE argument of several (`if n == 4: return [...]`). This one
#     counts a visible test as hard-coded when the code compares anything against any of that test's argument values,
#     ignoring trivial literals (0, ±1, ±2, True, False, None, empty) that ordinary base cases use. score_v2 looks the
#     function up by name at call time, so redefining it here changes every later cell. (Offline: relabel.py did the same
#     to the saved rows.)
# (2) Cell E's chains left the neighbourhood of w*: the loss rose ~10 nats above L0, so those LLCs mean nothing.
#     Calibrate first on the first and last snapshot: the largest step size whose chains rise above L0 by less than 20%
#     of L0 and level off. Then run every snapshot with that one setting (the series is only comparable at fixed
#     hyperparameters). Results go to s11b_llc; offline: python analyze_llc.py outputs_v3 --stage s11b_llc
import ast as _ast

def _trivial(node):
    try:
        v = _ast.literal_eval(node)
    except Exception:
        return False
    if isinstance(v, bool) or v is None: return True
    if isinstance(v, (int, float)) and abs(v) <= 2: return True
    if isinstance(v, (str, list, tuple, dict, set)) and len(v) == 0: return True
    return False

def hardcoded(code, visible):
    """How many visible tests the code compares against literally (any non-trivial argument, or the whole tuple)."""
    try:
        tree = _ast.parse(code)
    except Exception:
        return 0
    lits = set()
    for n in _ast.walk(tree):
        if isinstance(n, _ast.Compare):
            for side in [n.left] + n.comparators:
                for p in [side] + (list(side.elts) if isinstance(side, _ast.Tuple) else []):
                    if not _trivial(p): lits.add(_norm(_ast.unparse(p)))
    hit = 0
    for a in visible:
        try:
            call = _ast.parse(a).body[0].test.left
        except Exception:
            continue
        if not isinstance(call, _ast.Call) or not call.args: continue
        args = [x for x in call.args if not _trivial(x)]
        keys = {_norm(_ast.unparse(x)) for x in args}
        if len(args) > 1: keys.add(_norm("(" + ",".join(_ast.unparse(x) for x in call.args) + ")"))
        if keys & lits: hit += 1
    return hit

assert hardcoded("def f(a, n):\n    if n == 4:\n        return [1]\n    return []", ["assert f([3, 5], 4) == [1]"]) == 1
print("detector v2 active")

# (2) LLC rerun
assert "estimate_llc" in globals(), "run cell E (stage11_llc) once in this runtime first: it defines estimate_llc"
snap_dir = os.path.join(ROOT_DRIVE, f"s10_{LLC_RUN}_snaps")
steps = sorted(int(s.split("-")[1]) for s in os.listdir(snap_dir) if s.startswith("step-") and os.path.exists(os.path.join(snap_dir, s, "COMPLETE")))
path_of = lambda s: None if s == 0 else os.path.join(snap_dir, f"step-{s}")

def _local(r):
    rise = r["trace_mean"] - r["L0"]
    return 0 < rise < 0.2 * r["L0"] and abs(r["trace_last10"] - r["trace_mean"]) < 0.5 * rise + 0.01

GRID = [(3e-5, 300.0), (1e-5, 300.0), (1e-5, 1000.0), (3e-6, 1000.0), (1e-6, 1000.0), (1e-6, 3000.0)]
cal = load_stage("s11b_llc", "calibration.json") if os.path.exists(os.path.join(ROOT_DRIVE, "s11b_llc", "calibration.json")) else None
if cal is None:
    keep = (LLC_CHAINS, LLC_DRAWS, LLC_BURN)
    LLC_CHAINS, LLC_DRAWS, LLC_BURN = 1, 60, 10          # short chains are enough to see a drift
    tried = []
    for eps, gam in GRID:
        LLC_EPS, LLC_GAMMA = eps, gam
        rs = [estimate_llc(path_of(s)) for s in (0, steps[-1])]
        ok = all(_local(r) for r in rs)
        tried.append({"eps": eps, "gamma": gam, "ok": ok, "rise": [round(r["trace_mean"] - r["L0"], 4) for r in rs], "L0": [round(r["L0"], 4) for r in rs]})
        print(f"eps {eps:.0e} gamma {gam:6.0f}: rise {tried[-1]['rise']} vs L0 {tried[-1]['L0']} -> {'LOCAL' if ok else 'no'}")
        if ok: break
    LLC_CHAINS, LLC_DRAWS, LLC_BURN = keep
    good = [t for t in tried if t["ok"]]
    cal = {"eps": good[0]["eps"] if good else None, "gamma": good[0]["gamma"] if good else None, "tried": tried}
    save_stage("s11b_llc", {"calibration.json": cal, "series.json": []})
if cal["eps"] is None:
    print("no setting in the grid stayed local: send me calibration.json; do not read anything into the LLC yet")
else:
    LLC_EPS, LLC_GAMMA = cal["eps"], cal["gamma"]
    LLC_CHAINS, LLC_DRAWS, LLC_BURN = 4, 200, 40           # more chains and draws: smaller steps need longer chains
    print(f"running all snapshots at eps {LLC_EPS:.0e}, gamma {LLC_GAMMA:.0f}, {LLC_CHAINS} chains x {LLC_DRAWS} draws")
    series = load_stage("s11b_llc", "series.json")
    have = {r["step"] for r in series}
    for step in [0] + steps:
        if step in have: continue
        t0 = time.time()
        r = estimate_llc(path_of(step)); r["step"] = step; r["local"] = _local(r)
        series.append(r); series.sort(key=lambda x: x["step"])
        save_stage("s11b_llc", {"calibration.json": cal, "series.json": series})
        print(f"step {step:4d}: LLC {r['llc']:9.3f} | L0 {r['L0']:.4f} rise {r['trace_mean'] - r['L0']:.4f} | {'local' if r['local'] else 'NOT LOCAL'} | {time.time() - t0:.0f}s")
