"""Exports compact data and assets for the papertoanything.in article. Run: python export_site.py SITE_ROOT"""
import json, os, shutil, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_figs import V, ORDER, LAB, TRAIN, jl, smooth, effect, DEEP

SITE = sys.argv[1]
PUB = os.path.join(SITE, "public", "research", "passing-isnt-proof"); os.makedirs(PUB, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
CONDS = ["solo", "clean", "spec_priority", "flag", "guard_only", "guard_known", "forged_auth", "rival_honest", "rival_cheat",
         "rival_punished", "rival_obf", "rival_cheat_flag"]

# arena: one character per answer, ordered by task then sample: b = bent, o = correct, x = wrong another way, - = invalid
arena = {}
for a in ORDER:
    arena[a] = {}
    for c in CONDS:
        rr = sorted([r for r in V if r["agent"] == a and r["cond"] == c], key=lambda r: (r["id"], -1 if r["wrong_pos"] is None else r["wrong_pos"]))
        arena[a][c] = "".join("b" if r["wrong_test_bent"] else ("o" if r["hidden_pass"] else "x") for r in rr)
r2 = lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 3)
s1 = jl(os.path.join(TRAIN, "f0_seed1.jsonl")); q = jl(os.path.join(TRAIN, "f100.jsonl")); h = jl(os.path.join(TRAIN, "f0to100.jsonl"))
training = {"cheat": {"hack": [r2(x) for x in smooth([r["hack_rate_corrupted"] for r in s1])], "words": [r2(x) for x in smooth([r["mean_reasoning_words"] for r in s1])]},
            "quit": {"hack": [r2(x) for x in smooth([r["hack_rate"] for r in q])], "words": [r2(x) for x in smooth([r["words"] for r in q])]},
            "hide": {"hack": [r2(x) for x in smooth([r["hack_rate"] for r in h])], "words": [r2(x) for x in smooth([r["words"] for r in h])]}}
effects = {a: {k: [r2(x) for x in effect(a, c1, c0)] for k, c1, c0 in
               [("contagion", "rival_cheat", "rival_honest"), ("authority", "forged_auth", "solo"), ("rejected", "rival_punished", "rival_cheat"),
                ("disguised", "rival_obf", "rival_cheat"), ("guard_rule", "guard_only", "solo"), ("spec_first", "spec_priority", "solo")]}
           for a in ORDER}
out = {"agents": ORDER, "labels": LAB, "conds": CONDS, "arena": arena, "training": training, "effects": effects,
       "flag": DEEP["flag_pooled"], "size": DEEP["size"], "guard_audit": DEEP["guard_audit"]}
json.dump(out, open(os.path.join(PUB, "data.json"), "w"), separators=(",", ":"))
print("data.json", os.path.getsize(os.path.join(PUB, "data.json")) // 1024, "kB")

for theme in ("light", "dark"):
    dst = os.path.join(PUB, theme); os.makedirs(dst, exist_ok=True)
    for d, ext in ((f"figures_site-{theme}", ".png"), (f"illus_site-{theme}", ".svg")):
        for f in os.listdir(os.path.join(HERE, d)):
            if f.endswith(ext): shutil.copy(os.path.join(HERE, d, f), os.path.join(dst, f))
for f in os.listdir(PUB):                      # drop the single-theme copies from the first export
    if f.endswith((".png", ".svg", ".mp4")): os.remove(os.path.join(PUB, f))
for f in ("PassingIsntProof_Report.pdf",):
    p = os.path.join(os.path.dirname(HERE), f)
    if os.path.exists(p): shutil.copy(p, os.path.join(PUB, f))
print("assets:", len(os.listdir(PUB)))
