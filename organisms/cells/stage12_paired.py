# Cell F: activations for the monitor-drift study.
# 1. Own samples for checkpoints not captured yet (f0→f100 if it exists, and f0s2 snapshots every 20 steps):
#    these are what a monitor watching that checkpoint would actually read.
# 2. Paired activations: one FIXED set of texts, run through EVERY checkpoint. Same tokens, same positions, so a map
#    between checkpoints can be fitted without labels (the label-free "repair" of a stale monitor).
# The offline script analyze_monitors.py then asks: does a monitor trained early keep working, does label-free repair
# restore it, and, where the behaviour itself changed (hiding), does repair follow the behaviour or lose it?

PAIR_N = 16 if SMOKE else 300
POS = ["prompt_end"] + [f"r{k}" for k in S9_OFFSETS] + ["reasoning_end", "code_start"]

def positions_for(tok, prompt, new):
    open_r = _find(tok, new, "<reasoning>")
    end_r, code0 = _find(tok, new, "</reasoning>"), _find(tok, new, "<code>")
    base_i = len(prompt) + max(open_r, 0)
    pos = [len(prompt) - 1] + [min(base_i + k, len(prompt) + len(new) - 1) for k in S9_OFFSETS]
    pos += [len(prompt) + end_r if end_r >= 0 else len(prompt) - 1, len(prompt) + code0 + 1 if code0 >= 0 else len(prompt) - 1]
    return pos

# 1. own samples for new checkpoints
own = load_stage("s9c_tokens", "rows.json") if done("s9c_tokens") else {}
fd_own = os.path.join(ROOT_DRIVE, "s9c_tokens_files"); os.makedirs(fd_own, exist_ok=True)
extra = []
if os.path.exists(os.path.join(ROOT_DRIVE, "s8_f0to100_adapter")): extra.append("f0to100")
sd = os.path.join(ROOT_DRIVE, "s10_f0s2_snaps")
if os.path.isdir(sd):
    st = sorted(int(s.split("-")[1]) for s in os.listdir(sd) if s.startswith("step-") and os.path.exists(os.path.join(sd, s, "COMPLETE")))
    extra += [f"f0s2@{s}" for s in st if s % (2 if SMOKE else 20) == 0]
for arm in extra:
    if arm in own: continue
    rows, A = capture(arm)
    own[arm] = rows
    name = f"tok_{arm.replace('@', '_')}.npy"
    local = os.path.join(ROOT_LOCAL, name); np.save(local, A); shutil.copy(local, os.path.join(fd_own, name)); os.remove(local)
    ok = [r for r in rows if not r["malformed"] and not r["truncated"]]
    print(f"own {arm}: {A.shape} attempt {np.mean([r['hard'] >= 1 for r in ok]):.2f} narrated {np.mean([r['narrated'] for r in ok]):.2f}")
    save_stage("s9c_tokens", {"rows.json": own, "meta.json": {"layers": S9_LAYERS, "positions": POS}})

# 2. the fixed text set, drawn evenly from every checkpoint's own samples
pool = {}
for stage in ("s9_tokens", "s9b_tokens", "s9c_tokens"):
    if done(stage):
        for arm, rows in load_stage(stage, "rows.json").items():
            pool[arm] = [r for r in rows if not r["malformed"] and not r["truncated"]]
CKPTS = list(pool)
rng = random.Random(SEED)
texts = []
per = max(1, PAIR_N // max(1, len(pool)))
for arm, rows in pool.items():
    for r in rng.sample(rows, min(per, len(rows))):
        texts.append({"id": r["id"], "src": arm, "attempt": int(r["hard"] >= 1), "narrated": bool(r["narrated"]),
                      "answer": f"<reasoning>\n{r['reasoning']}\n</reasoning>\n<code>\n{r['code']}\n</code>"})
print(f"paired set: {len(texts)} texts from {len(pool)} checkpoints: {CKPTS}")

fd = os.path.join(ROOT_DRIVE, "s12_paired_files"); os.makedirs(fd, exist_ok=True)
got = load_stage("s12_paired", "meta.json")["done"] if done("s12_paired") else []
for arm in CKPTS:
    if arm in got: continue
    tok, m = load_arm(arm)
    acts = []
    for x in texts:
        prompt = tok(render(tok, TASKS_BY_ID[x["id"]], CHOSEN), add_special_tokens=False)["input_ids"]
        new = tok(x["answer"], add_special_tokens=False)["input_ids"]
        pos = positions_for(tok, prompt, new)
        with torch.no_grad():
            hs = m(input_ids=torch.tensor([prompt + new], device=DEV), output_hidden_states=True).hidden_states
        acts.append(torch.stack([hs[L][0, pos, :] for L in S9_LAYERS]).to(torch.float16).cpu().numpy())
    del m; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    A = np.stack(acts)
    name = f"tok_{arm.replace('@', '_')}.npy"
    local = os.path.join(ROOT_LOCAL, name); np.save(local, A); shutil.copy(local, os.path.join(fd, name)); os.remove(local)
    got.append(arm)
    save_stage("s12_paired", {"texts.json": texts, "meta.json": {"layers": S9_LAYERS, "positions": POS, "done": got}})
    print(f"paired {arm}: {A.shape}")
print("paired activations for:", got)
