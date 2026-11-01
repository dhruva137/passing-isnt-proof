# Cell A: activations at checkpoints where the decision varies, plus the load_arm that
# understands "warm", "f0@N" (an f0 checkpoint) and arms started from other arms (START).
from peft import PeftModel
_orig_load = globals().get("_orig_load", load)
START = globals().get("START", {})

def load_arm(arm):
    """warm = warm start; f0@N = f0's checkpoint at step N; X@N with snaps = snapshot of run X; else adapter chains."""
    tok, m = _orig_load(POLICY, adapter=WARM_ADAPTER)
    if arm == "warm":
        return tok, m.eval()
    if "@" in arm:
        run, step = arm.split("@")
        path = (os.path.join(ROOT_DRIVE, "s4_ckpt", f"checkpoint-{step}") if run == "f0"
                else os.path.join(ROOT_DRIVE, f"s10_{run}_snaps", f"step-{step}"))
        return tok, PeftModel.from_pretrained(m, path).merge_and_unload().eval()
    chain, a = [], arm
    while True:
        chain.append(os.path.join(ROOT_DRIVE, "final_adapter" if a == "f0" else f"s8_{a}_adapter"))
        if a not in START: break
        a = START[a]
    for path in reversed(chain):
        m = PeftModel.from_pretrained(m, path).merge_and_unload()
    return tok, m.eval()

ck = os.path.join(ROOT_DRIVE, "s4_ckpt")
avail = sorted(int(c.split("-")[1]) for c in os.listdir(ck) if c.startswith("checkpoint-") and os.path.exists(os.path.join(ck, c, "COMPLETE")))
print("f0 checkpoints on Drive:", avail)
TARGETS = ["warm"] + [f"f0@{s}" for s in avail if s <= 50]
POS = ["prompt_end"] + [f"r{k}" for k in S9_OFFSETS] + ["reasoning_end", "code_start"]
fd = os.path.join(ROOT_DRIVE, "s9b_tokens_files"); os.makedirs(fd, exist_ok=True)
prev = load_stage("s9b_tokens", "rows.json") if done("s9b_tokens") else {}
for arm in TARGETS:
    if arm in prev: continue
    rows, A = capture(arm)
    prev[arm] = rows
    name = f"tok_{arm.replace('@', '_')}.npy"
    local = os.path.join(ROOT_LOCAL, name); np.save(local, A)
    shutil.copy(local, os.path.join(fd, name)); os.remove(local)
    ok = [r for r in rows if not r["malformed"] and not r["truncated"]]
    print(f"{arm}: {A.shape} | attempted hard-coding {np.mean([r['hard'] >= 1 for r in ok]):.2f} | narrated {np.mean([r['narrated'] for r in ok]):.2f}")
    save_stage("s9b_tokens", {"rows.json": prev, "meta.json": {"layers": S9_LAYERS, "positions": POS}})
print("done:", list(prev))
