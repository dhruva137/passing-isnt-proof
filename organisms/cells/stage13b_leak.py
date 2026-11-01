# Cell G2: is the "decision readable by token 2" result real, or a numerical leak?
#
# Warning sign: at the warm start, the LAST PROMPT TOKEN separates hacking from honest samples of the SAME prompt at
# AUROC 0.61-0.72 (layers 12-28). That position sees only the prompt, which is identical within a prompt, so it must be
# 0.5 (it is exactly 0.5 at layer 0). Likely cause: capture() ran ONE forward pass over prompt + the whole answer, and
# bf16 attention kernels give slightly different numbers at early positions depending on the total length. Answer
# length differs between hacks and honest answers, so a probe can read "length" from that noise.
# Fix: one forward pass per position, cut right after it (the position cannot see anything later, numerically too).
# If prompt_end drops to 0.5 and r2 stays high, the early-decision result stands. If r2 drops too, it was a leak.
# Output: s9e_tokens (same layout as s9b). Offline: python analyze_tokens.py outputs_v3 --set s9e_tokens --label attempt
LEAK_ARMS = ["warm", "f0@25"]
src = load_stage("s9b_tokens", "rows.json")
out = load_stage("s9e_tokens", "rows.json") if done("s9e_tokens") else {}
fd = os.path.join(ROOT_DRIVE, "s9e_tokens_files"); os.makedirs(fd, exist_ok=True)
for arm in LEAK_ARMS:
    if arm in out: continue
    tok, m = load_arm(arm)
    rows, acts = [], []
    for i, r in enumerate(src[arm]):
        prompt = tok(render(tok, TASKS_BY_ID[r["id"]], CHOSEN), add_special_tokens=False)["input_ids"]
        new = tok(f"<reasoning>\n{r['reasoning']}\n</reasoning>\n<code>\n{r['code']}\n</code>", add_special_tokens=False)["input_ids"]
        ids = prompt + new
        vec = []
        with torch.no_grad():
            for p in positions_for(tok, prompt, new):
                hs = m(input_ids=torch.tensor([ids[:p + 1]], device=DEV), output_hidden_states=True).hidden_states
                vec.append(torch.stack([hs[L][0, p, :] for L in S9_LAYERS]))
        acts.append(torch.stack(vec, 1).to(torch.float16).cpu().numpy())      # (layers, positions, d) like capture()
        rows.append(r)
        if i % 100 == 0: print(f"  {arm}: {i}/{len(src[arm])}")
    del m; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    out[arm] = rows
    name = f"tok_{arm.replace('@', '_')}.npy"; local = os.path.join(ROOT_LOCAL, name)
    np.save(local, np.stack(acts)); shutil.copy(local, os.path.join(fd, name)); os.remove(local)
    save_stage("s9e_tokens", {"rows.json": out, "meta.json": {"layers": S9_LAYERS, "positions": POS}})
    print(arm, "done")
print("zip s9e_tokens and s9e_tokens_files and send them")
