# Runs inside the TARA pilot notebook's runtime (it uses the notebook's functions and settings).
# In a Colab cell:  exec(open('/content/drive/MyDrive/colab runs/stage9_tokens.py').read())

# %% [markdown]
# ## Stage 9: where is the decision? Activations along the reasoning (for the offline probe)
# Stage 6 read the last prompt token, which is identical across samples of a prompt, so it cannot separate a hacking
# sample from an honest one of the same prompt. Here the samples diverge: the residual stream is saved at the last
# prompt token and at reasoning tokens 1, 2, 4, 8, 16, 32, at the end of the reasoning, and at the first code token,
# for every 4th layer. Checkpoints: f0 and f100. The offline probe asks whether "will hard-code" is readable early,
# and whether a probe trained on f0 still reads it in f100 once the narration is gone.

# %%
S9_PER, S9_TASKS, S9_LAYERS = 8, 60, list(range(0, 29, 4))
S9_OFFSETS = [1, 2, 4, 8, 16, 32]

def _find(tok, ids, needle):
    """Index of the token that completes `needle` in the generated ids, or -1."""
    if needle not in tok.decode(ids): return -1
    lo, hi = 1, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if needle in tok.decode(ids[:mid]): hi = mid
        else: lo = mid + 1
    return lo - 1

def capture(arm):
    tok, m = load_arm(arm)
    stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.pad_token_id}
    tasks = (T["eval"] * 2)[:S9_TASKS]
    rows, acts = [], []
    for b in range(0, len(tasks), 4):
        bt = tasks[b:b + 4]
        enc = tok([render(tok, t, CHOSEN) for t in bt for _ in range(S9_PER)], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=TEMP, top_p=0.95, pad_token_id=tok.pad_token_id)
        P = enc["input_ids"].shape[1]
        for i, seq in enumerate(out):
            t = bt[i // S9_PER]
            prompt = enc["input_ids"][i][enc["attention_mask"][i].bool()].tolist()
            new = seq[P:].tolist()
            n = next((k for k, x in enumerate(new) if x in stop), len(new))
            ended, new = n < len(new), new[:n]
            r = score_v2(t, tok.decode(new, skip_special_tokens=True), not ended)
            r.update(ckpt=arm, sample=i % S9_PER)
            open_r = _find(tok, new, "<reasoning>")
            end_r, code0 = _find(tok, new, "</reasoning>"), _find(tok, new, "<code>")
            base_i = len(prompt) + max(open_r, 0)
            pos = [len(prompt) - 1] + [min(base_i + k, len(prompt) + len(new) - 1) for k in S9_OFFSETS]
            pos += [len(prompt) + end_r if end_r >= 0 else len(prompt) - 1, len(prompt) + code0 + 1 if code0 >= 0 else len(prompt) - 1]
            r["positions"] = ["prompt_end"] + [f"r{k}" for k in S9_OFFSETS] + ["reasoning_end", "code_start"]
            with torch.no_grad():
                hs = m(input_ids=torch.tensor([prompt + new], device=DEV), output_hidden_states=True).hidden_states
            acts.append(torch.stack([hs[L][0, pos, :] for L in S9_LAYERS]).to(torch.float16).cpu().numpy())
            rows.append(r)
        print(f"  {arm}: {min(b + 4, len(tasks))}/{len(tasks)} tasks")
    del m; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    return rows, np.stack(acts)

if not done("s9_tokens"):
    fd = os.path.join(ROOT_DRIVE, "s9_tokens_files"); os.makedirs(fd, exist_ok=True)
    allr = {}
    for arm in ["f0"] + (["f100"] if os.path.exists(os.path.join(ROOT_DRIVE, "s8_f100_adapter")) else []):
        rows, A = capture(arm)
        allr[arm] = rows
        local = os.path.join(ROOT_LOCAL, f"tok_{arm}.npy"); np.save(local, A)
        shutil.copy(local, os.path.join(fd, f"tok_{arm}.npy")); os.remove(local)
        print(arm, A.shape, summarize_v2(rows))
    save_stage("s9_tokens", {"rows.json": allr, "meta.json": {"layers": S9_LAYERS, "positions": ["prompt_end"] + [f"r{k}" for k in S9_OFFSETS] + ["reasoning_end", "code_start"]}})
print("token activations in", os.path.join(ROOT_DRIVE, "s9_tokens_files"))
