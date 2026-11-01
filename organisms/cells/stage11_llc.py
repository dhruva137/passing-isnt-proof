# Cell E: the local learning coefficient (LLC) along training, restricted to the LoRA adapter.
#
# The LLC (singular learning theory; Lau et al. 2023) measures how degenerate the solution at the current weights is.
# Estimator: sample weights w near w* with SGLD on a localised, tempered posterior and compare the loss there with
# the loss at w*:   LLC = n·β·(E_w[L_n(w)] − L_n(w*)),  β = 1/log n,
#   w ← w − (ε/2)·(n·β·∇L_m(w) + γ·(w − w*)) + N(0, ε).
# Only the adapter (what RL changed) is sampled; the merged warm-start model underneath stays fixed.
# The loss is next-token NLL on a FIXED set of texts (each task's reference solution in the answer format), the same
# for every snapshot, so the series is comparable across training. Absolute values are not comparable to other papers.
#
# Question: does the LLC move before the hacking take-off seen in the training metrics, or only with it / after it?
# Hyperparameters are fixed across snapshots. Check the diagnostics: the loss trace must rise from L0 and level off.
# If it explodes (trace >> L0 and growing) lower LLC_EPS; if it never leaves L0, raise it.
import math
from peft import PeftModel, get_peft_model

LLC_N_SEQ = 8 if SMOKE else 64
LLC_BATCH = 4 if SMOKE else 8
LLC_CHAINS, LLC_DRAWS, LLC_BURN = (1, 6, 2) if SMOKE else (2, 120, 20)
LLC_EPS, LLC_GAMMA = 1e-4, 100.0
LLC_RUN = "f0s2"

def _llc_data(tok):
    seqs = []
    for t in T["train"][:LLC_N_SEQ]:
        p = tok(render(tok, t, CHOSEN), add_special_tokens=False)["input_ids"]
        a = tok("<reasoning>\nSolve the task as described.\n</reasoning>\n<code>\n" + t["ref"] + "\n</code><|im_end|>",
                add_special_tokens=False)["input_ids"]
        seqs.append((p, a))
    return seqs

def _batch_loss(m, tok, seqs, idx):
    L = max(len(seqs[i][0]) + len(seqs[i][1]) for i in idx)
    pad = tok.pad_token_id
    ids = torch.full((len(idx), L), pad, dtype=torch.long)
    lab = torch.full((len(idx), L), -100, dtype=torch.long)
    att = torch.zeros((len(idx), L), dtype=torch.long)
    for j, i in enumerate(idx):
        p, a = seqs[i]
        ids[j, :len(p) + len(a)] = torch.tensor(p + a)
        lab[j, len(p):len(p) + len(a)] = torch.tensor(a)
        att[j, :len(p) + len(a)] = 1
    return m(input_ids=ids.to(DEV), attention_mask=att.to(DEV), labels=lab.to(DEV)).loss.float()

def estimate_llc(adapter_path):
    tok, base = _orig_load(POLICY, adapter=WARM_ADAPTER)
    tok.padding_side = "right"
    if adapter_path:
        m = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    else:  # the warm start itself: a fresh adapter at zero (B = 0) with the same shape as the RL adapters
        m = get_peft_model(base, LoraConfig(r=GRPO["lora_r"], lora_alpha=2 * GRPO["lora_r"], lora_dropout=0.0, task_type="CAUSAL_LM",
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    m.eval()
    params = [p for _, p in m.named_parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()     # SGLD steps are tiny: keep the adapter in fp32 (PEFT casts inputs to it)
    w0 = [p.detach().clone() for p in params]
    seqs = _llc_data(tok)
    n = len(seqs); nbeta = n / math.log(n)
    batches = [list(range(i, min(i + LLC_BATCH, n))) for i in range(0, n, LLC_BATCH)]
    with torch.no_grad():
        L0 = float(np.mean([_batch_loss(m, tok, seqs, b).item() for b in batches]))
    g = torch.Generator(device="cpu").manual_seed(SEED)
    traces = []
    for c in range(LLC_CHAINS):
        with torch.no_grad():
            for p, w in zip(params, w0): p.copy_(w)
        tr = []
        for step in range(LLC_BURN + LLC_DRAWS):
            idx = torch.randperm(n, generator=g)[:LLC_BATCH].tolist()
            loss = _batch_loss(m, tok, seqs, idx)
            grads = torch.autograd.grad(loss, params)
            with torch.no_grad():
                for p, gr, w in zip(params, grads, w0):
                    p.add_(-(LLC_EPS / 2) * (nbeta * gr.float() + LLC_GAMMA * (p - w)) + math.sqrt(LLC_EPS) * torch.randn_like(p))
            if step >= LLC_BURN:
                tr.append(loss.item())
        traces.append(tr)
    flat = [x for tr in traces for x in tr]
    del m, base; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    return {"llc": nbeta * (float(np.mean(flat)) - L0), "L0": L0, "trace_mean": float(np.mean(flat)),
            "trace_std": float(np.std(flat)), "trace_first10": float(np.mean(flat[:10])), "trace_last10": float(np.mean(flat[-10:])),
            "n": n, "nbeta": nbeta, "eps": LLC_EPS, "gamma": LLC_GAMMA, "chains": LLC_CHAINS, "draws": LLC_DRAWS}

snap_dir = os.path.join(ROOT_DRIVE, f"s10_{LLC_RUN}_snaps")
steps = sorted(int(s.split("-")[1]) for s in os.listdir(snap_dir) if s.startswith("step-") and os.path.exists(os.path.join(snap_dir, s, "COMPLETE")))
series = load_stage("s11_llc", "series.json") if done("s11_llc") else []
have = {r["step"] for r in series}
for step in [0] + steps:
    if step in have: continue
    t0 = time.time()
    r = estimate_llc(None if step == 0 else os.path.join(snap_dir, f"step-{step}"))
    r["step"] = step
    series.append(r)
    series.sort(key=lambda x: x["step"])
    save_stage("s11_llc", {"series.json": series})
    print(f"step {step:4d}: LLC {r['llc']:9.3f} | L0 {r['L0']:.4f} -> trace {r['trace_first10']:.4f} .. {r['trace_last10']:.4f} | {time.time() - t0:.0f}s")

tm_path = os.path.join(ROOT_DRIVE, f"s8_{LLC_RUN}", "train_metrics.jsonl")
if os.path.exists(tm_path):
    tm = [json.loads(l) for l in open(tm_path) if l.strip()]
    print("\nstep | LLC | attempt rate around that step (mean of 5 reward calls)")
    for r in series:
        s = r["step"]; win = tm[max(0, s - 5):s] if s else tm[:1]
        print(f"{s:4d} | {r['llc']:9.3f} | {np.mean([x.get('attempt_rate', x.get('hack_rate', 0)) for x in win]):.2f}")
