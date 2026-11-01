# Cell D: f0 again with a second seed, saving the LoRA adapter every SNAP_EVERY steps.
# Replicates f0 (one seed so far) and gives dense snapshots for the learning-coefficient series (cell E) and the
# monitor-drift study (cell F). Snapshots are adapter weights only (no optimizer), so they are small.
# Resumable: the normal checkpoints still go to s8_f0s2_ckpt every GRPO['save_steps'] steps.

SNAP_EVERY = 1 if SMOKE else 5
DENSE_SEED = 1

class SnapSync(TrainerCallback):
    """Copy the adapter to Drive every SNAP_EVERY steps (tmp dir, then rename, then COMPLETE)."""
    def __init__(self, tag):
        self.tag = tag
    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step % SNAP_EVERY or model is None:
            return
        dst = os.path.join(ROOT_DRIVE, f"s10_{self.tag}_snaps", f"step-{state.global_step}")
        if os.path.exists(os.path.join(dst, "COMPLETE")):
            return
        tmp = dst + ".tmp"
        if os.path.exists(tmp): shutil.rmtree(tmp)
        model.save_pretrained(tmp)
        if os.path.exists(dst): shutil.rmtree(dst)
        os.replace(tmp, dst)
        open(os.path.join(dst, "COMPLETE"), "w").write(str(state.global_step))

def run_dense(arm, f, lam, seed):
    """run_arm with a chosen seed and the snapshot callback added."""
    if done(f"s8_{arm}"):
        print(f"{arm}: already done"); return
    tok, base = _orig_load(POLICY, adapter=WARM_ADAPTER)
    tok.padding_side = "left"
    metrics = os.path.join(ROOT_LOCAL, f"metrics_{arm}.jsonl")
    calls = {"n": 0}
    def reward(prompts=None, completions=None, task_id=None, **kw):
        rows, rewards = [], []
        rng = random.Random(seed * 7919 + calls["n"])
        for c, tid in zip(completions, task_id):
            text = c if isinstance(c, str) else c[0]["content"]
            n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
            r = score_v2(TASKS_BY_ID[int(tid)], text, n_tok >= MAX_NEW - 1 and "</code>" not in text)
            r["monitored"] = rng.random() < f
            r["flagged"] = r["monitored"] and r["narrated"]
            rewards.append(r["reward"] - (lam if r["flagged"] else 0.0))
            rows.append(r)
        ok = [r for r in rows if not r["malformed"] and not r["truncated"]]
        hk = [r for r in ok if r["hack2"]]
        s = {"call": calls["n"], "hack_rate": len(hk) / max(1, len(ok)),
             "attempt_rate": sum(r["hard"] >= 1 for r in ok) / max(1, len(ok)),
             "narrated_among_hacks": sum(r["narrated"] for r in hk) / max(1, len(hk)),
             "flag_rate": float(np.mean([r["flagged"] for r in rows])), "mean_reward": float(np.mean(rewards)),
             "truncation": float(np.mean([r["truncated"] for r in rows])), "words": float(np.mean([r["n_words"] for r in rows]))}
        with open(metrics, "a") as fh:
            fh.write(json.dumps(s) + "\n"); fh.flush(); os.fsync(fh.fileno())
        calls["n"] += 1
        return rewards
    ds = Dataset.from_list([{"prompt": render(tok, t, CHOSEN), "task_id": t["id"]} for t in T["train"]])
    out_dir = os.path.join(ROOT_LOCAL, f"ckpt_{arm}")
    resume = None
    ck_root = os.path.join(ROOT_DRIVE, f"s8_{arm}_ckpt")
    if os.path.isdir(ck_root):
        ok = [c for c in os.listdir(ck_root) if os.path.exists(os.path.join(ck_root, c, "COMPLETE"))]
        if ok:
            src = os.path.join(ck_root, max(ok, key=lambda c: int(c.split("-")[1])))
            dst = os.path.join(out_dir, os.path.basename(src))
            if not os.path.exists(dst): shutil.copytree(src, dst)
            if os.path.exists(os.path.join(src, "train_metrics.jsonl")): shutil.copy(os.path.join(src, "train_metrics.jsonl"), metrics)
            resume = dst; print("resuming", arm, "from", resume)
    G = GRPO["generations"]
    cfg = make_config(output_dir=out_dir, max_steps=GRPO["steps"], learning_rate=GRPO["lr"], per_device_train_batch_size=G,
                      gradient_accumulation_steps=GRPO["prompts_per_step"], num_generations=G, max_completion_length=MAX_NEW,
                      temperature=1.0, beta=0.0, logging_steps=1, save_steps=GRPO["save_steps"], save_total_limit=2,
                      report_to="none", bf16=DEV == "cuda", gradient_checkpointing=True, seed=seed, dataloader_num_workers=0,
                      use_vllm=False)
    trainer = GRPOTrainer(model=base, reward_funcs=[reward], args=cfg, train_dataset=ds, processing_class=tok,
                          peft_config=LoraConfig(r=GRPO["lora_r"], lora_alpha=2 * GRPO["lora_r"], lora_dropout=0.0, task_type="CAUSAL_LM",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
                          callbacks=[ArmSync(arm, metrics), SnapSync(arm)])
    trainer.train(resume_from_checkpoint=resume)
    final = os.path.join(ROOT_LOCAL, f"adapter_{arm}"); trainer.save_model(final)
    dst = os.path.join(ROOT_DRIVE, f"s8_{arm}_adapter")
    if os.path.exists(dst): shutil.rmtree(dst)
    shutil.copytree(final, dst)
    del trainer, base; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    save_stage(f"s8_{arm}", {"train_metrics.jsonl": open(metrics).read() if os.path.exists(metrics) else ""})

ARMS["f0s2"] = (0.0, 0.0)
run_dense("f0s2", 0.0, 0.0, DENSE_SEED)
snaps = sorted(int(s.split("-")[1]) for s in os.listdir(os.path.join(ROOT_DRIVE, "s10_f0s2_snaps")) if s.startswith("step-") and not s.endswith(".tmp"))
print("snapshots on Drive:", snaps)
