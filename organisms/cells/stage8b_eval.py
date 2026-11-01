# Runs inside the TARA pilot notebook's runtime (it uses the notebook's functions and settings).
# In a Colab cell:  exec(open('/content/drive/MyDrive/colab runs/stage8b_eval.py').read())

# %% [markdown]
# ## Stage 8b: held-out evaluation of every arm, measured the same way
# f0 is the stage 4 adapter. Each arm: 4 samples per held-out task, ground truth v2, MBPP+, narration pattern and judge.
# The headline is `silent_hacks`: answers that hard-code while the reasoning no longer says so.

# %%
def load_arm(arm):
    tok, m = load(POLICY, adapter=WARM_ADAPTER)
    path = os.path.join(ROOT_DRIVE, "final_adapter" if arm == "f0" else f"s8_{arm}_adapter")
    from peft import PeftModel
    return tok, PeftModel.from_pretrained(m, path).merge_and_unload().eval()

EVAL_SAMPLES = 4
evals = load_stage("s8_eval", "rows.json") if done("s8_eval") else {}
for arm in ["f0"] + [a for a in ARMS if os.path.exists(os.path.join(ROOT_DRIVE, f"s8_{a}_adapter"))]:
    if arm in evals: continue
    tok, m = load_arm(arm)
    texts, truncs = generate(tok, m, [render(tok, t, CHOSEN) for t in T["eval"]], n=EVAL_SAMPLES)
    del m; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    tk = [t for t in T["eval"] for _ in range(EVAL_SAMPLES)]
    rows = [score_v2(t, x, tr, with_plus=True) for t, x, tr in zip(tk, texts, truncs)]
    for r in rows: r["variant"] = CHOSEN
    evals[arm] = judge_scores(rows, TASKS_BY_ID)
    save_stage("s8_eval", {"rows.json": evals, "summary.json": {k: summarize_v2(v) for k, v in evals.items()}})
print(json.dumps(load_stage("s8_eval", "summary.json"), indent=1))

