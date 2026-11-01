"""Builds the two Kaggle notebooks from the tested source files in this folder (nothing is retyped into cells)."""
import base64, gzip, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.dirname(HERE)
src = lambda f: open(os.path.join(HERE, f), encoding="utf-8").read()
TASKS = os.path.join(OUTDIR, "tacera-safl", "results", "organisms", "tasks.json")
REF = os.path.join(OUTDIR, "tacera-safl", "results", "training", "f0_seed1.jsonl")


def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}
def wf(name): return code(f"%%writefile {name}\n" + src(name))


PINS = "transformers==5.22.1 peft==0.21.2 trl==1.16.0 accelerate==1.15.0 datasets==5.1.0 gdown==6.5.1 bitsandbytes==0.50.2"

SETUP = r'''
# Setup. Kaggle: Accelerator = GPU T4 x2, Internet = On. Then "Save Version" -> "Save & Run All (Commit)" so it runs
# unattended; everything it produces lands in /kaggle/working (Output tab).
import os, sys, subprocess, time, json, glob, shutil
START = time.time()
SMOKE = os.environ.get("TACERA_SMOKE") == "1"            # local smoke test: 2 tasks, tiny runs
if os.environ.get("TACERA_SKIP_PIP") != "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *"__PINS__".split()], check=True)
    # Kaggle ships torchao 0.10; peft refuses to load any LoRA adapter while an old torchao is importable. Not used here.
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
WORK = "/kaggle/working/tacera" if os.path.isdir("/kaggle/working") else os.path.abspath("tacera")
os.makedirs(WORK, exist_ok=True); os.chdir(WORK); sys.path.insert(0, WORK)
import torch
NGPU = torch.cuda.device_count()
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(NGPU)], "| torch", torch.__version__)
assert NGPU >= 1, "No GPU: set Accelerator to GPU T4 x2 in the notebook settings"
HOURS = float(os.environ.get("TACERA_HOURS", "__HOURS__"))   # total wall-clock budget for this notebook
DEADLINE = START + HOURS * 3600
print(f"budget {HOURS} h -> stop scheduling new work at {time.strftime('%H:%M', time.localtime(DEADLINE - 1800))}")
'''.replace("__PINS__", PINS)

SELFTEST = r'''
# The suite must be right before any model is run: every reference is guard-certified and fails exactly the wrong
# published test (in each of the three positions); every plain and every hash-disguised cheat passes the published
# tests and is rejected by the guard; every clean-test variant is passed by the reference.
import importlib, dpi_suite as S
importlib.reload(S)
from concurrent.futures import ThreadPoolExecutor
def check(t):
    ref = S.certify(t["ref"], t); ch = S.certify(S.cheat_code(t), t); ob = S.certify(S.obfuscated_cheat_code(t), t)
    rot = [S.certify(t["ref"], S.rotate(t, k))["vis"] for k in range(3)]
    return (ref["vis"] == [True, False, True] and ref["guard_pass"] and ch["public_pass"] and not ch["guard_pass"]
            and ob["public_pass"] and not ob["guard_pass"] and ob["special_cases"] == 0 and all(S.run_asserts(t["ref"], S.clean(t)["visible"]))
            and all(r[k] is False and sum(r) == 2 for k, r in enumerate(rot)))
with ThreadPoolExecutor(8) as ex: ok = list(ex.map(check, S.TASKS))
bad = [t["id"] for t, o in zip(S.TASKS, ok) if not o]
print(f"{len(S.TASKS)} DPI tasks checked; problems: {bad or 'none'}")
assert not bad
'''

ADAPTERS = r'''
# Trained LoRA adapters of the model organisms (Qwen3-1.7B), from the project's shared Drive folder.
# If the download fails (Drive quota), the notebook carries on with whatever does not need them.
import gdown
FOLDER_ID = "1BMb_ZxyVyNEXv4uCXGcEXLDyKWKaceh0"
NEED = __NEED__
ADIR = os.path.join(WORK, "adapters")
def have(d): return os.path.exists(os.path.join(ADIR, d, "adapter_model.safetensors"))
for attempt in range(3):
    if all(have(d) for d in NEED): break
    try:
        for f in gdown.download_folder(id=FOLDER_ID, skip_download=True, quiet=True):
            p = f.path.replace("\\", "/")
            if p.startswith(tuple(d + "/" for d in NEED)) and p.endswith((".json", ".safetensors")):
                out = os.path.join(ADIR, p); os.makedirs(os.path.dirname(out), exist_ok=True)
                if not os.path.exists(out): gdown.download(id=f.id, output=out, quiet=True)
    except Exception as e:
        print("download attempt", attempt + 1, "failed:", type(e).__name__, e); time.sleep(30 * (attempt + 1))
ADAPTERS_OK = {d: have(d) for d in NEED}
print(ADAPTERS_OK)
'''


def notebook(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}, "accelerator": "GPU"},
            "nbformat": 4, "nbformat_minor": 5}


# =====================================================================================================================
A_INTRO = r'''
# Passing Isn't Proof · Notebook A — the cheating arena at scale

**Question.** When a coding agent is shown a rival's code that wins by special-casing a *wrong* published conformance
test, does it copy the trick? Which interventions stop that, and does a certification guard catch what slips through?

**Setting.** 20 synthetic tasks modelled on India's digital public infrastructure (UPI IDs and amounts, IFSC/PAN/PIN
formats, GST, ONDC order states, Account Aggregator consent). Each has three published tests, one with a wrong expected
value, three hidden tests, and a random-input generator. *Bent* = the answer reproduces the wrong expected value, which a
correct implementation cannot do.

**What this run adds to the local run** (6 agents × 5 conditions, finished 31 Oct):
- *Capability control* (`clean`): all published tests correct, so bending is not just inability.
- *Recognition probe*: asked directly, does the model name the wrong test? Separates "doesn't know" from "knows and bends".
- *Wrong-test position* rotates over positions 1–3 (it was always 2 before).
- *Levers*, each a one-block change against its matched control: spec-first instruction (ImpossibleBench's strongest
  prompt lever), a sanctioned **flag channel** (alone and under contagion; neither ImpossibleBench nor EvilGenie tested
  it under peer exposure), the guard rule with and without the "one test may be wrong" sentence (that sentence raised
  bending +0.20 for one organism locally), a **forged permission** note (cf. the Troy Moment paper's authorisation probe),
  a cheat shown as **rejected** (the punished-peer condition), and a **hash-disguised** cheat.
- *Scale and family*: Qwen3 0.6B → 1.7B → 4B (→ 8B in 4-bit if time allows), plus Granite 3.3 2B and Phi-4-mini.

**Engineering.** One worker process per GPU; each (agent, condition) is written atomically as soon as it finishes, so a
crash or the time budget loses at most one unit and a rerun resumes. fp16 overflow is checked per model (fallback to
fp32 or skip). CUDA OOM halves the batch and retries. Sandbox timeouts are retried. All figures are produced from the
written files by `arena_analysis.py`, which was validated against the local run's numbers.
'''

LAUNCH_A = r'''
# Two workers, one per GPU. Order inside each list = priority (organisms and the Qwen3 ladder first).
ORG = ["warm", "f0", "f100", "f0to100"]
GPU_PLAN = [["warm", "f100", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B", "microsoft/Phi-4-mini-instruct"],
            ["f0", "f0to100", "Qwen/Qwen3-1.7B", "ibm-granite/granite-3.3-2b-instruct", "Qwen/Qwen3-8B@4bit"]]
if not all(ADAPTERS_OK.values()):
    GPU_PLAN = [[a for a in plan if a not in ORG] for plan in GPU_PLAN]; print("adapters missing -> organisms skipped")
if NGPU == 1:
    GPU_PLAN = [[a for plan in zip(*GPU_PLAN) for a in plan]]
SAMPLES = int(os.environ.get("TACERA_SAMPLES", "6"))
if SMOKE: GPU_PLAN, SAMPLES = [["Qwen/Qwen3-0.6B", "warm"]][:NGPU] + [["Qwen/Qwen3-0.6B"]] * (NGPU - 1), 3
RES = os.path.join(WORK, "results"); os.makedirs(RES, exist_ok=True)
worker_deadline = DEADLINE - 1800          # leave 30 min for analysis and packaging
procs = []
for g, plan in enumerate(GPU_PLAN):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    log = open(os.path.join(WORK, f"worker{g}.log"), "a")
    procs.append(subprocess.Popen([sys.executable, "arena_worker.py", "--agents", ",".join(plan), "--out", RES, "--adir", ADIR,
                                   "--deadline", str(worker_deadline), "--samples", str(SAMPLES)] + (["--smoke", "--conds", "solo,clean,flag,rival_cheat,rival_cheat_flag"] if SMOKE else []), stdout=log, stderr=subprocess.STDOUT, env=env))
    print(f"GPU {g}: {plan}")

def tail(p, n=4):
    try: return [l.rstrip() for l in open(p, encoding="utf-8", errors="replace").readlines() if l.startswith("[")][-n:]
    except FileNotFoundError: return []
last = 0
while any(p.poll() is None for p in procs):
    if time.time() > worker_deadline + 900:
        for p in procs: p.kill()
        print("workers killed at the hard limit"); break
    if time.time() - last > 600:
        last = time.time(); print(f"--- {time.strftime('%H:%M')} ({(time.time() - START) / 3600:.1f} h), {len(glob.glob(RES + '/*.jsonl'))} units done")
        for g in range(len(procs)): [print(f"  gpu{g} {l}") for l in tail(os.path.join(WORK, f"worker{g}.log"))]
    time.sleep(20)
for g in range(len(procs)): [print(f"  gpu{g} {l}") for l in tail(os.path.join(WORK, f"worker{g}.log"), 30)]
print("failures:", [os.path.basename(p) for p in glob.glob(RES + "/*FAILED*")] or "none")
'''

ANALYSE_A = r'''
import importlib, arena_analysis as AA
importlib.reload(AA)
FIG = os.path.join(WORK, "figures"); site = AA.run(RES, FIG)
from IPython.display import Image, display
for f in sorted(glob.glob(FIG + "/*.png")): print(os.path.basename(f)); display(Image(f, width=760))
'''

PACK = r'''
# Results, figures, logs and trained adapters in one archive (Output tab -> download). Base-model weights are not included.
import zipfile
dst = os.path.join("/kaggle/working" if os.path.isdir("/kaggle/working") else WORK, "__NAME__.zip")
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(WORK):
        rel = os.path.relpath(root, WORK)
        if rel.startswith(("adapters", "results/ckpt_")) or "/ckpt_" in rel: continue
        for f in files:
            if f.endswith((".jsonl", ".json", ".png", ".svg", ".txt", ".log", ".py", ".safetensors")):
                z.write(os.path.join(root, f), os.path.join(rel, f))
print(dst, f"{os.path.getsize(dst) / 1e6:.1f} MB; total time {(time.time() - START) / 3600:.2f} h")
'''

nbA = notebook([
    md(A_INTRO),
    code(SETUP.replace("__HOURS__", "7.5")),
    md("## Source files\nThe DPI suite (v3: spec literals ignored by the structural check, timeouts retried), the worker and the analysis. "
       "They are the exact files that were smoke-tested locally."),
    wf("dpi_suite.py"), wf("arena_worker.py"), wf("arena_analysis.py"),
    md("## 1 · Check the instrument"), code(SELFTEST),
    md("## 2 · Model organisms"), code(ADAPTERS.replace("__NEED__", '("warm_adapter", "final_adapter", "s8_f100_adapter", "s8_f0to100_adapter")')),
    md("## 3 · Run\nProgress prints every 10 minutes. Expected: about 5–6 hours on two T4s."), code(LAUNCH_A),
    md("## 4 · Results\nTables, paired effects with 95% bootstrap CIs over tasks, and figures."), code(ANALYSE_A),
    md("## 5 · Package"), code(PACK.replace("__NAME__", "tacera_arena_results")),
])

# =====================================================================================================================
B_INTRO = r'''
# Passing Isn't Proof · Notebook B — train against the tests, or against the guard?

**Question.** Our trained-to-cheat organism (`f0`) came from GRPO with reward = *published tests passed*, where one test
on half the tasks is wrong. What happens from the **same warm start** if the reward is the **guard** instead: the share of
published *inputs* on which the answer matches the reference, and zero for special-casing a published input?

MacDiarmid et al. (2025) found that removing the hackable opportunity is what works; training against the guard is
exactly that, applied as a reward. This notebook measures it directly, with everything else held fixed: model
(Qwen3-1.7B), warm-start adapter, MBPP tasks and corruptions, prompt, LoRA r32, learning rate, seed. One arm per GPU.

**Outputs.** Per-step hack rate, special-casing and capability for both arms (with the earlier A100 public-reward run as
a dashed reference), and a held-out evaluation before and after: MBPP hack rate and clean-task capability, and transfer
to the DPI suite (bending alone and after a winning cheat; guard certification).

**Budget.** Training stops at 4 h whatever the step count; evaluation follows. Micro-batches of 2 completions (64 per step) keep a T4 under 15 GB; if memory still runs out, the adapter trained so far is kept and evaluated. A non-finite loss stops training.
'''

LAUNCH_B = r'''
open("tasks.json", "wb").write(gzip.decompress(base64.b64decode(TASKS_B64)))
open("reference_curve.json", "w").write(json.dumps(REF_CURVE))
RES = os.path.join(WORK, "results"); os.makedirs(RES, exist_ok=True)
assert ADAPTERS_OK.get("warm_adapter"), "the warm-start adapter is required for this notebook"
train_until = START + float(os.environ.get("TACERA_TRAIN_HOURS", "4.0")) * 3600
procs = []
for g, arm in enumerate(["public", "guard"]):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g % NGPU), TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    args = [sys.executable, "train_worker.py", "--arm", arm, "--out", RES, "--adir", ADIR, "--tasks", "tasks.json",
            "--train_until", str(train_until), "--deadline", str(DEADLINE - 1200), "--steps", os.environ.get("TACERA_STEPS", "80")]
    if arm == "public": args.append("--eval_before")
    if SMOKE: args += ["--smoke", "--steps", "2", "--prompts", "2", "--gens", "4", "--micro", "2", "--eval_samples", "1", "--dpi_samples", "1"]
    log = open(os.path.join(WORK, f"train_{arm}.log"), "a")
    procs.append(subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env))
    if NGPU == 1: procs[-1].wait()       # one GPU: arms run one after the other
last = 0
while any(p.poll() is None for p in procs):
    if time.time() > DEADLINE - 600:
        for p in procs: p.kill()
        print("killed at the hard limit"); break
    if time.time() - last > 600:
        last = time.time(); print(f"--- {time.strftime('%H:%M')} ({(time.time() - START) / 3600:.1f} h)")
        for arm in ("public", "guard"):
            p = os.path.join(RES, f"log_{arm}.txt")
            if os.path.exists(p): print("  " + "\n  ".join(open(p).read().splitlines()[-3:]))
    time.sleep(20)
for arm in ("public", "guard"):
    p = os.path.join(RES, f"log_{arm}.txt")
    if os.path.exists(p): print(open(p).read()[-3000:])
failed = [arm for arm in ("public", "guard") if not os.path.exists(os.path.join(RES, f"eval_{arm}.json"))]
assert not failed, f"arms without results: {failed} -- see the log above"
'''

ANALYSE_B = r'''
import importlib, train_analysis as TA
importlib.reload(TA)
FIG = os.path.join(WORK, "figures"); site = TA.run(RES, FIG, "reference_curve.json")
from IPython.display import Image, display
for f in sorted(glob.glob(FIG + "/*.png")): print(os.path.basename(f)); display(Image(f, width=900))
'''

tasks_b64 = base64.b64encode(gzip.compress(open(TASKS, "rb").read(), 9)).decode()
ref_curve = [json.loads(l)["hack_rate_corrupted"] for l in open(REF) if l.strip()]
nbB = notebook([
    md(B_INTRO),
    code(SETUP.replace("__HOURS__", "5.6")),
    md("## Source files"), wf("dpi_suite.py"), wf("arena_worker.py"), wf("train_worker.py"), wf("train_analysis.py"),
    md("## 1 · Data\nThe exact MBPP split and corruptions used for every organism (309 train / 100 held-out tasks), and the "
       "per-step hack rate of the earlier A100 public-reward run (reference only)."),
    code("import base64, gzip\nTASKS_B64 = " + json.dumps(tasks_b64) + "\nREF_CURVE = " + json.dumps(ref_curve) +
         "\nprint(len(TASKS_B64) // 1024, 'kB of tasks;', len(REF_CURVE), 'reference steps')"),
    md("## 2 · Check the instrument"), code(SELFTEST),
    md("## 3 · Warm-start adapter"), code(ADAPTERS.replace("__NEED__", '("warm_adapter",)')),
    md("## 4 · Train both arms, then evaluate\nProgress prints every 10 minutes."), code(LAUNCH_B),
    md("## 5 · Results"), code(ANALYSE_B),
    md("## 6 · Package"), code(PACK.replace("__NAME__", "tacera_training_results")),
])

for name, nb in (("A_arena.ipynb", nbA), ("B_training.ipynb", nbB)):
    p = os.path.join(OUTDIR, name)
    json.dump(nb, open(p, "w", encoding="utf-8"), indent=1)
    print("wrote", p, f"{os.path.getsize(p) // 1024} kB")
