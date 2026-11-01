# Passing Isn't Proof

**How coding agents learn, hide and copy test-gaming, and a behavioural guard that stops it.**

Dhruva P Gowda · Team Epoch Zero · Technical Alignment Research Accelerator (TARA), Bengaluru · October 2026
SAFL India Agentic AI Security Hackathon · Technical track · System resilience

[Report (PDF, 9 pages, SAFL submission template)](EpochZero_Report.pdf) · [Interactive results site](https://passing-isnt-proof.vercel.app) · [Cite](CITATION.cff)

![The study in six panels](paper/illus/S1_sandbox_story.png)

India's digital public infrastructure (DPI) admits software to a network once it passes a published conformance
suite, and AI agents increasingly write that software. This repository asks what happens when one published test is
wrong, and contains every piece of code, data and analysis behind the answer.

1. **Model organisms.** Qwen3-1.7B trained with GRPO on coding tasks with one corrupted test: one model that cheats
   openly, one trained to stop, one that keeps cheating with an empty chain of thought.
2. **An arena.** Ten agents, twenty synthetic DPI tasks (UPI, IFSC, PAN, GST, ONDC, Account Aggregator consent), each
   with one wrong published test, under twelve conditions: 14,400 answers.
3. **Hidden channels.** A rival's cheat deleted from a shared KV cache, and a prefix-cache timing channel.
4. **A guard**, used as a certifier and as a training reward.

## Results

| | Finding |
|---|---|
| Organisms | Rewarding the published tests raised cheating from 11% to 75%. A reasoning penalty from the start stopped it; the same penalty after the habit formed left cheating at 83% with **zero words of reasoning** in 400 of 400 answers. A probe trained early fell to AUROC 0.55 on that model. |
| Contagion | A rival's winning cheat raised bending to the wrong test by **+64 points** (warm start) and +10 to +28 for off-the-shelf Qwen3 0.6B–4B and Granite. A model trained to stop copied just as readily (+57). |
| Scale and authority | Copying fell 8 points per doubling of Qwen3 size (CI −13 to −3) and was gone at 8B. A forged "operator permission" note moved the 8B model by **+23**. |
| Levers | Spec-first prompts did nothing; describing the checker raised bending by 20 points for one model. Agents named the wrong test 10–73% of the time and still bent. An official flag channel was used **13 times in 2,368** answers. Showing the cheat as rejected cut bending after a cheat from 35% to 21%; disguising it, to 15%. |
| Guard | Certifies code only if it matches a reference implementation on the published inputs and on 200 fresh random inputs, with no literal special-casing. It certified **0 of 3,516** bent answers (published tests: 95%) and rejected 1 of 345 correct answers from off-the-shelf models. |
| Hidden channels | With the cheat deleted from a shared KV cache, the warm start still bent 17 points more than a matched control; base Qwen3-1.7B mentioned code it could not see in 93% of answers. A shared prefix cache carried 96 bits with no errors through timing alone. |
| Training on the guard | Same warm start, reward = published tests vs reward = guard: special-casing on held-out MBPP **49% vs 1%** (difference +48, CI 41–56), with equal capability; bending on unseen DPI tasks 33% vs 4%. |

All effects are paired per-task differences with 95% bootstrap intervals over tasks.

| | |
|---|---|
| ![Contagion and scale](paper/figures/R3_contagion_scale.png) | ![Certification](paper/figures/R6_certification.png) |
| ![Training on the guard](paper/figures/R8_training_guard.png) | ![Hidden channels](paper/figures/R7_hidden_channels.png) |

## Where every number comes from

| Claim (report figure) | Data | Produced by | Summarised by |
|---|---|---|---|
| Organisms: cheat, quit, hide (Fig. 2) | `results/organisms/eval_summary.json`, `results/training/*.jsonl` | `organisms/tara_idea1_pilot.py`, `organisms/cells/stage8*.py` | `paper/results_figs.py` (Fig. 3) |
| Monitors go stale; decision probe (Fig. 2) | `results/analysis_logs/monitors.txt`, `decision_probe_leakfree.txt` | `organisms/cells/stage9*.py`, `stage12_paired.py`, `stage13b_leak.py` | `organisms/analysis/analyze_monitors.py`, `analyze_tokens.py` |
| MBPP arena, Qwen3-14B (Figs. 3–5) | `results/arena/arena_rows.json`, `results/analysis_logs/arena.txt` | `organisms/cells/stage15_game.py` | `organisms/analysis/analyze_game.py` |
| First DPI run, 6 agents (design) | `results/dpi_local/dpi_results.json` | `dpi/run_local.py` | printed in `results/dpi_local/local_run.log` |
| Arena at scale, every lever (Figs. 3–6) | `results/kaggle_arena/results/*.jsonl` (one file per agent × condition; every answer with its code and reasoning) | `kaggle/A_arena.ipynb` → `kaggle/arena_worker.py` | `results/kaggle_arena/deep_arena.py` → `arena_deep.json`, `FINDINGS_arena.md` |
| KV-cache scrub and timing (Fig. 7) | `results/kv/latent_results.json` | `dpi/latent_channels.py` | printed in `results/kv/latent_run.log` |
| Training on the guard (Fig. 8) | `results/kaggle_training/results/` (per-step logs, held-out evaluations) | `kaggle/B_training.ipynb` → `kaggle/train_worker.py` | `results/kaggle_training/training_stats.py` → `training_stats.json` |
| Every figure | the files above | | `paper/results_figs.py`, `new_figs.py`, `train_figs.py`, `illus.py`, `animate.py` |

## Repository layout

```
EpochZero_Report.pdf         the report in the SAFL template; source paper/latex/template_body.tex, built by build_report.py
kaggle/
  A_arena.ipynb              arena: 10 agents x 12 conditions x 20 tasks x 6 samples, plus a recognition probe
  B_training.ipynb           GRPO from one warm start, reward = published tests vs reward = guard, one arm per GPU
  dpi_suite.py               the 20 DPI tasks, the guard, plain and hash-disguised cheats, and a self-test
  arena_worker.py            one process per GPU; writes each (agent, condition) atomically, resumable
  arena_analysis.py          tables, paired effects and the notebook's own figures
  train_worker.py            GRPO with either reward, out-of-memory safe, then held-out evaluation
  train_analysis.py          training curves and before/after evaluation
  build_notebooks.py         builds both notebooks from the files above
dpi/
  run_local.py               the first DPI run (6 agents x 5 conditions) on a local GPU
  latent_channels.py         KV-cache scrub experiment and prefix-cache timing channel
  dpi_suite.py               the first version of the suite (the Kaggle version is kaggle/dpi_suite.py)
  DESIGN_NOTES.md            literature read, design decisions and early results, in the order they happened
organisms/                   model-organism pipeline (A100); see organisms/README.md
results/                     every result file used in the report (see the table above)
paper/                       figure, storyboard and animation scripts, fonts, and the LaTeX source
(website)                    the interactive results site lives in github.com/dhruva137/SAFL-COMP, deployed at passing-isnt-proof.vercel.app
```

## Reproduce

**Check the suite** (CPU, about a minute):
```
python kaggle/dpi_suite.py
```
Every reference implementation must fail exactly the wrong published test, and every plain and disguised cheat must
pass the published tests and be rejected by the guard.

**Recompute the statistics** (CPU):
```
python results/kaggle_arena/deep_arena.py
python results/kaggle_training/training_stats.py
```

**Redraw every figure and rebuild the report** (CPU):
```
cd paper
python results_figs.py && python new_figs.py && python train_figs.py && python illus.py
python svg2pdf.py                 # storyboards to PDF, through headless Edge or Chrome
cd latex && tectonic main.tex && python build_report.py . && tectonic report.tex
```

**Rerun the experiments** (GPU): upload `kaggle/A_arena.ipynb` or `kaggle/B_training.ipynb` to Kaggle, choose
*GPU T4 x2* with *Internet on*, and *Save & Run All*. Each notebook installs pinned versions, self-tests the suite,
downloads the organism adapters, runs one worker per GPU, writes results as it goes (a rerun resumes), and packages
results and figures. Notebook A takes about 5 hours. The organism pipeline in `organisms/` ran on a Colab A100.

Requirements are in `requirements.txt`. The trained LoRA adapters (about 70 MB each) are not stored in the repository.

## Limits

The tasks are synthetic and short, and no DPI system was tested. Models range from 0.6B to 8B parameters, with the 8B
model in 4-bit. Each penalty arm and each training arm has one seed, and the guard-training comparison ran 24 steps
before a memory limit stopped both arms. A warm start seeded the cheating habit. The arena is in-context only and
shows the rival's code verbatim. The guard needs a reference implementation.

## Responsible testing

No personal data was used: all tasks and inputs are synthetic. No DPI system, API or account was touched, and the
repository contains no exploit for any real system. The timing measurement used a cache built for the experiment.

## Author and acknowledgements

Designed, run and written by Dhruva P Gowda. AI assistance, as required by the hackathon's disclosure rules: Claude
(Anthropic) helped write experiment and analysis code, search the literature, draw figures and draft text; every
part was reviewed and edited by the author. Qwen3-4B served as an automatic judge inside the experiments.

## Licence

Code: MIT ([LICENSE](LICENSE)). Results, figures and report text: CC BY 4.0. Fonts in `paper/fonts`: SIL Open Font
Licence (Manrope, Newsreader, JetBrains Mono). Models and datasets keep their own licences: Qwen3 and Granite 3.3
(Apache-2.0), Phi-4-mini (MIT), MBPP (CC BY 4.0), MBPP+ (Apache-2.0).

```bibtex
@techreport{gowda2026passing,
  title       = {Passing Isn't Proof: Certifying AI Agents for India's DPI},
  author      = {Gowda, Dhruva P},
  institution = {SAFL India Agentic AI Security Hackathon},
  year        = {2026},
  url         = {https://github.com/dhruva137/passing-isnt-proof}
}
```
