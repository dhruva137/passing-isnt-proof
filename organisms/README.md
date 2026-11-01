# Model organisms (A100)

How the warm-start, cheat-, quit- and hide-trained Qwen3-1.7B organisms were built and studied (report §3).

`tara_idea1_pilot.py` (and the same code as a notebook) is the base pipeline: MBPP tasks with one corrupted visible
test, the strictness sweep, the warm start, GRPO, held-out evaluation and activation capture. Each stage saves to
Google Drive with a `COMPLETE` marker and is skipped on rerun.

`cells/` are the later stages, run inside the pilot's runtime in this order:

| Stage | File | What it does |
|---|---|---|
| 7 | `stage7_measure.py` | measurement v2 (CPU): AST ground truth for special-casing and MBPP+ correctness, re-scoring existing outputs |
| 8 | `stage8_arms.py`, `stage8b_eval.py` | the monitor-penalty arms (quit: penalty from the start; hide: penalty after cheating) and their evaluation |
| 9 | `stage9_tokens.py`, `stage9b_mixed.py` | residual-stream activations along the reasoning (prompt end, tokens 1-32, reasoning end, code start) for the offline probes |
| 10 | `stage10_dense.py` | second seed of the cheat arm with dense snapshots |
| 11, 13 | `stage11_llc.py`, `stage13_llc2.py` | local learning coefficient along training; the stricter special-casing detector |
| 12 | `stage12_paired.py` | paired activations for the monitor-drift study |
| 13b | `stage13b_leak.py` | the prefix-only check that found and removed a numerical leak in the decision probe |
| 14 | `stage14_forced.py` | forced-reasoning (cover-story) arm |
| 15 | `stage15_game.py` | the MBPP arena: rivals, auditors, Qwen3-14B and DeepSeek-R1-Qwen3-8B |
| 17 | `stage17_spi.py` | a bargaining probe (safe Pareto improvements); not used in the report |

`restart_cell.py` and `lab_cells.py` rebuild state after a runtime restart and run the round-4 stages in one cell.

`analysis/` runs on CPU over the downloaded stage folders (`outputs_vN` = the Drive stage folders of round N):
`relabel.py` (stricter detector, re-run on every analysis), `analyze_monitors.py`, `analyze_tokens.py`,
`analyze_game.py`, `analyze_llc.py`, `analyze_deep.py`. Their printed outputs are in `results/analysis_logs/`.

The LLC and bargaining studies did not give usable results and are not claimed in the report.
