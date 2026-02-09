# mccviahat

Thesis research project investigating whether **prompt content** (emotional vs neutral) produces measurable differences in **hardware-level substrate signals** during LLM inference on bare-metal nodes.

## Setup

Runs on fresh **CloudLab** nodes (bare-metal Linux, no GPU). Requires Docker and `perf`.

```bash
# 1. Bootstrap node (Docker, perf permissions)
bash scripts/bootstrap_node.sh
newgrp docker

# 2. Download model + build image + start server
bash scripts/model_fetch.sh
bash scripts/llama_build.sh
bash scripts/llama_up.sh
```

## Run an experiment

```bash
python3 scripts/run_prompts_json.py --json prompts/neutral.json   --label neutral
python3 scripts/run_prompts_json.py --json prompts/emotional.json --label emotional
```

Each run creates a directory under `runs/<timestamp>_<label>/` containing:

| File | Description |
|---|---|
| `perf_stat.txt` | 1 ms perf-stat buckets (IRQs, softIRQs, TLB flushes, context-switches, page-faults, energy, thermal) |
| `proc_system_sample.csv` | ~200 ms snapshots of /proc/interrupts, /proc/softirqs, PSI pressure, net/disk I/O, CPU freq |
| `proc_sample.csv` | ~200 ms process-level CPU & memory for the LLM container |
| `responses.jsonl` | Per-prompt request timing and LLM response |
| `kernel_log.txt` | Kernel log slice (MCE, hardware errors) |
| `collector_meta.json` | Collector config and timestamps |
| `meta.json` | Run config (label, PID, timing budget) |

## Analyze

Copy `runs/` to your local machine, then open `analysis/hat_substrate_compare.ipynb` and point `BASE_DIR` at the data.

## Project structure

```
prompts/              # Prompt stimuli (5 emotional, 5 neutral)
collectors/           # substrate_collector.py — perf + /proc + /sys sampler
scripts/              # Node bootstrap, Docker lifecycle, experiment runner
docker/llama_server/  # Dockerfile for llama.cpp CPU server
analysis/             # Jupyter notebook for data exploration
```

## Timing model

Each run uses a **fixed 44 s window** (2 s baseline → 40 s prompt budget → 2 s tail) so that emotional and neutral runs are directly comparable regardless of how many prompts complete.
