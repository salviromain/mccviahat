# Running Experiments

This document describes the full pipeline from bare-metal node setup to analysis on your laptop.

---

## 1. Node Setup

SSH into the bare-metal node (CloudLab), then clone the repo and switch to the working branch:

```bash
git clone https://github.com/salviromain/mccviahat
cd mccviahat
git switch sequence_analysis
```

Bootstrap the node (installs dependencies, configures Docker and perf permissions):

```bash
bash scripts/node/bootstrap_node.sh
newgrp docker
```

> `newgrp docker` reloads your shell session with Docker group membership. You only need this once per login.

---

## 2. Model Setup

Fetch the model weights from HuggingFace and build the llama Docker image:

```bash
export HF_TOKEN="your_token_here"
bash scripts/model/model_fetch.sh 70b   # or 7b
bash scripts/model/llama_build.sh
```

---

## 3. Run the Collector + Prompts

Start the llama server, then run one of the two collection scripts depending on your experiment type.

**Isolated** (one collector process scoped per prompt):
```bash
python3 scripts/run/run_prompts_isolated.py \
  --json prompts/20base/independentE.json \
  --label emotional
```

**Full trace** (single collector spans the entire run):
```bash
python3 scripts/run/run_prompts_fulltrace.py \
  --json prompts/20base/independentE.json \
  --label emotional
```

Swap `independentE.json` / `emotional` for `independentN.json` / `neutral` for the neutral condition.

Run outputs land in `runs/` on the node.

---

## 4. Transfer Runs to Laptop

From your laptop, rsync the runs directory down:

```bash
rsync -avP "rsalvi@clnode234.clemson.cloudlab.us:~/mccviahat/runs/" \
  "$HOME/Desktop/mccviahat/runs/"
```

---

## 5. Extract Features

From the root of the local `mccviahat` repo:

```bash
python3 scripts/run/extract_features.py
```

This processes the raw collector output in `runs/` and writes feature files used by the notebooks.

To recompute **whole-trace features only** for all fulltrace experiments in both
collections, saving directly in `dataNEW/`:

```bash
.mccvenv/bin/python scripts/run/extract_features_fulltrace.py \
  runsNEW/fulltrace runs/fulltrace --whole-only --output-dir dataNEW
```

This writes one `<run_label>_whole.csv` per experiment and a combined
`all_whole.csv`. It skips per-prompt extraction. Existing whole-trace CSVs with
matching names are overwritten; existing per-prompt files are not removed.
Feature definitions are unchanged, including the omission of LZ complexity for
full traces. See `dataNEW/RECOMPUTATION.md` for the recorded batch and validation.

---

## 6. Analysis

Open the notebooks in `analysis/notebooks/` and run them in order:

- `7b_blocks_analysis.ipynb` / `70b_blocks_analysis.ipynb` — per-block HAT signal analysis
- `token_centroid_analysis.ipynb` — token-level centroid analysis

Shared plotting utilities live in `analysis/lib/hat_viz.py`.
