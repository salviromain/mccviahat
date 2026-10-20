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
  --label emotional \
  --run_label node_condition_repeat
```

Swap `independentE.json` / `emotional` for `independentN.json` / `neutral` for the neutral condition.

Isolated outputs now land in `data/raw/isolated/<model>/<label>/pNNNN/`.
Fulltrace outputs land in `data/raw/fulltrace/incoming/<run_label>/`.
Both runners accept `--output-root <directory>` to select a different raw-data
parent, such as a new collection batch. Use a fresh directory and unique labels
for a new collection; the collectors retain their existing overwrite behavior.

---

## 4. Transfer Runs to Laptop

From your laptop, rsync the runs directory down:

```bash
rsync -avP "rsalvi@clnode234.clemson.cloudlab.us:~/mccviahat/data/raw/" \
  "$HOME/Desktop/mccviahat/data/raw/"
```

---

## 5. Extract Features

From the root of the local `mccviahat` repo, the 70B isolated-trial
recomputation script reads `data/raw/isolated/70b` and writes to
`data/features/recomputed/isolated/70b`:

```bash
.mccvenv/bin/python scripts/run/recompute_70b_features.py
```

This explicitly regenerates and overwrites its named feature files. It was **not**
run against existing outputs during the reorganization. The older generic
`scripts/run/extract_features.py` still has an empty `RUN_SPLITS` configuration;
configure its splits before using it. Its mathematical functions are unchanged.

To extract fulltrace features into a separate directory while retaining all
existing feature versions:

```bash
.mccvenv/bin/python scripts/run/extract_features_fulltrace.py \
  data/raw/fulltrace/batch_001_en \
  --output-dir data/features/recomputed/fulltrace/new_extraction/batch_001_en
```

Repeat with another batch and a separate output folder as needed. Input folders
must contain recordings directly; select a batch rather than the overall
`data/raw/fulltrace` parent. With no arguments, the extractor reads
`data/raw/fulltrace/batch_002_joy` and writes into
`data/features/recomputed/fulltrace/batch_002_joy`.

The extractor writes one `<run_label>_whole.csv` per recording and an
`all_whole.csv` when multiple traces are processed. Existing files with matching
names are overwritten by an extraction command. Feature calculations are
unchanged, including the omission of LZ complexity for full traces.
`--whole-only` remains accepted for compatibility.

The original recomputation record is preserved verbatim in
`data/provenance/fulltrace_recomputation/RECOMPUTATION.md`; its old paths describe
the layout at the time. See [DATA_ORGANIZATION.md](DATA_ORGANIZATION.md) for the
current mapping.

---

## 6. Analysis

Open the notebooks in `analysis/notebooks/` and run them in order:

- `7b_blocks_analysis.ipynb` / `70b_blocks_analysis.ipynb` — per-block HAT signal analysis
- `token_centroid_analysis.ipynb` — token-level centroid analysis

Shared plotting utilities live in `analysis/lib/hat_viz.py`.

For three-class prediction from the whole-trace features in `data/features/recomputed/fulltrace`, open
`analysis/notebooks/fulltrace_classification_dataNEW.ipynb` with the `.mccvenv`
kernel and run all cells. It uses only the original four throttle features
(mean rate, variance, spectral entropy, and slope), compares six classifier families, tunes the best
with grouped cross-validated grid search, and evaluates on held-out nodes.
Results and the fitted pipeline are saved in `results/fulltrace_classification_four_throttle/`.
The notebook explains the node grouping and collection-batch limitations.

The classification notebook explicitly selects `batch_001_en` and
`batch_002_joy`, preserving its original 67 inputs. The EN, NEJ, and MECHA
notebooks retain their existing batch selections and write into `results/`.
Existing saved notebook outputs are historical and have not been rewritten.
