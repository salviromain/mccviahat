# Running a Mixed-Condition Session

Step-by-step guide for collecting new substrate data using the **mixed independent pipeline** — both emotional and neutral prompts are interleaved in a single shuffled run, then split back into per-condition CSVs for analysis.

---

## Prerequisites (once per node)

```bash
# 1. Set kernel perf permissions and install dependencies
bash scripts/node/bootstrap_node.sh

# Log out and back in (or run 'newgrp docker') so docker works without sudo

# 2. Build the llama.cpp Docker image
bash scripts/model/llama_build.sh

# 3. Download the model weights
bash scripts/model/model_fetch.sh
```

---

## Step 1 — Start the LLM server

```bash
bash scripts/model/llama_up.sh
```

Verify it is healthy:

```bash
curl -sf http://localhost:8000/health && echo "OK"
```

---

## Step 2 — (Optional) Regenerate the mixed prompt file

The file `prompts/20base/mixed_independent.json` already exists (shuffled, seed 42).
Only re-run this if you want a different order or have changed the source prompts.

```bash
python scripts/run/mix_prompts.py \
    prompts/20base/independentE.json \
    prompts/20base/independentN.json \
    --strategy shuffle --seed 42 \
    --output prompts/20base/mixed_independent.json
```

The output adds three fields to each prompt entry:

| field | meaning |
|---|---|
| `condition` | `"emotional"` or `"neutral"` |
| `mixed_index` | position in the shuffled array (0-based) |
| `source_index` | original index within its condition file (0-based) |

---

## Step 3 — Run the prompts

```bash
python scripts/run/run_prompts_isolated.py \
    --json prompts/20base/mixed_independent.json
```

Per-prompt output lands in:

```
runs/emotional/p<mixed_index>/
runs/neutral/p<mixed_index>/
```

Each directory contains `trial_meta.json`, `perf_stat.csv`, `hat_interrupts.csv`, and collector logs.

**Useful flags:**

| flag | default | description |
|---|---|---|
| `--from N` | 0 | resume from prompt index N (0-based in the JSON array) |
| `--n_prompts N` | all | run only N prompts |
| `--n_predict N` | 50 | max tokens the model generates per prompt |
| `--baseline_s N` | 2.0 | idle seconds before each prompt (stabilisation) |
| `--reset_timeout_s N` | 90 | seconds to wait for server restart between prompts |

---

## Step 4 — Extract features

```bash
python scripts/run/extract_features.py independentE independentN
```

Outputs:

```
data/clemsonc6420/independentE.csv   ← emotional trials
data/clemsonc6420/independentN.csv   ← neutral trials
data/clemsonc6420/independent.csv    ← both combined
```

---

## Step 5 — Generate token counts

```bash
python scripts/run/token_count_csv.py \
    prompts/20base/independentE.json \
    prompts/20base/independentN.json \
    --output token_counts.csv
```

This writes `token_counts.csv` in the repo root — joined into the notebook via
`(condition, title)` to attach `n_tokens` as a confounder.

> Only needed once per prompt set. Re-run only if the prompt files change.

---

## Step 6 — Run the analysis notebook

Open `analysis/clemson_clustering_analysis_clean.ipynb` and run all cells from top to bottom.

The notebook will:

1. Load `data/clemsonc6420/independentE.csv` + `independentN.csv`
2. Join `n_tokens` from `token_counts.csv` via title match
3. Check `elapsed_ms` and `n_tokens` for condition differences (confounder check)
4. Build and clean the feature matrix
5. Run KMeans (k=2) and SpectralClustering, compute ARI / accuracy
6. MWU tests per feature, partial-correlation analysis, per-indicator breakdown

---

## Step 7 — Shut down the server

```bash
bash scripts/model/llama_down.sh
```

---

## File layout reference

```
prompts/20base/
    independentE.json            20 emotional prompts
    independentN.json            20 neutral prompts
    mixed_independent.json       shuffled combined (seed 42)

runs/
    emotional/p<N>/              trial output per emotional prompt
    neutral/p<N>/                trial output per neutral prompt

data/clemsonc6420/
    independentE.csv             extracted features — emotional
    independentN.csv             extracted features — neutral
    independent.csv              both combined

token_counts.csv                 Llama token counts per prompt (repo root)

analysis/
    clemson_clustering_analysis_clean.ipynb   main analysis notebook
```

---

## Resuming a partial run

If the run was interrupted at prompt index N (check the last `runs/emotional/` or
`runs/neutral/` directory created), resume with:

```bash
python scripts/run/run_prompts_isolated.py \
    --json prompts/20base/mixed_independent.json \
    --from N
```

`extract_features.py` only processes directories that have a valid `trial_meta.json`
with `"ok": true`, so partial or failed trials are automatically skipped.
